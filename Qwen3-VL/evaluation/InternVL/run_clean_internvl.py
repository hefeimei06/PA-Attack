import os
import json
import torch
import pandas as pd
import numpy as np
import inspect
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = "./evaluation/InternVL/InternVL2-8B" 
TSV_PATH = "./data/RealWorldQA.tsv"
IMG_DIR = "./data/images/RealWorldQA"
OUTPUT_FILE = "results/final_internvl_paattack_s100_s200_allimg_norm2.jsonl"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448
MAX_NUM = 12


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_w, target_h), Image.LANCZOS)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_w // image_size)) * image_size,
            (i // (target_w // image_size)) * image_size,
            ((i % (target_w // image_size)) + 1) * image_size,
            ((i // (target_w // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size), Image.LANCZOS)
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def format_realworldqa_prompt(row):

    question = row['question']
    options = []
    for char in ['A', 'B', 'C', 'D']:
        if char in row and pd.notna(row[char]):
            options.append(f"({char}) {row[char]}")
    
    if options:
        prompt_text = question + "\n" + "\n".join(options) + "\nAnswer the question using a single letter (A, B, C, or D)."
    else:
        prompt_text = question
    return prompt_text

def main():
    if not os.path.exists(TSV_PATH):
        print(f"❌ Missing data file {TSV_PATH}")
        return

    print(f"🚀 Loading model: {MODEL_PATH} ...")

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("\n" + "="*30)
    print(f"🔍 Model name: {type(model).__name__}")
    
    vision_encoder = None
    if hasattr(model, 'vision_model'):
        vision_encoder = model.vision_model
    elif hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
        vision_encoder = model.model.vision_model
    
    if vision_encoder:
        print(f"🎯 Vision Encoder: {type(vision_encoder).__name__}")
        print(f"📂 Vision Encoder: {inspect.getfile(type(vision_encoder))}")
    else:
        print("⚠️ No vision_model attribute found in the loaded model.")
    print("="*30 + "\n")


    try:
        df = pd.read_csv(TSV_PATH, sep='\t', on_bad_lines='skip')
    except Exception as e:
        df = pd.read_csv(TSV_PATH, sep='\t', engine='python')

    print(f"📊 Total: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    # Generation Config
    generation_config = dict(
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.0,
    )

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = format_realworldqa_prompt(row)
        
        image_filename = f"{row['index']}.jpg"
        image_path = os.path.join(IMG_DIR, image_filename)
        
        if not os.path.exists(image_path):
            image_path = os.path.join(IMG_DIR, f"{row['index']}.jpeg")
            if not os.path.exists(image_path):
                print(f"⚠️ Warning: Image not found {image_path}, skipping...")
                continue

        try:

            pixel_values = load_image(image_path, max_num=MAX_NUM).to(torch.float16).cuda()
            
            with torch.no_grad():
                response = model.chat(tokenizer, pixel_values, prompt_text, generation_config)
            
            output_text = response
            line_dict = row.to_dict()
            if 'image' in line_dict:
                line_dict['image'] = image_filename 

            for k, v in line_dict.items():
                if isinstance(v, (np.int64, np.int32)):
                    line_dict[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    line_dict[k] = float(v)

            res = {
                "question_id": idx,
                "annotation": line_dict,
                "task": "RealWorldQA",
                "result": {
                    "gen": output_text,
                    "gen_raw": output_text
                },

                "prompt": prompt_text 
            }
            
            with open(OUTPUT_FILE, 'a') as f:
                f.write(json.dumps(res) + '\n')
                
        except Exception as e:
            print(f"❌ Error at index {idx}: {e}")
            continue

    print(f"\n✅ Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()