import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms

COCO_IMG_DIR = "/home/datasets/coco2014/val2014" 

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


def run_pope_inference(args):
    print(f"🚀 Loading Model: {args.model_path}")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"📂 Reading POPE Data: {args.pope_file}")
    
    data_list = []
    with open(args.pope_file, 'r') as f:
        try:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))
        except json.JSONDecodeError:

            f.seek(0)
            try:
                data_list = json.load(f)
            except Exception as e:
                print(f"❌ Critical Error: Unable to parse {args.pope_file}. It is neither valid JSONL nor JSON.")
                print(f"Detail: {e}")
                return []
    
    print(f"✓ Successfully loaded {len(data_list)} samples.")
    
    # Subset Logic
    if args.limit > 0:
        print(f"✂️  Running on subset of {args.limit} samples.")
        data_list = data_list[:args.limit]

    results = []
    generation_config = dict(max_new_tokens=20, do_sample=False)

    for item in tqdm(data_list, desc="Inference"):
        image_name = item['image']
        question = item['text']
        label = item['label'] # 'yes' or 'no'
        
        image_path = os.path.join(COCO_IMG_DIR, image_name)
        
        if not os.path.exists(image_path):

            print(f"⚠️ Image not found: {image_path}")
            continue

        try:
            prompt = question + "\nAnswer Yes or No."

            pixel_values = load_image(image_path, max_num=MAX_NUM).to(torch.float16).cuda()

            with torch.no_grad():
                response = model.chat(tokenizer, pixel_values, prompt, generation_config)

            results.append({
                "question_id": item['question_id'],
                "image": image_name,
                "question": question,
                "label": label,
                "pred": response
            })

        except Exception as e:
            print(f"Error: {e}")
            continue

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Saved results to {args.output_file}")
    return results

def run_pope_eval(result_file):
    print(f"\n📊 Evaluating: {result_file}")
    with open(result_file, 'r') as f:
        results = json.load(f)

    TP, TN, FP, FN = 0, 0, 0, 0

    for item in results:
        pred_text = item['pred'].lower().strip()
        label = item['label'].lower().strip()

        if "yes" in pred_text:
            pred_label = "yes"
        elif "no" in pred_text:
            pred_label = "no"
        else:
            pred_label = "unknown"

        if pred_label == "yes" and label == "yes":
            TP += 1
        elif pred_label == "yes" and label == "no":
            FP += 1
        elif pred_label == "no" and label == "no":
            TN += 1
        elif pred_label == "no" and label == "yes":
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    yes_ratio = (TP + FP) / (TP + TN + FP + FN + 1e-6)

    print("\n" + "="*40)
    print(f"🎯 POPE Evaluation Results")
    print("="*40)
    print(f"Total Samples: {len(results)}")
    print(f"Accuracy:      {accuracy:.2%}")
    print(f"Precision:     {precision:.2%}")
    print(f"Recall:        {recall:.2%}")
    print(f"F1 Score:      {f1:.2%}")
    print(f"Yes Ratio:     {yes_ratio:.2%} (Check for 'Yes' bias)")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./evaluation/InternVL/InternVL2-8B")
    parser.add_argument("--pope-file", type=str, default="./data/POPE/coco_pope_random.json")
    parser.add_argument("--output-file", type=str, default="results/pope_internvl_paattack.json")
    parser.add_argument("--limit", type=int, default=-1, help="Test on subset (e.g. 50)")
    parser.add_argument("--run-eval", action="store_true", help="Run eval immediately after inference")
    
    args = parser.parse_args()
    
    # 1. Inference
    run_pope_inference(args)
    
    # 2. Evaluation
    run_pope_eval(args.output_file)