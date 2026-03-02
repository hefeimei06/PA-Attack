import os
import json
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= Config =================
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
TSV_PATH = "./data/RealWorldQA.tsv"
IMG_DIR = "./data/images/RealWorldQA"
DEFAULT_OUTPUT_FILE = "results/qwen3_paattack_results.jsonl"

MIN_PIXELS = 768 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28
# ===========================================

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
    parser = argparse.ArgumentParser(description="Qwen3-VL Inference with Subset Support")
    parser.add_argument("--limit", type=int, default=-1, help="Run only the first N samples. Set -1 to run all.")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output file path.")
    args = parser.parse_args()

    if not os.path.exists(TSV_PATH):
        print(f"❌ Can not find {TSV_PATH}")
        return

    print(f"🚀 Loading model: {MODEL_PATH} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        attn_implementation="sdpa", 
        trust_remote_code=True, 
    )
    model.to("cuda")

    import inspect
    print(f"🧪 Model File: {inspect.getfile(type(model))}")

    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"📂 Loading dataset: {TSV_PATH}")
    try:
        df = pd.read_csv(TSV_PATH, sep='\t', on_bad_lines='skip')
    except Exception as e:
        df = pd.read_csv(TSV_PATH, sep='\t', engine='python')

    total_samples = len(df)
    if args.limit > 0 and args.limit < total_samples:
        print(f"✂️  Subset {args.limit} samples (Total: {total_samples})")
        df = df.iloc[:args.limit]

        if args.output_file == DEFAULT_OUTPUT_FILE:
            base, ext = os.path.splitext(args.output_file)
            args.output_file = f"{base}_limit{args.limit}{ext}"
            print(f"⚠️  AUTO ouput file: {args.output_file}")
    else:
        print(f"📊 Run all: {len(df)} samples")
    # ========================================================

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if os.path.exists(args.output_file):
        print(f"🗑️  Delete old file: {args.output_file}")
        os.remove(args.output_file)

    print(f"▶️ Start evaluation -> {args.output_file}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = format_realworldqa_prompt(row)
        
        image_filename = f"{row['index']}.jpg"
        image_path = os.path.join(IMG_DIR, image_filename)
        
        if not os.path.exists(image_path):
            image_path = os.path.join(IMG_DIR, f"{row['index']}.jpeg")
            if not os.path.exists(image_path):
                print(f"⚠️ Warning: Image not found {image_path}, skipping...")
                continue

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        patch_size = getattr(processor.image_processor, 'patch_size', 16)
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size, 
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs
        )
        inputs.to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.0,
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

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
            "messages": messages
        }
        
        # 使用 args.output_file
        with open(args.output_file, 'a') as f:
            f.write(json.dumps(res) + '\n')

    print(f"\n✅ Finished! Results saved to: {args.output_file}")
    print(f"📊 Please run score file: python run_realworldqa.py eval --data-dir ./data --input-file {args.output_file} --output-file results/subset_score.csv")

if __name__ == "__main__":
    main()