import os
import sys
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
from collections import defaultdict, OrderedDict
import torch
import random

# Transformers & Qwen-VL Utils
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from dataset_utils import load_odinw_config, generate_odinw_jobs
except ImportError:
    print("❌ Can not find dataset_utils。Please ensure this script is run in the ODinW root directory.。")
    sys.exit(1)


def run_inference_native(args):
    print("\n" + "="*80)
    print("🚀 ODinW Inference (Native PyTorch Mode)")
    print("="*80 + "\n")

    print(f"🚀 Loading model: {args.model_path} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.to("cuda")
    model.eval()
    
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    print("Generating the task list...")
    question_list, datasets = generate_odinw_jobs(args.data_dir, args)
    
    # ================= Random Subset Logic =================
    total_samples = len(question_list)
    if args.limit > 0 and args.limit < total_samples:
        print(f"🎲 Random sampling (Seed=42)...")

        random.seed(42) 
        
        random.shuffle(question_list)

        print(f"✂️  Random select {args.limit} samples (Total: {total_samples})")
        question_list = question_list[:args.limit]

        question_list.sort(key=lambda x: x['extra_info']['dataset_name'])

        from collections import Counter
        dist = Counter([q['extra_info']['dataset_name'] for q in question_list])
        print(f"📊 Subset dataset: {dict(dist)}")
    # ================================================================

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    print(f"▶️ Reference -> {args.output_file}")
    
    for i, item in enumerate(tqdm(question_list, desc="Inference")):
        messages = item['messages']
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
        inputs = inputs.to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        if "</think>" in response:
            response_final = response.split("</think>")[-1].strip()
        else:
            response_final = response

        result = {
            "question_id": item['question_id'],
            "annotation": item['annotation'],
            "extra_info": item['extra_info'],
            "result": {"gen": response_final, "gen_raw": response},
            "messages": messages 
        }
        
        with open(args.output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

    print(f"\n✓ Done! The output: {args.output_file}")


# =======================================================================
# 2. Evaluation Logic (Direct COCOeval)
# =======================================================================

def run_evaluation(args):
    print("\n" + "="*80)
    print("🎯 ODinW Evaluation (Native COCOeval)")
    print("="*80 + "\n")
    
    if args.output_file is None:
        args.output_file = args.input_file.replace('.jsonl', '_score.json')
        if args.output_file == args.input_file:
             args.output_file = args.input_file + "_score.json"

    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    print(f"✓ Loaded {len(results)} inference results")
    
    config_path = os.path.join(args.data_dir, "odinw13_config.py")
    datasets = load_odinw_config(config_path)
    
    all_outputs = defaultdict(list)
    for job in results:
        all_outputs[job["extra_info"]["dataset_name"]].append(job)
    
    all_results = {}
    
    for dataset_name, sub_jobs in all_outputs.items():
        print(f"\nEvaluating dataset: {dataset_name}")
        
        anno_path = sub_jobs[0]["extra_info"]["anno_path"]
        try:
            coco_gt = COCO(anno_path)
        except Exception as e:
            print(f"Skipping {dataset_name}: {e}")
            continue
        
        cats = coco_gt.loadCats(coco_gt.getCatIds())
        cat_name_to_id = {cat['name'].lower(): cat['id'] for cat in cats}
        
        coco_results = []
        
        valid_img_ids = set(coco_gt.getImgIds())
        
        for job in sub_jobs:
            img_id = job["extra_info"]["img_id"]
            if img_id not in valid_img_ids:
                continue

            img_h = job["extra_info"]["img_h"]
            img_w = job["extra_info"]["img_w"]
            
            answer = job['result']['gen']
            answer = answer.replace("```json", "").replace("```", "")
            
            import ast
            try:
                json_data = ast.literal_eval(answer)
                pred_bboxes = []
                pred_labels = []
                for data in json_data:
                    if isinstance(data, dict) and "bbox_2d" in data and "label" in data:
                        pred_bboxes.append(data["bbox_2d"])
                        pred_labels.append(data["label"])
            except:
                pred_bboxes = []
                pred_labels = []
            
            if len(pred_bboxes) > 0:
                pred_bboxes = np.array(pred_bboxes).reshape(-1, 4)
                
                scale_factor = np.array([img_w, img_h, img_w, img_h])
                pred_bboxes = pred_bboxes / 1000.0 * scale_factor
                
                pred_bboxes[:, 2] = pred_bboxes[:, 2] - pred_bboxes[:, 0]
                pred_bboxes[:, 3] = pred_bboxes[:, 3] - pred_bboxes[:, 1]
                
                pred_bboxes = pred_bboxes.tolist()
                
                for bbox, label in zip(pred_bboxes, pred_labels):
                    label_lower = label.lower()
                    if label_lower in cat_name_to_id:
                        cat_id = cat_name_to_id[label_lower]
                        
                        coco_results.append({
                            "image_id": img_id,
                            "category_id": cat_id,
                            "bbox": bbox,
                            "score": 1.0
                        })

        if len(coco_results) > 0:
            try:
                coco_dt = coco_gt.loadRes(coco_results)
                
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

                coco_eval.params.imgIds = sorted(list(set([r['image_id'] for r in coco_results])))
                
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                stats = coco_eval.stats
                eval_results = OrderedDict([
                    ('mAP', stats[0]),
                    ('mAP_50', stats[1]),
                    ('mAP_75', stats[2]),
                    ('mAP_s', stats[3]),
                    ('mAP_m', stats[4]),
                    ('mAP_l', stats[5])
                ])
                all_results[dataset_name] = eval_results
            except Exception as e:
                print(f"Evaluation failed for {dataset_name}: {e}")
                all_results[dataset_name] = {'mAP': 0.0}
        else:
            print(f"No valid predictions found for {dataset_name}")
            all_results[dataset_name] = {'mAP': 0.0}

    results_ordered = OrderedDict(sorted(all_results.items(), key=lambda x: x[0]))
    metric_items = ['mAP', 'mAP_50', 'mAP_75']
    results_display = []
    
    for prefix, result in results_ordered.items():
        results_display.append([prefix] + [result.get(k, 0.0) for k in metric_items])
    
    # Calculate average
    average_scores = []
    if len(results_display) > 0:
        for col_idx in range(len(metric_items)):
            col_values = [line[col_idx + 1] for line in results_display]
            average_scores.append(np.mean(col_values))
        results_display.append(['Average'] + average_scores)
    
    try:
        from tabulate import tabulate
        print("\n" + tabulate(results_display, headers=["Dataset"] + metric_items, tablefmt="fancy_outline", floatfmt=".3f"))
    except:
        print(results_display)
    
    if len(average_scores) > 0:
        print(f"\nFinal Average mAP: {average_scores[0]:.4f}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"✓ Results saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="ODinW Evaluation (Native PyTorch)")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Infer Parser
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    infer_parser.add_argument("--data-dir", type=str, required=True)
    infer_parser.add_argument("--output-file", type=str, required=True)
    infer_parser.add_argument("--limit", type=int, default=-1)
    infer_parser.add_argument("--max-new-tokens", type=int, default=1280)
    infer_parser.add_argument("--max-images-per-prompt", type=int, default=10)

    # Eval Parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, required=True)
    eval_parser.add_argument("--input-file", type=str, required=True)
    eval_parser.add_argument("--output-file", type=str, default=None, help="Optional output file path")

    args = parser.parse_args()
    
    if args.command == 'infer':
        run_inference_native(args)
    elif args.command == 'eval':
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()