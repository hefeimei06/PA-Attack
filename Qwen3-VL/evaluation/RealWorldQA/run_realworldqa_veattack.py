import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
TSV_PATH = "./data/RealWorldQA.tsv"
IMG_DIR = "./data/images/RealWorldQA"
OUTPUT_FILE = "results/qwen3_veattack_results.jsonl"

# 官方默认分辨率参数
MIN_PIXELS = 768 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28
# ===========================================

def format_realworldqa_prompt(row):
    """
    模拟官方 build_realworldqa_prompt 的逻辑。
    """
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
        print(f"❌ 错误：未找到数据文件 {TSV_PATH}")
        return

    print(f"🚀 正在加载模型: {MODEL_PATH} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        attn_implementation="sdpa", 
        trust_remote_code=True, 
        # device_map="auto"
    )
    model.to("cuda")

    # 2. 打印它的真实类名
    print(type(model).__name__)

    # 3. 打印它对应的物理文件路径 (核心攻击点！)
    import inspect
    print(inspect.getfile(type(model)))

    model.eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"📂 读取数据集: {TSV_PATH}")
    # 增加容错参数，防止 bad lines 报错
    try:
        df = pd.read_csv(TSV_PATH, sep='\t', on_bad_lines='skip')
    except Exception as e:
        # 如果读取失败，尝试用 python 引擎
        df = pd.read_csv(TSV_PATH, sep='\t', engine='python')

    print(f"📊 总样本数: {len(df)}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print("▶️ 开始推理 (Native PyTorch)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = format_realworldqa_prompt(row)
        
        # [核心修正点] 
        # 不要用 row['image'] (那是Base64字符串)
        # 而是用 index 构造文件名。通常是 {index}.jpg
        image_filename = f"{row['index']}.jpg"
        image_path = os.path.join(IMG_DIR, image_filename)
        
        # 双重保险：检查图片文件是否存在
        if not os.path.exists(image_path):
            # 尝试一下 jpeg 后缀
            image_path = os.path.join(IMG_DIR, f"{row['index']}.jpeg")
            if not os.path.exists(image_path):
                # 如果还找不到，说明 dump_image 没成功，跳过
                print(f"⚠️ Warning: Image not found {image_path}, skipping...")
                continue

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image_path, # 这里传入的是真实的文件路径
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 动态获取 patch size
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
        
        # 官方对齐的生成参数
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

        # 结果写入
        line_dict = row.to_dict()
        # 清理 Base64 字符串，防止 output 文件过大且没必要
        if 'image' in line_dict:
            line_dict['image'] = image_filename # 替换为文件名，而不是 Base64

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
        
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(res) + '\n')

    print(f"\n✅ 推理完成！")
    print(f"请运行评分命令：python run_realworldqa.py eval --data-dir ./data --input-file {OUTPUT_FILE} --output-file results/clean_score.csv")

if __name__ == "__main__":
    main()