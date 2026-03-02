import os
import json
import base64
import io
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoProcessor

# ================= 配置区域 =================
# 建议保持 trust_remote_code=True，以确保加载最新的 InternVL2.5 特性
# 虽然本地有文件夹，但本地版本可能旧于 2.5 的权重文件
MODEL_PATH = "OpenGVLab/InternVL2_5-8B" 
TSV_PATH = "./data/RealWorldQA.tsv"
OUTPUT_FILE = "./results/internvl_rwqa_results.jsonl"
# ==========================================

def format_realworldqa_prompt(row):
    """构造标准 Prompt"""
    question = str(row['question'])
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
    print(f"🚀 [InternVL] Loading Model & Processor: {MODEL_PATH}")
    
    # 1. 加载模型 (Model)
    # 使用 AutoModel 加载，它会自动对应到 modeling_internvl.py 中的结构
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True # 推荐 True，防止本地 transformers 版本过老不兼容 2.5 权重
    ).eval().cuda()

    # 2. 打印它的真实类名
    print(type(model).__name__)

    # 3. 打印它对应的物理文件路径 (核心攻击点！)
    import inspect
    print(inspect.getfile(type(model)))

    # 2. 加载处理器 (Processor) - 这就是你截图里 processing_internvl.py 的入口
    # 它会自动处理切图、归一化、Token 拼接，不用我们手写了！
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 确保使用 InternVL 2.5 的高分辨率配置 (max_dynamic_patch=12)
    # 这一步是为了保险，防止默认配置过低
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_dynamic_patch = 12

    # 3. 读取数据 (Pandas)
    if not os.path.exists(TSV_PATH):
        print(f"❌ Error: File not found {TSV_PATH}")
        return

    print(f"📖 Reading TSV with Pandas: {TSV_PATH}")
    try:
        df = pd.read_csv(TSV_PATH, sep='\t', on_bad_lines='skip')
    except Exception as e:
        print(f"⚠️ Pandas failed, retrying with python engine: {e}")
        df = pd.read_csv(TSV_PATH, sep='\t', engine='python')
    
    print(f"📊 Total samples: {len(df)}")

    # 4. 准备结果文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)

    success_count = 0
    fail_count = 0

    # 5. 推理循环
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        try:
            # --- A. 获取图片 (Base64) ---
            image_content = row.get('image', None)
            pil_image = None
            
            if isinstance(image_content, str) and len(image_content) > 100:
                try:
                    image_bytes = base64.b64decode(image_content)
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    if fail_count < 3: print(f"⚠️ [ID:{idx}] Base64 decode failed: {e}")
                    fail_count += 1
                    continue
            elif isinstance(image_content, str) and os.path.exists(image_content):
                pil_image = Image.open(image_content).convert("RGB")
            else:
                if fail_count < 3: print(f"⚠️ [ID:{idx}] No valid image data found.")
                fail_count += 1
                continue

            # --- B. 构造 Prompt ---
            question_text = format_realworldqa_prompt(row)
            
            # 构造符合 Processor 要求的消息格式
            # Processor 会自动寻找 <image> 占位符并替换为 visual tokens
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"}, 
                        {"type": "text", "text": question_text}
                    ]
                }
            ]
            
            # 应用对话模版 -> "<|im_start|>user\n<image>\nQuestion...<|im_end|>\n<|im_start|>assistant\n"
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            # --- C. 处理输入 (The Magic Step) ---
            # 这一步直接调用了你截图里 processing_internvl.py 的逻辑
            inputs = processor(
                text=text_prompt,
                images=pil_image,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")

            # --- D. 生成 (Generate) ---
            # 使用标准的 generate 接口，避开 chat() 的 bug
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id
                )

            # --- E. 解码与保存 ---
            # 这里的切片是为了去掉输入的 prompt 部分，只保留回答
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            # 结果整理
            line_dict = row.to_dict()
            if 'image' in line_dict: del line_dict['image'] # 移除 Base64
            
            for k, v in line_dict.items():
                if isinstance(v, (np.int64, np.int32)): line_dict[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)): line_dict[k] = float(v)

            res = {
                "question_id": idx,
                "question": question_text,
                "annotation": line_dict,
                "task": "RealWorldQA",
                "result": {
                    "gen": response.strip(),
                    "gen_raw": response
                }
            }

            with open(OUTPUT_FILE, 'a') as f:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
            success_count += 1

        except Exception as e:
            if fail_count < 5: print(f"❌ Error at index {idx}: {e}")
            fail_count += 1
            continue

    print(f"\n✅ Completed! Success: {success_count}, Failed: {fail_count}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()