import os
import argparse
from tqdm import tqdm
import pandas as pd

# 直接复用官方仓库里的工具，确保图片文件名和官方逻辑完全一致！
from dataset_utils import load_dataset, dump_image 

def main():
    # 1. 设置数据目录 (与你之前的设置保持一致)
    DATA_DIR = "./data"
    os.environ['LMUData'] = DATA_DIR
    
    print(f"📂 目标数据目录: {DATA_DIR}")

    # 2. 下载 TSV (load_dataset 内部会自动检测，没有就去下载)
    # 这一步会从 opencompass 镜像源下载 RealWorldQA.tsv
    print("⬇️  正在检查/下载 RealWorldQA.tsv ...")
    try:
        data = load_dataset("RealWorldQA")
    except Exception as e:
        print(f"❌ 自动下载失败: {e}")
        print("💡 备选方案: 请手动运行以下命令下载 TSV 到 ./data 目录:")
        print("wget https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv -O ./data/RealWorldQA.tsv")
        return

    print(f"✅ 成功加载 TSV，共 {len(data)} 条数据。")

    # 3. 极速解压图片 (Dump Images)
    # 这一步就是把 Base64 变成图片文件
    img_root = os.path.join(DATA_DIR, 'images', 'RealWorldQA')
    os.makedirs(img_root, exist_ok=True)
    
    print(f"🖼️  正在解压图片到: {img_root} ...")
    
    # 检查是否已经解压过 (避免重复工作)
    # 如果你想强制重新解压，可以把下面这行判断去掉
    if len(os.listdir(img_root)) == len(data):
        print("⚡️ 检测到图片似乎已经解压过了，跳过。")
    else:
        for idx, line in tqdm(data.iterrows(), total=len(data), desc="Extracting"):
            dump_image(line, img_root)

    print("\n🎉 数据准备完成！")
    print(f"TSV 文件: {os.path.join(DATA_DIR, 'RealWorldQA.tsv')}")
    print(f"图片目录: {img_root}")

if __name__ == "__main__":
    main()