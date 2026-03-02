from transformers import AutoModelForImageTextToText

# 1. 加载模型配置 (不需要下载权重，飞快)
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)

# 2. 打印它的真实类名
print(type(model).__name__)
# 输出通常是: Qwen2_5_VLForConditionalGeneration

# 3. 打印它对应的物理文件路径 (核心攻击点！)
import inspect
print(inspect.getfile(type(model)))
# 输出路径会指向: .../transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py