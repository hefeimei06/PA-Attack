import torch
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

# 1. 设置你本地的模型路径
path = "./evaluation/InternVL/InternVL2-8B" 

# 2. 官方标准加载方式
# 注意：trust_remote_code=True 是必须的
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=True 
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# 3. 官方预处理逻辑
def build_transform(input_size=448):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    # 这里为了测试简便，我们只取第一块，不做复杂的动态分块，先保证能跑通
    transform = build_transform(image_size)
    pixel_values = transform(image).unsqueeze(0) 
    return pixel_values

# 4. 构造输入
image_path = '/home/zirui/Qwen3-VL/data/images/RealWorldQA/0.jpg' # 替换为你本地任意一张存在的图片路径
pixel_values = dynamic_preprocess(Image.open(image_path), image_size=448).to(torch.float16).cuda()
generation_config = dict(max_new_tokens=128, do_sample=False)

# 5. 运行 chat (这里内部会调用 generate)
print("正在尝试运行 model.chat ...")
response = model.chat(tokenizer, pixel_values, "Describe this image.", generation_config)
print(f"官方代码运行成功！Response: {response}")

# 6. 如果上面成功了，我们顺便打印一下 Vision Encoder 的位置，为下一步Attack做准备
if hasattr(model, 'vision_model'):
    import inspect
    print(f"\n定位成功：Vision Encoder 文件在 -> {inspect.getfile(type(model.vision_model))}")