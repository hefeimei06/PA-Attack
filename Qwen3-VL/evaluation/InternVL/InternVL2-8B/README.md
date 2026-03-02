---
license: mit
pipeline_tag: image-text-to-text
library_name: transformers
base_model:
  - OpenGVLab/InternViT-300M-448px
  - internlm/internlm2_5-7b-chat
new_version: OpenGVLab/InternVL2_5-8B
base_model_relation: merge
language:
  - multilingual
tags:
  - internvl
  - custom_code
---

# InternVL2-8B

[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVL)  [\[📜 InternVL 1.0\]](https://huggingface.co/papers/2312.14238)  [\[📜 InternVL 1.5\]](https://huggingface.co/papers/2404.16821)  [\[📜 Mini-InternVL\]](https://arxiv.org/abs/2410.16261)  [\[📜 InternVL 2.5\]](https://huggingface.co/papers/2412.05271)

[\[🆕 Blog\]](https://internvl.github.io/blog/)  [\[🗨️ Chat Demo\]](https://internvl.opengvlab.com/)  [\[🤗 HF Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL)  [\[🚀 Quick Start\]](#quick-start)  [\[📖 Documents\]](https://internvl.readthedocs.io/en/latest/)

<div align="center">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/64006c09330a45b03605bba3/zJsd2hqd3EevgXo6fNgC-.png">
</div>

## Introduction

We are excited to announce the release of InternVL 2.0, the latest addition to the InternVL series of multimodal large language models. InternVL 2.0 features a variety of **instruction-tuned models**, ranging from 1 billion to 108 billion parameters. This repository contains the instruction-tuned InternVL2-8B model.

Compared to the state-of-the-art open-source multimodal large language models, InternVL 2.0 surpasses most open-source models. It demonstrates competitive performance on par with proprietary commercial models across various capabilities, including document and chart comprehension, infographics QA, scene text understanding and OCR tasks, scientific and mathematical problem solving, as well as cultural understanding and integrated multimodal capabilities.

InternVL 2.0 is trained with an 8k context window and utilizes training data consisting of long texts, multiple images, and videos, significantly improving its ability to handle these types of inputs compared to InternVL 1.5. For more details, please refer to our [blog](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) and [GitHub](https://github.com/OpenGVLab/InternVL).

|      Model Name      |                                     Vision Part                                     |                                        Language Part                                         |                             HF Link                              |                                MS Link                                 |
| :------------------: | :---------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------------: |
|     InternVL2-1B     |    [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)    |            [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)            |     [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-1B)     |     [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-1B)     |
|     InternVL2-2B     |    [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)    |          [internlm2-chat-1_8b](https://huggingface.co/internlm/internlm2-chat-1_8b)          |     [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-2B)     |     [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-2B)     |
|     InternVL2-4B     |    [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)    |    [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)     |     [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-4B)     |     [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-4B)     |
|     InternVL2-8B     |    [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)    |          [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat)          |     [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-8B)     |     [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-8B)     |
|    InternVL2-26B     | [InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) |           [internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)           |    [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-26B)     |    [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-26B)     |
|    InternVL2-40B     | [InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) |       [Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B)       |    [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-40B)     |    [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-40B)     |
| InternVL2-Llama3-76B | [InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5) | [Hermes-2-Theta-Llama-3-70B](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B) | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B) | [🤖 link](https://modelscope.cn/models/OpenGVLab/InternVL2-Llama3-76B) |

## Model Details

InternVL 2.0 is a multimodal large language model series, featuring models of various sizes. For each size, we release instruction-tuned models optimized for multimodal tasks. InternVL2-8B consists of [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px), an MLP projector, and [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat).

## Performance

### Image Benchmarks

|          Benchmark           | MiniCPM-Llama3-V-2_5 | InternVL-Chat-V1-5 | InternVL2-8B |
| :--------------------------: | :------------------: | :----------------: | :----------: |
|          Model Size          |         8.5B         |       25.5B        |     8.1B     |
|                              |                      |                    |              |
|    DocVQA<sub>test</sub>     |         84.8         |        90.9        |     91.6     |
|    ChartQA<sub>test</sub>    |          -           |        83.8        |     83.3     |
|    InfoVQA<sub>test</sub>    |          -           |        72.5        |     74.8     |
|    TextVQA<sub>val</sub>     |         76.6         |        80.6        |     77.4     |
|           OCRBench           |         725          |        724         |     794      |
|      MME<sub>sum</sub>       |        2024.6        |       2187.8       |    2210.3    |
|         RealWorldQA          |         63.5         |        66.0        |     64.4     |
|     AI2D<sub>test</sub>      |         78.4         |        80.7        |     83.8     |
|      MMMU<sub>val</sub>      |         45.8         |        46.8        |     51.8     |
|  MMBench-EN<sub>test</sub>   |         77.2         |        82.2        |     81.7     |
|  MMBench-CN<sub>test</sub>   |         74.2         |        82.0        |     81.2     |
|    CCBench<sub>dev</sub>     |         45.9         |        69.8        |     75.9     |
|  MMVet<sub>GPT-4-0613</sub>  |          -           |        62.8        |     60.0     |
| MMVet<sub>GPT-4-Turbo</sub>  |         52.8         |        55.4        |     54.2     |
|          SEED-Image          |         72.3         |        76.0        |     76.2     |
|   HallBench<sub>avg</sub>    |         42.4         |        49.3        |     45.2     |
| MathVista<sub>testmini</sub> |         54.3         |        53.5        |     58.3     |
|  OpenCompass<sub>avg</sub>   |         58.8         |        61.7        |     64.1     |

- For more details and evaluation reproduction, please refer to our [Evaluation Guide](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html).

- We simultaneously use [InternVL](https://github.com/OpenGVLab/InternVL) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) repositories for model evaluation. Specifically, the results reported for DocVQA, ChartQA, InfoVQA, TextVQA, MME, AI2D, MMBench, CCBench, MMVet (GPT-4-0613), and SEED-Image were tested using the InternVL repository. MMMU, OCRBench, RealWorldQA, HallBench, MMVet (GPT-4-Turbo), and MathVista were evaluated using the VLMEvalKit.

### Video Benchmarks

|          Benchmark          | VideoChat2-HD-Mistral | Video-CCAM-9B | InternVL2-4B | InternVL2-8B |
| :-------------------------: | :-------------------: | :-----------: | :----------: | :----------: |
|         Model Size          |          7B           |      9B       |     4.2B     |     8.1B     |
|                             |                       |               |              |              |
|           MVBench           |         60.4          |     60.7      |     63.7     |     66.4     |
| MMBench-Video<sub>8f</sub>  |           -           |       -       |     1.10     |     1.19     |
| MMBench-Video<sub>16f</sub> |           -           |       -       |     1.18     |     1.28     |
|    Video-MME<br>w/o subs    |         42.3          |     50.6      |     51.4     |     54.0     |
|     Video-MME<br>w subs     |         54.6          |     54.9      |     53.4     |     56.9     |

- We evaluate our models on MVBench and Video-MME by extracting 16 frames from each video, and each frame was resized to a 448x448 image.

### Grounding Benchmarks

|             Model              | avg. | RefCOCO<br>(val) | RefCOCO<br>(testA) | RefCOCO<br>(testB) | RefCOCO+<br>(val) | RefCOCO+<br>(testA) | RefCOCO+<br>(testB) | RefCOCO‑g<br>(val) | RefCOCO‑g<br>(test) |
| :----------------------------: | :--: | :--------------: | :----------------: | :----------------: | :---------------: | :-----------------: | :-----------------: | :----------------: | :-----------------: |
| UNINEXT-H<br>(Specialist SOTA) | 88.9 |       92.6       |        94.3        |        91.5        |       85.2        |        89.6         |        79.8         |        88.7        |        89.4         |
|                                |      |                  |                    |                    |                   |                     |                     |                    |                     |
| Mini-InternVL-<br>Chat-2B-V1-5 | 75.8 |       80.7       |        86.7        |        72.9        |       72.5        |        82.3         |        60.8         |        75.6        |        74.9         |
| Mini-InternVL-<br>Chat-4B-V1-5 | 84.4 |       88.0       |        91.4        |        83.5        |       81.5        |        87.4         |        73.8         |        84.7        |        84.6         |
|       InternVL‑Chat‑V1‑5       | 88.8 |       91.4       |        93.7        |        87.1        |       87.0        |        92.3         |        80.9         |        88.5        |        89.3         |
|                                |      |                  |                    |                    |                   |                     |                     |                    |                     |
|          InternVL2‑1B          | 79.9 |       83.6       |        88.7        |        79.8        |       76.0        |        83.6         |        67.7         |        80.2        |        79.9         |
|          InternVL2‑2B          | 77.7 |       82.3       |        88.2        |        75.9        |       73.5        |        82.8         |        63.3         |        77.6        |        78.3         |
|          InternVL2‑4B          | 84.4 |       88.5       |        91.2        |        83.9        |       81.2        |        87.2         |        73.8         |        84.6        |        84.6         |
|          InternVL2‑8B          | 82.9 |       87.1       |        91.1        |        80.7        |       79.8        |        87.9         |        71.4         |        82.7        |        82.7         |
|         InternVL2‑26B          | 88.5 |       91.2       |        93.3        |        87.4        |       86.8        |        91.0         |        81.2         |        88.5        |        88.6         |
|         InternVL2‑40B          | 90.3 |       93.0       |        94.7        |        89.2        |       88.5        |        92.8         |        83.6         |        90.3        |        90.6         |
|    InternVL2-<br>Llama3‑76B    | 90.0 |       92.2       |        94.8        |        88.4        |       88.8        |        93.1         |        82.8         |        89.5        |        90.3         |

- We use the following prompt to evaluate InternVL's grounding ability: `Please provide the bounding box coordinates of the region this sentence describes: <ref>{}</ref>`

Limitations: Although we have made efforts to ensure the safety of the model during the training process and to encourage the model to generate text that complies with ethical and legal requirements, the model may still produce unexpected outputs due to its size and probabilistic generation paradigm. For example, the generated responses may contain biases, discrimination, or other harmful content. Please do not propagate such content. We are not responsible for any consequences resulting from the dissemination of harmful information.

## Quick Start

We provide an example code to run `InternVL2-8B` using `transformers`.

> Please use transformers>=4.37.2 to ensure the model works normally.

### Model Loading

#### 16-bit (bf16 / fp16)

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```

#### BNB 8-bit Quantization

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

#### Multiple GPUs

The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors.

```python
import math
import torch
from transformers import AutoTokenizer, AutoModel

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

path = "OpenGVLab/InternVL2-8B"
device_map = split_model('InternVL2-8B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```

### Inference with Transformers

```python
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)

# pure-text conversation (纯文本对话)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# single-image single-round conversation (单图单轮对话)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# single-image multi-round conversation (单图多轮对话)
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Please write a poem according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

question = '<image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# batch inference, single image per sample (单图批处理)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

video_path = './examples/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
question = video_prefix + 'What is the red panda doing?'
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Describe this video in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
```

#### Streaming Output

Besides this method, you can also use the following code to get streamed output.

```python
from transformers import TextIteratorStreamer
from threading import Thread

# Initialize the streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
# Define the generation configuration
generation_config = dict(max_new_tokens=1024, do_sample=False, streamer=streamer)
# Start the model chat in a separate thread
thread = Thread(target=model.chat, kwargs=dict(
    tokenizer=tokenizer, pixel_values=pixel_values, question=question,
    history=None, return_history=False, generation_config=generation_config,
))
thread.start()

# Initialize an empty string to store the generated text
generated_text = ''
# Loop through the streamer to get the new text as it is generated
for new_text in streamer:
    if new_text == model.conv_template.sep:
        break
    generated_text += new_text
    print(new_text, end='', flush=True)  # Print each new chunk of generated text on the same line
```

## Finetune

Many repositories now support fine-tuning of the InternVL series models, including [InternVL](https://github.com/OpenGVLab/InternVL), [SWIFT](https://github.com/modelscope/ms-swift), [XTurner](https://github.com/InternLM/xtuner), and others. Please refer to their documentation for more details on fine-tuning.

## Deployment

### LMDeploy

LMDeploy is a toolkit for compressing, deploying, and serving LLMs & VLMs.

```sh
pip install lmdeploy>=0.5.3
```

LMDeploy abstracts the complex inference process of multi-modal Vision-Language Models (VLM) into an easy-to-use pipeline, similar to the Large Language Model (LLM) inference pipeline.

#### A 'Hello, world' Example

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-8B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

If `ImportError` occurs while executing this case, please install the required dependency packages as prompted.

#### Multi-images Inference

When dealing with multiple images, you can put them all in one list. Keep in mind that multiple images will lead to a higher number of input tokens, and as a result, the size of the context window typically needs to be increased.

> Warning: Due to the scarcity of multi-image conversation data, the performance on multi-image tasks may be unstable, and it may require multiple attempts to achieve satisfactory results.

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

#### Batch Prompts Inference

Conducting inference with batch prompts is quite straightforward; just place them within a list structure:

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

#### Multi-turn Conversation

There are two ways to do the multi-turn conversations with the pipeline. One is to construct messages according to the format of OpenAI and use above introduced method, the other is to use the `pipeline.chat` interface.

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

#### Service

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2-8B --server-port 23333
```

To use the OpenAI-style interface, you need to install OpenAI:

```shell
pip install openai
```

Then, use the code below to make the API call:

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```

## License

This project is released under the MIT License. This project uses the pre-trained internlm2_5-7b-chat as a component, which is licensed under the Apache License 2.0.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2024expanding,
  title={Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  author={Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2412.05271},
  year={2024}
}
@article{gao2024mini,
  title={Mini-internvl: A flexible-transfer pocket multimodal model with 5\% parameters and 90\% performance},
  author={Gao, Zhangwei and Chen, Zhe and Cui, Erfei and Ren, Yiming and Wang, Weiyun and Zhu, Jinguo and Tian, Hao and Ye, Shenglong and He, Junjun and Zhu, Xizhou and others},
  journal={arXiv preprint arXiv:2410.16261},
  year={2024}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
@inproceedings{chen2024internvl,
  title={Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24185--24198},
  year={2024}
}
```
