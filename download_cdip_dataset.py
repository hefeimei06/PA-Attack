import os
from datasets import load_dataset
from PIL import Image

save_folder = "rvl_cdip_guidance"
target_count = 3000

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

dataset = load_dataset("rvl_cdip", split="train", streaming=True)

for i, sample in enumerate(dataset):
    if i >= target_count:
        break
    
    try:
        image = sample['image']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        file_path = os.path.join(save_folder, f"image_{i}.png")
        
        image.save(file_path)

        if (i + 1) % 100 == 0:
            print(f"Save {i + 1}/{target_count} 张")
            
    except Exception as e:
        print(f"Error in Image {i}: {e}")

print(f"Done!")