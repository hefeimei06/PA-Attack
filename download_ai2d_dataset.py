import os
from datasets import load_dataset
from PIL import Image

save_folder = "scienceqa_guidance"
target_count = 3000

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


dataset = load_dataset("derek-thomas/ScienceQA", split="train", streaming=True)


count = 0
for i, sample in enumerate(dataset):
    if count >= target_count:
        break

    image = sample.get('image')
    
    if image is not None:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            file_path = os.path.join(save_folder, f"scienceqa_{count}.png")
            image.save(file_path)
            
            count += 1
            
            if count % 100 == 0:
                print(f"Save {count}/{target_count}")
                
        except Exception as e:
            print(f"Error: {e}")

print(f"Done!")