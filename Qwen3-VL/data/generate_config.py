import os
import json
import glob

DATA_ROOT = "/home/Qwen3-VL/data/odinw"

TARGET_DATASETS = [
    "AerialMaritimeDrone",
    "Aquarium",
    "Cottontail Rabbits",
    "EgoHands",
    "NorthAmerica Mushrooms",
    "Packages",
    "Pascal VOC",
    "Pistols",
    "Pothole",
    "Raccoon",
    "ShellfishOpenImages",
    "Thermal Dogs and People",
    "Vehicles OpenImages"
]

def find_annotation_file(dataset_path):
    search_patterns = [
        "test/**/_annotations.coco.json",
        "test/**/*.json",
        "valid/**/_annotations.coco.json",
        "valid/**/*.json",
        "**/*test*/**/*.json",
    ]
    for pattern in search_patterns:
        files = glob.glob(os.path.join(dataset_path, pattern), recursive=True)
        if files:
            for f in files:
                if "annotations" in f or "coco" in f:
                    return f
    return None

def main():
    
    config_entries = []     
    valid_datasets = []     
    
    for dataset_name in TARGET_DATASETS:
        dataset_path = os.path.join(DATA_ROOT, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"❌ Can not find: {dataset_name}")
            continue
            
        ann_file_abs = find_annotation_file(dataset_path)
        if not ann_file_abs:
            print(f"⚠️ No annotation file found for: {dataset_name}")
            continue
            
        try:
            with open(ann_file_abs, 'r') as f:
                data = json.load(f)
            categories = data.get('categories', [])
            categories.sort(key=lambda x: x['id'])
            class_names = tuple([c['name'] for c in categories])
        except Exception as e:
            print(f"   Read JSON fail: {e}")
            continue

        rel_ann_path = os.path.relpath(ann_file_abs, dataset_path)

        if not rel_ann_path.startswith("/"):
            rel_ann_path = "/" + rel_ann_path
            
        img_dir_rel = os.path.dirname(rel_ann_path) 

        rel_img_prefix = img_dir_rel + "/"
        
        print(f"✅ {dataset_name} -> Path: ...{rel_ann_path}")

        entry = "{\n"
        entry += f"    'metainfo': {{'classes': {class_names}}},\n"
        entry += f"    'data_root': '{dataset_path}',\n"
        entry += f"    'ann_file': '{rel_ann_path}',\n"
        entry += f"    'data_prefix': {{'img': '{rel_img_prefix}'}}\n"
        entry += "}"
        
        config_entries.append(entry)
        valid_datasets.append(dataset_name)

    final_content = []
    
    entries_str = ",\n".join(config_entries)
    
    final_content.append(f"datasets = [\n{entries_str}\n]")
    final_content.append("")
    final_content.append(f"dataset_prefixes = {valid_datasets}")

    output_path = os.path.join(DATA_ROOT, "odinw13_config.py")
    with open(output_path, 'w') as f:
        f.write("\n".join(final_content))
        
    print(f"\n🎉 Finished config generation: {output_path}")

if __name__ == "__main__":
    main()