import os
import torch
import glob
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= Config =================
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
COCO_IMG_DIR = "/home/datasets/coco2014/val2014"
SAVE_DIR = "Prototype/prototypes_qwen3"
SAVE_NAME = "prototypes_global_3000_20_1024.pt"

N_SAMPLES = 3000
N_PROTOTYPES = 20
PCA_DIM = 1024

MIN_PIXELS = 768 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28
# ===========================================

def extract_features_qwen3(model, processor, image_files, device):

    global_features = []
    
    print(f"🚀 Start extract features in {len(image_files)} images...")
 
    dummy_text = "Describe this image."

    for img_path in tqdm(image_files, desc="Extracting"):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": img_path,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    },
                    {"type": "text", "text": dummy_text},
                ],
            }
        ]

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
        

        pixel_values = inputs["pixel_values"].to(device, dtype=model.dtype)
        image_grid_thw = inputs["image_grid_thw"].to(device)
        
        with torch.no_grad():

            visual_tokens = model.visual(
                pixel_values, 
                grid_thw=image_grid_thw, 
                visual_feature=True
            )

            global_feat = visual_tokens.mean(dim=0) 

            global_features.append(global_feat.cpu())

    all_features = torch.stack(global_features, dim=0)
    return all_features

def build_prototypes_with_pca(features, n_proto=20, pca_dim=1024, random_state=42):

    print(f"📊 Start PCA: Input Shape {features.shape}, PCA={pca_dim}, K={n_proto}")
    
    X = features.numpy() # [N, D]
    D = X.shape[1]

    if pca_dim < D:
        print(f"   -> Running PCA ({D} -> {pca_dim})...")
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X_pca = pca.fit_transform(X)
    else:
        print("   -> Skipping PCA (dim is small enough).")
        X_pca = X

    # 3. KMeans
    print(f"   -> Running KMeans (K={n_proto})...")
    kmeans = KMeans(n_clusters=n_proto, random_state=random_state, n_init="auto")
    cluster_ids = kmeans.fit_predict(X_pca)

    proto_list = []

    features_norm = F.normalize(features, dim=-1).numpy()
    
    for cid in range(n_proto):
        members = features_norm[cluster_ids == cid]
        
        if len(members) > 0:
            proto_vec = members.mean(axis=0) # [D]
        else:
            print(f"[Warning] Empty cluster {cid}")
            proto_vec = np.zeros((D,), dtype=np.float16)
            
        proto_list.append(proto_vec)

    prototypes = torch.tensor(np.stack(proto_list), dtype=torch.float16)
    print(f"✅ Prototypes: {prototypes.shape}")
    
    return prototypes, cluster_ids

def main():
    if not os.path.exists(COCO_IMG_DIR):
        print(f"❌ Can not find COCO images in {COCO_IMG_DIR}")
        return

    print(f"🚀 Loading Qwen3-VL: {MODEL_PATH}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.to("cuda")
    model.eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    image_files = glob.glob(os.path.join(COCO_IMG_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(COCO_IMG_DIR, "*.jpeg"))
    image_files.sort()
    
    if len(image_files) > N_SAMPLES:
        print(f"⚠️ Use {N_SAMPLES} images for Prototypes")
        image_files = image_files[:N_SAMPLES]
    else:
        print(f"⚠️ No enough {N_SAMPLES} images, using all {len(image_files)} images")

    features = extract_features_qwen3(model, processor, image_files, device="cuda")

    prototypes, _ = build_prototypes_with_pca(
        features, 
        n_proto=N_PROTOTYPES, 
        pca_dim=PCA_DIM
    )
    print("Shape of prototype:", prototypes.shape)
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    torch.save(prototypes, save_path)
    print(f"🎉 Successfully saved Prototypes to: {save_path}")

if __name__ == "__main__":
    main()