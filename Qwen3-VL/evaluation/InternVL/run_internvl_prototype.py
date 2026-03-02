import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torchvision import transforms
from transformers import AutoModel


MODEL_PATH = "./evaluation/InternVL/InternVL2-8B" 

COCO_IMG_DIR = "/home/datasets/coco2014/val2014" 

SAVE_DIR = "Prototype/prototypes_internvl"
SAVE_NAME = "prototypes_internvl_3000_20_1024.pt"

N_IMAGES = 3000
N_PROTOTYPES = 20 
PCA_DIM = 1024

INPUT_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):

    resized_img = image.resize((image_size, image_size), Image.LANCZOS)
    return [resized_img]

def load_image_for_prototype(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=1)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values) # [1, 3, 448, 448]
    return pixel_values


def extract_structural_features(model, image_files, device):
    all_features = []
    
    print(f"🚀 Start extracting features for {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc="Extracting"):
        try:
            pixel_values = load_image_for_prototype(img_path, input_size=INPUT_SIZE)
            pixel_values = pixel_values.to(torch.float16).to(device)
            
            with torch.no_grad():
                outputs = model.vision_model(pixel_values)

                hidden_state = outputs.last_hidden_state
                
                features = hidden_state[0, 1:, :] 
                
                all_features.append(features)
                
        except Exception as e:
            print(f"Error: {e}")
            continue

    all_features = torch.stack(all_features, dim=0)
    print(f"✅ Shape: {all_features.shape}")
    return all_features


def build_prototypes_with_pca(tokens_feats, n_proto=30, pca_dim=64, random_state=0):
    N, T, D = tokens_feats.shape
    X = tokens_feats.reshape(N, T * D).cpu().numpy()
    print(f"   -> Running PCA ({T*D} -> {pca_dim})...")
    pca = PCA(n_components=pca_dim, random_state=random_state, svd_solver="randomized")
    X_pca = pca.fit_transform(X)
    print(f"   -> Running KMeans (K={n_proto})...")
    kmeans = KMeans(n_clusters=n_proto, random_state=random_state, n_init="auto")
    cluster_ids = kmeans.fit_predict(X_pca)
    print("   -> Reconstructing structural prototypes...")
    proto_tokens = []
    tokens_feats_norm = F.normalize(tokens_feats, dim=-1)
    tokens_feats_np = tokens_feats_norm.cpu().numpy()
    for cid in range(n_proto):
        members = tokens_feats_np[cluster_ids == cid]
        if len(members) > 0:
            proto_token = members.mean(axis=0)  # [T, D]
        else:
            print(f"[Warning] Empty cluster {cid}")
            proto_token = np.zeros((T, D), dtype=np.float16)
        proto_tokens.append(proto_token)

    proto_tokens = torch.tensor(np.stack(proto_tokens), dtype=torch.float16)

    return proto_tokens, cluster_ids


def main():
    if not os.path.exists(COCO_IMG_DIR):
        print(f"❌ Can not find {COCO_IMG_DIR}")
        return

    print(f"🚀 Loading InternVL: {MODEL_PATH}")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()

    image_files = glob.glob(os.path.join(COCO_IMG_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(COCO_IMG_DIR, "*.jpeg"))
    image_files.sort()
    
    if len(image_files) > N_IMAGES:
        image_files = image_files[:N_IMAGES]
    
    features = extract_structural_features(model, image_files, device="cuda")
    
    prototypes, cluster_ids = build_prototypes_with_pca(
        features, 
        n_proto=N_PROTOTYPES, 
        pca_dim=PCA_DIM
    )
    print("Prototype Shape:", prototypes.shape)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    torch.save(prototypes, save_path)
    print(f"🎉 Saved Structural Prototypes to: {save_path}")

if __name__ == "__main__":
    main()