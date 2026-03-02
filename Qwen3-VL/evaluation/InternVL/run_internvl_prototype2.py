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
    flatten_dim = T * D
    X_gpu = tokens_feats.reshape(N, T * D).cuda().float()
    print(f"   -> Running GPU PCA ({flatten_dim} -> {pca_dim})...")

    mean_vec = X_gpu.mean(dim=0)
    X_gpu -= mean_vec

    U, S, V = torch.pca_lowrank(X_gpu, q=pca_dim, center=False, niter=2)
    X_pca_gpu = torch.matmul(X_gpu, V[:, :pca_dim])

    X_pca_np = X_pca_gpu.cpu().numpy()
    
    del X_gpu, U, S, V, X_pca_gpu, mean_vec
    torch.cuda.empty_cache()
    print("   -> PCA Done. GPU memory cleared.")

    print(f"   -> Running KMeans (K={n_proto})...")
    kmeans = KMeans(n_clusters=n_proto, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(X_pca_np)

    print("   -> Reconstructing structural prototypes...")
    proto_tokens = []

    tokens_feats_norm = F.normalize(tokens_feats.float(), dim=-1)
    
    for cid in tqdm(range(n_proto), desc="Centroids"):
        indices = np.where(cluster_ids == cid)[0]
        
        if len(indices) > 0:
            members = tokens_feats_norm[indices]
            proto_token = members.mean(dim=0) 
        else:
            print(f"[Warning] Empty cluster {cid}")
            proto_token = torch.zeros((T, D), dtype=torch.float16)
            
        proto_tokens.append(proto_token)

    proto_tokens = torch.stack(proto_tokens).to(torch.float16)
    print(f"✅ Prototypes: {proto_tokens.shape}")

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

    del model
    torch.cuda.empty_cache()
    
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