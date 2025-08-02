# download_models.py
import os
import requests
from tqdm import tqdm

MODEL_URLS = {
    "clip/open_clip_pytorch_model.bin": "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin",
    "beit3/beit3.spm": "https://github.com/addf400/files/releases/download/beit3/beit3.spm",
    "beit3/beit3_base_patch16_384_coco_retrieval.pth": "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth",  # Back to Base
    "co_detr/co_dino_5scale_swin_large_16e_o365tococo.pth": "https://github.com/Sense-X/Co-DETR/releases/download/v0.1.0/co_dino_5scale_swin_large_16e_o365tococo.pth" # Thay bằng URL checkpoint của bạn nếu khác
}

def download_file(url, dest_path):
    """Tải file với thanh tiến trình."""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {os.path.basename(dest_path)} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

if __name__ == "__main__":
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    for dest, url in MODEL_URLS.items():
        full_dest_path = os.path.join(base_dir, dest)
        os.makedirs(os.path.dirname(full_dest_path), exist_ok=True)
        download_file(url, full_dest_path)
    
    print("\n[✔] All models downloaded successfully.")