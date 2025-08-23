# ==============================================================================
# STAGE 1: Môi trường cơ sở và các thư viện hệ thống
# ==============================================================================
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/unilm/beit3:/app/Co_DETR
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common git wget curl ca-certificates libgl1-mesa-glx libglib2.0-0 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-distutils python3.8-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 && \
    pip install --no-cache-dir --upgrade pip gdown

WORKDIR /app

# ==============================================================================
# STAGE 2: Cài đặt và tải xuống
# ==============================================================================

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
COPY mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl /tmp/mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl

# Cài đặt mmcv-full từ wheel và xoá file sau khi cài
RUN pip install --no-cache-dir /tmp/mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl \
    && rm -f /tmp/mmcv_full-1.7.0-cp38-cp38-manylinux1_x86_64.whl

# --- Tải xuống TẤT CẢ các model cần thiết ---
RUN mkdir -p /app/models

# Tải các model từ nguồn công khai
RUN wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin -O /app/models/clip_model.bin
RUN wget https://github.com/addf400/files/releases/download/beit3/beit3.spm -O /app/models/beit3.spm
RUN wget https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth -O /app/models/beit3_base_patch16_384_coco_retrieval.pth

# [CẬP NHẬT] Tải model Co-DETR từ Google Drive bằng gdown với File ID mới
# File ID mới: 1ffDz9lGNAjEF7iXzINZezZ4alx6S0KcO
#RUN gdown '1ffDz9lGNAjEF7iXzINZezZ4alx6S0KcO' -O /app/models/co_dino_5scale_swin_large_16e_o365tococo.pth

COPY co_dino_5scale_swin_large_16e_o365tococo.pth /app/models/co_dino_5scale_swin_large_16e_o365tococo.pth
# --- Cài đặt Co-DETR ---
# Hạ pip/setuptools/wheel TƯƠNG THÍCH Torch 1.9 (giữ nguyên)
RUN pip install --no-cache-dir --upgrade "pip<24" "setuptools==59.5.0" "wheel<0.41" "packaging==21.3"

# --- Cài đặt Co-DETR (KHÔNG downgrade setuptools lần nữa, KHÔNG cài packaging lẻ) ---
RUN git clone https://github.com/Sense-X/Co-DETR.git /app/Co-DETR-temp && \
    pip install --no-cache-dir -e /app/Co-DETR-temp && \
    mv /app/Co-DETR-temp /app/Co_DETR && \
    pip install --no-cache-dir timm==0.4.12

# --- Cài đặt BEiT-3 ---
RUN git clone https://github.com/microsoft/unilm.git /app/unilm && \
    pip install --no-cache-dir -r /app/unilm/beit3/requirements.txt && \
    pip uninstall -y protobuf && pip install --no-cache-dir protobuf==3.20.3
RUN pip install icecream
# ==============================================================================
# STAGE 3: Cấu hình và chạy ứng dụng
# ==============================================================================
#FROM base AS runtime
#WORKDIR /app
COPY ./app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]