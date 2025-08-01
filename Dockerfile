# Dockerfile để tái tạo môi trường từ retrieval-full.py và chạy ứng dụng FastAPI
# PHIÊN BẢN: Tuân thủ nghiêm ngặt thứ tự cài đặt từ script gốc.

# ==============================================================================
# STAGE 1: Môi trường cơ sở và các thư viện hệ thống (Giữ nguyên)
# ==============================================================================
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONPATH=/app:/app/unilm/beit3:/app/Co_DETR

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git wget curl ca-certificates libgl1-mesa-glx libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.8 python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

WORKDIR /app

# ==============================================================================
# STAGE 2: Cài đặt và tải xuống theo đúng thứ tự của retrieval-full.py
# ==============================================================================

# Bước 1-3: Cài đặt các framework ML cốt lõi
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Bước 4: Tải model CLIP
RUN mkdir -p /app/models
RUN wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin -O /app/models/clip_model.bin

# Bước 5: Clone Co-DETR
RUN git clone https://github.com/Sense-X/Co-DETR.git /app/Co-DETR-temp

# --- [SỬA LỖI QUAN TRỌNG] ---
# Dựa trên phân tích lỗi, `torch 1.9.0` cần gói `packaging` khi cài đặt các extension C++.
# Gói này phải được cài TRƯỚC khi `pip install -e Co-DETR` được chạy.
RUN pip install "packaging>=20.0"

# Bước 6: Cài đặt Co-DETR
RUN pip install -e /app/Co-DETR-temp

# Bước 7: Đổi tên thư mục Co-DETR (giống lệnh `mv` trong script)
RUN mv /app/Co-DETR-temp /app/Co_DETR

# Bước 8: Xử lý phiên bản `timm`
RUN pip uninstall -y timm && pip install timm==0.4.12

# Bước 9: Cài đặt `scikit-learn`
RUN pip install scikit-learn

# Bước 10: Cài đặt các thư viện hỗ trợ và thư viện cho FastAPI App
# Gộp các thư viện này lại vì chúng không có thứ tự nghiêm ngặt với nhau trong script gốc.
RUN pip install \
    moviepy \
    opensearch-py \
    requests-aws4auth \
    boto3 \
    nbimporter \
    transformers \
    pillow \
    open-clip-torch \
    pymilvus \
    lmdb \
    nbformat \
    ipython \
    "fastapi>=0.95.0" \
    "uvicorn[standard]" \
    "pydantic-settings"

# Bước 11: Xử lý phiên bản `setuptools`
RUN pip install "setuptools<58.0.0"

# Bước 12: Clone unilm (cho BEiT-3)
RUN git clone https://github.com/microsoft/unilm.git /app/unilm

# Bước 13: Cài đặt requirements cho BEiT-3
RUN pip install -r /app/unilm/beit3/requirements.txt

# Bước 14: Xử lý phiên bản `protobuf`
RUN pip uninstall -y protobuf && pip install protobuf==3.20.3

# Bước 15: Tải các model và tokenizer còn lại của BEiT-3
RUN wget https://github.com/addf400/files/releases/download/beit3/beit3.spm -O /app/models/beit3.spm
RUN wget https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth -O /app/models/beit3_base_patch16_384_coco_retrieval.pth

# ==============================================================================
# STAGE 3: Cấu hình và chạy ứng dụng
# ==============================================================================

# Sao chép toàn bộ mã nguồn ứng dụng (bao gồm các file .py) vào image
COPY ./app /app/app

# Expose port mà FastAPI sẽ chạy
EXPOSE 8000

# Lệnh mặc định để chạy ứng dụng FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]