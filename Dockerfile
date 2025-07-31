# Dockerfile để tái tạo môi trường từ retrieval-full.py

# ==============================================================================
# STAGE 1: Môi trường cơ sở và các thư viện hệ thống
# ==============================================================================
# Sử dụng image base có sẵn CUDA 11.1 và Ubuntu 18.04 để khớp với môi trường Kaggle
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Đặt các biến môi trường
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Cài đặt các dependencies hệ thống cơ bản và Python 3.8
# Đây là bước cần thiết để có thể cài các gói python sau này
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.8 \
    python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật symbolic link để python3 trỏ tới python3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Cài đặt pip cho python 3.8
RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

WORKDIR /app

# ==============================================================================
# STAGE 2: Cài đặt các thư viện ML/AI theo đúng phiên bản trong script gốc
# Các lệnh được giữ nguyên cấu trúc để đảm bảo tính tương thích
# ==============================================================================

# 1. Cài đặt các thư viện ML Frameworks cốt lõi (PyTorch, Detectron2, MMCV)
# Đây là bước quan trọng nhất và phải tuân thủ nghiêm ngặt phiên bản.
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 2. Xử lý các xung đột phiên bản và ép phiên bản (Dependency Fixing)
# Các bước này được lấy trực tiếp từ script gốc.
RUN pip uninstall -y timm && pip install timm==0.4.12
RUN pip uninstall -y protobuf && pip install protobuf==3.20.3
RUN pip install "setuptools<58.0.0"

# 3. Clone và cài đặt các repo phụ thuộc (Co-DETR, BEiT-3)
RUN git clone https://github.com/Sense-X/Co-DETR.git /app/Co-DETR
# Lưu ý: Script gốc có lệnh `mv`, ta đổi tên trực tiếp khi clone hoặc sau đó.
# Để đơn giản, ta sẽ dùng đường dẫn /app/Co-DETR và chỉnh PYTHONPATH
RUN pip install -e /app/Co-DETR

RUN git clone https://github.com/microsoft/unilm.git /app/unilm
RUN pip install -r /app/unilm/beit3/requirements.txt

# 4. Cài đặt các thư viện hỗ trợ còn lại
# Các thư viện này không có phiên bản cụ thể trong script gốc, nên ta cũng không chỉ định ở đây.
# Việc cài sau các gói cốt lõi giúp pip giải quyết xung đột tốt hơn.
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
    scikit-learn

# 5. Tải các file model và tokenizer cần thiết
# Đặt wget vào cuối để tận dụng cache của Docker tốt hơn.
RUN wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin -O /app/clip_model.bin
RUN wget https://github.com/addf400/files/releases/download/beit3/beit3.spm -O /app/beit3.spm
RUN wget https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth -O /app/beit3_base_patch16_384_coco_retrieval.pth

# ==============================================================================
# STAGE 3: Cài đặt ứng dụng FastAPI và cấu hình môi trường chạy
# ==============================================================================

# Sao chép file requirements.txt của FastAPI
COPY requirements.txt .

# Cài đặt các thư viện của FastAPI
RUN pip install -r requirements.txt

# Sao chép toàn bộ mã nguồn ứng dụng (bao gồm các file .py của bạn)
# Phải đặt sau các lệnh cài đặt và tải xuống để tận dụng cache.
COPY . .

# Expose port mà FastAPI sẽ chạy
EXPOSE 8000

# Set PYTHONPATH để các module từ Co-DETR và BEiT-3 có thể được import đúng cách
# Script gốc có đổi tên Co-DETR -> Co_DETR, ta cần đảm bảo đường dẫn này đúng.
RUN mv /app/Co-DETR /app/Co_DETR
ENV PYTHONPATH=/app:/app/unilm/beit3:/app/Co_DETR

# Lệnh mặc định để chạy ứng dụng FastAPI
# Giả sử file chính của bạn là `main.py` nằm trong thư mục gốc `/app`
# và đối tượng FastAPI có tên là `app`
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]