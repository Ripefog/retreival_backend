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
    software-properties-common git wget curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    build-essential ninja-build python3.8-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-distutils python3.8-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    curl https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# (2) Pin pip/setuptools/wheel/packaging SỚM
#    (đảm bảo pkg_resources hợp với cách mmcv-full import packaging)
RUN python -m pip install --no-cache-dir --upgrade \
    "pip<24" "setuptools==59.5.0" "wheel<0.41" "packaging==21.3" gdown

# (3) Tùy chọn: giảm lỗi SSL bằng cách tin các host này (áp dụng cho lệnh pip kế tiếp)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# ================== STAGE 2 ==================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Torch/cu111 + Detectron2 (có thể thêm --trusted-host cho repo detectron2)
RUN pip install --no-cache-dir \
    torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir \
    --trusted-host dl.fbaipublicfiles.com \
    detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# (4) CÀI MMCV-FULL nhưng bỏ qua verify SSL cho openmmlab
#     --trusted-host làm pip chấp nhận HTTPS host này không verify cert
RUN pip install --no-cache-dir \
    --trusted-host download.openmmlab.com \
    mmcv-full==1.7.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# --- Tải xuống TẤT CẢ các model cần thiết ---
RUN mkdir -p /app/models

# Tải các model từ nguồn công khai
RUN wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin -O /app/models/clip_model.bin
RUN wget https://github.com/addf400/files/releases/download/beit3/beit3.spm -O /app/models/beit3.spm
RUN wget https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth -O /app/models/beit3_base_patch16_384_coco_retrieval.pth

# [CẬP NHẬT] Tải model Co-DETR từ Google Drive bằng gdown với File ID mới
# File ID mới: 1ffDz9lGNAjEF7iXzINZezZ4alx6S0KcO
RUN gdown '1ffDz9lGNAjEF7iXzINZezZ4alx6S0KcO' -O /app/models/co_dino_5scale_swin_large_16e_o365tococo.pth
#RUN pip install --no-cache-dir --upgrade "pip<24" "setuptools==59.5.0" "wheel<0.41" "packaging==21.3"

# --- Cài đặt Co-DETR ---
# Cần unzip và curl
RUN apt-get update && apt-get install -y --no-install-recommends unzip curl && rm -rf /var/lib/apt/lists/*

# Tải mã nguồn Co-DETR dạng ZIP từ codeload (ổn định, hỗ trợ retry)
RUN curl -L --retry 6 --retry-delay 5 --connect-timeout 15 --max-time 0 \
      -o /tmp/co-detr.zip \
      https://codeload.github.com/Sense-X/Co-DETR/zip/refs/heads/master && \
    unzip /tmp/co-detr.zip -d /tmp && \
    mv /tmp/Co-DETR-master /app/Co_DETR && \
    rm -f /tmp/co-detr.zip

# Cài đặt editable
RUN pip install --no-cache-dir -e /app/Co_DETR

# timm đúng version
RUN pip install --no-cache-dir timm==0.4.12



# --- Cài đặt BEiT-3 ---
RUN git clone https://github.com/microsoft/unilm.git /app/unilm && \
    pip install --no-cache-dir -r /app/unilm/beit3/requirements.txt && \
    pip uninstall -y protobuf && pip install --no-cache-dir protobuf==3.20.3
RUN pip install icecream

RUN mkdir -p /data
COPY video2pair_21_26.txt /data/video2pair_21_26.txt
# ==============================================================================
# STAGE 3: Cấu hình và chạy ứng dụng
# ==============================================================================
#FROM base AS runtime
#WORKDIR /app
COPY ./app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]