# ==============================================================================
# STAGE 1: Môi trường cơ sở và các thư viện hệ thống
# ==============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/unilm/beit3
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common git wget curl ca-certificates libgl1-mesa-glx libglib2.0-0 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3.9-venv python3.9-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    apt-get update && apt-get install -y python3-pip && \
    pip3 install --no-cache-dir --upgrade pip

WORKDIR /app

# ==============================================================================
# STAGE 2: Cài đặt dependencies và models
# ==============================================================================

# Copy requirements và cài đặt Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MetaCLIP 2 notebook baseline: PyTorch 2.1 with CUDA 11.8.
RUN pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Cài đặt Gemini API và MetaCLIP 2 runtime
RUN pip install -U langchain-google-genai && \
    pip install --no-cache-dir "transformers==4.56.2" safetensors accelerate --force-reinstall

# Tạo thư mục models
RUN mkdir -p /app/models

# Copy các model files. MetaCLIP 2 is downloaded by Transformers on first use
# (or restored from a mounted Hugging Face cache).
COPY models/beit3.spm /app/models/beit3.spm
COPY models/beit3_base_patch16_384_coco_retrieval.pth /app/models/beit3_base_patch16_384_coco_retrieval.pth

# --- Cài đặt BEiT-3 ---
RUN git clone https://github.com/microsoft/unilm.git /app/unilm && \
    sed -i '/deepspeed/d' /app/unilm/beit3/requirements.txt && \
    sed -i '/numpy/d' /app/unilm/beit3/requirements.txt && \
    pip install --no-cache-dir -r /app/unilm/beit3/requirements.txt && \
    pip uninstall -y protobuf && pip install --no-cache-dir protobuf==3.20.3 && \
    find /app/unilm/beit3 -name "*.py" -exec sed -i 's/from torch._six import inf/inf = float("inf")/g' {} \; && \
    echo "=== Files in /app/unilm/beit3 ===" && \
    find /app/unilm/beit3 -maxdepth 1 -name "*.py" -type f && \
    echo "================================="

# Cài đặt icecream cho debugging
RUN pip install icecream

# Pin numpy cuối cùng để đảm bảo tương thích
RUN pip install --no-cache-dir --force-reinstall "numpy==1.24.4"

# ==============================================================================
# STAGE 3: Cấu hình và chạy ứng dụng
# ==============================================================================
# Copy application code
COPY ./app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
