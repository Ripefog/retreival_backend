# --- START OF FILE app/config.py ---

import os
from pydantic_settings import BaseSettings
from typing import Optional
import logging
import torch

# Thiết lập logger cho module này
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Quản lý tập trung tất cả các cấu hình cho ứng dụng.
    Sử dụng Pydantic để tự động đọc từ biến môi trường hoặc file .env.
    """
    
    # --- Milvus Configuration ---
    # Có thể ghi đè bằng cách set biến môi trường, ví dụ: export MILVUS_HOST="my-milvus.com"
    MILVUS_HOST: str = "0.tcp.ap.ngrok.io"
    MILVUS_PORT: int = 19220
    MILVUS_ALIAS: str = "default"
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    
    # --- Elasticsearch/OpenSearch Configuration ---
    ELASTICSEARCH_HOST: str = "0.tcp.ap.ngrok.io"
    ELASTICSEARCH_PORT: int = 16056
    ELASTICSEARCH_USERNAME: Optional[str] = None
    ELASTICSEARCH_PASSWORD: Optional[str] = None
    ELASTICSEARCH_USE_SSL: bool = False
    
    # --- Milvus Collection Names ---
    CLIP_COLLECTION: str = 'arch_clip_image_v3'
    BEIT3_COLLECTION: str = 'arch_beit3_image_v3'
    OBJECT_COLLECTION: str = 'arch_object_name_v3'
    COLOR_COLLECTION: str = 'arch_color_name_v3'
    
    # --- Elasticsearch Index Names ---
    METADATA_INDEX: str = 'video_retrieval_metadata_v3'
    OCR_INDEX: str = 'ocr'
    ASR_INDEX: str = 'video_transcripts'
    
    # --- Model Paths ---
    # Đường dẫn có thể là tuyệt đối (container) hoặc tương đối (local)
    CLIP_MODEL_PATH: str = os.environ.get("CLIP_MODEL_PATH", "models/clip_model.bin")
    BEIT3_MODEL_PATH: str = os.environ.get("BEIT3_MODEL_PATH", "models/beit3_base_patch16_384_coco_retrieval.pth")
    BEIT3_SPM_PATH: str = os.environ.get("BEIT3_SPM_PATH", "models/beit3.spm")
    CO_DETR_CONFIG_PATH: str = os.environ.get("CO_DETR_CONFIG_PATH", "Co_DETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py")
    CO_DETR_CHECKPOINT_PATH: str = os.environ.get("CO_DETR_CHECKPOINT_PATH", "models/co_dino_5scale_swin_large_16e_o365tococo.pth")

    # --- Model & Processing Configuration ---
    # Tự động phát hiện CUDA nếu có, nếu không thì dùng CPU.
    # Không nên set cứng là "cuda", để code tự quyết định sẽ linh hoạt hơn.
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Search Configuration ---
    DEFAULT_TOP_K: int = 100
    MAX_TOP_K: int = 1000
    
    # --- API Configuration ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    class Config:
        # Pydantic sẽ tìm file .env trong thư mục gốc của dự án
        env_file = ".env"
        # Đảm bảo phân biệt chữ hoa, chữ thường cho các biến môi trường
        case_sensitive = True

    # --- Các phương thức kiểm tra cấu hình (tùy chọn nhưng hữu ích) ---
    def check_milvus_config(self) -> bool:
        """Kiểm tra cấu hình Milvus cơ bản."""
        if not self.MILVUS_HOST or not self.MILVUS_PORT:
            logger.error("Milvus host or port is not configured.")
            return False
        return True
    
    def check_elasticsearch_config(self) -> bool:
        """Kiểm tra cấu hình Elasticsearch cơ bản."""
        if not self.ELASTICSEARCH_HOST or not self.ELASTICSEARCH_PORT:
            logger.error("Elasticsearch host or port is not configured.")
            return False
        return True

    def check_model_paths(self) -> bool:
        """Kiểm tra sự tồn tại của các file model."""
        paths_to_check = [
            self.CLIP_MODEL_PATH,
            self.BEIT3_MODEL_PATH,
            self.BEIT3_SPM_PATH,
            self.CO_DETR_CONFIG_PATH,
            self.CO_DETR_CHECKPOINT_PATH
        ]
        all_exist = True
        for path in paths_to_check:
            if not os.path.exists(path):
                logger.error(f"Model path not found: {path}")
                all_exist = False
        return all_exist

# Tạo một instance duy nhất của Settings để import và sử dụng trong toàn bộ ứng dụng
settings = Settings()

# Log các thông tin cấu hình quan trọng khi khởi động
# Điều này rất hữu ích cho việc gỡ lỗi
logger.info("--- Application Configuration Loaded ---")
logger.info(f"Device: {settings.DEVICE.upper()}")
logger.info(f"Milvus Endpoint: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
logger.info(f"Elasticsearch Endpoint: {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}")
logger.info(f"API will run on: http://{settings.API_HOST}:{settings.API_PORT}")
logger.info("------------------------------------")

# Chạy kiểm tra đường dẫn model để cảnh báo sớm nếu có lỗi
if not settings.check_model_paths():
    logger.warning("One or more model paths are invalid. The application might fail to start.")

# --- END OF FILE app/config.py ---