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
    MILVUS_HOST: str = "0.tcp.ap.ngrok.io"
    MILVUS_PORT: int = 15380
    MILVUS_ALIAS: str = "default"
    MILVUS_USER: str = "root"
    MILVUS_PASSWORD: str = "aiostorm"
    MILVUS_USE_SECURE: bool = False  # Thêm option này cho SSL/TLS
    
    # --- Elasticsearch/OpenSearch Configuration ---
    ELASTICSEARCH_HOST: str = "0.tcp.ap.ngrok.io"
    ELASTICSEARCH_PORT: int = 10184
    ELASTICSEARCH_USERNAME: str = "elastic"
    ELASTICSEARCH_PASSWORD: str = "aiostorm"
    ELASTICSEARCH_USE_SSL: bool = False
    ELASTICSEARCH_VERIFY_CERTS: bool = False  # Thêm option này
    
    # --- Milvus Collection Names ---
    CLIP_COLLECTION: str = 'arch_clip_image_v100'
    BEIT3_COLLECTION: str = 'arch_beit3_image_v100'
    OBJECT_COLLECTION: str = 'arch_object_name_v100'
    
    # --- Elasticsearch Index Names ---
    METADATA_INDEX: str = 'video_retrieval_metadata_v3'
    OCR_INDEX: str = 'ocr_v6'
    ASR_INDEX: str = 'video_transcripts_v6'
    
    # --- Model Paths ---
    CLIP_MODEL_PATH: str = os.environ.get("CLIP_MODEL_PATH", "models/clip_model.bin")
    BEIT3_MODEL_PATH: str = os.environ.get("BEIT3_MODEL_PATH", "models/beit3_base_patch16_384_coco_retrieval.pth")
    BEIT3_SPM_PATH: str = os.environ.get("BEIT3_SPM_PATH", "models/beit3.spm")
    CO_DETR_CONFIG_PATH: str = os.environ.get("CO_DETR_CONFIG_PATH", "Co_DETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py")
    CO_DETR_CHECKPOINT_PATH: str = os.environ.get("CO_DETR_CHECKPOINT_PATH", "models/co_dino_5scale_swin_large_16e_o365tococo.pth")

    # --- Model & Processing Configuration ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Search Configuration ---
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100
    
    # --- API Configuration ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # --- Cache Configuration ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    ENABLE_REDIS_CACHE: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    # --- Connection Helper Methods ---
    def get_milvus_connection_params(self) -> dict:
        """Trả về dictionary chứa các tham số kết nối Milvus."""
        params = {
            "host": self.MILVUS_HOST,
            "port": self.MILVUS_PORT,
            "alias": self.MILVUS_ALIAS
        }
        
        # Chỉ thêm user/password nếu được cấu hình
        if self.MILVUS_USER:
            params["user"] = self.MILVUS_USER
        if self.MILVUS_PASSWORD:
            params["password"] = self.MILVUS_PASSWORD
        if self.MILVUS_USE_SECURE:
            params["secure"] = self.MILVUS_USE_SECURE
            
        return params
    
    def get_elasticsearch_connection_params(self) -> dict:
        """Trả về dictionary chứa các tham số kết nối Elasticsearch."""
        params = {
            "hosts": [f"{self.ELASTICSEARCH_HOST}:{self.ELASTICSEARCH_PORT}"],
            "http_auth": (self.ELASTICSEARCH_USERNAME, self.ELASTICSEARCH_PASSWORD) if self.ELASTICSEARCH_USERNAME and self.ELASTICSEARCH_PASSWORD else None,
            "use_ssl": self.ELASTICSEARCH_USE_SSL,
            "verify_certs": self.ELASTICSEARCH_VERIFY_CERTS,
            "ssl_show_warn": False
        }
        
        # Loại bỏ các giá trị None
        return {k: v for k, v in params.items() if v is not None}

    # --- Validation Methods ---
    def check_milvus_config(self) -> bool:
        """Kiểm tra cấu hình Milvus."""
        if not self.MILVUS_HOST or not self.MILVUS_PORT:
            logger.error("Milvus host or port is not configured.")
            return False
        if not self.MILVUS_USER or not self.MILVUS_PASSWORD:
            logger.warning("Milvus authentication not configured. Using default connection.")
        return True
    
    def check_elasticsearch_config(self) -> bool:
        """Kiểm tra cấu hình Elasticsearch."""
        if not self.ELASTICSEARCH_HOST or not self.ELASTICSEARCH_PORT:
            logger.error("Elasticsearch host or port is not configured.")
            return False
        if not self.ELASTICSEARCH_USERNAME or not self.ELASTICSEARCH_PASSWORD:
            logger.warning("Elasticsearch authentication not configured. Using default connection.")
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
                logger.warning(f"Model path not found: {path}")
                all_exist = False
        return all_exist

# Tạo instance settings
settings = Settings()

# Log thông tin cấu hình
logger.info("--- Application Configuration Loaded ---")
logger.info(f"Device: {settings.DEVICE.upper()}")
logger.info(f"Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT} (User: {settings.MILVUS_USER})")
logger.info(f"Elasticsearch: {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT} (User: {settings.ELASTICSEARCH_USERNAME})")
logger.info(f"API: http://{settings.API_HOST}:{settings.API_PORT}")
logger.info("------------------------------------")

# Kiểm tra cấu hình
settings.check_milvus_config()
settings.check_elasticsearch_config()
if not settings.check_model_paths():
    logger.warning("Some model paths are invalid. Check your model configuration.")
# --- END OF FILE app/config.py ---