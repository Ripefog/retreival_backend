import os
from pydantic_settings import BaseSettings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Milvus Configuration
    MILVUS_HOST: str = "0.tcp.ap.ngrok.io"
    MILVUS_PORT: int = 17897
    MILVUS_ALIAS: str = "default"
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    
    # Elasticsearch Configuration
    ELASTICSEARCH_HOST: str = "0.tcp.ap.ngrok.io"
    ELASTICSEARCH_PORT: int = 19043
    ELASTICSEARCH_USERNAME: Optional[str] = None
    ELASTICSEARCH_PASSWORD: Optional[str] = None
    ELASTICSEARCH_USE_SSL: bool = False
    
    # Collection names
    CLIP_COLLECTION: str = 'arch_clip_image_v3'
    BEIT3_COLLECTION: str = 'arch_beit3_image_v3'
    OBJECT_COLLECTION: str = 'arch_object_name_v3'
    COLOR_COLLECTION: str = 'arch_color_name_v3'
    
    # Index names
    METADATA_INDEX: str = 'video_retrieval_metadata_v3'
    OCR_INDEX: str = 'ocr'
    ASR_INDEX: str = 'video_transcripts'
    
    CLIP_MODEL_PATH: str = "/app/models/clip_model.bin"
    BEIT3_MODEL_PATH: str = "/app/models/beit3_base_patch16_384_coco_retrieval.pth"
    BEIT3_SPM_PATH: str = "/app/models/beit3.spm"
    # Model Configuration
    DEVICE: str = "cuda"  # or "cpu"
    
    # Search Configuration
    DEFAULT_TOP_K: int = 100
    MAX_TOP_K: int = 1000
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def check_milvus_config(self) -> bool:
        """Kiểm tra cấu hình Milvus"""
        if not self.MILVUS_HOST or not self.MILVUS_PORT:
            logger.error("Milvus host or port not configured")
            return False
        return True
    
    def check_elasticsearch_config(self) -> bool:
        """Kiểm tra cấu hình Elasticsearch"""
        if not self.ELASTICSEARCH_HOST or not self.ELASTICSEARCH_PORT:
            logger.error("Elasticsearch host or port not configured")
            return False
        return True

# Tạo instance settings
settings = Settings()

# Log cấu hình
logger.info(f"Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
logger.info(f"Elasticsearch: {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}")
logger.info(f"Device: {settings.DEVICE}") 