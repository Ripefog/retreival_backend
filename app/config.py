# --- START OF FILE app/config.py ---

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import logging
import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Thiết lập logger cho module này
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Quản lý tập trung tất cả các cấu hình cho ứng dụng.
    Sử dụng Pydantic để tự động đọc từ biến môi trường hoặc file .env.
    """
    # model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # --- Milvus Configuration ---
    # Có thể ghi đè bằng cách set biến môi trường, ví dụ: export MILVUS_HOST="my-milvus.com"
    MILVUS_HOST: str = "1.53.19.130"
    MILVUS_PORT: int = 19530
    MILVUS_ALIAS: str = "default"
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    MILVUS_START: int  = 25
    MILVUS_END: int  = 26
    # --- Elasticsearch/OpenSearch Configuration ---
    ELASTICSEARCH_HOST: str = "1.53.19.130"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_USERNAME: Optional[str] = None
    ELASTICSEARCH_PASSWORD: Optional[str] = None
    ELASTICSEARCH_USE_SSL: bool = False
    ELASTICSEARCH_START: int  = 25
    ELASTICSEARCH_END: int  = 26
    # --- Milvus Collection Names ---
    CLIP_COLLECTION: str = 'arch_clip_image_v2'
    BEIT3_COLLECTION: str = 'arch_beit3_image_v2'
    OBJECT_COLLECTION: str = 'arch_object_name_v2'
    COLOR_COLLECTION: str = 'arch_color_name_v2'
    
    # --- Elasticsearch Index Names ---
    METADATA_INDEX: str = 'video_retrieval_metadata_v3'
    OCR_INDEX: str = 'ocr_v2'
    ASR_INDEX: str = 'video_transcripts_v2'
    
    # --- Model Paths ---
    # Đường dẫn có thể là tuyệt đối (container) hoặc tương đối (local)
    CLIP_MODEL_PATH: str = os.environ.get("CLIP_MODEL_PATH", "models/clip_model.bin")
    BEIT3_MODEL_PATH: str = os.environ.get("BEIT3_MODEL_PATH", "models/beit3_base_patch16_384_coco_retrieval.pth")
    BEIT3_SPM_PATH: str = os.environ.get("BEIT3_SPM_PATH", "models/beit3.spm")
    CO_DETR_CONFIG_PATH: str = os.environ.get("CO_DETR_CONFIG_PATH", "Co_DETR/projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py")
    CO_DETR_CHECKPOINT_PATH: str = os.environ.get("CO_DETR_CHECKPOINT_PATH", "models/co_dino_5scale_swin_large_16e_o365tococo.pth")
    KEYFRAME_ROOT_DIR: str  = r"/data/30v"
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


class VectorSchema:
    # Schema unchanged
    KEYFRAME_ID, VECTOR, VIDEO_ID, TIMESTAMP, LABEL = "keyframe_id", "vector", "video_id", "timestamp", "label"

    @staticmethod
    def get_fields(dim, has_label=False):
        fields = [
            FieldSchema(name=VectorSchema.KEYFRAME_ID, dtype=DataType.VARCHAR, max_length=512, is_primary=True,
                        auto_id=False),
            FieldSchema(name=VectorSchema.VECTOR, dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name=VectorSchema.VIDEO_ID, dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name=VectorSchema.TIMESTAMP, dtype=DataType.DOUBLE),
        ]

        if has_label:
            fields.append(FieldSchema(name=VectorSchema.LABEL, dtype=DataType.VARCHAR, max_length=255))  # Lưu màu LAB
        return fields


# --- START OF FIX ---
# Correct the dimensions for all collections that use the CLIP model.
# CLIP ViT-H-14 -> 1024 dimensions
# BEIT-3 Base   -> 768 dimensions
# BEIT-3 Large   -> 1024 dimensions
settings = Settings()
SCHEMA_CONFIG = {
    settings.CLIP_COLLECTION: {'dim': 1024, 'has_label': False},  # Corrected from 768
    settings.BEIT3_COLLECTION: {'dim': 768, 'has_label': False},  # This was already correct
    settings.COLOR_COLLECTION: {'dim': 3, 'has_label': True},  # Màu nền không có label, chỉ có màu
    settings.OBJECT_COLLECTION: {'dim': 1024, 'has_label': True},  # Đã sửa lại để có 'has_label'
}


# --- END OF FIX ---

class MilvusManager:
    def __init__(self, collection_name: str, config: Settings):
        self.collection_name = collection_name
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )

        s_config = SCHEMA_CONFIG.get(collection_name)
        if not s_config: raise ValueError(f"No schema config for collection: {collection_name}")

        self.has_label = s_config['has_label']

        # Tiếp tục phần còn lại của khởi tạo collection
        if not utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' not found. Creating with new schema (dim={s_config['dim']})...")
            schema_def = CollectionSchema(fields=VectorSchema.get_fields(**s_config))
            self.collection = Collection(name=self.collection_name, schema=schema_def)
            self.collection.create_index(VectorSchema.VECTOR,
                                         {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
            print(f"Created Milvus collection and index '{self.collection_name}'.")
        else:
            self.collection = Collection(name=self.collection_name)
        self.collection.load()
        print(f"Milvus collection '{self.collection_name}' loaded.")

    def insert(self, data: list):
        expected_len = 4  # Không có 'label', chỉ có 4 trường dữ liệu
        if self.has_label:
            expected_len += 1  # Nếu có màu sắc, thì thêm trường màu (màu LAB)

        if len(data) != expected_len:
            raise ValueError(
                f"Data for '{self.collection_name}' has wrong number of fields. Expected {expected_len}, got {len(data)}.")
        self.collection.insert(data)

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