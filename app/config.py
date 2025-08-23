# --- START OF FILE app/config.py ---

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import logging
import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Thiết lập logger cho module này
logger = logging.getLogger(__name__)
def load_video2user_mapping(filename):
    video_to_pair = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # line: L01_V000: Gia Nguyên, Duy Bảo
            video_id, pair_str = line.strip().split(": ")
            persons = tuple(p.strip() for p in pair_str.split(","))
            video_to_pair[video_id] = persons
    return video_to_pair

filename = "/data/data/video2pair_21_26.txt"
with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()
num_lines = len(lines)
print(f"Số dòng trong file (~tổng số video): {num_lines}")
pair_counts = Counter()
for line in lines:
    # line dạng: L01_V000: Gia Nguyên, Duy Bảo
    # tách phần sau dấu ': '
    pair_str = line.strip().split(": ")[1]
    # Đếm
    pair_counts[pair_str] += 1

print("\nSố video cho mỗi cặp:")
for pair, count in pair_counts.items():
    print(f"{pair}: {count}")



# Đếm số video cho mỗi người
person_counts = Counter()

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        # Dòng dạng: L01_V000: Gia Nguyên, Duy Bảo
        pair_str = line.strip().split(": ")[1]  # "Gia Nguyên, Duy Bảo"
        persons = [p.strip() for p in pair_str.split(",")]
        for p in persons:
            person_counts[p] += 1

# In kết quả
print("\nSố video cho mỗi người:")
for person, count in person_counts.items():
    print(f"{person}: {count}")

class Settings(BaseSettings):
    KEYFRAME_ROOT_DIR: str  = r"/data"
    """
    Quản lý tập trung tất cả các cấu hình cho ứng dụng.
    Sử dụng Pydantic để tự động đọc từ biến môi trường hoặc file .env.
    """
    # model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # --- Milvus Configuration ---
    # Có thể ghi đè bằng cách set biến môi trường, ví dụ: export MILVUS_HOST="my-milvus.com"
    MILVUS_HOST: str = "0.tcp.ap.ngrok.io"
    MILVUS_PORT: int = 19636
    MILVUS_ALIAS: str = "default"
    MILVUS_USER: Optional[str] = None
    MILVUS_PASSWORD: Optional[str] = None
    MILVUS_START: int  = 25
    MILVUS_END: int  = 26
    # --- Elasticsearch/OpenSearch Configuration ---
    ELASTICSEARCH_HOST: str = "0.tcp.ap.ngrok.io"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_USERNAME: Optional[str] = None
    ELASTICSEARCH_PASSWORD: Optional[str] = None
    ELASTICSEARCH_USE_SSL: bool = False
    ELASTICSEARCH_START: int  = 25
    ELASTICSEARCH_END: int  = 26
    # --- Milvus Collection Names ---
    CLIP_COLLECTION: str = 'arch_clip_image_v100'
    BEIT3_COLLECTION: str = 'arch_beit3_image_v100'
    OBJECT_COLLECTION: str = 'arch_object_name_v100'
    VIDEO_TO_USER = load_video2user_mapping(filename)
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


class ClipBeitFields:
    KEYFRAME_ID = "keyframe_id"
    VECTOR = "vector"
    TIMESTAMP = "timestamp"
    OBJECT_IDS = "object_ids"      # VARCHAR (JSON hoặc CSV)
    LAB_COLORS = "lab_colors"      # VARCHAR (JSON hoặc CSV)
    USER = "user"


class ObjectFields:
    OBJECT_ID = "object_id"
    VECTOR = "vector"
    BBOX_XYXY = "bbox_xyxy"        # VARCHAR (JSON hoặc CSV)
    COLOR_LAB = "color_lab"        # VARCHAR (JSON hoặc CSV)
    NAME = "name"

# ==============================
# Schema Factory
# ==============================
class SchemaFactory:
    MAX_OBJECT_IDS = 256
    LAB_COLORS_CAPACITY = 18
    BBOX_CAPACITY = 4
    LAB3_CAPACITY = 3

    @staticmethod
    def clip_schema(dim: int = 1024) -> CollectionSchema:
        fields = [
            FieldSchema(
                name=ClipBeitFields.KEYFRAME_ID,
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False
            ),
            FieldSchema(
                name=ClipBeitFields.VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=dim
            ),
            FieldSchema(
                name=ClipBeitFields.TIMESTAMP,
                dtype=DataType.DOUBLE
            ),
            FieldSchema(
                name=ClipBeitFields.OBJECT_IDS,
                dtype=DataType.VARCHAR,
                max_length=4096  # chứa JSON hoặc CSV
            ),
            FieldSchema(
                name=ClipBeitFields.LAB_COLORS,
                dtype=DataType.VARCHAR,
                max_length=1024  # chứa JSON hoặc CSV
            ),
            FieldSchema(
                name=ClipBeitFields.USER,
                dtype=DataType.VARCHAR,
                max_length=64
            )
        ]
        return CollectionSchema(fields=fields)

    @staticmethod
    def beit3_schema(dim: int = 768) -> CollectionSchema:
        fields = [
            FieldSchema(
                name=ClipBeitFields.KEYFRAME_ID,
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                auto_id=False
            ),
            FieldSchema(
                name=ClipBeitFields.VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=dim
            ),
            FieldSchema(
                name=ClipBeitFields.TIMESTAMP,
                dtype=DataType.DOUBLE
            ),
            FieldSchema(
                name=ClipBeitFields.OBJECT_IDS,
                dtype=DataType.VARCHAR,
                max_length=4096
            ),
            FieldSchema(
                name=ClipBeitFields.LAB_COLORS,
                dtype=DataType.VARCHAR,
                max_length=1024
            ),
            FieldSchema(
                name=ClipBeitFields.USER,
                dtype=DataType.VARCHAR,
                max_length=64
            )
        ]
        return CollectionSchema(fields=fields)

    @staticmethod
    def object_schema(dim: int = 1024) -> CollectionSchema:
        fields = [
            FieldSchema(
                name=ObjectFields.OBJECT_ID,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name=ObjectFields.VECTOR,
                dtype=DataType.FLOAT_VECTOR,
                dim=dim
            ),
            FieldSchema(
                name=ObjectFields.BBOX_XYXY,
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name=ObjectFields.COLOR_LAB,
                dtype=DataType.VARCHAR,
                max_length=128
            ),
            FieldSchema(
                name=ObjectFields.NAME,
                dtype=DataType.VARCHAR,
                max_length=256
            ),
        ]
        return CollectionSchema(fields=fields)


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

def default_index_params(metric: str = "L2", index_type: str = "IVF_FLAT", nlist: int = 1024):
    return {
        "metric_type": metric,
        "index_type": index_type,
        "params": {"nlist": nlist}
    }
# --- END OF FIX ---
class MilvusManager:
    """
    - Kết nối Milvus
    - Tự tạo collection nếu chưa có, với schema chính xác theo tên collection
    - Tạo index mặc định cho field 'vector'
    - Cung cấp các hàm insert có kiểm tra dữ liệu
    """

    def __init__(self, collection_name: str, config: Config_Milvus):
        self.collection_name = collection_name

        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )

        # Chọn schema theo collection
        if collection_name == config.CLIP_COLLECTION:
            schema = SchemaFactory.clip_schema(dim=1024)
            vector_dim = 1024
            vector_field = ClipBeitFields.VECTOR
        elif collection_name == config.BEIT3_COLLECTION:
            schema = SchemaFactory.beit3_schema(dim=768)
            vector_dim = 768
            vector_field = ClipBeitFields.VECTOR
        elif collection_name == config.OBJECT_COLLECTION:
            schema = SchemaFactory.object_schema(dim=1024)
            vector_dim = 1024
            vector_field = ObjectFields.VECTOR
        else:
            raise ValueError(f"Unknown collection: {collection_name}")

        # Tạo / mở collection
        if not utility.has_collection(collection_name):
            print(f"Creating collection '{collection_name}'...")
            self.collection = Collection(name=collection_name, schema=schema)
            # Tạo index cho vector
            self.collection.create_index(vector_field, default_index_params())
            print(f"Created collection '{collection_name}' with index on '{vector_field}' (dim={vector_dim}).")
        else:
            self.collection = Collection(name=collection_name)
            print(f"Opened existing collection '{collection_name}'.")

        self.collection.load()
        print(f"Milvus collection '{collection_name}' loaded.")

        # Lưu tên field để tiện validate khi insert
        self.vector_field = vector_field
        self.vector_dim = vector_dim
        self.config = config

    # --------- Insert helpers cho từng loại collection ---------
    def _join_csv(self, values):
        """
        Chuyển list thành chuỗi CSV. Trả "" nếu None/[].
        Lưu ý: không xử lý trường hợp phần tử có chứa dấu phẩy.
        """
        if not values:
            return ""
        return ",".join(str(x) for x in values)

    def get_video_id(self, filename: str):
        name = os.path.splitext(filename)[0]  # bỏ .jpg
        parts = name.split("_")

        if len(parts) >= 3:
            # Nếu phần 0 và phần 1 giống nhau => dạng L02_L02_V002
            if parts[0] == parts[1]:
                return f"{parts[0]}_{parts[2]}"
            else:
                # Dạng L02_V002_0000.00s => lấy 2 phần đầu
                return f"{parts[0]}_{parts[1]}"
        return name  # fallback

    def insert_clip_or_beit(
            self,
            keyframe_id: str,
            vector: list[float],
            timestamp: float,
            object_ids,  # chấp nhận list[int] hoặc sẵn CSV (str)
            lab_colors_flat18  # chấp nhận list[float] hoặc sẵn CSV (str)
    ):
        if len(vector) != self.vector_dim:
            raise ValueError(f"vector dim mismatch: expected {self.vector_dim}, got {len(vector)}")

        # ép về CSV nếu là list
        object_ids_csv = object_ids if isinstance(object_ids, str) else self._join_csv(object_ids)
        lab_colors_csv = lab_colors_flat18 if isinstance(lab_colors_flat18, str) else self._join_csv(lab_colors_flat18)

        video_id = self.get_video_id(keyframe_id)
        pair = self.config.VIDEO_TO_USER[video_id]
        user_str = ", ".join(pair)

        entities = [
            [keyframe_id],
            [vector],
            [timestamp],
            [object_ids_csv],
            [lab_colors_csv],
            [user_str],
        ]
        self.collection.insert(entities)

    def insert_objects_batch(
            self,
            names: List[str],
            vectors: List[List[float]],
            bboxes_xyxy: List[List[float]],
            colors_lab: List[List[float]]
    ) -> List[int]:
        if self.collection_name != self.config.OBJECT_COLLECTION:
            raise ValueError("insert_objects_batch chỉ dùng cho OBJECT_COLLECTION.")

        n = len(names)
        if not (n and n == len(vectors) == len(bboxes_xyxy) == len(colors_lab)):
            raise ValueError(
                f"Batch size mismatch: names={len(names)}, vectors={len(vectors)}, "
                f"bboxes_xyxy={len(bboxes_xyxy)}, colors_lab={len(colors_lab)}"
            )

        # Validate kích thước
        for v in vectors:
            if len(v) != self.vector_dim:
                raise ValueError(f"vector dim mismatch: expected {self.vector_dim}, got {len(v)}")
        for b in bboxes_xyxy:
            if len(b) != SchemaFactory.BBOX_CAPACITY:
                raise ValueError(f"bbox length must be {SchemaFactory.BBOX_CAPACITY}")
        for c in colors_lab:
            if len(c) != SchemaFactory.LAB3_CAPACITY:
                raise ValueError(f"color_lab length must be {SchemaFactory.LAB3_CAPACITY}")

        # CSV hoá bbox & color theo thiết kế hiện tại
        bboxes_csv = [self._join_csv(b) for b in bboxes_xyxy]
        colors_csv = [self._join_csv(c) for c in colors_lab]
        names = [str(x) for x in names]
        # Thứ tự entities PHẢI khớp schema (bỏ PK auto_id):
        # NAME, VECTOR, BBOX_XYXY, COLOR_LAB
        entities = [
            vectors,
            bboxes_csv,
            colors_csv,
            names,
        ]
        mr = self.collection.insert(entities)
        return [int(pk) for pk in mr.primary_keys]

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

