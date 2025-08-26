# --- START OF FILE app/retrieval_engine.py ---

import logging
import time
import os
import sys
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import numpy as np
import torch
from PIL import Image

# Performance optimization imports
from scipy.optimize import linear_sum_assignment
try:
    import colorspacious
    HAS_COLORSPACIOUS = True
except ImportError:
    HAS_COLORSPACIOUS = False
    logging.warning("colorspacious not available, falling back to faster Euclidean distance")

# Thêm đường dẫn tới các repo phụ thuộc mà không có trong PyPI
sys.path.append('/app/Co_DETR')
sys.path.append('/app/unilm/beit3')

# Import từ các thư viện ML
import open_clip
import sentencepiece as spm
from torchvision import transforms
from modeling_finetune import BEiT3ForRetrieval

# Imports cho ObjectColorDetector
import cv2
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from icecream import ic

# Import từ các module của ứng dụng
from .config import settings
from .database import db_manager
from .caching import CacheManager, cached_embedding, cached_search_results
from .vectorization import vectorized_ops
import numpy as np
from rapidfuzz import fuzz
if not hasattr(np, "asscalar"): 
    np.asscalar = lambda a: a.item() if hasattr(a, "item") else np.asarray(a).item()

logger = logging.getLogger(__name__)

# OPTIMIZATION: Global cache manager instance
cache_manager = CacheManager(
    redis_host=getattr(settings, 'REDIS_HOST', 'localhost'),
    redis_port=getattr(settings, 'REDIS_PORT', 6379),
    memory_cache_size=1000,
    enable_redis=getattr(settings, 'ENABLE_REDIS_CACHE', True)
)

# --- Cấu hình cho BEiT-3 (lấy từ repo gốc) ---
class BEiT3Config:
    def __init__(self):
        self.encoder_embed_dim = 768
        self.encoder_attention_heads = 12
        self.encoder_layers = 12
        self.encoder_ffn_embed_dim = 3072
        self.img_size = 384
        self.patch_size = 16
        self.in_chans = 3
        self.vocab_size = 64010
        self.num_max_bpe_tokens = 64
        self.max_source_positions = 1024
        self.multiway = True
        self.share_encoder_input_output_embed = False
        self.no_scale_embedding = False
        self.layernorm_embedding = False
        self.normalize_output = True
        self.no_output_layer = True
        self.drop_path_rate = 0.1
        self.dropout = 0.0
        self.attention_dropout = 0.0
        self.drop_path = 0.1
        self.activation_dropout = 0.0
        self.max_position_embeddings = 1024
        self.encoder_normalize_before = True
        self.activation_fn = "gelu"
        self.encoder_learned_pos = True
        self.xpos_rel_pos = False
        self.xpos_scale_base = 512
        self.checkpoint_activations = False
        self.deepnorm = False
        self.subln = True
        self.rel_pos_buckets = 0
        self.max_rel_pos = 0
        self.bert_init = False
        self.moe_freq = 0
        self.moe_expert_count = 0
        self.moe_top1_expert = False
        self.moe_gating_use_fp32 = True
        self.moe_eval_capacity_token_fraction = 0.25
        self.moe_second_expert_policy = "random"
        self.moe_normalize_gate_prob_before_dropping = False
        self.use_xmoe = False
        self.fsdp = False
        self.ddp_rank = 0
        self.flash_attention = False
        self.scale_length = 2048
        self.layernorm_eps = 1e-5


class ObjectColorDetector:
    """Sử dụng Co-DETR để phát hiện đối tượng và màu sắc chính của chúng."""

    def __init__(self, device):
        logger.info("Initializing OPTIMIZED ObjectColorDetector (Co-DETR)...")
        self.device = device
        self.model = init_detector(
            Config.fromfile(settings.CO_DETR_CONFIG_PATH),
            settings.CO_DETR_CHECKPOINT_PATH,
            device=self.device
        )
        # Bảng tra cứu màu cơ bản
        self.basic_colors = {
            'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 
            'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 
            'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (128, 128, 128), 
            'orange': (255, 165, 0), 'brown': (165, 42, 42), 'pink': (255, 192, 203), 
            'purple': (128, 0, 128)
        }
        
        # OPTIMIZATION: Precompute LAB colors using vectorization
        self.basic_colors_lab = self._precompute_basic_colors_lab()
        self.color_names = list(self.basic_colors.keys())
        self.color_tree = KDTree(np.array(list(self.basic_colors.values())))
        logger.info("✅ Optimized Co-DETR model loaded with precomputed color space.")

    def _precompute_basic_colors_lab(self) -> dict:
        """OPTIMIZED: Batch convert basic colors to LAB using vectorization."""
        rgb_array = np.array(list(self.basic_colors.values()), dtype=np.float32)
        lab_array = vectorized_ops.batch_color_conversion_rgb_to_lab(rgb_array)
        
        lab_dict = {}
        for i, name in enumerate(self.basic_colors.keys()):
            lab_dict[name] = LabColor(*lab_array[i])
        return lab_dict

    def _convert_basic_colors_to_lab(self) -> dict:
        """DEPRECATED: Use _precompute_basic_colors_lab instead."""
        return self._precompute_basic_colors_lab()

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """OPTIMIZED: Cached RGB to LAB conversion using vectorization."""
        # Check cache first
        cached_result = cache_manager.get_cached_color_conversion(rgb)
        if cached_result is not None:
            return cached_result
        
        # Compute using vectorized operation
        rgb_array = np.array([rgb], dtype=np.float32)
        lab_array = vectorized_ops.batch_color_conversion_rgb_to_lab(rgb_array)
        lab_tuple = tuple(lab_array[0])
        
        # Cache result
        cache_manager.cache_color_conversion(rgb, lab_tuple)
        return lab_tuple

    def _get_closest_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """OPTIMIZED: Tìm tên màu gần nhất với vectorized distance calculation."""
        lab_color = self._rgb_to_lab(rgb)
        lab_ref_values = [
            (lab_obj.lab_l, lab_obj.lab_a, lab_obj.lab_b) 
            for lab_obj in self.basic_colors_lab.values()
        ]
        
        # Use vectorized distance calculation
        distances = vectorized_ops.vectorized_color_distances([lab_color], lab_ref_values)[0]
        min_idx = np.argmin(distances)
        return self.color_names[min_idx]

    def detect(self, image_path: str) -> Tuple[
        List[Tuple[float, float, float]],
        Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]]
    ]:
        try:
            result = inference_detector(self.model, image_path)
            if isinstance(result, tuple):
                result = result[0]

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Không đọc được ảnh.")

            # --- MÀU CHỦ ĐẠO TOÀN ẢNH ---
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            flat_pixels = img_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(flat_pixels)
            dominant_rgb = kmeans.cluster_centers_.astype(int)

            dominant_colors_lab = [self._rgb_to_lab(tuple(color)) for color in dominant_rgb]

            # --- MÀU CỦA TỪNG OBJECT ---
            object_colors_lab = {}

            for class_id, bboxes in enumerate(result):
                if class_id >= len(self.model.CLASSES):
                    continue
                class_name = self.model.CLASSES[class_id]
                for bbox in bboxes:
                    if bbox[4] < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3)
                    kmeans_obj = KMeans(n_clusters=1, random_state=0, n_init=10).fit(crop_rgb)
                    dom_rgb = kmeans_obj.cluster_centers_[0].astype(int)
                    lab_color = self._rgb_to_lab(tuple(dom_rgb))

                    if class_name not in object_colors_lab:
                        object_colors_lab[class_name] = []
                    # Thêm vị trí bounding box vào kết quả
                    object_colors_lab[class_name].append((lab_color, (x1, y1, x2, y2)))

            return dominant_colors_lab, object_colors_lab

        except Exception as e:
            print(f"Error during detection: {e}")
            return [], {}


class HybridRetriever:
    """Công cụ truy xuất lai, kết hợp các model AI và tìm kiếm đa phương thức."""

    def __init__(self):
        self.db_manager = db_manager
        self.device = settings.DEVICE
        self.initialized = False
        # Placeholders for models and tokenizers
        self.clip_model, self.clip_preprocess, self.clip_tokenizer = None, None, None
        self.beit3_model, self.beit3_preprocess, self.beit3_sp_model = None, None, None
        self.object_detector: Optional[ObjectColorDetector] = None
        
        # OPTIMIZATION: Precomputed embeddings for common objects
        self.precomputed_embeddings = {}

    async def initialize(self):
        """OPTIMIZED: Khởi tạo retriever với precomputed embeddings."""
        if self.initialized: 
            return
        logger.info("Initializing OPTIMIZED Hybrid Retriever engine...")
        if not self.db_manager.milvus_connected or not self.db_manager.elasticsearch_connected:
            raise RuntimeError("Database connections must be established before initializing the retriever.")
        self._load_models()
        await self._precompute_common_embeddings()
        self.initialized = True
        logger.info("✅ Optimized Hybrid Retriever initialized successfully.")

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB tuple to CIELAB tuple."""
        rgb_obj = sRGBColor(*rgb, is_upscaled=True)
        lab_obj = convert_color(rgb_obj, LabColor)
        return (lab_obj.lab_l, lab_obj.lab_a, lab_obj.lab_b)

    def _load_models(self):
        """Tải tất cả các mô hình AI cần thiết vào đúng device."""
        logger.info(f"Loading AI models onto device: '{self.device}'")
        
        # 1. Tải CLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-H-14', pretrained=settings.CLIP_MODEL_PATH, device=self.device)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        logger.info("  - CLIP model loaded.")
        
        # 2. Tải BEiT-3
        self.beit3_model = BEiT3ForRetrieval(BEiT3Config())
        checkpoint = torch.load(settings.BEIT3_MODEL_PATH, map_location="cpu")
        self.beit3_model.load_state_dict(checkpoint["model"])
        self.beit3_model = self.beit3_model.to(self.device).eval()
        self.beit3_sp_model = spm.SentencePieceProcessor()
        self.beit3_sp_model.load(settings.BEIT3_SPM_PATH)
        self.beit3_preprocess = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        logger.info("  - BEiT-3 model loaded.")
        
        # 3. Tải Co-DETR
        self.object_detector = ObjectColorDetector(device=self.device)

    async def _precompute_common_embeddings(self):
        """OPTIMIZATION: Precompute embeddings for common objects and cache them."""
        common_objects = [
            "person", "man", "woman", "child", "people",
            "car", "truck", "bus", "motorcycle", "vehicle", 
            "building", "house", "office", "store", "shop",
            "tree", "grass", "sky", "cloud", "water",
            "street", "road", "sidewalk", "bridge",
            "food", "table", "chair", "computer", "phone"
        ]
        
        logger.info(f"Precomputing embeddings for {len(common_objects)} common objects...")
        
        for obj_name in common_objects:
            # Check if already cached
            cached_embedding = cache_manager.get_cached_object_embedding(obj_name)
            if cached_embedding is not None:
                self.precomputed_embeddings[obj_name] = cached_embedding
            else:
                # Compute and cache
                embedding = self.get_clip_text_embedding_direct(obj_name)
                self.precomputed_embeddings[obj_name] = embedding
                cache_manager.cache_object_embedding(obj_name, embedding)
        
        logger.info(f"✅ Precomputed {len(self.precomputed_embeddings)} object embeddings")

    # --- CÁC HÀM MÃ HÓA (EMBEDDING) ---
    def get_clip_text_embedding_direct(self, text: str) -> np.ndarray:
        """Direct embedding computation without caching decorator."""
        with torch.no_grad():
            tokens = self.clip_tokenizer([text]).to(self.device)
            text_emb = self.clip_model.encode_text(tokens).cpu().numpy()[0]
            return text_emb / np.linalg.norm(text_emb, axis=0)

    @cached_embedding(cache_manager, 'clip')
    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        """OPTIMIZED: Cached CLIP text embedding."""
        return self.get_clip_text_embedding_direct(text)

    @cached_embedding(cache_manager, 'beit3')
    def get_beit3_text_embedding(self, text: str) -> np.ndarray:
        """OPTIMIZED: Cached BEiT-3 text embedding."""
        with torch.no_grad():
            text_ids = self.beit3_sp_model.encode_as_ids(text)
            text_padding_mask = [0] * len(text_ids)
            text_ids_tensor = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            text_padding_mask_tensor = torch.tensor(text_padding_mask, dtype=torch.long).unsqueeze(0).to(self.device)
            _, text_emb = self.beit3_model(
                text_description=text_ids_tensor, 
                text_padding_mask=text_padding_mask_tensor,
                only_infer=True
            )
            return text_emb.cpu().numpy()[0]

    def get_object_embedding_fast(self, object_name: str) -> np.ndarray:
        """OPTIMIZED: Fast object embedding lookup with precomputed cache."""
        obj_name_lower = object_name.lower()
        
        # Check precomputed embeddings first
        if obj_name_lower in self.precomputed_embeddings:
            return self.precomputed_embeddings[obj_name_lower].copy()
        
        # Check cache
        cached_embedding = cache_manager.get_cached_object_embedding(obj_name_lower)
        if cached_embedding is not None:
            return cached_embedding
        
        # Compute and cache
        embedding = self.get_clip_text_embedding(object_name)
        cache_manager.cache_object_embedding(obj_name_lower, embedding)
        return embedding

    def _vectorized_color_distances(self, colors1: List[Tuple[float, float, float]], 
                                   colors2: List[Tuple[float, float, float]]) -> np.ndarray:
        """OPTIMIZED: Use vectorized operations for color distances."""
        return vectorized_ops.vectorized_color_distances(colors1, colors2)

    def _compare_color(self, color1: Tuple[float, float, float], color2: Tuple[float, float, float]) -> float:
        """
        OPTIMIZED: Fallback single color comparison.
        Uses vectorized method internally for consistency.
        """
        distances = self._vectorized_color_distances([color1], [color2])
        return distances[0, 0] if distances.size > 0 else 0.0

    def _compare_bbox(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """OPTIMIZED: Single bbox comparison using vectorized backend."""
        iou_matrix = vectorized_ops.batch_bbox_iou(np.array([bbox1]), np.array([bbox2]))
        return float(iou_matrix[0, 0])

    def _compare_bbox_vectorized(self, bboxes1: List[Tuple[int, int, int, int]], 
                               bboxes2: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """OPTIMIZED: Vectorized IoU computation for multiple bounding boxes."""
        return vectorized_ops.batch_bbox_iou(np.array(bboxes1), np.array(bboxes2))

    def _parse_video_id_from_kf(self, kf: str) -> Tuple[str, str]:
        """
        Nhận keyframe: L02_L02_V002_1130.04s.jpg, K05_V002_1130.04s.jpg, hoặc L02_V002_1130.04s.jpg
        Trả về: (video_id, kf_id) dạng ('L02_V002', 'L02_V002_1130.04s.jpg')
        """
        name = os.path.splitext(os.path.basename(kf))[0]
        parts = name.split("_")

        # bỏ trùng L02_L02 -> giữ 1
        unique_parts = []
        for p in parts:
            if not unique_parts or unique_parts[-1] != p:
                unique_parts.append(p)

        # timestamp là phần cuối
        timestamp = unique_parts[-1]
        # lấy mã sequence (Lxx, Kxx, ...) và Vxxx
        sequence_code = None
        v_code = None
        for p in unique_parts:
            # Tìm pattern: chữ cái + số (ví dụ: L02, K05, M10, ...)
            if len(p) >= 2 and p[0].isalpha() and p[1:].isdigit():
                if p.upper().startswith("V"):
                    v_code = p.upper()
                elif sequence_code is None:  # Lấy mã sequence đầu tiên (L, K, M, ...)
                    sequence_code = p.upper()

        video_id = f"{sequence_code}_{v_code}"
        kf_id = f"{video_id}_{timestamp}.jpg"
        return video_id, kf_id

    def _split_csv_ints(self, s: Optional[str]) -> List[int]:
        """'1,2,3' -> [1,2,3]; bỏ qua phần tử rỗng/lỗi."""
        if not s:
            return []
        out = []
        for p in s.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                out.append(int(p))
            except ValueError:
                # phòng trường hợp '12.0'
                try:
                    out.append(int(float(p)))
                except ValueError:
                    pass
        return out

    def _split_csv_floats(self, s: Optional[str]) -> List[float]:
        """'1.2,3,4.5' -> [1.2,3.0,4.5]; bỏ qua phần tử rỗng/lỗi."""
        if not s:
            return []
        out = []
        for p in s.split(","):
            p = p.strip()
            if p == "":
                continue
            try:
                out.append(float(p))
            except ValueError:
                pass
        return out

    def _parse_lab_colors18(self, s: Optional[str]) -> List[Tuple[float, float, float]]:
        """
        Nhận chuỗi 18 số (L,a,b * 6 màu), trả về list 6 tuple (L,a,b).
        Nếu thiếu thì pad 0.0; nếu thừa thì cắt bớt.
        """
        vals = self._split_csv_floats(s)
        if len(vals) < 18:
            vals += [0.0] * (18 - len(vals))
        elif len(vals) > 18:
            vals = vals[:18]

        lab6 = []
        i = 0
        while i + 2 < 18:
            lab6.append((vals[i], vals[i + 1], vals[i + 2]))
            i += 3
        return lab6

    # --- LOGIC TÌM KIẾM CHÍNH ---
    @cached_search_results(cache_manager)
    async def search(self, text_query: str, mode: str, user_query: str, object_filters: Optional[Dict],
                    color_filters: Optional[List], ocr_query: Optional[str], asr_query: Optional[str], 
                    top_k: int) -> List[Dict[str, Any]]:
        if not self.initialized: 
            raise RuntimeError("Retriever is not initialized.")
        
        start_time = time.time()
        candidate_info: Dict[str, Dict[str, Any]] = {}
        num_initial_candidates = top_k

        # BƯỚC 1: LẤY ỨNG VIÊN BAN ĐẦU
        tasks = []
        
        if mode in ['hybrid', 'clip']:
            clip_vector = self.get_clip_text_embedding(text_query).tolist()
            tasks.append(self._search_milvus_async(settings.CLIP_COLLECTION, clip_vector, 
                                                  num_initial_candidates, None, user_query, 'clip'))
        if mode == 'beit3':
            beit3_vector = self.get_beit3_text_embedding(text_query).tolist()
            tasks.append(self._search_milvus_async(settings.BEIT3_COLLECTION, beit3_vector, 
                                                  num_initial_candidates, None, user_query, 'beit3'))
        # OPTIMIZED: Parallel execution of vector searches
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        # Process search results
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Search task {i} failed: {result}")
                continue

            search_type = 'clip' if (mode in ['hybrid', 'clip'] and i == 0) else 'beit3'
            self._process_search_results(result, candidate_info, search_type)
        # BƯỚC 2: TINH CHỈNH (chỉ cho mode hybrid)
        refinement_tasks = []
        if mode == 'hybrid' and candidate_info:
            refinement_tasks.append(self._async_hybrid_reranking(candidate_info, text_query))
        # BƯỚC 3: TĂNG ĐIỂM với Object/Color (parallel)
        if object_filters or color_filters:
            refinement_tasks.append(self._apply_object_color_filters_optimized(
                candidate_info, object_filters, color_filters, top_k))

        # BƯỚC 4: LỌC CỨNG với OCR (parallel)
        if ocr_query:
            refinement_tasks.append(self._async_apply_ocr_filter(candidate_info, ocr_query))

        # OPTIMIZED: Execute all refinement steps in parallel
        if refinement_tasks:
            await asyncio.gather(*refinement_tasks, return_exceptions=True)

        # BƯỚC 5: XẾP HẠNG VÀ TRẢ VỀ
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        final_results = self._format_results(sorted_results[:top_k])
        
        search_time = time.time() - start_time
        logger.info(f"OPTIMIZED Search completed in {search_time:.2f}s with {len(final_results)} results")
        
        # Log cache performance for non-cached queries
        if search_time > 0.1:  # Only log for non-cached queries
            cache_stats = cache_manager.get_cache_stats()
            logger.info(f"Cache stats: embedding_hit_rate={cache_stats.get('embedding_hit_rate', 0):.1f}%, search_hit_rate={cache_stats.get('search_hit_rate', 0):.1f}%")
        
        return final_results

    async def _search_milvus_async(self, collection_name: str, vector: List[float], top_k: int, 
                                  expr: Optional[str] = None, user_query: str = "", 
                                  search_type: str = "") -> List[Dict]:
        """Async wrapper for Milvus search to enable parallel execution."""
        return await self._search_milvus(collection_name, vector, top_k, expr, user_query)

    def _process_search_results(self, search_results: List[Dict], candidate_info: Dict[str, Dict[str, Any]], 
                               search_type: str):
        """Process search results and populate candidate_info."""
        for hit in search_results:
            kf_id = hit["entity"]["keyframe_id"]
            vid, kf_id = self._parse_video_id_from_kf(kf_id)
            #score = 1.0 / (1.0 + hit['distance'])
            score = max(0.0, 1.0 - hit['distance'])  # COSINE distance in [0,2], similarity in [1,-1]
            obj_ids = self._split_csv_ints(hit['entity']['object_ids'])
            lab6 = self._parse_lab_colors18(hit['entity']['lab_colors'])

            if kf_id in candidate_info:
                # Already exists from another search
                candidate_info[kf_id][f'{search_type}_score'] = score
                candidate_info[kf_id]['score'] += score
                candidate_info[kf_id]['reasons'].append(f"{search_type.upper()} match ({score:.3f})")
                
                # Fill missing info
                if not candidate_info[kf_id].get('object_ids') and obj_ids:
                    candidate_info[kf_id]['object_ids'] = obj_ids
                if not candidate_info[kf_id].get('lab_colors6') and lab6:
                    candidate_info[kf_id]['lab_colors6'] = lab6
            else:
                # New entry
                candidate_info[kf_id] = {
                    "keyframe_id": kf_id,
                    "timestamp": hit['entity']['timestamp'],
                    "object_ids": obj_ids,
                    "lab_colors6": lab6,
                    f"{search_type}_score": score,
                    "score": score,
                    "reasons": [f"{search_type.upper()} match ({score:.3f})"],
                }

    def build_expr(self, expr: Optional[str] = None, user_query: str = "") -> Optional[str]:
        user_list = [
            "Gia Nguyên, Duy Bảo", "Gia Nguyên, Duy Khương", "Gia Nguyên, Minh Tâm", "Gia Nguyên, Lê Hiếu",
            "Duy Bảo, Duy Khương", "Duy Bảo, Minh Tâm", "Duy Bảo, Lê Hiếu",
            "Duy Khương, Minh Tâm", "Duy Khương, Lê Hiếu", "Minh Tâm, Lê Hiếu"
        ]

        if user_query:
            filtered_users = [u for u in user_list if user_query in u]
            if filtered_users:
                user_expr = f'user in {filtered_users}'
                if expr:
                    expr = f'({expr}) && ({user_expr})'
                else:
                    expr = user_expr
        return expr

    async def _search_milvus(self, collection_name: str, vector: List[float], top_k: int, 
                            expr: Optional[str] = None, user_query: str = "") -> List[Dict]:
        # Tải sẵn các collection (idempotent)
        await self.db_manager._load_milvus_collections()

        collection = self.db_manager.get_collection(collection_name)
        if not collection:
            return []

        # Chọn output_fields theo collection
        if collection_name in (settings.CLIP_COLLECTION, settings.BEIT3_COLLECTION):
            output_fields = ["keyframe_id", "timestamp", "object_ids", "lab_colors", "user"]
        elif collection_name == settings.OBJECT_COLLECTION:
            output_fields = ["object_id", "bbox_xyxy", "color_lab"]
        else:
            output_fields = []
        
        expr_new = self.build_expr(expr, user_query)

        search_results = collection.search(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=expr_new,
            output_fields=output_fields,
        )[0]

        hits = []
        for hit in search_results:
            hit_data = hit.entity.to_dict()["entity"]

            if collection_name in (settings.CLIP_COLLECTION, settings.BEIT3_COLLECTION):
                raw_kf = hit_data.get("keyframe_id", "")
                if isinstance(raw_kf, str):
                    pos = raw_kf.lower().find(".jpg")
                    kf_clean = raw_kf[:pos + 4] if pos != -1 else raw_kf
                else:
                    kf_clean = raw_kf

                vid, kf_normalized = self._parse_video_id_from_kf(kf_clean)

                hits.append({
                    "id": kf_normalized,
                    "distance": hit.distance,
                    "entity": {
                        "keyframe_id": kf_normalized,
                        "timestamp": hit_data.get("timestamp"),
                        "object_ids": hit_data.get("object_ids"),  # CSV string
                        "lab_colors": hit_data.get("lab_colors"),  # CSV string
                    },
                })

            elif collection_name == settings.OBJECT_COLLECTION:
                hits.append({
                    "id": hit_data.get("object_id"),
                    "distance": hit.distance,
                    "entity": {
                        "bbox_xyxy": hit_data.get("bbox_xyxy"),  # CSV string
                        "color_lab": hit_data.get("color_lab"),  # CSV string
                    },
                })

        return hits

    async def _async_hybrid_reranking(self, candidate_info: Dict[str, Dict], text_query: str):
        """OPTIMIZED: Async hybrid reranking with better error handling."""
        beit3_collection = self.db_manager.get_collection(settings.BEIT3_COLLECTION)
        if not beit3_collection:
            logger.warning("BEIT-3 collection not available for reranking")
            return
            
        candidate_kf_ids = list(candidate_info.keys())
        if not candidate_kf_ids:
            return
            
        try:
            logger.debug(f"Hybrid reranking for keyframes: {candidate_kf_ids[:5]}...")

            # Convert normalized keyframe IDs back to database format for querying
            all_possible_kf_ids = set()
            for kf_id in candidate_kf_ids:
                all_possible_kf_ids.add(kf_id)  # normalized format
                # Try to generate database format (with potential duplicate prefix)
                parts = kf_id.replace('.jpg', '').split('_')
                if len(parts) >= 3 and parts[0].startswith('L') and parts[1].startswith('V'):
                    # L02_V002_123.45s -> L02_L02_V002_123.45s
                    db_format = f"{parts[0]}_{parts[0]}_{parts[1]}_{parts[2]}.jpg"
                    all_possible_kf_ids.add(db_format)

            kf_ids_list = list(all_possible_kf_ids)
            res = beit3_collection.query(
                expr=f'keyframe_id in {kf_ids_list}', 
                output_fields=["keyframe_id", "vector"]
            )

            # Map both database format and normalized format to vectors
            beit3_vector_map = {}
            for item in res:
                db_kf_id = item['keyframe_id']
                vid, normalized_kf_id = self._parse_video_id_from_kf(db_kf_id)
                beit3_vector_map[normalized_kf_id] = item['vector']

            # Debug: log found vs missing keyframes
            found_kfs = set(beit3_vector_map.keys())
            missing_kfs = set(candidate_kf_ids) - found_kfs
            if missing_kfs:
                logger.warning(f"BEIT-3 vectors missing for {len(missing_kfs)}/{len(candidate_kf_ids)} keyframes")
            else:
                logger.info(f"Found BEIT-3 vectors for all {len(candidate_kf_ids)} keyframes")

            beit3_query_vector = np.array(self.get_beit3_text_embedding(text_query))
            
            # OPTIMIZED: Vectorized distance computation
            kf_vectors = []
            kf_ids_ordered = []
            for kf_id in candidate_kf_ids:
                if kf_id in beit3_vector_map:
                    kf_vectors.append(beit3_vector_map[kf_id])
                    kf_ids_ordered.append(kf_id)
            
            if kf_vectors:
                kf_matrix = np.array(kf_vectors)  # shape: (n, embedding_dim)
                # Compute all distances at once
                distances = np.linalg.norm(kf_matrix - beit3_query_vector[np.newaxis, :], axis=1)
                
                for i, kf_id in enumerate(kf_ids_ordered):
                    dist = distances[i]
                    beit3_score = 1.0 / (1.0 + dist)
                    info = candidate_info[kf_id]
                    info['score'] = (0.4 * info.get('clip_score', 0)) + (0.6 * beit3_score)
                    info['beit3_score'] = beit3_score
                    info['reasons'].append(f"BEIT-3 refine ({beit3_score:.3f})")
            
            # Handle missing vectors
            for kf_id in candidate_kf_ids:
                if kf_id not in beit3_vector_map:
                    candidate_info[kf_id]['score'] *= 0.8
                    candidate_info[kf_id]['reasons'].append("BEIT-3 vector missing")
                    
        except Exception as e:
            logger.error(f"BEIT-3 reranking failed: {e}", exc_info=True)

    async def _batch_search_milvus_objects(self, all_object_ids: List[int], obj_vector: List[float]) -> Dict[int, Dict]:
        """
        OPTIMIZED: Single batch query for all object IDs instead of individual queries.
        Returns dict mapping object_id -> search result data
        """
        if not all_object_ids:
            return {}
            
        # Remove duplicates while preserving order
        unique_obj_ids = list(dict.fromkeys(all_object_ids))
        
        try:
            expr = f"object_id in [{','.join(map(str, unique_obj_ids))}]"
            limit = max(len(unique_obj_ids), 1)
            obj_hits = await self._search_milvus(settings.OBJECT_COLLECTION, obj_vector, limit, expr=expr)
            
            # Create lookup dict for fast access
            results = {}
            for hit in obj_hits:
                obj_id = hit.get("id")
                if obj_id:
                    results[obj_id] = hit
                    
            return results
            
        except Exception as e:
            logger.error(f"Batch object search failed: {e}", exc_info=True)
            return {}

    async def _apply_object_color_filters_optimized(
        self,
        candidate_info: Dict[str, Dict[str, Any]],
        object_filters: Optional[Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]]],
        color_filters: Optional[List[Tuple[float, float, float]]],
        top_k: int
    ):
        """OPTIMIZED: Parallel processing with vectorized operations and batch queries."""
        
        # Performance parameters
        W_VEC = 0.6
        W_COLOR = 0.3
        W_BBOX = 0.4
        SIGMA_COLOR = 20.0
        MAX_DELTA_E = 50.0
        MIN_IOU = 0.30
        ALPHA = 0.7
        BETA = 0.3
        TAU_S = 0.5
        W_OBJ = 0.20

        def _ensure_lab(c: Optional[Tuple[float, float, float]]):
            """Convert RGB to LAB if needed."""
            if c is None:
                return None
            L, A, B = c
            # Heuristics: if all values are in [0..255] and at least one > 1 -> assume RGB
            if 0 <= L <= 255 and 0 <= A <= 255 and 0 <= B <= 255 and (L > 1 or A > 1 or B > 1):
                return self._rgb_to_lab((int(L), int(A), int(B)))
            return (float(L), float(A), float(B))  # assume already LAB

        def _sim_from_delta(d: float, sigma: float = 20.0) -> float:
            """Similarity from ΔE: exp(-(ΔE/σ)²)."""
            return np.exp(-(d / sigma) ** 2)

        # ===== COMPREHENSIVE OBJECT FILTERS =====
        if object_filters:
            norm_object_filters = self._normalize_object_filters(object_filters)

            for obj_label, constraint_list in norm_object_filters.items():
                logger.info(f"Processing object filter: '{obj_label}' with {len(constraint_list)} constraints")
                
                # Get object embedding
                obj_vector = self.get_object_embedding_fast(obj_label).tolist()
                
                # Collect all object IDs for batch processing
                all_object_ids = []
                candidate_objects = {}
                
                for kf_id, info in candidate_info.items():
                    obj_ids = info.get("object_ids") or []
                    if obj_ids:
                        candidate_objects[kf_id] = obj_ids
                        all_object_ids.extend(obj_ids)
                
                if not all_object_ids:
                    continue
                
                # Batch query objects
                batch_results = await self._batch_search_milvus_objects(all_object_ids, obj_vector)
                
                # Process each candidate keyframe
                for kf_id, obj_ids in candidate_objects.items():
                    obj_hits = [batch_results[obj_id] for obj_id in obj_ids if obj_id in batch_results]
                    
                    if not obj_hits:
                        continue

                    # Apply comprehensive constraint matching
                    final_boost, match_details = self._apply_comprehensive_object_matching(
                        constraint_list, obj_hits, obj_label
                    )
                    
                    if final_boost > 0:
                        candidate_info[kf_id]["score"] += final_boost
                        candidate_info[kf_id].setdefault("reasons", []).extend(match_details)

        # ===== COLOR FILTERS (OPTIMIZED) =====
        if color_filters:
            queries_lab = []
            for qc in color_filters:
                if qc is not None:
                    queries_lab.append(self._rgb_to_lab(tuple(qc)))
                    
            if queries_lab:
                alpha, beta = 0.7, 0.3
                w_color = 0.15
                tau = 15.0

                for kf_id, info in candidate_info.items():
                    palette = info.get("lab_colors6") or []
                    if not palette:
                        continue

                    m = len(queries_lab)
                    n = len(palette)

                    # OPTIMIZED: Vectorized distance matrix computation
                    distance_matrix = self._vectorized_color_distances(queries_lab, palette)  # (m, n)
                    similarity_matrix = np.exp(-(distance_matrix / 20.0) ** 2)  # Vectorized
                    cost_matrix = 1.0 - similarity_matrix

                    # OPTIMIZED: scipy Hungarian algorithm
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    # Calculate metrics
                    sim_sum = 0.0
                    real_pairs = 0
                    
                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if row_idx < m and col_idx < n:
                            sim_sum += similarity_matrix[row_idx, col_idx]
                            real_pairs += 1

                    S_hung = (sim_sum / m) if m > 0 else 0.0

                    # Coverage: vectorized minimum distance computation
                    min_distances = np.min(distance_matrix, axis=1)  # min distance for each query color
                    covered = np.sum(min_distances <= tau)
                    C = (covered / m) if m > 0 else 0.0

                    S_color = alpha * S_hung + beta * C
                    boost = w_color * S_color

                    if boost > 0:
                        info["score"] += boost
                        info.setdefault("reasons", []).append(
                            f"Color match (Hungarian): +{boost:.3f} (S={S_color:.3f}, cov={C:.2f})"
                        )

    def _normalize_object_filters(self, object_filters: Dict) -> Dict:
        """
        Comprehensive object filters normalization supporting all cases:
        1. Empty array: object-only search -> {"person": []}
        2. Color only: color constraint -> {"person": [[L,a,b]]}  
        3. BBox only: spatial constraint -> {"person": [[x1,y1,x2,y2]]}
        4. Full constraint: color + bbox -> {"person": [[[L,a,b], [x1,y1,x2,y2]]]}
        5. Mixed constraints: multiple conditions -> {"person": [[], [L,a,b], [[L,a,b], [x1,y1,x2,y2]]]}
        """
        norm: Dict[str, List[Any]] = {}
        
        for obj, items in object_filters.items():
            if not isinstance(items, (list, tuple)):
                logger.warning(f"Invalid object_filters['{obj}'] format: expected array, got {type(items)}")
                continue
                
            # Handle empty array case: object-only search
            if len(items) == 0:
                norm[obj] = [{'type': 'object_only'}]
                logger.info(f"Object filter '{obj}': object-only search (no constraints)")
                continue
            
            normalized_items = []
            
            for idx, item in enumerate(items):
                try:
                    normalized_item = self._normalize_single_object_constraint(obj, item, idx)
                    if normalized_item:
                        normalized_items.append(normalized_item)
                except Exception as e:
                    logger.warning(f"Failed to normalize object_filters['{obj}'][{idx}]: {e}")
                    continue
            
            if normalized_items:
                norm[obj] = normalized_items
                logger.info(f"Object filter '{obj}': {len(normalized_items)} constraints normalized")
            else:
                logger.warning(f"No valid constraints for object '{obj}', skipping")
                
        return norm

    def _normalize_single_object_constraint(self, obj_name: str, item: Any, index: int) -> Optional[Dict]:
        """
        Normalize a single object constraint item.
        Supports flexible input formats:
        
        Input formats:
        - [] -> object-only search  
        - [L, a, b] -> color constraint (LAB or RGB auto-detected)
        - [x1, y1, x2, y2] -> bbox constraint
        - [[L, a, b], [x1, y1, x2, y2]] -> full constraint
        - {"color": [L, a, b], "bbox": [x1, y1, x2, y2]} -> dict format
        """
        
        if not isinstance(item, (list, tuple, dict)):
            return None
            
        # Handle dict format
        if isinstance(item, dict):
            return self._normalize_dict_constraint(item)
            
        # Handle list/tuple formats
        if len(item) == 0:
            return {'type': 'object_only'}
            
        elif len(item) == 3:
            # Could be color [L,a,b] or [R,G,B]
            try:
                color_vals = [float(x) for x in item]
                # Auto-detect RGB vs LAB
                if all(0 <= x <= 255 for x in color_vals) and any(x > 1 for x in color_vals):
                    # Likely RGB, convert to LAB
                    lab_color = self._rgb_to_lab(tuple(int(x) for x in color_vals))
                    return {
                        'type': 'color_only',
                        'color_lab': lab_color,
                        'color_original': tuple(color_vals),
                        'color_space': 'RGB->LAB'
                    }
                else:
                    # Assume LAB
                    return {
                        'type': 'color_only', 
                        'color_lab': tuple(color_vals),
                        'color_original': tuple(color_vals),
                        'color_space': 'LAB'
                    }
            except (ValueError, TypeError):
                return None
                
        elif len(item) == 4:
            # Bbox [x1, y1, x2, y2]
            try:
                bbox_vals = [int(x) for x in item]
                return {
                    'type': 'bbox_only',
                    'bbox': tuple(bbox_vals)
                }
            except (ValueError, TypeError):
                return None
                
        elif len(item) == 2:
            # Full constraint [[color], [bbox]]
            color_part, bbox_part = item[0], item[1]
            
            # Validate color part
            if not isinstance(color_part, (list, tuple)) or len(color_part) != 3:
                return None
            # Validate bbox part  
            if not isinstance(bbox_part, (list, tuple)) or len(bbox_part) != 4:
                return None
                
            try:
                color_vals = [float(x) for x in color_part]
                bbox_vals = [int(x) for x in bbox_part]
                
                # Auto-detect color space
                if all(0 <= x <= 255 for x in color_vals) and any(x > 1 for x in color_vals):
                    lab_color = self._rgb_to_lab(tuple(int(x) for x in color_vals))
                    color_space = 'RGB->LAB'
                else:
                    lab_color = tuple(color_vals)
                    color_space = 'LAB'
                    
                return {
                    'type': 'full_constraint',
                    'color_lab': lab_color,
                    'color_original': tuple(color_vals),
                    'color_space': color_space,
                    'bbox': tuple(bbox_vals)
                }
            except (ValueError, TypeError):
                return None
        
        return None
    
    def _normalize_dict_constraint(self, item: Dict) -> Optional[Dict]:
        """Handle dict-based constraint format"""
        result = {'type': 'mixed'}
        
        # Handle color
        if 'color' in item:
            color = item['color']
            if isinstance(color, (list, tuple)) and len(color) == 3:
                try:
                    color_vals = [float(x) for x in color]
                    if all(0 <= x <= 255 for x in color_vals) and any(x > 1 for x in color_vals):
                        result['color_lab'] = self._rgb_to_lab(tuple(int(x) for x in color_vals))
                        result['color_space'] = 'RGB->LAB'
                    else:
                        result['color_lab'] = tuple(color_vals)
                        result['color_space'] = 'LAB'
                    result['color_original'] = tuple(color_vals)
                except (ValueError, TypeError):
                    pass
        
        # Handle bbox
        if 'bbox' in item:
            bbox = item['bbox']
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    result['bbox'] = tuple(int(x) for x in bbox)
                except (ValueError, TypeError):
                    pass
        
        # Determine final type
        has_color = 'color_lab' in result
        has_bbox = 'bbox' in result
        
        if has_color and has_bbox:
            result['type'] = 'full_constraint'
        elif has_color:
            result['type'] = 'color_only'
        elif has_bbox:
            result['type'] = 'bbox_only'
        else:
            result['type'] = 'object_only'
            
        return result

    def _apply_comprehensive_object_matching(self, constraint_list: List[Dict], obj_hits: List[Dict], 
                                           obj_label: str) -> Tuple[float, List[str]]:
        """
        Apply comprehensive object matching supporting all constraint types.
        Returns (final_boost, match_details)
        """
        if not constraint_list or not obj_hits:
            return 0.0, []
            
        # Parse object hits data
        O_vec_sim = []
        O_color_lab = []
        O_bbox = []
        
        for hit in obj_hits:
            ent = hit.get("entity", {})
            distance = float(hit.get("distance", 0.0))
            vec_similarity = 1.0 / (1.0 + distance)
            O_vec_sim.append(vec_similarity)
            
            # Parse color
            color_csv = ent.get("color_lab", "")
            color_vals = self._split_csv_floats(color_csv)
            O_color_lab.append(tuple(color_vals) if len(color_vals) == 3 else None)
            
            # Parse bbox
            bbox_csv = ent.get("bbox_xyxy", "")
            bbox_vals = self._split_csv_floats(bbox_csv)
            O_bbox.append(tuple(int(x) for x in bbox_vals) if len(bbox_vals) == 4 else None)
        
        n_objects = len(O_vec_sim)
        if n_objects == 0:
            return 0.0, []
        
        # Process each constraint type
        constraint_results = []
        match_details = []
        
        for idx, constraint in enumerate(constraint_list):
            constraint_type = constraint.get('type', 'unknown')
            
            if constraint_type == 'object_only':
                boost, detail = self._match_object_only(O_vec_sim, obj_label)
                
            elif constraint_type == 'color_only':
                boost, detail = self._match_color_only(
                    constraint, O_vec_sim, O_color_lab, obj_label
                )
                
            elif constraint_type == 'bbox_only':
                boost, detail = self._match_bbox_only(
                    constraint, O_vec_sim, O_bbox, obj_label
                )
                
            elif constraint_type == 'full_constraint':
                boost, detail = self._match_full_constraint(
                    constraint, O_vec_sim, O_color_lab, O_bbox, obj_label
                )
                
            else:
                logger.warning(f"Unknown constraint type: {constraint_type}")
                continue
            
            if boost > 0:
                constraint_results.append(boost)
                match_details.append(detail)
                
        # Aggregate results - use max boost to avoid over-boosting
        final_boost = max(constraint_results) if constraint_results else 0.0
        
        return final_boost, match_details
    
    def _match_object_only(self, vec_similarities: List[float], obj_label: str) -> Tuple[float, str]:
        """Match object without any constraints - pure semantic similarity"""
        if not vec_similarities:
            return 0.0, ""
            
        max_similarity = max(vec_similarities)
        
        # Object-only boost weights
        W_OBJ_ONLY = 0.15
        SIMILARITY_THRESHOLD = 0.5
        
        if max_similarity >= SIMILARITY_THRESHOLD:
            boost = W_OBJ_ONLY * max_similarity
            detail = f"Object '{obj_label}' match +{boost:.3f} (similarity={max_similarity:.3f})"
            return boost, detail
        
        return 0.0, ""
    
    def _match_color_only(self, constraint: Dict, vec_similarities: List[float], 
                         object_colors: List[Optional[Tuple]], obj_label: str) -> Tuple[float, str]:
        """Match object with color constraint only"""
        query_color = constraint.get('color_lab')
        if not query_color or not vec_similarities:
            return 0.0, ""
        
        # Find best matching object by color + vector similarity
        best_combined_score = 0.0
        best_vec_sim = 0.0
        best_color_sim = 0.0
        
        W_VEC = 0.6
        W_COLOR = 0.4
        MAX_DELTA_E = 50.0
        SIGMA_COLOR = 20.0
        
        for i, (vec_sim, obj_color) in enumerate(zip(vec_similarities, object_colors)):
            if obj_color is None:
                continue
                
            # Color similarity
            color_distance = self._compare_color(query_color, obj_color)
            if color_distance <= MAX_DELTA_E:
                color_sim = np.exp(-(color_distance / SIGMA_COLOR) ** 2)
                combined_score = W_VEC * vec_sim + W_COLOR * color_sim
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_vec_sim = vec_sim
                    best_color_sim = color_sim
        
        if best_combined_score > 0.5:
            W_COLOR_ONLY = 0.12
            boost = W_COLOR_ONLY * best_combined_score
            detail = f"Object '{obj_label}' color match +{boost:.3f} (vec={best_vec_sim:.3f}, color={best_color_sim:.3f})"
            return boost, detail
            
        return 0.0, ""
    
    def _match_bbox_only(self, constraint: Dict, vec_similarities: List[float],
                        object_bboxes: List[Optional[Tuple]], obj_label: str) -> Tuple[float, str]:
        """Match object with bbox constraint only"""
        query_bbox = constraint.get('bbox')
        if not query_bbox or not vec_similarities:
            return 0.0, ""
        
        best_combined_score = 0.0
        best_vec_sim = 0.0
        best_iou = 0.0
        
        W_VEC = 0.7
        W_BBOX = 0.3
        MIN_IOU = 0.3
        
        for i, (vec_sim, obj_bbox) in enumerate(zip(vec_similarities, object_bboxes)):
            if obj_bbox is None:
                continue
                
            iou = self._compare_bbox(query_bbox, obj_bbox)
            if iou >= MIN_IOU:
                combined_score = W_VEC * vec_sim + W_BBOX * iou
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_vec_sim = vec_sim
                    best_iou = iou
        
        if best_combined_score > 0.5:
            W_BBOX_ONLY = 0.10
            boost = W_BBOX_ONLY * best_combined_score
            detail = f"Object '{obj_label}' bbox match +{boost:.3f} (vec={best_vec_sim:.3f}, IoU={best_iou:.3f})"
            return boost, detail
            
        return 0.0, ""
    
    def _match_full_constraint(self, constraint: Dict, vec_similarities: List[float],
                             object_colors: List[Optional[Tuple]], 
                             object_bboxes: List[Optional[Tuple]], obj_label: str) -> Tuple[float, str]:
        """Match object with both color and bbox constraints"""
        query_color = constraint.get('color_lab')
        query_bbox = constraint.get('bbox')
        
        if not query_color or not query_bbox or not vec_similarities:
            return 0.0, ""
        
        best_combined_score = 0.0
        best_details = {}
        
        W_VEC = 0.5
        W_COLOR = 0.3
        W_BBOX = 0.2
        MAX_DELTA_E = 50.0
        SIGMA_COLOR = 20.0
        MIN_IOU = 0.3
        
        for i, (vec_sim, obj_color, obj_bbox) in enumerate(zip(vec_similarities, object_colors, object_bboxes)):
            if obj_color is None or obj_bbox is None:
                continue
                
            # Color similarity
            color_distance = self._compare_color(query_color, obj_color)
            if color_distance > MAX_DELTA_E:
                continue
            color_sim = np.exp(-(color_distance / SIGMA_COLOR) ** 2)
            
            # Bbox similarity
            iou = self._compare_bbox(query_bbox, obj_bbox)
            if iou < MIN_IOU:
                continue
                
            # Combined score
            combined_score = W_VEC * vec_sim + W_COLOR * color_sim + W_BBOX * iou
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_details = {
                    'vec_sim': vec_sim,
                    'color_sim': color_sim,
                    'iou': iou,
                    'color_distance': color_distance
                }
        
        if best_combined_score > 0.6:  # Higher threshold for full constraint
            W_FULL = 0.20
            boost = W_FULL * best_combined_score
            detail = f"Object '{obj_label}' full match +{boost:.3f} (vec={best_details.get('vec_sim', 0):.3f}, color={best_details.get('color_sim', 0):.3f}, IoU={best_details.get('iou', 0):.3f})"
            return boost, detail
            
        return 0.0, ""

    def _normalize_text(self, s: str) -> str:
        """Normalize text for fuzzy matching."""
        return " ".join((s or "").lower().split())

    async def _async_apply_ocr_filter(self, candidate_info: Dict, ocr_query: str):
        """OPTIMIZED: Async OCR filtering with fuzzy matching."""
        if not ocr_query or not candidate_info:
            return

        es_client = self.db_manager.es_client
        if not es_client:
            logger.error("Elasticsearch client không khả dụng. Bỏ qua bộ lọc OCR.")
            return

        kf_ids_to_fetch = list(candidate_info.keys())
        ocr_texts_from_es = {}
        
        # Note: Elasticsearch queries are commented out in original code
        # This maintains the same behavior
        
        FUZZ_THRESHOLD = 70
        q = self._normalize_text(ocr_query)

        matched_count = 0
        for kf_id, info in candidate_info.items():
            ocr_text = ocr_texts_from_es.get(kf_id)
            if not ocr_text:
                continue

            t = self._normalize_text(ocr_text)

            # Use fuzzy matching for robust OCR text comparison
            score_partial = fuzz.partial_ratio(q, t)
            score_token_set = fuzz.token_set_ratio(q, t)
            score_token_sort = fuzz.token_sort_ratio(q, t)
            score = max(score_partial, score_token_set, score_token_sort)

            if score >= FUZZ_THRESHOLD:
                info['score'] += 0.5
                info['reasons'].append(f"OCR fuzzy match (score={int(score)})")
                matched_count += 1

        logger.info(f"OCR filter: {matched_count}/{len(candidate_info)} candidates boosted")

    def _format_results(self, sorted_candidates: List[Tuple[str, Dict]]) -> List[Dict]:
        """Format final search results."""
        return [{
            "keyframe_id": kf_id,
            "video_id": info.get('video_id', ''),
            "timestamp": info.get('timestamp', 0.0),
            "score": round(info.get('score', 0.0), 4),
            "reasons": info.get('reasons', []),
            "metadata": {
                "rank": rank + 1,
                "clip_score": round(info.get('clip_score', 0.0), 4),
                "beit3_score": round(info.get('beit3_score', 0.0), 4)
            }
        } for rank, (kf_id, info) in enumerate(sorted_candidates)]

    # --- CÁC PHƯƠNG THỨC TIỆN ÍCH ĐƯỢC GỌI TỪ API ---
    def detect_objects_in_image(self, image_path: str) -> Tuple[List[str], List[str]]:
        if not self.object_detector:
            raise RuntimeError("Object detector is not initialized.")
        return self.object_detector.detect(image_path)

    def check_milvus_connection(self) -> Dict[str, Any]:
        return self.db_manager.check_milvus_connection()

    def check_elasticsearch_connection(self) -> Dict[str, Any]:
        return self.db_manager.check_elasticsearch_connection()

    # --- OPTIMIZATION MONITORING ---
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        return {
            'cache_stats': cache_manager.get_cache_stats(),
            'precomputed_embeddings': len(self.precomputed_embeddings),
            'vectorization_enabled': True,
            'device': str(self.device),
            'optimizations_active': [
                'embedding_caching',
                'search_result_caching', 
                'vectorized_operations',
                'precomputed_embeddings',
                'parallel_processing'
            ]
        }

    def clear_cache(self, cache_type: str = 'all'):
        """Clear optimization caches."""
        cache_manager.clear_cache(cache_type)
        if cache_type in ['all', 'embeddings']:
            self.precomputed_embeddings.clear()
        logger.info(f"Cache cleared: {cache_type}")

# --- END OF FILE app/retrieval_engine.py ---
                    