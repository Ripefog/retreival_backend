# --- START OF FILE app/retrieval_engine.py ---

import logging
import time
import os
import sys
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
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
import numpy as np
from rapidfuzz import fuzz

if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item() if hasattr(a, "item") else np.asarray(a).item()

logger = logging.getLogger(__name__)


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
        logger.info("Initializing ObjectColorDetector (Co-DETR)...")
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
        self.color_names = list(self.basic_colors.keys())
        self.color_tree = KDTree(np.array(list(self.basic_colors.values())))
        logger.info("✅ Co-DETR model loaded.")

    def _convert_basic_colors_to_lab(self) -> dict:
        """Chuyển basic_colors sang CIELAB để so sánh nhanh hơn."""
        lab_dict = {}
        for name, rgb in self.basic_colors.items():
            rgb_obj = sRGBColor(*rgb, is_upscaled=True)
            lab_obj = convert_color(rgb_obj, LabColor)
            lab_dict[name] = lab_obj
        return lab_dict

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB tuple to CIELAB tuple."""
        rgb_obj = sRGBColor(*rgb, is_upscaled=True)
        lab_obj = convert_color(rgb_obj, LabColor)
        return (lab_obj.lab_l, lab_obj.lab_a, lab_obj.lab_b)

    def _get_closest_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Tìm tên màu gần nhất theo thị giác (CIELAB + Delta E CIEDE2000)."""
        rgb_color = sRGBColor(*rgb, is_upscaled=True)
        lab_color = convert_color(rgb_color, LabColor)

        min_delta = float('inf')
        closest_name = None

        for name, lab_ref in self.basic_colors_lab.items():
            delta = delta_e_cie2000(lab_color, lab_ref)
            if delta < min_delta:
                min_delta = delta
                closest_name = name
        return closest_name

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

    async def initialize(self):
        """Khởi tạo retriever: kết nối DB và tải model một cách an toàn."""
        if self.initialized:
            return
        logger.info("Initializing Hybrid Retriever engine...")
        if not self.db_manager.milvus_connected or not self.db_manager.elasticsearch_connected:
            raise RuntimeError("Database connections must be established before initializing the retriever.")
        self._load_models()
        self.initialized = True
        logger.info("✅ Hybrid Retriever initialized successfully.")

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

    # --- CÁC HÀM MÃ HÓA (EMBEDDING) ---
    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.clip_tokenizer([text]).to(self.device)
            text_emb = self.clip_model.encode_text(tokens).cpu().numpy()[0]
            return text_emb / np.linalg.norm(text_emb, axis=0)

    def get_beit3_text_embedding(self, text: str) -> np.ndarray:
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

    def _vectorized_color_distances(self, colors1: List[Tuple[float, float, float]],
                                    colors2: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        OPTIMIZED: Vectorized color distance calculation using NumPy.
        Returns distance matrix of shape (len(colors1), len(colors2))
        """
        if not colors1 or not colors2:
            return np.array([[]])

        # Convert to numpy arrays for vectorized operations
        c1_array = np.array(colors1)  # shape: (m, 3)
        c2_array = np.array(colors2)  # shape: (n, 3)

        if HAS_COLORSPACIOUS:
            # Use optimized colorspacious for accurate CIEDE2000
            distances = np.zeros((len(colors1), len(colors2)))
            for i, color1 in enumerate(colors1):
                for j, color2 in enumerate(colors2):
                    distances[i, j] = colorspacious.deltaE(color1, color2, input_space="CIELab")
            return distances
        else:
            # Fallback: Euclidean distance in LAB space (much faster, reasonably accurate)
            # Broadcast to compute all pairwise distances at once
            c1_expanded = c1_array[:, np.newaxis, :]  # shape: (m, 1, 3)
            c2_expanded = c2_array[np.newaxis, :, :]  # shape: (1, n, 3)

            # Euclidean distance in LAB space
            distances = np.sqrt(np.sum((c1_expanded - c2_expanded) ** 2, axis=2))
            return distances

    def _compare_color(self, color1: Tuple[float, float, float], color2: Tuple[float, float, float]) -> float:
        """
        OPTIMIZED: Fallback single color comparison.
        Uses vectorized method internally for consistency.
        """
        distances = self._vectorized_color_distances([color1], [color2])
        return distances[0, 0] if distances.size > 0 else 0.0

    def _compare_bbox(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """So sánh hai bounding box bằng IoU (Intersection over Union)."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Tính diện tích giao nhau
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap

        # Tính diện tích của mỗi bbox
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Tính IoU
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        return iou

    def _parse_video_id_from_kf(self, kf: str) -> Tuple[str, str]:
        """
        Nhận keyframe: L02_L02_V002_1130.04s.jpg hoặc L02_V002_1130.04s.jpg
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
        # lấy Lxx và Vxxx
        l_code = None
        v_code = None
        for p in unique_parts:
            if p.upper().startswith("L") and p[1:].isdigit():
                l_code = p.upper()
            if p.upper().startswith("K") and p[1:].isdigit():
                l_code = p.upper()
            if p.upper().startswith("V") and p[1:].isdigit():
                v_code = p.upper()

        video_id = f"{l_code}_{v_code}"
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

        # BƯỚC 4.5: LỌC VÀ CỘNG ĐIỂM với ASR (parallel)
        if asr_query:
            refinement_tasks.append(self._async_apply_asr_filter(candidate_info, asr_query))

        # OPTIMIZED: Execute all refinement steps in parallel
        if refinement_tasks:
            await asyncio.gather(*refinement_tasks, return_exceptions=True)

        # BƯỚC 5: XẾP HẠNG VÀ TRẢ VỀ
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        final_results = self._format_results(sorted_results[:top_k])

        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.2f}s with {len(final_results)} results")
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
            # score = max(0.0, 1.0 - hit['distance'])
            score = hit['distance']
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
            user_db = hit_data.get("user") or ""
            # if user_query and user_query not in user_db.split(","):
            #     continue

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

        # ===== OBJECT FILTERS =====
        if object_filters:
            norm_object_filters = self._normalize_object_filters(object_filters)

            for obj_label, filter_spec in norm_object_filters.items():
                obj_vector = self.get_clip_text_embedding(obj_label).tolist()

                # OPTIMIZED: Collect all object IDs from all candidates for batch query
                all_object_ids = []
                candidate_objects = {}  # kf_id -> list of object_ids

                for kf_id, info in candidate_info.items():
                    obj_ids = info.get("object_ids") or []
                    if obj_ids:
                        candidate_objects[kf_id] = obj_ids
                        all_object_ids.extend(obj_ids)

                if not all_object_ids:
                    continue

                # OPTIMIZED: Single batch query instead of per-candidate queries
                batch_results = await self._batch_search_milvus_objects(all_object_ids, obj_vector)

                # Process each candidate based on filter type
                for kf_id, obj_ids in candidate_objects.items():
                    # Get results for this candidate's objects
                    obj_hits = [batch_results[obj_id] for obj_id in obj_ids if obj_id in batch_results]

                    if not obj_hits:
                        continue

                    # Route to appropriate processing logic
                    if filter_spec.get('type') == 'count_aware':
                        detected_count = len(obj_hits)
                        required_count = filter_spec.get('count')

                        # Debug logging
                        logger.info(
                            f"Count-aware processing: {obj_label} detected={detected_count}, required={required_count}")

                        # EXACT COUNT ENFORCEMENT: Apply strong penalty for wrong count
                        if not self._validate_count_constraint(detected_count, required_count):
                            logger.info(
                                f"Count mismatch: {obj_label} count {detected_count} doesn't match requirement {required_count}")
                            # Apply severe penalty instead of skipping
                            penalty = -0.5  # Large negative boost
                            candidate_info[kf_id]["score"] += penalty
                            candidate_info[kf_id].setdefault("reasons", []).append(
                                f"Count penalty: '{obj_label}' {penalty:.3f} (Count: {detected_count}, req: {required_count})"
                            )
                            continue  # Skip positive boost processing

                        boost = self._calculate_count_aware_score(obj_hits, filter_spec, obj_label)
                        reason_detail = f"Count: {detected_count}"
                        if required_count is not None:
                            reason_detail += f" (req: {required_count})"
                    else:
                        # Legacy processing (existing logic)
                        boost = self._calculate_legacy_object_score(obj_hits, filter_spec, obj_label)
                        reason_detail = f"Legacy match"

                    if boost > 0:
                        candidate_info[kf_id]["score"] += boost
                        candidate_info[kf_id].setdefault("reasons", []).append(
                            f"Object match: '{obj_label}' +{boost:.3f} ({reason_detail})"
                        )

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
        """Normalize and validate object filters with count-aware support."""
        norm = {}

        for obj, filter_spec in object_filters.items():
            # New count-aware format
            if isinstance(filter_spec, dict) and any(key in filter_spec for key in
                                                     ['count', 'exact_count', 'min_count', 'max_count', 'constraints',
                                                      'all_match']):
                norm[obj] = self._normalize_count_aware_filter(filter_spec)
            else:
                # Legacy format - backward compatibility
                norm[obj] = self._normalize_legacy_filter(filter_spec)

        return norm

    def _normalize_count_aware_filter(self, filter_spec: Dict) -> Dict:
        """Normalize count-aware filter specification."""
        # Handle different count format variations
        count_constraint = None
        if 'count' in filter_spec:
            count_constraint = filter_spec['count']
        elif 'exact_count' in filter_spec:
            count_constraint = filter_spec['exact_count']
        elif 'min_count' in filter_spec and 'max_count' in filter_spec:
            count_constraint = [filter_spec['min_count'], filter_spec['max_count']]
        elif 'min_count' in filter_spec:
            count_constraint = f">={filter_spec['min_count']}"
        elif 'max_count' in filter_spec:
            count_constraint = f"<={filter_spec['max_count']}"

        normalized = {
            'type': 'count_aware',
            'count': count_constraint,
            'constraints': [],
            'all_match': None
        }

        # Parse individual constraints
        if 'constraints' in filter_spec:
            for constraint in filter_spec['constraints']:
                if isinstance(constraint, dict):
                    normalized['constraints'].append(constraint)
                elif isinstance(constraint, list):
                    # Handle legacy format within constraints
                    if len(constraint) == 0:
                        normalized['constraints'].append({})  # Any object
                    elif len(constraint) == 3:
                        normalized['constraints'].append({'color': constraint})
                    elif len(constraint) == 4:
                        normalized['constraints'].append({'bbox': constraint})
                    elif len(constraint) == 2:
                        color, bbox = constraint
                        if len(color) == 3 and len(bbox) == 4:
                            normalized['constraints'].append({'color': color, 'bbox': bbox})

        # Parse all_match constraint
        if 'all_match' in filter_spec:
            normalized['all_match'] = filter_spec['all_match']

        return normalized

    def _normalize_legacy_filter(self, items) -> Dict:
        """Normalize legacy filter format for backward compatibility."""
        constraints = []

        if not isinstance(items, list):
            return {'type': 'legacy', 'constraints': []}

        for item in items:
            if isinstance(item, list):
                if len(item) == 0:
                    constraints.append({})  # Any object
                elif len(item) == 2:
                    color, bbox = item
                    if (isinstance(color, (list, tuple)) and len(color) == 3 and
                            isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                        constraints.append({
                            'color': tuple(float(x) for x in color),
                            'bbox': tuple(int(x) for x in bbox)
                        })
                elif len(item) == 3:
                    # Color only
                    constraints.append({'color': tuple(float(x) for x in item)})
                elif len(item) == 4:
                    # BBox only
                    constraints.append({'bbox': tuple(int(x) for x in item)})
            elif isinstance(item, dict):
                constraints.append(item)

        return {'type': 'legacy', 'constraints': constraints}

    def _validate_count_constraint(self, detected_count: int, constraint) -> bool:
        """Validate if detected count satisfies constraint."""
        if constraint is None:
            return True  # No constraint

        if isinstance(constraint, int):
            return detected_count == constraint  # Exact count
        elif isinstance(constraint, list) and len(constraint) == 2:
            return constraint[0] <= detected_count <= constraint[1]  # Range
        elif isinstance(constraint, str):
            # Parse expressions like ">=2", "<=5", "!=1"
            try:
                if constraint.startswith(">="):
                    return detected_count >= int(constraint[2:])
                elif constraint.startswith("<="):
                    return detected_count <= int(constraint[2:])
                elif constraint.startswith("!="):
                    return detected_count != int(constraint[2:])
                elif constraint.startswith(">"):
                    return detected_count > int(constraint[1:])
                elif constraint.startswith("<"):
                    return detected_count < int(constraint[1:])
                elif constraint.startswith("=="):
                    return detected_count == int(constraint[2:])
            except ValueError:
                logger.warning(f"Invalid count constraint expression: {constraint}")

        return True  # Default: no constraint or unparseable

    def _calculate_count_accuracy_score(self, detected: int, required) -> float:
        """Calculate accuracy score for count matching."""
        if required is None:
            return 1.0  # No constraint = perfect score

        if isinstance(required, int):
            # Exact count: exponential decay from perfect match
            if detected == required:
                return 1.0
            else:
                error_ratio = abs(detected - required) / max(required, 1)
                return max(0.0, np.exp(-2 * error_ratio))  # Steep penalty for wrong count

        elif isinstance(required, list) and len(required) == 2:  # Range [min, max]
            min_count, max_count = required
            if min_count <= detected <= max_count:
                # Perfect if in range, bonus for being in middle
                if max_count == min_count:
                    return 1.0
                range_center = (min_count + max_count) / 2
                distance_from_center = abs(detected - range_center) / (max_count - min_count)
                return 1.0 - 0.2 * distance_from_center  # Small penalty for being off-center
            else:
                # Penalty for being outside range
                if detected < min_count:
                    return max(0.0, 0.5 * detected / min_count)
                else:  # detected > max_count
                    return max(0.0, 0.5 * max_count / detected)

        elif isinstance(required, str):
            # For expression constraints, if it passes validation, give full score
            if self._validate_count_constraint(detected, required):
                return 1.0
            else:
                return 0.0

        return 1.0  # Default: no constraint

    def _calculate_count_aware_score(
            self,
            obj_hits: List[Dict],
            filter_spec: Dict,
            obj_label: str
    ) -> float:
        """Calculate score for count-aware object filter."""
        detected_count = len(obj_hits)
        count_constraint = filter_spec.get('count')

        # Scoring weights
        SCORE_WEIGHTS = {
            'count_accuracy': 0.4,  # How well detected count matches required count
            'semantic_match': 0.3,  # CLIP semantic similarity
            'constraint_match': 0.2,  # Color/bbox constraint satisfaction
            'count_bonus': 0.1  # Bonus for exact count match
        }

        # 1. Count validation - hard filter
        if not self._validate_count_constraint(detected_count, count_constraint):
            return 0.0  # Hard filter: doesn't meet count requirement

        # 2. Count accuracy score
        count_accuracy = self._calculate_count_accuracy_score(detected_count, count_constraint)

        # 3. Count bonus (exact match gets bonus)
        count_bonus = 0.0
        if isinstance(count_constraint, int) and detected_count == count_constraint:
            count_bonus = 1.0

        # 4. Semantic similarity score
        semantic_scores = []
        for hit in obj_hits:
            distance = hit.get('distance', 0.0)
            semantic_score = 1.0 / (1.0 + distance)
            semantic_scores.append(semantic_score)
        avg_semantic_score = np.mean(semantic_scores) if semantic_scores else 0.0

        # 5. Constraint matching score
        constraint_score = 0.0
        individual_constraints = filter_spec.get('constraints', [])
        all_match_constraint = filter_spec.get('all_match')

        if all_match_constraint:
            # All objects must satisfy the all_match constraint
            constraint_score = self._evaluate_all_match_constraint(obj_hits, all_match_constraint)
            if constraint_score < 0.7:  # Threshold for all_match
                return 0.0  # Hard filter if all_match fails
        elif individual_constraints:
            # Individual constraint matching using Hungarian algorithm
            constraint_score = self._evaluate_individual_constraints(obj_hits, individual_constraints)
        else:
            # No specific constraints beyond count
            constraint_score = 1.0

        # 6. Final weighted score
        final_score = (
                SCORE_WEIGHTS['count_accuracy'] * count_accuracy +
                SCORE_WEIGHTS['semantic_match'] * avg_semantic_score +
                SCORE_WEIGHTS['constraint_match'] * constraint_score +
                SCORE_WEIGHTS['count_bonus'] * count_bonus
        )

        # BOOST PERFECT EXACT MATCHES
        boost_multiplier = 0.25  # Base weight
        if isinstance(count_constraint, int) and detected_count == count_constraint:
            boost_multiplier *= 2.0  # Double boost for exact count matches
            logger.info(f"Perfect count match bonus: {obj_label} exact count {detected_count}")

        return final_score * boost_multiplier

    def _evaluate_all_match_constraint(self, obj_hits: List[Dict], constraint: Dict) -> float:
        """Evaluate if all objects satisfy the constraint."""
        if not obj_hits:
            return 1.0

        total_score = 0.0
        for hit in obj_hits:
            entity = hit.get('entity', {})
            score = self._evaluate_single_constraint(entity, constraint)
            total_score += score

        return total_score / len(obj_hits)

    def _evaluate_individual_constraints(self, obj_hits: List[Dict], constraints: List[Dict]) -> float:
        """Evaluate individual constraints using simplified matching."""
        if not constraints or not obj_hits:
            return 1.0

        m = len(constraints)  # Number of constraint queries
        n = len(obj_hits)  # Number of detected objects

        if m == 0:
            return 1.0

        # Build similarity matrix
        S = np.zeros((m, n))

        for i, constraint in enumerate(constraints):
            for j, hit in enumerate(obj_hits):
                entity = hit.get('entity', {})

                # Semantic score
                distance = hit.get('distance', 0.0)
                semantic_score = 1.0 / (1.0 + distance)

                # Constraint score
                constraint_score = self._evaluate_single_constraint(entity, constraint)

                # Combined score (50-50 weight)
                S[i, j] = 0.5 * semantic_score + 0.5 * constraint_score

        # Use Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        cost_matrix = 1.0 - S
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Calculate final score
        total_similarity = 0.0
        matched_constraints = 0

        for row_idx, col_idx in zip(row_indices, col_indices):
            if row_idx < m and col_idx < n:
                similarity = S[row_idx, col_idx]
                total_similarity += similarity
                if similarity >= 0.5:  # Threshold for "good match"
                    matched_constraints += 1

        avg_similarity = (total_similarity / m) if m > 0 else 0.0
        coverage = (matched_constraints / m) if m > 0 else 0.0

        # Combine similarity and coverage
        return 0.7 * avg_similarity + 0.3 * coverage

    def _evaluate_single_constraint(self, entity: Dict, constraint: Dict) -> float:
        """Evaluate single object against constraint."""
        if not constraint:  # Empty constraint = any object matches
            return 1.0

        score = 1.0
        constraint_count = 0

        # Color constraint
        if 'color' in constraint:
            constraint_count += 1
            required_color = constraint['color']
            entity_color_str = entity.get('color_lab', '')

            if entity_color_str:
                try:
                    entity_color = self._split_csv_floats(entity_color_str)
                    if len(entity_color) >= 3:
                        entity_lab = tuple(entity_color[:3])
                        required_lab = self._ensure_lab(tuple(required_color)) if required_color else None

                        if required_lab and entity_lab:
                            distance = self._compare_color(required_lab, entity_lab)
                            color_score = np.exp(-(distance / 20.0) ** 2)  # Gaussian similarity
                            score *= color_score
                        else:
                            score *= 0.5  # Partial penalty for missing color data
                    else:
                        score *= 0.5
                except:
                    score *= 0.5
            else:
                score *= 0.5

        # Bbox constraint
        if 'bbox' in constraint:
            constraint_count += 1
            required_bbox = constraint['bbox']
            entity_bbox_str = entity.get('bbox_xyxy', '')

            if entity_bbox_str and required_bbox:
                try:
                    entity_bbox_list = self._split_csv_floats(entity_bbox_str)
                    if len(entity_bbox_list) >= 4:
                        entity_bbox = tuple(int(x) for x in entity_bbox_list[:4])
                        required_bbox_tuple = tuple(required_bbox)

                        iou = self._compare_bbox(required_bbox_tuple, entity_bbox)
                        if iou >= 0.3:  # Minimum IoU threshold
                            score *= iou
                        else:
                            score *= 0.1  # Low score for poor spatial match
                    else:
                        score *= 0.5
                except:
                    score *= 0.5
            else:
                score *= 0.5

        # If no constraints specified, perfect match
        if constraint_count == 0:
            return 1.0

        return score

    def _calculate_legacy_object_score(
            self,
            obj_hits: List[Dict],
            filter_spec: Dict,
            obj_label: str
    ) -> float:
        """Calculate score for legacy object filter format."""
        constraints = filter_spec.get('constraints', [])

        if not constraints:
            # No constraints, just semantic matching
            semantic_scores = []
            for hit in obj_hits:
                distance = hit.get('distance', 0.0)
                semantic_score = 1.0 / (1.0 + distance)
                semantic_scores.append(semantic_score)
            return np.mean(semantic_scores) * 0.20 if semantic_scores else 0.0

        # Use individual constraint evaluation for legacy format
        return self._evaluate_individual_constraints(obj_hits, constraints) * 0.20

    def _normalize_text(self, s: str) -> str:
        """Normalize text for fuzzy matching."""
        return " ".join((s or "").lower().split())

    async def _async_apply_ocr_filter(self, candidate_info: Dict, ocr_query: str):
        """OPTIMIZED: Async OCR filtering with fuzzy matching."""
        ic(ocr_query)
        if not ocr_query or not candidate_info:
            logger.info(f"OCR filter skipped: ocr_query='{ocr_query}', candidates={len(candidate_info) if candidate_info else 0}")
            return

        es_client = self.db_manager.es_client
        if not es_client:
            logger.error("Elasticsearch client không khả dụng. Bỏ qua bộ lọc OCR.")
            return
        # ic(es_client)
        logger.info(f"Starting OCR filter with query: '{ocr_query}' for {len(candidate_info)} candidates")

        kf_ids_to_fetch = list(candidate_info.keys())
        ocr_texts_from_es = {}

        # Query Elasticsearch để lấy OCR text cho các keyframes
        try:
            if kf_ids_to_fetch:
                # ic(f"Fetching OCR texts for keyframes: {kf_ids_to_fetch[:5]}...")
                
                query_body = {
                    "query": {
                        "terms": {
                            "keyframe_id": kf_ids_to_fetch
                        }
                    },
                    # Bỏ "_source" để lấy tất cả fields
                    "size": len(kf_ids_to_fetch)
                }

                # ic(f"Querying OCR index '{settings.OCR_INDEX}' with {len(kf_ids_to_fetch)} keyframe IDs")
                response = es_client.search(
                    index=settings.OCR_INDEX,  # = 'ocr_v6'
                    body=query_body
                )

                # ic(f"ES OCR query returned {len(response['hits']['hits'])} hits")

                for hit in response['hits']['hits']:
                    # ic(f"All available fields in _source: {hit['_source']}")
                    kf_id = hit['_source']['keyframe_id']
                    
                    # Thử các tên trường khác nhau
                    possible_ocr_fields = ['text', 'ocr_text', 'ocr', 'content', 'ocr_content', 'extracted_text']
                    ocr_text = ''
                    
                    for field in possible_ocr_fields:
                        if field in hit['_source']:
                            ocr_text = hit['_source'].get(field, '')
                            # ic(f"Found OCR text in field '{field}': {bool(ocr_text)}")
                            break
                    
                    if ocr_text:
                        ocr_texts_from_es[kf_id] = ocr_text
                    #     ic(f"OCR text for {kf_id}: '{ocr_text[:50]}...'")
                    # else:
                    #     ic(f"No OCR text found for {kf_id} in any field")

                # ic(f"Loaded OCR text for {len(ocr_texts_from_es)}/{len(kf_ids_to_fetch)} keyframes")

        except Exception as e:
            logger.error(f"Failed to query OCR texts from Elasticsearch: {e}", exc_info=True)
            return

        FUZZ_THRESHOLD = 70
        q = self._normalize_text(ocr_query)
        # ic(f"Starting OCR fuzzy matching with normalized query: '{q}' (threshold: {FUZZ_THRESHOLD})")

        matched_count = 0
        for kf_id, info in candidate_info.items():
            ocr_text = ocr_texts_from_es.get(kf_id)
            if not ocr_text:
                continue

            t = self._normalize_text(ocr_text)
            # ic(f"Comparing query '{q}' with text '{t[:100]}...'")

            # Use fuzzy matching for robust OCR text comparison
            score_partial = fuzz.partial_ratio(q, t)
            score_token_set = fuzz.token_set_ratio(q, t)
            score_token_sort = fuzz.token_sort_ratio(q, t)
            score = max(score_partial, score_token_set, score_token_sort)

            # ic(f"Fuzzy scores for {kf_id}: partial={score_partial}, token_set={score_token_set}, token_sort={score_token_sort}, max={score}")

            if score >= FUZZ_THRESHOLD:
                info['score'] += 0.5
                info['reasons'].append(f"OCR fuzzy match (score={int(score)})")
                matched_count += 1
            #     ic(f"OCR match found: {kf_id} with score {score}")
            # else:
            #     ic(f"No match for {kf_id}: score {score} < threshold {FUZZ_THRESHOLD}")

        # ic(f"OCR filter: {matched_count}/{len(candidate_info)} candidates boosted")

    async def _async_apply_asr_filter(self, candidate_info: Dict, asr_query: str):
        """
        Lọc và tăng điểm ứng viên dựa trên ASR (lời thoại).
        Hàm sẽ tìm các đoạn ASR trong một cửa sổ thời gian quanh mỗi keyframe,
        ghép nối chúng lại và dùng fuzzy matching để so sánh với query.
        """
        if not asr_query or not candidate_info:
            return

        es_client = self.db_manager.es_client
        if not es_client:
            logger.error("Elasticsearch client không khả dụng. Bỏ qua bộ lọc ASR.")
            return

        # --- Bước 1: Chuẩn bị query cho Elasticsearch ---
        time_window_sec = 15.0  # Tăng từ 5.0 lên 15.0 giây (±15s = 30s total)
        es_should_clauses = []
        candidate_map = {}

        for kf_id, info in candidate_info.items():
            timestamp = info.get("timestamp")
            if timestamp is None:
                continue

            video_id, _ = self._parse_video_id_from_kf(kf_id)
            kf_start = float(timestamp) - time_window_sec
            kf_end = float(timestamp) + time_window_sec

            candidate_map[kf_id] = {
                "video_id": video_id,
                "window_start": kf_start,
                "window_end": kf_end
            }

            # Điều kiện để một segment ASR [start, end] giao với cửa sổ [kf_start, kf_end]
            # là: start <= kf_end AND end >= kf_start
            es_should_clauses.append({
                "bool": {
                    "must": [
                        {"term": {"video_id": video_id}},
                        {"range": {"start": {"lte": kf_end}}},
                        {"range": {"end": {"gte": kf_start}}}
                    ]
                }
            })

        if not es_should_clauses:
            return

        # --- Bước 2: Lấy dữ liệu ASR từ Elasticsearch ---
        kf_asr_texts = {}
        try:
            query_body = {
                "query": {"bool": {"should": es_should_clauses, "minimum_should_match": 1}},
                "_source": ["video_id", "text", "start", "end"],
                "size": 1000,
                "sort": ["video_id", "start"]
            }

            response = es_client.search(
                index=settings.ASR_INDEX,
                body=query_body
            )

            # --- Bước 3: Ghép nối ASR text cho từng keyframe ---
            asr_segments_by_video = {}
            for hit in response['hits']['hits']:
                source = hit['_source']
                vid = source['video_id']
                if vid not in asr_segments_by_video:
                    asr_segments_by_video[vid] = []
                asr_segments_by_video[vid].append(source)

            for kf_id, data in candidate_map.items():
                video_id = data["video_id"]
                if video_id not in asr_segments_by_video:
                    continue

                overlapping_texts = []
                for segment in asr_segments_by_video[video_id]:
                    if segment['start'] <= data['window_end'] and segment['end'] >= data['window_start']:
                        overlapping_texts.append(segment['text'])

                if overlapping_texts:
                    kf_asr_texts[kf_id] = " ".join(overlapping_texts)

        except Exception as e:
            logger.error(f"Lỗi khi truy vấn ASR text từ Elasticsearch: {e}", exc_info=True)
            return

        # --- Bước 4: So sánh và cộng điểm ---
        ASR_FUZZ_THRESHOLD = 75
        BASE_BOOST = 0.5
        q_normalized = self._normalize_text(asr_query)
        matched_count = 0

        for kf_id, info in candidate_info.items():
            full_asr_text = kf_asr_texts.get(kf_id)
            if not full_asr_text:
                continue

            t_normalized = self._normalize_text(full_asr_text)

            score = max(
                fuzz.partial_ratio(q_normalized, t_normalized),
                fuzz.token_set_ratio(q_normalized, t_normalized),
                fuzz.token_sort_ratio(q_normalized, t_normalized)
            )

            if score >= ASR_FUZZ_THRESHOLD:
                # Cộng điểm động
                score_normalized = (score - ASR_FUZZ_THRESHOLD) / (100 - ASR_FUZZ_THRESHOLD)
                boost = BASE_BOOST * score_normalized

                info['score'] += boost
                info.setdefault('reasons', []).append(f"ASR dynamic match (score={int(score)}, boost={boost:.3f})")
                matched_count += 1

        logger.info(f"ASR filter: {matched_count}/{len(candidate_info)} candidates boosted")

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

# --- END OF FILE app/retrieval_engine.py ---