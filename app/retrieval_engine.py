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
        await self.db_manager._load_milvus_collections()
        self.initialized = True
        logger.info("✅ Hybrid Retriever initialized successfully.")

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB tuple to CIELAB tuple."""
        rgb_obj = sRGBColor(*rgb, is_upscaled=True)
        lab_obj = convert_color(rgb_obj, LabColor)
        return lab_obj.lab_l, lab_obj.lab_a, lab_obj.lab_b

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

    # --- CÁC HÀM MÃ HÓA (EMBEDDING) ---
    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.clip_tokenizer([text]).to(self.device)
            text_emb = self.clip_model.encode_text(tokens).cpu().numpy()[0]
            return text_emb / np.linalg.norm(text_emb, axis=0)
        return None

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
        return None

    def _vectorized_color_distances(self, colors1: List[Tuple[float, float, float]],
                                    colors2: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        OPTIMIZED: Vectorized color distance calculation using NumPy.
        Returns distance matrix of shape (len(colors1), len(colors2))
        """
        if not colors1 or not colors2:
            return np.array([[]])

        c1_array = np.array(colors1)
        c2_array = np.array(colors2)

        if HAS_COLORSPACIOUS:
            distances = np.zeros((len(colors1), len(colors2)))
            for i, color1 in enumerate(colors1):
                for j, color2 in enumerate(colors2):
                    distances[i, j] = colorspacious.deltaE(color1, color2, input_space="CIELab")
            return distances
        else:
            c1_expanded = c1_array[:, np.newaxis, :]
            c2_expanded = c2_array[np.newaxis, :, :]
            distances = np.sqrt(np.sum((c1_expanded - c2_expanded) ** 2, axis=2))
            return distances
            
    ### MERGED ###
    # Thêm lại hàm so sánh màu đơn lẻ để các hàm tính điểm cũ có thể sử dụng
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

        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        return iou

    def _parse_video_id_from_kf(self, kf: str) -> Tuple[str, str]:
        name = os.path.splitext(os.path.basename(kf))[0]
        parts = name.split("_")
        unique_parts = []
        for p in parts:
            if not unique_parts or unique_parts[-1] != p:
                unique_parts.append(p)

        timestamp = unique_parts[-1]
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
        if not s: return []
        out = []
        for p in s.split(","):
            p = p.strip()
            if not p: continue
            try:
                out.append(int(p))
            except ValueError:
                try: out.append(int(float(p)))
                except ValueError: pass
        return out

    def _split_csv_floats(self, s: Optional[str]) -> List[float]:
        if not s: return []
        out = []
        for p in s.split(","):
            p = p.strip()
            if p == "": continue
            try: out.append(float(p))
            except ValueError: pass
        return out

    def _parse_lab_colors18(self, s: Optional[str]) -> List[Tuple[float, float, float]]:
        vals = self._split_csv_floats(s)
        if len(vals) < 18: vals += [0.0] * (18 - len(vals))
        elif len(vals) > 18: vals = vals[:18]
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

        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Search task {i} failed: {result}")
                continue
            search_type = 'clip' if (mode in ['hybrid', 'clip'] and i == 0) else 'beit3'
            self._process_search_results(result, candidate_info, search_type)

        # BƯỚC 2, 3, 4: TINH CHỈNH, LỌC VÀ TĂNG ĐIỂM
        refinement_tasks = []
        if mode == 'hybrid' and candidate_info:
            refinement_tasks.append(self._async_hybrid_reranking(candidate_info, text_query))
        if object_filters or color_filters:
            refinement_tasks.append(self._apply_object_color_filters_optimized(
                candidate_info, object_filters, color_filters, top_k))
        if ocr_query:
            refinement_tasks.append(self._async_apply_ocr_filter(candidate_info, ocr_query))
        if asr_query:
            refinement_tasks.append(self._async_apply_asr_filter(candidate_info, asr_query))

        if refinement_tasks:
            await asyncio.gather(*refinement_tasks, return_exceptions=True)

        # BƯỚC 5: XẾP HẠNG VÀ TRẢ VỀ
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        final_results = self._format_results(sorted_results[:top_k])

        search_time = time.time() - start_time
        #logger.info(f"Search completed in {search_time:.2f}s with {len(final_results)} results")
        return final_results

    async def _search_milvus_async(self, collection_name: str, vector: List[float], top_k: int,
                                   expr: Optional[str] = None, user_query: str = "",
                                   search_type: str = "") -> List[Dict]:
        return await self._search_milvus(collection_name, vector, top_k, expr, user_query)

    def _process_search_results(self, search_results: List[Dict], candidate_info: Dict[str, Dict[str, Any]],
                                search_type: str):
        for hit in search_results:
            kf_id = hit["entity"]["keyframe_id"]
            vid, kf_id = self._parse_video_id_from_kf(kf_id)
            score = hit['distance']
            obj_ids = self._split_csv_ints(hit['entity']['object_ids'])
            lab6 = self._parse_lab_colors18(hit['entity']['lab_colors'])

            if kf_id in candidate_info:
                candidate_info[kf_id][f'{search_type}_score'] = score
                candidate_info[kf_id]['score'] += score
                candidate_info[kf_id]['reasons'].append(f"{search_type.upper()} match ({score:.3f})")
                if not candidate_info[kf_id].get('object_ids') and obj_ids:
                    candidate_info[kf_id]['object_ids'] = obj_ids
                if not candidate_info[kf_id].get('lab_colors6') and lab6:
                    candidate_info[kf_id]['lab_colors6'] = lab6
            else:
                candidate_info[kf_id] = {
                    "keyframe_id": kf_id, "timestamp": hit['entity']['timestamp'], "object_ids": obj_ids,
                    "lab_colors6": lab6, f"{search_type}_score": score, "score": score,
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
                if expr: expr = f'({expr}) && ({user_expr})'
                else: expr = user_expr
        return expr

    async def _search_milvus(self, collection_name: str, vector: List[float], top_k: int,
                             expr: Optional[str] = None, user_query: str = "") -> List[Dict]:
        collection = self.db_manager.get_collection(collection_name)
        if not collection: return []

        if collection_name in (settings.CLIP_COLLECTION, settings.BEIT3_COLLECTION):
            output_fields = ["keyframe_id", "timestamp", "object_ids", "lab_colors", "user"]
        elif collection_name == settings.OBJECT_COLLECTION:
            output_fields = ["object_id", "bbox_xyxy", "color_lab"]
        else: output_fields = []

        expr_new = self.build_expr(expr, user_query)
        search_results = collection.search(
            data=[vector], anns_field="vector", param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k, expr=expr_new, output_fields=output_fields,
        )[0]

        hits = []
        for hit in search_results:
            hit_data = hit.entity.to_dict()["entity"]
            if collection_name in (settings.CLIP_COLLECTION, settings.BEIT3_COLLECTION):
                raw_kf = hit_data.get("keyframe_id", "")
                kf_clean = raw_kf[:raw_kf.lower().find(".jpg") + 4] if isinstance(raw_kf, str) and ".jpg" in raw_kf.lower() else raw_kf
                vid, kf_normalized = self._parse_video_id_from_kf(kf_clean)
                hits.append({"id": kf_normalized, "distance": hit.distance, "entity": {"keyframe_id": kf_normalized, "timestamp": hit_data.get("timestamp"), "object_ids": hit_data.get("object_ids"), "lab_colors": hit_data.get("lab_colors"),},})
            elif collection_name == settings.OBJECT_COLLECTION:
                hits.append({"id": hit_data.get("object_id"), "distance": hit.distance, "entity": {"bbox_xyxy": hit_data.get("bbox_xyxy"), "color_lab": hit_data.get("color_lab"),},})
        return hits

    async def _async_hybrid_reranking(self, candidate_info: Dict[str, Dict], text_query: str):
        beit3_collection = self.db_manager.get_collection(settings.BEIT3_COLLECTION)
        if not beit3_collection:
            logger.warning("BEIT-3 collection not available for reranking")
            return
        candidate_kf_ids = list(candidate_info.keys())
        if not candidate_kf_ids: return

        try:
            all_possible_kf_ids = set()
            for kf_id in candidate_kf_ids:
                all_possible_kf_ids.add(kf_id)
                parts = kf_id.replace('.jpg', '').split('_')
                if len(parts) >= 3 and parts[0].startswith('L') and parts[1].startswith('V'):
                    db_format = f"{parts[0]}_{parts[0]}_{parts[1]}_{parts[2]}.jpg"
                    all_possible_kf_ids.add(db_format)

            kf_ids_list = list(all_possible_kf_ids)
            res = beit3_collection.query(expr=f'keyframe_id in {kf_ids_list}', output_fields=["keyframe_id", "vector"])
            beit3_vector_map = {}
            for item in res:
                db_kf_id = item['keyframe_id']
                vid, normalized_kf_id = self._parse_video_id_from_kf(db_kf_id)
                beit3_vector_map[normalized_kf_id] = item['vector']
            
            beit3_query_vector = np.array(self.get_beit3_text_embedding(text_query))
            kf_vectors = [beit3_vector_map[kf_id] for kf_id in candidate_kf_ids if kf_id in beit3_vector_map]
            kf_ids_ordered = [kf_id for kf_id in candidate_kf_ids if kf_id in beit3_vector_map]
            
            if kf_vectors:
                kf_matrix = np.array(kf_vectors)
                distances = np.linalg.norm(kf_matrix - beit3_query_vector[np.newaxis, :], axis=1)
                for i, kf_id in enumerate(kf_ids_ordered):
                    dist = distances[i]
                    beit3_score = 1.0 / (1.0 + dist)
                    info = candidate_info[kf_id]
                    info['score'] = (0.4 * info.get('clip_score', 0)) + (0.6 * beit3_score)
                    info['beit3_score'] = beit3_score
                    info['reasons'].append(f"BEIT-3 refine ({beit3_score:.3f})")
            
            for kf_id in candidate_kf_ids:
                if kf_id not in beit3_vector_map:
                    candidate_info[kf_id]['score'] *= 0.8
                    candidate_info[kf_id]['reasons'].append("BEIT-3 vector missing")
        except Exception as e:
            logger.error(f"BEIT-3 reranking failed: {e}", exc_info=True)

    async def _batch_search_milvus_objects(self, all_object_ids: List[int], obj_vector: List[float]) -> Dict[int, Dict]:
        if not all_object_ids: return {}
        unique_obj_ids = list(dict.fromkeys(all_object_ids))
        try:
            expr = f"object_id in [{','.join(map(str, unique_obj_ids))}]"
            limit = max(len(unique_obj_ids), 1)
            obj_hits = await self._search_milvus(settings.OBJECT_COLLECTION, obj_vector, limit, expr=expr)
            return {hit.get("id"): hit for hit in obj_hits if hit.get("id")}
        except Exception as e:
            logger.error(f"Batch object search failed: {e}", exc_info=True)
            return {}
            
    ### MERGED ###
    # Toàn bộ logic lọc đối tượng và màu sắc được thay thế bằng phiên bản mạnh mẽ hơn.
    async def _apply_object_color_filters_optimized(
            self,
            candidate_info: Dict[str, Dict[str, Any]],
            object_filters: Optional[Dict],
            color_filters: Optional[List[Tuple[float, float, float]]],
            top_k: int
    ):
        """OPTIMIZED & MERGED: Hỗ trợ cả logic 'count-aware' và logic 'instance-matching'."""

        # ===== OBJECT FILTERS =====
        if object_filters:
            # Sử dụng hàm normalize mạnh mẽ từ code cũ
            norm_object_filters = self._normalize_object_filters(object_filters)

            for obj_label, filter_spec in norm_object_filters.items():
                obj_vector = self.get_clip_text_embedding(obj_label).tolist()

                all_object_ids = []
                candidate_objects = {}
                for kf_id, info in candidate_info.items():
                    obj_ids = info.get("object_ids") or []
                    if obj_ids:
                        candidate_objects[kf_id] = obj_ids
                        all_object_ids.extend(obj_ids)

                if not all_object_ids:
                    continue

                batch_results = await self._batch_search_milvus_objects(all_object_ids, obj_vector)

                for kf_id, obj_ids in candidate_objects.items():
                    obj_hits = [batch_results[obj_id] for obj_id in obj_ids if obj_id in batch_results]
                    if not obj_hits:
                        continue
                        
                    # Phân nhánh logic dựa trên loại bộ lọc
                    if filter_spec.get('type') == 'count_aware':
                        detected_count = len(obj_hits)
                        required_count = filter_spec.get('count')
                        
                        # Logic phạt nếu không khớp số lượng
                        if not self._validate_count_constraint(detected_count, required_count):
                            penalty = -0.5
                            candidate_info[kf_id]["score"] += penalty
                            candidate_info[kf_id].setdefault("reasons", []).append(
                                f"Count penalty: '{obj_label}' {penalty:.3f} (Count: {detected_count}, req: {required_count})"
                            )
                            continue 
                        
                        # Logic tính điểm thưởng nếu khớp
                        boost = self._calculate_count_aware_score(obj_hits, filter_spec, obj_label)
                        reason_detail = f"Count: {detected_count}"
                        if required_count is not None:
                            reason_detail += f" (req: {required_count})"
                    else: # Legacy/Instance-matching logic
                        boost = self._calculate_legacy_object_score(obj_hits, filter_spec, obj_label)
                        reason_detail = f"Legacy match"

                    if boost > 0:
                        candidate_info[kf_id]["score"] += boost
                        candidate_info[kf_id].setdefault("reasons", []).append(
                            f"Object match: '{obj_label}' +{boost:.3f} ({reason_detail})"
                        )

        # ===== COLOR FILTERS (OPTIMIZED) - Giữ nguyên từ code mới =====
        if color_filters:
            queries_lab = [self._rgb_to_lab(tuple(qc)) for qc in color_filters if qc is not None]

            if queries_lab:
                alpha, beta, w_color, tau = 0.7, 0.3, 0.15, 15.0
                for kf_id, info in candidate_info.items():
                    palette = info.get("lab_colors6") or []
                    if not palette: continue

                    m, n = len(queries_lab), len(palette)
                    distance_matrix = self._vectorized_color_distances(queries_lab, palette)
                    similarity_matrix = np.exp(-(distance_matrix / 20.0) ** 2)
                    cost_matrix = 1.0 - similarity_matrix
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    sim_sum = sum(similarity_matrix[r, c] for r, c in zip(row_indices, col_indices) if r < m and c < n)
                    S_hung = (sim_sum / m) if m > 0 else 0.0

                    covered = np.sum(np.min(distance_matrix, axis=1) <= tau)
                    C = (covered / m) if m > 0 else 0.0

                    S_color = alpha * S_hung + beta * C
                    boost = w_color * S_color

                    if boost > 0:
                        info["score"] += boost
                        info.setdefault("reasons", []).append(
                            f"Color match (Hungarian): +{boost:.3f} (S={S_color:.3f}, cov={C:.2f})"
                        )

    ### MERGED ###
    # Thêm các hàm normalize và tính điểm từ code cũ.
    def _normalize_object_filters(self, object_filters: Dict) -> Dict:
        """Normalize and validate object filters with count-aware support."""
        norm = {}
        for obj, filter_spec in object_filters.items():
            if isinstance(filter_spec, dict) and any(key in filter_spec for key in ['count', 'exact_count', 'min_count', 'max_count', 'constraints', 'all_match']):
                norm[obj] = self._normalize_count_aware_filter(filter_spec)
            else:
                norm[obj] = self._normalize_legacy_filter(filter_spec)
        return norm

    def _normalize_count_aware_filter(self, filter_spec: Dict) -> Dict:
        count_constraint = None
        if 'count' in filter_spec: count_constraint = filter_spec['count']
        elif 'exact_count' in filter_spec: count_constraint = filter_spec['exact_count']
        elif 'min_count' in filter_spec and 'max_count' in filter_spec: count_constraint = [filter_spec['min_count'], filter_spec['max_count']]
        elif 'min_count' in filter_spec: count_constraint = f">={filter_spec['min_count']}"
        elif 'max_count' in filter_spec: count_constraint = f"<={filter_spec['max_count']}"
            
        normalized = {'type': 'count_aware', 'count': count_constraint, 'constraints': [], 'all_match': None}

        if 'constraints' in filter_spec:
            for constraint in filter_spec['constraints']:
                if isinstance(constraint, dict): normalized['constraints'].append(constraint)
                elif isinstance(constraint, list):
                    if len(constraint) == 0: normalized['constraints'].append({})
                    elif len(constraint) == 3: normalized['constraints'].append({'color': constraint})
                    elif len(constraint) == 4: normalized['constraints'].append({'bbox': constraint})
                    elif len(constraint) == 2:
                        color, bbox = constraint
                        if len(color) == 3 and len(bbox) == 4:
                            normalized['constraints'].append({'color': color, 'bbox': bbox})

        if 'all_match' in filter_spec: normalized['all_match'] = filter_spec['all_match']
        return normalized

    def _normalize_legacy_filter(self, items) -> Dict:
        constraints = []
        if not isinstance(items, list): return {'type': 'legacy', 'constraints': []}
        for item in items:
            if isinstance(item, list):
                if len(item) == 0: constraints.append({})
                elif len(item) == 2:
                    color, bbox = item
                    if (isinstance(color, (list, tuple)) and len(color) == 3 and isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                        constraints.append({'color': tuple(float(x) for x in color), 'bbox': tuple(int(x) for x in bbox)})
                elif len(item) == 3: constraints.append({'color': tuple(float(x) for x in item)})
                elif len(item) == 4: constraints.append({'bbox': tuple(int(x) for x in item)})
            elif isinstance(item, dict): constraints.append(item)
        return {'type': 'legacy', 'constraints': constraints}

    def _validate_count_constraint(self, detected_count: int, constraint) -> bool:
        if constraint is None: return True
        if isinstance(constraint, int): return detected_count == constraint
        elif isinstance(constraint, list) and len(constraint) == 2: return constraint[0] <= detected_count <= constraint[1]
        elif isinstance(constraint, str):
            try:
                if constraint.startswith(">="): return detected_count >= int(constraint[2:])
                elif constraint.startswith("<="): return detected_count <= int(constraint[2:])
                elif constraint.startswith("!="): return detected_count != int(constraint[2:])
                elif constraint.startswith(">"): return detected_count > int(constraint[1:])
                elif constraint.startswith("<"): return detected_count < int(constraint[1:])
                elif constraint.startswith("=="): return detected_count == int(constraint[2:])
            except ValueError: logger.warning(f"Invalid count constraint expression: {constraint}")
        return True

    def _calculate_count_accuracy_score(self, detected: int, required) -> float:
        if required is None: return 1.0
        if isinstance(required, int):
            if detected == required: return 1.0
            else: return max(0.0, np.exp(-2 * (abs(detected - required) / max(required, 1))))
        elif isinstance(required, list) and len(required) == 2:
            min_c, max_c = required
            if min_c <= detected <= max_c:
                if max_c == min_c: return 1.0
                return 1.0 - 0.2 * (abs(detected - (min_c + max_c) / 2) / (max_c - min_c))
            else:
                if detected < min_c: return max(0.0, 0.5 * detected / min_c)
                else: return max(0.0, 0.5 * max_c / detected)
        elif isinstance(required, str):
            return 1.0 if self._validate_count_constraint(detected, required) else 0.0
        return 1.0

    def _calculate_count_aware_score(self, obj_hits: List[Dict], filter_spec: Dict, obj_label: str) -> float:
        detected_count = len(obj_hits)
        count_constraint = filter_spec.get('count')
        SCORE_WEIGHTS = {'count_accuracy': 0.4, 'semantic_match': 0.3, 'constraint_match': 0.2, 'count_bonus': 0.1}
        
        if not self._validate_count_constraint(detected_count, count_constraint): return 0.0
        
        count_accuracy = self._calculate_count_accuracy_score(detected_count, count_constraint)
        count_bonus = 1.0 if isinstance(count_constraint, int) and detected_count == count_constraint else 0.0
        
        semantic_scores = [1.0 / (1.0 + hit.get('distance', 0.0)) for hit in obj_hits]
        avg_semantic_score = np.mean(semantic_scores) if semantic_scores else 0.0
        
        individual_constraints = filter_spec.get('constraints', [])
        all_match_constraint = filter_spec.get('all_match')
        
        if all_match_constraint:
            constraint_score = self._evaluate_all_match_constraint(obj_hits, all_match_constraint)
            if constraint_score < 0.7: return 0.0
        elif individual_constraints:
            constraint_score = self._evaluate_individual_constraints(obj_hits, individual_constraints)
        else:
            constraint_score = 1.0
        
        final_score = (SCORE_WEIGHTS['count_accuracy'] * count_accuracy +
                       SCORE_WEIGHTS['semantic_match'] * avg_semantic_score +
                       SCORE_WEIGHTS['constraint_match'] * constraint_score +
                       SCORE_WEIGHTS['count_bonus'] * count_bonus)
        
        boost_multiplier = 0.25
        if isinstance(count_constraint, int) and detected_count == count_constraint:
            boost_multiplier *= 2.0
        
        return final_score * boost_multiplier

    def _evaluate_all_match_constraint(self, obj_hits: List[Dict], constraint: Dict) -> float:
        if not obj_hits: return 1.0
        total_score = sum(self._evaluate_single_constraint(hit.get('entity', {}), constraint) for hit in obj_hits)
        return total_score / len(obj_hits)

    def _evaluate_individual_constraints(self, obj_hits: List[Dict], constraints: List[Dict]) -> float:
        if not constraints or not obj_hits: return 1.0
        m, n = len(constraints), len(obj_hits)
        if m == 0: return 1.0
            
        S = np.zeros((m, n))
        for i, constraint in enumerate(constraints):
            for j, hit in enumerate(obj_hits):
                semantic_score = 1.0 / (1.0 + hit.get('distance', 0.0))
                constraint_score = self._evaluate_single_constraint(hit.get('entity', {}), constraint)
                S[i, j] = 0.5 * semantic_score + 0.5 * constraint_score
        
        cost_matrix = 1.0 - S
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        total_similarity, matched_constraints = 0.0, 0
        for r, c in zip(row_indices, col_indices):
            if r < m and c < n:
                similarity = S[r, c]
                total_similarity += similarity
                if similarity >= 0.5: matched_constraints += 1
        
        avg_similarity = (total_similarity / m) if m > 0 else 0.0
        coverage = (matched_constraints / m) if m > 0 else 0.0
        return 0.7 * avg_similarity + 0.3 * coverage

    def _evaluate_single_constraint(self, entity: Dict, constraint: Dict) -> float:
        if not constraint: return 1.0
        score, constraint_count = 1.0, 0
        
        if 'color' in constraint:
            constraint_count += 1
            required_color, entity_color_str = constraint['color'], entity.get('color_lab', '')
            if entity_color_str:
                try:
                    entity_color = self._split_csv_floats(entity_color_str)
                    if len(entity_color) >= 3:
                        entity_lab = tuple(entity_color[:3])
                        r, g, b = required_color[:3]
                        required_lab = self._rgb_to_lab((int(r), int(g), int(b))) if 0 <= r <= 255 and (r > 1 or g > 1 or b > 1) else (float(r), float(g), float(b))
                        distance = self._compare_color(required_lab, entity_lab)
                        score *= np.exp(-(distance / 20.0) ** 2)
                    else: score *= 0.5
                except Exception: score *= 0.5
            else: score *= 0.5
        
        if 'bbox' in constraint:
            constraint_count += 1
            required_bbox, entity_bbox_str = constraint['bbox'], entity.get('bbox_xyxy', '')
            if entity_bbox_str and required_bbox:
                try:
                    entity_bbox_list = self._split_csv_floats(entity_bbox_str)
                    if len(entity_bbox_list) >= 4:
                        entity_bbox = tuple(int(x) for x in entity_bbox_list[:4])
                        iou = self._compare_bbox(tuple(required_bbox), entity_bbox)
                        score *= iou if iou >= 0.3 else 0.1
                    else: score *= 0.5
                except Exception: score *= 0.5
            else: score *= 0.5
        
        return 1.0 if constraint_count == 0 else score

    def _calculate_legacy_object_score(self, obj_hits: List[Dict], filter_spec: Dict, obj_label: str) -> float:
        constraints = filter_spec.get('constraints', [])
        if not constraints:
            semantic_scores = [1.0 / (1.0 + hit.get('distance', 0.0)) for hit in obj_hits]
            return np.mean(semantic_scores) * 0.20 if semantic_scores else 0.0
        return self._evaluate_individual_constraints(obj_hits, constraints) * 0.20

    # --- Các hàm lọc khác (OCR, ASR) giữ nguyên từ code mới ---
    def _normalize_text(self, s: str) -> str:
        return " ".join((s or "").lower().split())

    async def _async_apply_ocr_filter(self, candidate_info: Dict, ocr_query: str):
        if not ocr_query or not candidate_info: return
        es_client = self.db_manager.es_client
        if not es_client:
            logger.error("Elasticsearch client không khả dụng. Bỏ qua bộ lọc OCR.")
            return

        kf_ids_to_fetch = list(candidate_info.keys())
        ocr_texts_from_es = {}
        try:
            if kf_ids_to_fetch:
                response = es_client.search(index=settings.OCR_INDEX, body={"query": {"terms": {"keyframe_id": kf_ids_to_fetch}}, "size": len(kf_ids_to_fetch)})
                for hit in response['hits']['hits']:
                    kf_id = hit['_source']['keyframe_id']
                    ocr_text = next((hit['_source'].get(f, '') for f in ['text', 'ocr_text', 'ocr'] if f in hit['_source']), '')
                    if ocr_text: ocr_texts_from_es[kf_id] = ocr_text
        except Exception as e:
            logger.error(f"Failed to query OCR texts from Elasticsearch: {e}", exc_info=True)
            return

        FUZZ_THRESHOLD = 70
        q = self._normalize_text(ocr_query)
        matched_count = 0
        for kf_id, info in candidate_info.items():
            ocr_text = ocr_texts_from_es.get(kf_id)
            if not ocr_text: continue
            t = self._normalize_text(ocr_text)
            score = max(fuzz.partial_ratio(q, t), fuzz.token_set_ratio(q, t), fuzz.token_sort_ratio(q, t))
            if score >= FUZZ_THRESHOLD:
                info['score'] += 0.5
                info['reasons'].append(f"OCR fuzzy match (score={int(score)})")
                matched_count += 1
        #logger.info(f"OCR filter: {matched_count}/{len(candidate_info)} candidates boosted")

    async def _async_apply_asr_filter(self, candidate_info: Dict, asr_query: str):
        if not asr_query or not candidate_info: return
        es_client = self.db_manager.es_client
        if not es_client:
            #logger.error("Elasticsearch client không khả dụng. Bỏ qua bộ lọc ASR.")
            return

        time_window_sec = 15.0
        es_should_clauses, candidate_map = [], {}
        for kf_id, info in candidate_info.items():
            timestamp = info.get("timestamp")
            if timestamp is None: continue
            video_id, _ = self._parse_video_id_from_kf(kf_id)
            kf_start, kf_end = float(timestamp) - time_window_sec, float(timestamp) + time_window_sec
            candidate_map[kf_id] = {"video_id": video_id, "window_start": kf_start, "window_end": kf_end}
            es_should_clauses.append({"bool": {"must": [{"term": {"video_id": video_id}}, {"range": {"start": {"lte": kf_end}}}, {"range": {"end": {"gte": kf_start}}}]}})

        if not es_should_clauses: return
        kf_asr_texts = {}
        try:
            query_body = {"query": {"bool": {"should": es_should_clauses, "minimum_should_match": 1}}, "_source": ["video_id", "text", "start", "end"], "size": 1000, "sort": ["video_id", "start"]}
            response = es_client.search(index=settings.ASR_INDEX, body=query_body)
            asr_segments_by_video = {}
            for hit in response['hits']['hits']:
                vid = hit['_source']['video_id']
                if vid not in asr_segments_by_video: asr_segments_by_video[vid] = []
                asr_segments_by_video[vid].append(hit['_source'])
            for kf_id, data in candidate_map.items():
                if data["video_id"] in asr_segments_by_video:
                    overlapping_texts = [seg['text'] for seg in asr_segments_by_video[data["video_id"]] if seg['start'] <= data['window_end'] and seg['end'] >= data['window_start']]
                    if overlapping_texts: kf_asr_texts[kf_id] = " ".join(overlapping_texts)
        except Exception as e:
            #logger.error(f"Lỗi khi truy vấn ASR text từ Elasticsearch: {e}", exc_info=True)
            return

        ASR_FUZZ_THRESHOLD, BASE_BOOST = 75, 0.5
        q_normalized = self._normalize_text(asr_query)
        matched_count = 0
        for kf_id, info in candidate_info.items():
            full_asr_text = kf_asr_texts.get(kf_id)
            if not full_asr_text: continue
            t_normalized = self._normalize_text(full_asr_text)
            score = max(fuzz.partial_ratio(q_normalized, t_normalized), fuzz.token_set_ratio(q_normalized, t_normalized), fuzz.token_sort_ratio(q_normalized, t_normalized))
            if score >= ASR_FUZZ_THRESHOLD:
                boost = BASE_BOOST * ((score - ASR_FUZZ_THRESHOLD) / (100 - ASR_FUZZ_THRESHOLD))
                info['score'] += boost
                info.setdefault('reasons', []).append(f"ASR dynamic match (score={int(score)}, boost={boost:.3f})")
                matched_count += 1
        #logger.info(f"ASR filter: {matched_count}/{len(candidate_info)} candidates boosted")

    def _format_results(self, sorted_candidates: List[Tuple[str, Dict]]) -> List[Dict]:
        return [{
            "keyframe_id": kf_id, "video_id": self._parse_video_id_from_kf(kf_id)[0],
            "timestamp": info.get('timestamp', 0.0), "score": round(info.get('score', 0.0), 4),
            "reasons": info.get('reasons', []),
            "metadata": {"rank": rank + 1, "clip_score": round(info.get('clip_score', 0.0), 4), "beit3_score": round(info.get('beit3_score', 0.0), 4)}
        } for rank, (kf_id, info) in enumerate(sorted_candidates)]

# --- END OF FILE app/retrieval_engine.py ---