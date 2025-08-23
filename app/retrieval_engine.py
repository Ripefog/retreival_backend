# --- START OF FILE app/retrieval_engine.py ---

import logging
import time
import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import torch
from PIL import Image

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
if not hasattr(np, "asscalar"): np.asscalar = lambda a: a.item() if hasattr(a, "item") else np.asarray(a).item()

logger = logging.getLogger(__name__)


# --- Cấu hình cho BEiT-3 (lấy từ repo gốc) ---
class BEiT3Config:
    def __init__(self):
        self.encoder_embed_dim = 768;
        self.encoder_attention_heads = 12;
        self.encoder_layers = 12
        self.encoder_ffn_embed_dim = 3072;
        self.img_size = 384;
        self.patch_size = 16;
        self.in_chans = 3
        self.vocab_size = 64010;
        self.num_max_bpe_tokens = 64;
        self.max_source_positions = 1024
        self.multiway = True;
        self.share_encoder_input_output_embed = False;
        self.no_scale_embedding = False
        self.layernorm_embedding = False;
        self.normalize_output = True;
        self.no_output_layer = True
        self.drop_path_rate = 0.1;
        self.dropout = 0.0;
        self.attention_dropout = 0.0;
        self.drop_path = 0.1
        self.activation_dropout = 0.0;
        self.max_position_embeddings = 1024;
        self.encoder_normalize_before = True
        self.activation_fn = "gelu";
        self.encoder_learned_pos = True;
        self.xpos_rel_pos = False
        self.xpos_scale_base = 512;
        self.checkpoint_activations = False;
        self.deepnorm = False;
        self.subln = True
        self.rel_pos_buckets = 0;
        self.max_rel_pos = 0;
        self.bert_init = False;
        self.moe_freq = 0
        self.moe_expert_count = 0;
        self.moe_top1_expert = False;
        self.moe_gating_use_fp32 = True
        self.moe_eval_capacity_token_fraction = 0.25;
        self.moe_second_expert_policy = "random"
        self.moe_normalize_gate_prob_before_dropping = False;
        self.use_xmoe = False;
        self.fsdp = False
        self.ddp_rank = 0;
        self.flash_attention = False;
        self.scale_length = 2048;
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
        self.basic_colors = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
                             'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'black': (0, 0, 0),
                             'white': (255, 255, 255), 'gray': (128, 128, 128), 'orange': (255, 165, 0),
                             'brown': (165, 42, 42), 'pink': (255, 192, 203), 'purple': (128, 0, 128)}
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
        if self.initialized: return
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
        self.clip_model.eval();
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        logger.info("  - CLIP model loaded.")
        # 2. Tải BEiT-3
        self.beit3_model = BEiT3ForRetrieval(BEiT3Config())
        checkpoint = torch.load(settings.BEIT3_MODEL_PATH, map_location="cpu")
        self.beit3_model.load_state_dict(checkpoint["model"])
        self.beit3_model = self.beit3_model.to(self.device).eval()
        self.beit3_sp_model = spm.SentencePieceProcessor();
        self.beit3_sp_model.load(settings.BEIT3_SPM_PATH)
        self.beit3_preprocess = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        logger.info("  - BEiT-3 model loaded.")
        # 3. Tải Co-DETR
        self.object_detector = ObjectColorDetector(device=self.device)

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
            _, text_emb = self.beit3_model(text_description=text_ids_tensor, text_padding_mask=text_padding_mask_tensor,
                                           only_infer=True)
            return text_emb.cpu().numpy()[0]
        return None

    def _compare_color(self, color1, color2):
        """So sánh hai màu sắc CIELAB bằng CIEDE2000."""
        color1_lab = LabColor(*color1)  # color1 là tuple (L, A, B)
        color2_lab = LabColor(*color2)  # color2 là tuple (L, A, B)
        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        return delta_e


    # @staticmethod
    def _compare_bbox(self, bbox1, bbox2):
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

    def _parse_video_id_from_kf(self, kf: str):
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
    async def search(self, text_query: str, mode: str, user_query: Optional[str], object_filters: Optional[Dict[str, Any]],
               color_filters: Optional[List[Any]],
               ocr_query: Optional[str], asr_query: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        if not self.initialized: raise RuntimeError("Retriever is not initialized.")
        start_time = time.time()
        #logger.info(f"--- [STARTING SEARCH] Query: '{text_query}', Mode: {mode.upper()} ---")
        #ic(f"--- [STARTING SEARCH] Query: '{text_query}', Mode: {mode.upper()} ---")
        candidate_info: Dict[str, Dict[str, Any]] = {}
        num_initial_candidates = top_k # Lấy số lượng ứng viên ban đầu lớn hơn
        # GĐ1: LẤY ỨNG VIÊN BAN ĐẦU
        if mode in ['hybrid', 'clip']:
            clip_vector = self.get_clip_text_embedding(text_query).tolist()
            clip_candidates =  await self._search_milvus(settings.CLIP_COLLECTION, clip_vector, num_initial_candidates, None, user_query)
            #ic("********************************************************************************")
            # ic(clip_candidates)
            for hit in clip_candidates:
                kf_id = hit["entity"]["keyframe_id"]
                #ic(hit)
                # user = hit["entity"]["user"].split(",")
                vid, kf_id = self._parse_video_id_from_kf(kf_id)
                score = 1.0 / (1.0 + hit['distance'])
                obj_ids = self._split_csv_ints(hit['entity']['object_ids'])
                lab6 = self._parse_lab_colors18(hit['entity']['lab_colors'])
                candidate_info[kf_id] = {
                    "keyframe_id": kf_id,
                    # "user": user,
                    "timestamp": hit['entity']['timestamp'],
                    "object_ids": obj_ids,  # list[int]
                    "lab_colors6": lab6,  # [(L,a,b)*6]
                    "clip_score": score,
                    "score": score,
                    "reasons": [f"CLIP match ({score:.3f})"],
                }
        elif mode == 'beit3':
            beit3_vector = self.get_beit3_text_embedding(text_query).tolist()
            beit3_candidates = self._search_milvus(settings.BEIT3_COLLECTION, beit3_vector, num_initial_candidates, None, user_query)
            #ic("************** BEIT3 **************")
            # ic(beit3_candidates)

            for hit in beit3_candidates:
                kf_id = hit["entity"]["keyframe_id"]
                user_db = hit["entity"]["user"]
                vid, kf_id = self._parse_video_id_from_kf(kf_id)
                score = 1.0 / (1.0 + hit['distance'])
                obj_ids = self._split_csv_ints(hit['entity']['object_ids'])
                lab6 = self._parse_lab_colors18(hit['entity']['lab_colors'])

                if kf_id in candidate_info:
                    # đã có từ CLIP → cộng dồn điểm & bổ sung info nếu thiếu
                    candidate_info[kf_id]['beit3_score'] = score
                    candidate_info[kf_id]['score'] += score
                    candidate_info[kf_id]['reasons'].append(f"BEIT-3 match ({score:.3f})")

                    if not candidate_info[kf_id].get('object_ids') and obj_ids:
                        candidate_info[kf_id]['object_ids'] = obj_ids
                    if not candidate_info[kf_id].get('lab_colors6') and lab6:
                        candidate_info[kf_id]['lab_colors6'] = lab6
                    if candidate_info[kf_id].get('timestamp') is None:
                        candidate_info[kf_id]['timestamp'] = ent.get('timestamp')
                else:
                    # chưa có → tạo mới entry
                    candidate_info[kf_id] = {
                        "keyframe_id": kf_id,
                        "user": user,
                        "timestamp": hit["entity"]['timestamp'],
                        "object_ids": obj_ids,  # list[int]
                        "lab_colors6": lab6,  # [(L,a,b)*6]
                        "beit3_score": score,
                        "score": score,
                        "reasons": [f"BEIT-3 match ({score:.3f})"],
                    }

        # GĐ2: TINH CHỈNH (chỉ cho mode hybrid)
        if mode == 'hybrid' and candidate_info:
            # ic("trước:::::::: ", candidate_info)
            self._hybrid_reranking(candidate_info, text_query)
            #ic("sau:::::::: ", candidate_info)
        # GĐ3: TĂNG ĐIỂM với Object/Color
        if object_filters or color_filters:
            # ic(object_filters)
            await self._apply_object_color_filters(candidate_info, object_filters, color_filters, top_k)

        # GĐ4: LỌC CỨNG với OCR/ASR
        if ocr_query:
            candidate_info = self._apply_ocr_filter_on_candidates(candidate_info, ocr_query)

        # GĐ5: XẾP HẠNG VÀ TRẢ VỀ
        # ic(candidate_info)
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        final_results = self._format_results(sorted_results[:top_k])
        search_time = time.time() - start_time
        #logger.info(f"--- [SEARCH FINISHED] Found {len(final_results)} results in {search_time:.2f}s ---")
        return final_results


    def build_expr(self, expr: Optional[str] = None, user_query: Optional[str] = None) -> Optional[str]:
        user_list = [
            "Gia Nguyên, Duy Bảo",
            "Gia Nguyên, Duy Khương",
            "Gia Nguyên, Minh Tâm",
            "Gia Nguyên, Lê Hiếu",
            "Duy Bảo, Duy Khương",
            "Duy Bảo, Minh Tâm",
            "Duy Bảo, Lê Hiếu",
            "Duy Khương, Minh Tâm",
            "Duy Khương, Lê Hiếu",
            "Minh Tâm, Lê Hiếu"
        ]

        if user_query:
            # Lọc list các user chứa user_query
            filtered_users = [u for u in user_list if user_query in u]
            # Nếu có user nào match
            if filtered_users:
                user_expr = f'user in {filtered_users}'
                # Nếu đã có expr cũ, cộng dồn bằng AND
                if expr:
                    expr = f'({expr}) && ({user_expr})'
                else:
                    expr = user_expr

        return expr


    async def _search_milvus(self, collection_name: str, vector: List[float], top_k: int, expr: Optional[str] = None, user_query: Optional[str] = None) -> List[
        Dict]:
        # ic("alo00", collection_name)
        # Tải sẵn các collection (idempotent)
        await self.db_manager._load_milvus_collections()

        collection = self.db_manager.get_collection(collection_name)
        # ic(collection)
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
            param={"metric_type": "L2", "params": {"nprobe": 16}},
            limit=top_k,
            expr=expr_new,  # <-- quan trọng: giới hạn theo object_id in [...]
            output_fields=output_fields,
        )[0]
        #ic(collection_name, search_results)

        hits = []
        for hit in search_results:
            hit_data = hit.entity.to_dict()["entity"]
            user_db = hit_data.get("user") or ""
            if user_query and user_query not in user_db.split(","):
                continue
            if collection_name in (settings.CLIP_COLLECTION, settings.BEIT3_COLLECTION):
                raw_kf = hit_data.get("keyframe_id", "")
                if isinstance(raw_kf, str):
                    pos = raw_kf.lower().find(".jpg")
                    kf_clean = raw_kf[:pos + 4] if pos != -1 else raw_kf
                else:
                    kf_clean = raw_kf

                # Apply consistent keyframe_id cleaning using existing parser
                vid, kf_normalized = self._parse_video_id_from_kf(kf_clean)

                hits.append({
                    "id": kf_normalized,
                    "distance": hit.distance,
                    "entity": {
                        "keyframe_id": kf_normalized,
                        # "user": hit_data.get("user"),
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

    def _hybrid_reranking(self, candidate_info: Dict[str, Dict], text_query: str):
        beit3_collection = self.db_manager.get_collection(settings.BEIT3_COLLECTION)
        if not beit3_collection:
            logger.warning("BEIT-3 collection not available for reranking")
            return
        candidate_kf_ids = list(candidate_info.keys());
        if not candidate_kf_ids: return
        try:
            # Debug: log candidate keyframe IDs
            logger.debug(f"Hybrid reranking for keyframes: {candidate_kf_ids[:5]}...")

            # Convert normalized keyframe IDs back to database format for querying
            # Database has: L02_L02_V002_0488.36s.jpg
            # Normalized has: L02_V002_0488.36s.jpg
            # We need to query both formats to handle inconsistencies

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
            res = beit3_collection.query(expr=f'keyframe_id in {kf_ids_list}', output_fields=["keyframe_id", "vector"])

            # Map both database format and normalized format to vectors
            beit3_vector_map = {}
            for item in res:
                db_kf_id = item['keyframe_id']
                # Normalize the database keyframe_id for consistent lookup
                vid, normalized_kf_id = self._parse_video_id_from_kf(db_kf_id)
                beit3_vector_map[normalized_kf_id] = item['vector']

            # Debug: log found vs missing keyframes
            found_kfs = set(beit3_vector_map.keys())
            missing_kfs = set(candidate_kf_ids) - found_kfs
            if missing_kfs:
                logger.warning(
                    f"BEIT-3 vectors missing for {len(missing_kfs)}/{len(candidate_kf_ids)} keyframes. Examples: {list(missing_kfs)[:3]}")
            else:
                logger.info(f"Found BEIT-3 vectors for all {len(candidate_kf_ids)} keyframes")

            beit3_query_vector = np.array(self.get_beit3_text_embedding(text_query))
            for kf_id, info in candidate_info.items():
                if kf_id in beit3_vector_map:
                    dist = np.linalg.norm(beit3_query_vector - np.array(beit3_vector_map[kf_id]))
                    beit3_score = 1.0 / (1.0 + dist)
                    info['score'] = (0.4 * info.get('clip_score', 0)) + (0.6 * beit3_score)
                    info['beit3_score'] = beit3_score;
                    info['reasons'].append(f"BEIT-3 refine ({beit3_score:.3f})")
                else:
                    info['score'] *= 0.8;
                    info['reasons'].append("BEIT-3 vector missing")
        except Exception as e:
            logger.error(f"BEIT-3 reranking failed: {e}", exc_info=True)

    # color_filters: List[Tuple[float, float, float]],
    # object_filters: Dict[str, List[Tuple[float, float, float, Tuple[int, int, int, int]]]]
    # def _apply_object_color_filters(self, candidate_info: Dict, object_filters: Optional[List], color_filters: Optional[List], top_k: int):
    #     if object_filters:
    #         for obj in object_filters:
    #             obj_hits = self._search_milvus(settings.OBJECT_COLLECTION, self.get_clip_text_embedding(obj).tolist(), top_k * 10)
    #             for hit in obj_hits:
    #                 kf_id = '_'.join(hit['id'].split('_')[:-2])
    #                 if kf_id in candidate_info: candidate_info[kf_id]['score'] += 0.1; candidate_info[kf_id]['reasons'].append(f"Object match: '{obj}'")
    #     if color_filters:
    #         for color in color_filters:
    #             color_hits = self._search_milvus(settings.COLOR_COLLECTION, self.get_clip_text_embedding(color).tolist(), top_k * 10)
    #             for hit in color_hits:
    #                 kf_id = '_'.join(hit['id'].split('_')[:-2])
    #                 if kf_id in candidate_info: candidate_info[kf_id]['score'] += 0.05; candidate_info[kf_id]['reasons'].append(f"Color match: '{color}'")

    # Giả định: bạn đã có 2 hàm này
    # compare_color((L,A,B), (L,A,B)) -> float (DeltaE, càng nhỏ càng giống)
    # compare_bbox((xmin,ymin,xmax,ymax), (xmin,ymin,xmax,ymax)) -> float (IoU 0..1)

    def _safe_get_label(self, entity: Dict[str, Any]) -> Optional[str]:
        """
        Một số kết quả Milvus có thể bọc 'entity' 2 lớp. Hàm này tìm 'label' an toàn.
        """
        if not entity:
            return None
        # Trường hợp phổ biến: hit['entity']['label']
        if isinstance(entity, dict) and 'label' in entity and entity['label']:
            return entity['label']
        # Trường hợp bị lồng: hit['entity']['entity']['label']
        inner = entity.get('entity') if isinstance(entity, dict) else None
        if isinstance(inner, dict) and 'label' in inner and inner['label']:
            return inner['label']
        return None

    def _parse_color_bbox_from_label(self, label: str) -> Tuple[
        Optional[Tuple[float, float, float]], Optional[Tuple[int, int, int, int]]]:
        """
        Trả về (LAB | None, BBOX | None)
        label: 'L,A,B' hoặc 'xmin,ymin,xmax,ymax' hoặc 'L,A,B,xmin,ymin,xmax,ymax'
        """
        if not label:
            return None, None

        parts = [p.strip() for p in label.split(",") if p.strip() != ""]
        try:
            nums = list(map(float, parts))
        except ValueError:
            return None, None

        if len(nums) == 3:
            # Chỉ LAB
            return (nums[0], nums[1], nums[2]), None
        elif len(nums) == 4:
            # Chỉ bbox
            return None, (int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]))
        elif len(nums) >= 7:
            # Cả LAB và bbox
            return (nums[0], nums[1], nums[2]), (int(nums[3]), int(nums[4]), int(nums[5]), int(nums[6]))
        else:
            return None, None

    def _normalize_object_filters(self,
                                  object_filters: Dict[str, Any]
                                  ) -> Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]]:
        """
        Nhận vào: {obj: [((L,A,B),(xmin,ymin,xmax,ymax)), ...]}
        Trả ra:   y hệt, nhưng có validate cơ bản.
        """
        norm: Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]] = {}
        
        if not isinstance(object_filters, dict):
            logger.warning("object_filters is not a dict, skipping")
            return norm
            
        for obj, items in object_filters.items():
            if not isinstance(items, (list, tuple)):
                logger.warning(f"object_filters['{obj}'] is not a list/tuple, skipping")
                continue
                
            fixed: List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]] = []
            for it in items:
                try:
                    # kỳ vọng: it = ((L,A,B),(x1,y1,x2,y2))
                    if (not isinstance(it, (list, tuple))) or len(it) != 2:
                        logger.debug(f"Invalid item format in object_filters['{obj}'], expected 2-tuple")
                        continue
                    lab, bbox = it[0], it[1]
                    if not (isinstance(lab, (list, tuple)) and len(lab) == 3):
                        logger.debug(f"Invalid LAB format in object_filters['{obj}'], expected 3-tuple")
                        continue
                    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                        logger.debug(f"Invalid bbox format in object_filters['{obj}'], expected 4-tuple")
                        continue
                    lab_t = (float(lab[0]), float(lab[1]), float(lab[2]))
                    bbox_t = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    fixed.append((lab_t, bbox_t))
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Error parsing object_filters['{obj}'] item: {e}")
                    continue
            if fixed:
                norm[obj] = fixed
        return norm

    async def _apply_object_color_filters(
            self,
            candidate_info: Dict[str, Dict[str, Any]],
            object_filters: Optional[Dict[str, Any]],
            color_filters: Optional[List[Any]],
            top_k: int
    ):
        # ====== helpers nội bộ ======
        import math

        def _split_csv_floats(s: Optional[str]) -> List[float]:
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

        def _ensure_lab(c: Optional[Tuple[float, float, float]]):
            """Nhận (R,G,B) 0..255 hoặc (L,a,b). Nếu có vẻ là RGB → convert sang LAB."""
            if c is None:
                return None
            L, A, B = c
            # heuristics: nếu tất cả trong [0..255] và có ít nhất 1 > 1 → xem như RGB
            if 0 <= L <= 255 and 0 <= A <= 255 and 0 <= B <= 255 and (L > 1 or A > 1 or B > 1):
                return self._rgb_to_lab((int(L), int(A), int(B)))
            return (float(L), float(A), float(B))  # giả định đã là LAB

        def _sim_from_delta(d: float, sigma: float = 20.0) -> float:
            """Similarity từ ΔE: exp(-(ΔE/σ)^2)."""
            return math.exp(- (d / sigma) ** 2)

        def _hungarian_min_cost(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
            """Hungarian tối giản (min-cost). Tự pad ma trận thành vuông."""
            r = len(cost_matrix)
            c = len(cost_matrix[0]) if r > 0 else 0
            n = max(r, c)
            # cost dummy = 1.0 (tương ứng sim=0) cho ô pad
            cost = [[1.0 for _ in range(n)] for __ in range(n)]
            for i in range(r):
                for j in range(c):
                    cost[i][j] = float(cost_matrix[i][j])

            u = [0.0] * (n + 1)
            v = [0.0] * (n + 1)
            p = [0] * (n + 1)
            way = [0] * (n + 1)

            for i in range(1, n + 1):
                p[0] = i
                j0 = 0
                minv = [float("inf")] * (n + 1)
                used = [False] * (n + 1)
                while True:
                    used[j0] = True
                    i0 = p[j0]
                    delta = float("inf")
                    j1 = 0
                    for j in range(1, n + 1):
                        if not used[j]:
                            cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                            if cur < minv[j]:
                                minv[j] = cur
                                way[j] = j0
                            if minv[j] < delta:
                                delta = minv[j]
                                j1 = j
                    for j in range(0, n + 1):
                        if used[j]:
                            u[p[j]] += delta
                            v[j] -= delta
                        else:
                            minv[j] -= delta
                    j0 = j1
                    if p[j0] == 0:
                        break
                while True:
                    j1 = way[j0]
                    p[j0] = p[j1]
                    j0 = j1
                    if j0 == 0:
                        break

            assignment = []
            for j in range(1, n + 1):
                i = p[j]
                if i != 0:
                    assignment.append((i - 1, j - 1))
            return assignment

        # ====== tham số scoring ======
        # trọng số gốc cho từng thành phần (sẽ re-normalize theo query có gì)
        W_VEC = 0.6
        W_COLOR = 0.3
        W_BBOX = 0.4
        # tham số color
        SIGMA_COLOR = 20.0  # cho similarity exp(-(ΔE/σ)^2)
        MAX_DELTA_E = 50.0  # gate: nếu ΔE > ngưỡng → coi như 0
        # tham số bbox
        MIN_IOU = 0.10  # gate: IoU >= 0.10 mới tính điểm bbox
        # kết hợp assignment & coverage
        ALPHA = 0.7
        BETA = 0.3
        TAU_S = 0.5  # ngưỡng S_ij để tính "covered"
        # trọng số boost vào tổng score keyframe
        W_OBJ = 0.20

        # ===== OBJECT FILTERS =====
        if object_filters:
            norm_object_filters = self._normalize_object_filters(object_filters)
            #ic(norm_object_filters)

            for obj_label, queries in norm_object_filters.items():
                # vector của nhãn cần tìm (chung cho các truy vấn con của label này)
                obj_vector = self.get_clip_text_embedding(obj_label).tolist()

                # duyệt từng keyframe ứng viên
                for kf_id, info in candidate_info.items():
                    obj_ids = info.get("object_ids") or []
                    if not obj_ids:
                        continue

                    # lấy toàn bộ object của frame này
                    expr = f"object_id in [{','.join(map(str, obj_ids))}]"
                    limit = max(len(obj_ids), 1)
                    obj_hits = await self._search_milvus(settings.OBJECT_COLLECTION, obj_vector, limit, expr=expr)
                    if not obj_hits:
                        continue

                    # chuẩn bị m (số query con) & n (số object thật)
                    # mỗi query: (query_color_lab | None, query_bbox | None)
                    Q = []
                    for (q_color, q_bbox) in queries:
                        q_lab = _ensure_lab(q_color) if q_color is not None else None
                        q_bb = tuple(q_bbox) if q_bbox is not None else None
                        Q.append((q_lab, q_bb))
                    m = len(Q)
                    if m == 0:
                        continue

                    # parse hits (mảng per object)
                    O_vec_sim = []  # vector similarity theo hit.distance
                    O_color_lab = []  # [L,a,b] hoặc None
                    O_bbox = []  # [x1,y1,x2,y2] hoặc None
                    for h in obj_hits:
                        ent = h.get("entity", {})
                        # vector similarity từ distance: 1/(1+d)
                        d = float(h.get("distance", 0.0))
                        s_vec = 1.0 / (1.0 + d)
                        O_vec_sim.append(s_vec)
                        # parse màu & bbox
                        cl = _split_csv_floats(ent.get("color_lab"))
                        bl = _split_csv_floats(ent.get("bbox_xyxy"))
                        O_color_lab.append(cl if len(cl) == 3 else None)
                        O_bbox.append(tuple(bl) if len(bl) == 4 else None)

                    n = len(O_vec_sim)
                    if n == 0:
                        continue

                    # xây ma trận similarity S_ij (m x n)
                    S = [[0.0 for _ in range(n)] for __ in range(m)]
                    for i in range(m):
                        q_lab, q_bb = Q[i]
                        # xác định các thành phần có mặt
                        use_vec = True  # vector luôn có (do từ search)
                        use_color = (q_lab is not None)
                        use_bbox = (q_bb is not None)

                        # re-normalize trọng số theo phần có mặt
                        w_sum = 0.0
                        wv = W_VEC if use_vec else 0.0
                        wc = W_COLOR if use_color else 0.0
                        wb = W_BBOX if use_bbox else 0.0
                        w_sum = wv + wc + wb
                        if w_sum == 0:
                            # không có tín hiệu gì (trường hợp hiếm) → bỏ qua query này
                            continue
                        wv /= w_sum;
                        wc /= w_sum;
                        wb /= w_sum

                        for j in range(n):
                            s_total = 0.0

                            # vector sim
                            s_total += wv * O_vec_sim[j]

                            # color sim
                            if use_color:
                                p_lab = O_color_lab[j]
                                if p_lab is not None:
                                    de = self._compare_color(q_lab, tuple(p_lab))  # ΔE2000
                                    if de <= MAX_DELTA_E:
                                        s_col = _sim_from_delta(de, SIGMA_COLOR)
                                        s_total += wc * s_col
                                    else:
                                        # gate out nếu quá khác
                                        pass

                            # bbox sim
                            if use_bbox:
                                p_bb = O_bbox[j]
                                if p_bb is not None:
                                    iou = float(self._compare_bbox(q_bb, p_bb))
                                    if iou >= MIN_IOU:
                                        s_total += wb * iou
                                    else:
                                        pass

                            S[i][j] = s_total

                    # Hungarian: cost = 1 - S
                    cost = [[1.0 - S[i][j] for j in range(n)] for i in range(m)]
                    assignment = _hungarian_min_cost(cost)  # list of (i,j) theo ma trận pad

                    # tổng hợp điểm cho keyframe: S_match & coverage
                    sim_sum = 0.0
                    match_pairs = 0
                    covered = 0
                    for (i, j) in assignment:
                        if i < m and j < n:
                            sij = S[i][j]
                            sim_sum += sij
                            match_pairs += 1
                            if sij >= TAU_S:
                                covered += 1

                    # Chuẩn hoá theo số query (m) để công bằng
                    S_match = (sim_sum / m) if m > 0 else 0.0
                    C = (covered / m) if m > 0 else 0.0
                    S_obj = ALPHA * S_match + BETA * C
                    boost = W_OBJ * S_obj

                    if boost > 0:
                        info["score"] += boost
                        info.setdefault("reasons", []).append(
                            f"Object match: '{obj_label}' +{boost:.3f} (S={S_obj:.3f}, cov={C:.2f})"
                        )

        # ===== COLOR FILTERS (độc lập) =====
        if color_filters:
            # 1) Chuẩn bị: RGB -> LAB cho toàn bộ màu truy vấn
            queries_lab = []
            for qc in color_filters:
                if qc is None:
                    continue
                # qc có dạng [R,G,B] hoặc tuple
                queries_lab.append(self._rgb_to_lab(tuple(qc)))
            if not queries_lab:
                pass  # không có màu truy vấn thì bỏ qua
            else:
                import math

                # Gaussian similarity từ ΔE (CIEDE2000)
                def _sim_from_delta(d, sigma=20.0):
                    return math.exp(- (d / sigma) ** 2)

                # Hungarian (min-cost) cho ma trận vuông; tự pad nếu hình chữ nhật
                def _hungarian_min_cost(cost_matrix):
                    # Pad thành vuông
                    r = len(cost_matrix)
                    c = len(cost_matrix[0]) if r > 0 else 0
                    n = max(r, c)
                    # dùng cost_dummy=1.0 (tương ứng sim=0) cho ô trống
                    cost = [[1.0 for _ in range(n)] for __ in range(n)]
                    for i in range(r):
                        for j in range(c):
                            cost[i][j] = float(cost_matrix[i][j])

                    # Thuật toán Hungarian (phiên bản ngắn gọn)
                    u = [0.0] * (n + 1)
                    v = [0.0] * (n + 1)
                    p = [0] * (n + 1)
                    way = [0] * (n + 1)

                    for i in range(1, n + 1):
                        p[0] = i
                        j0 = 0
                        minv = [float("inf")] * (n + 1)
                        used = [False] * (n + 1)
                        while True:
                            used[j0] = True
                            i0 = p[j0]
                            delta = float("inf")
                            j1 = 0
                            for j in range(1, n + 1):
                                if not used[j]:
                                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                                    if cur < minv[j]:
                                        minv[j] = cur
                                        way[j] = j0
                                    if minv[j] < delta:
                                        delta = minv[j]
                                        j1 = j
                            for j in range(0, n + 1):
                                if used[j]:
                                    u[p[j]] += delta
                                    v[j] -= delta
                                else:
                                    minv[j] -= delta
                            j0 = j1
                            if p[j0] == 0:
                                break
                        # rebuild matching
                        while True:
                            j1 = way[j0]
                            p[j0] = p[j1]
                            j0 = j1
                            if j0 == 0:
                                break

                    # Kết quả: ghép i->j
                    assignment = []
                    # p[j] = i  (hàng i gán cho cột j)
                    for j in range(1, n + 1):
                        i = p[j]
                        if i != 0:
                            assignment.append((i - 1, j - 1))
                    return assignment  # list[(row_idx, col_idx)]

                # 2) Chấm điểm theo Hungarian cho từng keyframe
                alpha, beta = 0.7, 0.3  # trọng số kết hợp sim & coverage
                w_color = 0.15  # trọng số boost vào tổng điểm
                tau = 15.0  # ngưỡng coverage (ΔE <= tau)
                for kf_id, info in candidate_info.items():
                    palette = info.get("lab_colors6") or []  # list[(L,a,b), ...] (tối đa 6)
                    if not palette:
                        continue

                    m = len(queries_lab)
                    n = len(palette)

                    # Xây cost matrix = 1 - sim(ΔE) (minimize cost = maximize sim)
                    cost = []
                    for i in range(m):
                        row = []
                        for j in range(n):
                            d = self._compare_color(queries_lab[i], palette[j])  # ΔE2000
                            s = _sim_from_delta(d)
                            row.append(1.0 - s)
                        cost.append(row)

                    # Hungarian cho kích thước min(m,n) (pad nội bộ)
                    assignment = _hungarian_min_cost(cost)  # (i,j) theo ma trận pad

                    # Tính sim trung bình trên các cặp thực (i<m, j<n)
                    sim_sum = 0.0
                    real_pairs = 0
                    for i, j in assignment:
                        if i < m and j < n:
                            d = self._compare_color(queries_lab[i], palette[j])
                            sim_sum += _sim_from_delta(d)
                            real_pairs += 1
                    # Chuẩn hoá theo số màu query để không ưu tiên frame “ít bị ghép”
                    S_hung = (sim_sum / m) if m > 0 else 0.0

                    # Coverage: tỉ lệ màu query “gần” một màu palette (ΔE <= tau)
                    covered = 0
                    for i in range(m):
                        best_d = min(self._compare_color(queries_lab[i], p) for p in palette)
                        if best_d <= tau:
                            covered += 1
                    C = (covered / m) if m > 0 else 0.0

                    S_color = alpha * S_hung + beta * C
                    boost = w_color * S_color

                    if boost > 0:
                        info["score"] += boost
                        info.setdefault("reasons", []).append(
                            f"Color match (Hungarian): +{boost:.3f} (S={S_color:.3f}, cov={C:.2f})"
                        )

    # def _apply_ocr_filter_on_candidates(self, candidate_info: Dict, ocr_query: str) -> Dict:
    #     """
    #     Áp dụng bộ lọc OCR dựa trên danh sách ứng viên hiện có.
    #
    #     Hàm này thực hiện các bước sau:
    #     1. Lấy tất cả keyframe_id từ các ứng viên đầu vào.
    #     2. Gửi MỘT truy vấn duy nhất đến Elasticsearch để lấy văn bản OCR cho TẤT CẢ các keyframe đó.
    #     3. Lặp qua từng ứng viên, so sánh văn bản OCR của nó với ocr_query.
    #     4. Nếu khớp, tăng điểm và ghi lại lý do.
    #     5. Trả về danh sách ứng viên đã được cập nhật điểm.
    #     """
    #     if not ocr_query or not candidate_info:
    #         return candidate_info
    #
    #     es_client = self.db_manager.es_client
    #     if not es_client:
    #         logger.error("Elasticsearch client không khả dụng. Bỏ qua bộ lọc OCR.")
    #         return candidate_info
    #
    #     kf_ids_to_fetch = list(candidate_info.keys())
    #     logger.debug(f"Chuẩn bị áp dụng bộ lọc OCR cho {len(kf_ids_to_fetch)} ứng viên.")
    #
    #     ocr_texts_from_es = {}
    #     try:
    #         ic(f"ES: Lấy văn bản OCR cho {len(kf_ids_to_fetch)} keyframes.")
    #         res = es_client.search(
    #             index=settings.OCR_INDEX,
    #             body={
    #                 "query": {
    #                     # === SỬA ĐỔI QUAN TRỌNG NHẤT NẰM Ở ĐÂY ===
    #                     # Bỏ ".keyword" vì mapping đã định nghĩa trường là "keyword"
    #                     "terms": {"keyframe_id": kf_ids_to_fetch}
    #                 },
    #                 "_source": ["keyframe_id", "text"],
    #                 "size": len(kf_ids_to_fetch)
    #             }
    #         )
    #
    #         for hit in res['hits']['hits']:
    #             source = hit.get('_source', {})
    #             kf_id = source.get('keyframe_id')
    #             text = source.get('text')
    #             if kf_id and text:
    #                 ocr_texts_from_es[kf_id] = text
    #         ic(f"ES: Tìm thấy văn bản cho {len(ocr_texts_from_es)}/{len(kf_ids_to_fetch)} keyframes.")
    #
    #     except Exception as e:
    #         logger.error(f"ES OCR search để lấy văn bản thất bại: {e}", exc_info=True)
    #         return candidate_info
    #
    #     matched_count = 0
    #     for kf_id, info in candidate_info.items():
    #         ocr_text = ocr_texts_from_es.get(kf_id)
    #
    #         if ocr_text:
    #             if ocr_query.lower() in ocr_text.lower():
    #                 info['score'] += 0.5
    #                 info['reasons'].append("OCR match")
    #                 matched_count += 1
    #
    #     logger.info(f"Bộ lọc OCR: {matched_count}/{len(candidate_info)} ứng viên được tăng điểm.")
    #     ic(f"Bộ lọc OCR: {matched_count}/{len(candidate_info)} ứng viên được tăng điểm.")
    #
    #     return candidate_info

    def _normalize_text(self, s: str) -> str:
        # đủ dùng: lower + strip + rút gọn khoảng trắng
        # (nếu cần bóc dấu thì thêm unidecode, nhưng bạn bảo chỉ thêm rapidfuzz nên mình giữ tối thiểu)
        return " ".join((s or "").lower().split())

    def _apply_ocr_filter_on_candidates(self, candidate_info: Dict, ocr_query: str) -> Dict:
        """
        Áp dụng bộ lọc OCR dựa trên danh sách ứng viên hiện có.

        Hàm này thực hiện các bước sau:
        1. Lấy tất cả keyframe_id từ các ứng viên đầu vào.
        2. Gửi MỘT truy vấn duy nhất đến Elasticsearch để lấy văn bản OCR cho TẤT CẢ các keyframe đó.
        3. Lặp qua từng ứng viên, so sánh văn bản OCR của nó với ocr_query.
        4. Nếu khớp, tăng điểm và ghi lại lý do.
        5. Trả về danh sách ứng viên đã được cập nhật điểm.
        """
        if not ocr_query or not candidate_info:
            return candidate_info

        es_client = self.db_manager.es_client
        if not es_client:
            logger.error("Elasticsearch client không khả dụng. Bỏ qua bộ lọc OCR.")
            return candidate_info

        kf_ids_to_fetch = list(candidate_info.keys())
        #logger.debug(f"Chuẩn bị áp dụng bộ lọc OCR cho {len(kf_ids_to_fetch)} ứng viên.")

        ocr_texts_from_es = {}
        #try:
            #ic(f"ES: Lấy văn bản OCR cho {len(kf_ids_to_fetch)} keyframes.")
            #cmt elastic
            # res = es_client.search(
            #     index=settings.OCR_INDEX,
            #     body={
            #         "query": {
            #             # Bỏ ".keyword" vì mapping đã định nghĩa trường là "keyword"
            #             "terms": {"keyframe_id": kf_ids_to_fetch}
            #         },
            #         "_source": ["keyframe_id", "text"],
            #         "size": len(kf_ids_to_fetch)
            #     }
            # )
            #
            # for hit in res['hits']['hits']:
            #     source = hit.get('_source', {})
            #     kf_id = source.get('keyframe_id')
            #     text = source.get('text')
            #     if kf_id and text:
            #         ocr_texts_from_es[kf_id] = text
            # ic(f"ES: Tìm thấy văn bản cho {len(ocr_texts_from_es)}/{len(kf_ids_to_fetch)} keyframes.")

        #except Exception as e:
            #logger.error(f"ES OCR search để lấy văn bản thất bại: {e}", exc_info=True)
            #return candidate_info

        # Ngưỡng fuzzy (có thể điều chỉnh nếu cần, giữ nguyên hành vi cộng điểm khi đạt ngưỡng)
        FUZZ_THRESHOLD = 70  # hạ ngưỡng để phù hợp partial/token_set
        q = self._normalize_text(ocr_query)

        matched_count = 0
        for kf_id, info in candidate_info.items():
            ocr_text = ocr_texts_from_es.get(kf_id)
            if not ocr_text:
                continue

            t = self._normalize_text(ocr_text)

            # Dùng các scorer phù hợp với case "query ngắn" vs "văn bản dài"
            score_partial = fuzz.partial_ratio(q, t)
            score_token_set = fuzz.token_set_ratio(q, t)
            score_token_sort = fuzz.token_sort_ratio(q, t)  # tùy chọn
            score = max(score_partial, score_token_set, score_token_sort)

            if score >= FUZZ_THRESHOLD:
                info['score'] += 0.5
                info['reasons'].append(f"OCR fuzzy match (score={int(score)})")
                matched_count += 1

        #logger.info(f"Bộ lọc OCR: {matched_count}/{len(candidate_info)} ứng viên được tăng điểm.")
        #ic(f"Bộ lọc OCR: {matched_count}/{len(candidate_info)} ứng viên được tăng điểm.")
        return candidate_info

    def _format_results(self, sorted_candidates: List[Tuple[str, Dict]]) -> List[Dict]:
        # ic(sorted_candidates)
        return [{
            "keyframe_id": kf_id, "video_id": info.get('video_id', ''), "timestamp": info.get('timestamp', 0.0),
            "score": round(info.get('score', 0.0), 4), "reasons": info.get('reasons', []),
            "metadata": {"rank": rank + 1, "clip_score": round(info.get('clip_score', 0.0), 4),
                         "beit3_score": round(info.get('beit3_score', 0.0), 4)}
        } for rank, (kf_id, info) in enumerate(sorted_candidates)]

    # --- CÁC PHƯƠNG THỨC TIỆN ÍCH ĐƯỢC GỌI TỪ API ---
    def detect_objects_in_image(self, image_path: str) -> Tuple[List[str], List[str]]:
        # ic(self.object_detector)
        if not self.object_detector: raise RuntimeError("Object detector is not initialized.")
        return self.object_detector.detect(image_path)

    def check_milvus_connection(self) -> Dict[str, Any]:
        return self.db_manager.check_milvus_connection()

    def check_elasticsearch_connection(self) -> Dict[str, Any]:
        return self.db_manager.check_elasticsearch_connection()
# --- END OF FILE app/retrieval_engine.py ---


#
# {
#   "text_query": "There are two men with a boat in the river.",
#   "mode": "hybrid",
#   "object_filters": {
#     "boat": [
#       [[46.353206722122124, -4.904548028573375, 5.2011766925410985], [213, 393, 550, 483]]
#     ],
#     "person": [
#       [[37.71225526048862, -1.0217397637916903, 4.189403003138226], [476, 374, 547, 454]],
#       [[47.46675155121266, 3.4061168538457864, 11.794993111213802], [200, 338, 252, 422]]
#     ]
#   },
#   "color_filters": [
#     [76.65912653528162, -0.5829774648731245, -19.05410702358592],
#     [18.828432444742575, -6.5353777349142215, 12.744719724371178],
#     [59.29708671909229, -6.493880177980859, 3.162971364802103],
#     [60.69111602558509, 50.29951194603682, 14.683318638437793],
#     [34.61410230810931, 58.08672236684348, 44.34712642128737],
#     [44.31269057964889, -10.4675878934003, 17.099308966327964]
#   ],
#   "ocr_query": "string",
#   "asr_query": "string",
#   "top_k": 20
# }