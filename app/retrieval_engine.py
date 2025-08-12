# --- START OF FILE app/retrieval_engine.py ---

import logging
import time
import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import torch
from PIL import Image
import random
import itertools

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
# Import từ các module của ứng dụng
from .config import settings
from .database import db_manager

logger = logging.getLogger(__name__)

# --- Cấu hình cho BEiT-3 (lấy từ repo gốc) ---
class BEiT3Config:
    def __init__(self):
        self.encoder_embed_dim = 768; self.encoder_attention_heads = 12; self.encoder_layers = 12
        self.encoder_ffn_embed_dim = 3072; self.img_size = 384; self.patch_size = 16; self.in_chans = 3
        self.vocab_size = 64010; self.num_max_bpe_tokens = 64; self.max_source_positions = 1024
        self.multiway = True; self.share_encoder_input_output_embed = False; self.no_scale_embedding = False
        self.layernorm_embedding = False; self.normalize_output = True; self.no_output_layer = True
        self.drop_path_rate = 0.1; self.dropout = 0.0; self.attention_dropout = 0.0; self.drop_path = 0.1
        self.activation_dropout = 0.0; self.max_position_embeddings = 1024; self.encoder_normalize_before = True
        self.activation_fn = "gelu"; self.encoder_learned_pos = True; self.xpos_rel_pos = False
        self.xpos_scale_base = 512; self.checkpoint_activations = False; self.deepnorm = False; self.subln = True
        self.rel_pos_buckets = 0; self.max_rel_pos = 0; self.bert_init = False; self.moe_freq = 0
        self.moe_expert_count = 0; self.moe_top1_expert = False; self.moe_gating_use_fp32 = True
        self.moe_eval_capacity_token_fraction = 0.25; self.moe_second_expert_policy = "random"
        self.moe_normalize_gate_prob_before_dropping = False; self.use_xmoe = False; self.fsdp = False
        self.ddp_rank = 0; self.flash_attention = False; self.scale_length = 2048; self.layernorm_eps = 1e-5

class ObjectColorDetector:
    def __init__(self, device):
        logger.info("Initializing ObjectColorDetector (Co-DETR)...")
        self.device = device
        self.model = init_detector(
            Config.fromfile(settings.CO_DETR_CONFIG_PATH),
            settings.CO_DETR_CHECKPOINT_PATH,
            device=self.device
        )
        self.basic_colors = {'red':(255,0,0), 'green':(0,255,0), 'blue':(0,0,255), 'yellow':(255,255,0), 'cyan':(0,255,255), 'magenta':(255,0,255), 'black':(0,0,0), 'white':(255,255,255), 'gray':(128,128,128), 'orange':(255,165,0), 'brown':(165,42,42), 'pink':(255,192,203), 'purple': (128,0,128)}
        
        # === SỬA LỖI Ở ĐÂY: KHỞI TẠO self.basic_colors_lab ===
        self.basic_colors_lab = self._convert_basic_colors_to_lab()
        logger.info("✅ Co-DETR model loaded.")

    def _convert_basic_colors_to_lab(self) -> dict:
        lab_dict = {}
        for name, rgb in self.basic_colors.items():
            rgb_obj = sRGBColor(*rgb, is_upscaled=True)
            lab_obj = convert_color(rgb_obj, LabColor)
            lab_dict[name] = lab_obj
        return lab_dict

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        rgb_obj = sRGBColor(*rgb, is_upscaled=True)
        lab_obj = convert_color(rgb_obj, LabColor)
        return (lab_obj.lab_l, lab_obj.lab_a, lab_obj.lab_b)

    def _get_closest_color_name(self, rgb_query: Tuple[int, int, int]) -> str:
        lab_query = self._rgb_to_lab(rgb_query)
        query_color_obj = LabColor(lab_query[0], lab_query[1], lab_query[2])
        
        min_delta = float('inf')
        closest_name = "unknown"
        for name, lab_ref_obj in self.basic_colors_lab.items():
            delta = delta_e_cie2000(query_color_obj, lab_ref_obj)
            if delta < min_delta:
                min_delta = delta
                closest_name = name
        return closest_name

    def detect(self, image_path: str) -> Tuple[List[Tuple[float, float, float]], Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]]]:
        try:
            result = inference_detector(self.model, image_path)
            if isinstance(result, tuple): result = result[0]
            img = cv2.imread(image_path)
            if img is None: raise ValueError("Không đọc được ảnh.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            flat_pixels = img_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(flat_pixels)
            dominant_rgb = kmeans.cluster_centers_.astype(int)
            dominant_colors_lab = [self._rgb_to_lab(tuple(color)) for color in dominant_rgb]
            object_colors_lab = {}
            for class_id, bboxes in enumerate(result):
                if class_id >= len(self.model.CLASSES): continue
                class_name = self.model.CLASSES[class_id]
                for bbox in bboxes:
                    if bbox[4] < 0.5: continue
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3)
                    kmeans_obj = KMeans(n_clusters=1, random_state=0, n_init=10).fit(crop_rgb)
                    dom_rgb = kmeans_obj.cluster_centers_[0].astype(int)
                    lab_color = self._rgb_to_lab(tuple(dom_rgb))
                    if class_name not in object_colors_lab: object_colors_lab[class_name] = []
                    object_colors_lab[class_name].append((lab_color, (x1, y1, x2, y2)))
            return dominant_colors_lab, object_colors_lab
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], {}

class HybridRetriever:
    # ... (các hàm __init__, initialize, _load_models, get_*_embedding, _compare_*, search, ... giữ nguyên)
    def __init__(self):
        self.db_manager = db_manager; self.device = settings.DEVICE; self.initialized = False
        self.clip_model, self.clip_preprocess, self.clip_tokenizer = None, None, None
        self.beit3_model, self.beit3_preprocess, self.beit3_sp_model = None, None, None
        self.object_detector: Optional[ObjectColorDetector] = None

    async def initialize(self):
        if self.initialized: return
        logger.info("Initializing Hybrid Retriever engine...")
        if not self.db_manager.milvus_connected or not self.db_manager.elasticsearch_connected: raise RuntimeError("Database connections must be established before initializing the retriever.")
        self._load_models()
        self.initialized = True
        logger.info("✅ Hybrid Retriever initialized successfully.")

    def _load_models(self):
        logger.info(f"Loading AI models onto device: '{self.device}'")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(model_name='ViT-H-14', pretrained=settings.CLIP_MODEL_PATH, device=self.device)
        self.clip_model.eval(); self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        logger.info("  - CLIP model loaded.")
        self.beit3_model = BEiT3ForRetrieval(BEiT3Config())
        checkpoint = torch.load(settings.BEIT3_MODEL_PATH, map_location="cpu")
        self.beit3_model.load_state_dict(checkpoint["model"])
        self.beit3_model = self.beit3_model.to(self.device).eval()
        self.beit3_sp_model = spm.SentencePieceProcessor(); self.beit3_sp_model.load(settings.BEIT3_SPM_PATH)
        self.beit3_preprocess = transforms.Compose([transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        logger.info("  - BEiT-3 model loaded.")
        self.object_detector = ObjectColorDetector(device=self.device)

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
            _, text_emb = self.beit3_model(text_description=text_ids_tensor, text_padding_mask=text_padding_mask_tensor, only_infer=True)
            return text_emb.cpu().numpy()[0]

    def _compare_color(self, color1, color2):
        color1_lab = LabColor(*color1); color2_lab = LabColor(*color2)
        return delta_e_cie2000(color1_lab, color2_lab)

    def _compare_bbox(self, bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = bbox1; x2_min, y2_min, x2_max, y2_max = bbox2
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)); y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        area1 = (x1_max - x1_min) * (y1_max - y1_min); area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def search(self, text_query: str, mode: str, object_filters: Optional[Dict], color_filters: Optional[List], ocr_query: Optional[str], asr_query: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        if not self.initialized: raise RuntimeError("Retriever is not initialized.")
        start_time = time.time(); logger.info(f"--- [STARTING SEARCH] Query: '{text_query}', Mode: {mode.upper()} ---")
        candidate_info: Dict[str, Dict[str, Any]] = {}; num_initial_candidates = top_k * 5
        
        if mode in ['hybrid', 'clip']:
            clip_vector = self.get_clip_text_embedding(text_query).tolist()
            clip_candidates = self._search_milvus(settings.CLIP_COLLECTION, clip_vector, num_initial_candidates)
            for hit in clip_candidates:
                if hit['entity']['keyframe_id']:
                    score = 1.0 / (1.0 + hit['distance'])
                    candidate_info[hit['entity']['keyframe_id']] = {'video_id': hit['entity']['video_id'], 'timestamp': hit['entity']['timestamp'], 'clip_score': score, 'score': score, 'reasons': [f"CLIP match ({score:.3f})"]}
        elif mode == 'beit3':
            beit3_vector = self.get_beit3_text_embedding(text_query).tolist()
            beit3_candidates = self._search_milvus(settings.BEIT3_COLLECTION, beit3_vector, num_initial_candidates)
            for hit in beit3_candidates:
                 if hit['entity']['keyframe_id']:
                    score = 1.0 / (1.0 + hit['distance'])
                    candidate_info[hit['entity']['keyframe_id']] = {'video_id': hit['entity']['video_id'], 'timestamp': hit['entity']['timestamp'], 'beit3_score': score, 'score': score, 'reasons': [f"BEIT-3 match ({score:.3f})"]}
        
        if mode == 'hybrid' and candidate_info: self._hybrid_reranking(candidate_info, text_query)
        
        if object_filters or color_filters: 
            self._apply_object_color_filters(candidate_info, object_filters, color_filters, top_k)
            
        if ocr_query or asr_query:
            ocr_kf_ids, asr_video_ids = self._search_es(ocr_query, asr_query)
            candidate_info = self._apply_text_filters(candidate_info, ocr_kf_ids, asr_video_ids)
        
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        final_results = self._format_results(sorted_results[:top_k])
        search_time = time.time() - start_time
        logger.info(f"--- [SEARCH FINISHED] Found {len(final_results)} results in {search_time:.2f}s ---")
        return final_results

    def _search_milvus(self, collection_name: str, vector: List[float], top_k: int) -> List[Dict]:
        collection = self.db_manager.get_collection(collection_name)
        if not collection: return []
        search_results = collection.search(data=[vector], anns_field="vector", param={"metric_type": "L2", "params": {"nprobe": 16}}, limit=top_k, output_fields=["keyframe_id", "video_id", "timestamp"])[0]
        hits = [{'id': hit.id, 'distance': hit.distance, 'entity': {'keyframe_id': hit.entity.get('keyframe_id'), 'video_id': hit.entity.get('video_id'), 'timestamp': hit.entity.get('timestamp')}} for hit in search_results]
        return hits

    def _hybrid_reranking(self, candidate_info: Dict[str, Dict], text_query: str):
        beit3_collection = self.db_manager.get_collection(settings.BEIT3_COLLECTION)
        if not beit3_collection: return
        candidate_kf_ids = list(candidate_info.keys());
        if not candidate_kf_ids: return
        try:
            res = beit3_collection.query(expr=f'keyframe_id in {candidate_kf_ids}', output_fields=["keyframe_id", "vector"])
            beit3_vector_map = {item['keyframe_id']: item['vector'] for item in res}
            beit3_query_vector = np.array(self.get_beit3_text_embedding(text_query))
            for kf_id, info in candidate_info.items():
                if kf_id in beit3_vector_map:
                    dist = np.linalg.norm(beit3_query_vector - np.array(beit3_vector_map[kf_id]))
                    beit3_score = 1.0 / (1.0 + dist)
                    info['score'] = (0.4 * info.get('clip_score', 0)) + (0.6 * beit3_score)
                    info['beit3_score'] = beit3_score; info['reasons'].append(f"BEIT-3 refine ({beit3_score:.3f})")
                else: info['score'] *= 0.8; info['reasons'].append("BEIT-3 vector missing")
        except Exception as e: logger.error(f"BEIT-3 reranking failed: {e}")

    def _safe_get_label(self, entity: Dict[str, Any]) -> Optional[str]:
        if not entity: return None
        if isinstance(entity, dict) and 'label' in entity and entity['label']: return entity['label']
        inner = entity.get('entity') if isinstance(entity, dict) else None
        if isinstance(inner, dict) and 'label' in inner and inner['label']: return inner['label']
        return None

    def _parse_color_bbox_from_label(self, label: str) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[int, int, int, int]]]:
        if not label: return None, None
        parts = [p.strip() for p in label.split(",") if p.strip() != ""]
        try: nums = list(map(float, parts))
        except ValueError: return None, None
        if len(nums) == 3: return (nums[0], nums[1], nums[2]), None
        elif len(nums) == 4: return None, (int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]))
        elif len(nums) >= 7: return (nums[0], nums[1], nums[2]), (int(nums[3]), int(nums[4]), int(nums[5]), int(nums[6]))
        else: return None, None

    def _normalize_object_filters(self, object_filters: Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]]) -> Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]]:
        norm = {}
        if not isinstance(object_filters, dict): return norm
        for obj, items in object_filters.items():
            if not isinstance(items, list): continue
            fixed = []
            for it in items:
                if (not isinstance(it, (list, tuple))) or len(it) != 2: continue
                lab, bbox = it[0], it[1]
                if not (isinstance(lab, (list, tuple)) and len(lab) == 3): continue
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4): continue
                try:
                    lab_t = (float(lab[0]), float(lab[1]), float(lab[2]))
                    bbox_t = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    fixed.append((lab_t, bbox_t))
                except (ValueError, TypeError): continue
            if fixed: norm[obj] = fixed
        return norm

    def _apply_object_color_filters(self, candidate_info: Dict[str, Dict[str, Any]], object_filters: Optional[Dict], color_filters: Optional[List], top_k: int):
        if object_filters:
            norm_object_filters = self._normalize_object_filters(object_filters)
            for obj, queries in norm_object_filters.items():
                obj_vector = self.get_clip_text_embedding(obj).tolist()
                obj_hits = self._search_milvus(settings.OBJECT_COLLECTION, obj_vector, top_k * 10)
                for hit in obj_hits:
                    kf_id_parts = hit['id'].split('_'); kf_id = '_'.join(kf_id_parts[:-2]) if len(kf_id_parts) > 2 else hit['id']
                    if kf_id not in candidate_info: continue
                    label = self._safe_get_label(hit.get('entity', {})); stored_color, stored_bbox = self._parse_color_bbox_from_label(label) if label else (None, None)
                    for query_color, query_bbox in queries:
                        delta_e, iou, color_term, bbox_term = None, None, 0.0, 0.0
                        if query_color and stored_color:
                            delta_e = self._compare_color(tuple(query_color), tuple(stored_color))
                            color_term = 0.1 * (1.0 / (1.0 + float(delta_e)))
                        if query_bbox and stored_bbox:
                            iou = self._compare_bbox(tuple(query_bbox), tuple(stored_bbox))
                            bbox_term = 0.2 * float(iou)
                        score_boost = color_term + bbox_term
                        if score_boost > 0:
                            candidate_info[kf_id]['score'] += score_boost
                            candidate_info[kf_id]['reasons'].append(f"Object match: '{obj}'" + (f", ΔE={delta_e:.3f}" if delta_e is not None else "") + (f", IoU={iou:.3f}" if iou is not None else ""))
        if color_filters:
            for query_color in color_filters:
                if not (isinstance(query_color, (list, tuple)) and len(query_color) == 3): continue
                
                # === SỬA LỖI Ở ĐÂY ===
                # Chuyển đổi màu LAB sang RGB số nguyên để dùng cho _get_closest_color_name
                try:
                    lab_obj = LabColor(lab_l=query_color[0], lab_a=query_color[1], lab_b=query_color[2])
                    rgb_obj = convert_color(lab_obj, sRGBColor)
                    # Chuyển đổi sang tuple số nguyên (0-255)
                    rgb_tuple_int = tuple(int(c * 255) for c in rgb_obj.get_value_tuple())
                    closest_color_name = self.object_detector._get_closest_color_name(rgb_tuple_int)
                    color_vector = self.get_clip_text_embedding(f"the color {closest_color_name}").tolist()
                    color_hits = self._search_milvus(settings.COLOR_COLLECTION, color_vector, top_k * 10)
                    for hit in color_hits:
                        kf_id_parts = hit['id'].split('_'); kf_id = '_'.join(kf_id_parts[:-2]) if len(kf_id_parts) > 2 else hit['id']
                        if kf_id in candidate_info:
                            candidate_info[kf_id]['score'] += 0.05
                            candidate_info[kf_id]['reasons'].append(f"Color match for {query_color}")
                except Exception as e:
                    logger.warning(f"Could not process color filter for {query_color}: {e}")
                    continue

    def _apply_text_filters(self, candidate_info: Dict, ocr_kf_ids: Optional[Set[str]], asr_video_ids: Optional[Set[str]]) -> Dict:
        if not ocr_kf_ids and not asr_video_ids: return candidate_info
        final_candidates = {}
        for kf_id, info in candidate_info.items():
            is_match = False
            if ocr_kf_ids is not None and kf_id in ocr_kf_ids: info['score'] += 0.5; info['reasons'].append("OCR match"); is_match = True
            if asr_video_ids is not None and info['video_id'] in asr_video_ids: info['score'] += 0.3; info['reasons'].append("ASR match"); is_match = True
            if is_match: final_candidates[kf_id] = info
        return final_candidates

    def _search_es(self, ocr_query: Optional[str], asr_query: Optional[str]) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        ocr_kf_ids, asr_video_ids, es_client = None, None, self.db_manager.es_client
        if not es_client: return ocr_kf_ids, asr_video_ids
        if ocr_query:
            ocr_kf_ids = set()
            try:
                res = es_client.search(index=settings.OCR_INDEX, body={"query": {"match": {"text": ocr_query}}, "_source": ["keyframe_id"]}, size=10000)
                for hit in res['hits']['hits']:
                    if '_source' in hit and 'keyframe_id' in hit['_source']: ocr_kf_ids.add(hit['_source']['keyframe_id'])
            except Exception as e: logger.error(f"OCR search failed: {e}")
        if asr_query:
            asr_video_ids = set()
            try:
                res = es_client.search(index=settings.ASR_INDEX, body={"query": {"match": {"text": asr_query}}, "_source": ["video_id"]}, size=10000)
                for hit in res['hits']['hits']:
                    if '_source' in hit and 'video_id' in hit['_source']: asr_video_ids.add(hit['_source']['video_id'])
            except Exception as e: logger.error(f"ASR search failed: {e}")
        return ocr_kf_ids, asr_video_ids

    def _format_results(self, sorted_candidates: List[Tuple[str, Dict]]) -> List[Dict]:
        return [{
            "keyframe_id": kf_id, "video_id": info.get('video_id', ''), "timestamp": info.get('timestamp', 0.0),
            "score": round(info.get('score', 0.0), 4), "reasons": info.get('reasons', []),
            "metadata": {"rank": rank + 1, "clip_score": round(info.get('clip_score', 0.0), 4), "beit3_score": round(info.get('beit3_score', 0.0), 4)}
        } for rank, (kf_id, info) in enumerate(sorted_candidates)]
        
    def detect_objects_in_image(self, image_path: str) -> Tuple[Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]], List[Tuple[float, float, float]]]:
        if not self.object_detector: raise RuntimeError("Object detector is not initialized.")
        dominant_colors, object_colors = self.object_detector.detect(image_path)
        return object_colors, dominant_colors
    
    def get_all_video_ids(self) -> List[str]:
        if not self.initialized: raise RuntimeError("Retriever is not initialized.")
        video_ids = set(); es_client = self.db_manager.es_client
        try:
            if es_client and es_client.indices.exists(index=settings.METADATA_INDEX):
                res = es_client.search(index=settings.METADATA_INDEX, scroll="2m", size=1000, query={"match_all": {}}, _source=["video_id"])
                scroll_id = res['_scroll_id']; hits = res['hits']['hits']
                while hits:
                    for hit in hits:
                        if '_source' in hit and 'video_id' in hit['_source']: video_ids.add(hit['_source']['video_id'])
                    res = es_client.scroll(scroll_id=scroll_id, scroll='2m')
                    scroll_id = res['_scroll_id']; hits = res['hits']['hits']
                es_client.clear_scroll(scroll_id=scroll_id)
            if not video_ids:
                logger.warning("No video IDs in Elasticsearch. Falling back to Milvus.")
                clip_collection = self.db_manager.get_collection(settings.CLIP_COLLECTION)
                if clip_collection:
                    expr = "video_id != ''"
                    res_iterator = clip_collection.query_iterator(expr=expr, output_fields=["video_id"], batch_size=1000)
                    while True:
                        res_batch = res_iterator.next()
                        if not res_batch: break
                        for item in res_batch:
                             if 'video_id' in item: video_ids.add(item['video_id'])
                    res_iterator.close()
            logger.info(f"Found {len(video_ids)} unique video IDs.")
            return sorted(list(video_ids))
        except Exception as e:
            logger.error(f"Failed to get all video IDs: {e}", exc_info=True)
            return []
    
    def distribute_videos_to_workers(self, user_list: List[str]) -> Dict[str, Any]:
        logger.info(f"Starting video distribution for {len(user_list)} users: {user_list}")
        unique_users = sorted(list(set(user_list)))
        if len(unique_users) < 2:
            logger.error(f"Distribution requires at least 2 unique users, but got {len(unique_users)}.")
            raise ValueError("Cần ít nhất 2 người dùng duy nhất để tạo cặp.")
        all_videos = self.get_all_video_ids()
        if not all_videos:
            logger.warning("No videos found to distribute.")
            return {"summary": "Không tìm thấy video nào trong cơ sở dữ liệu để phân phối.", "user_counts": {user: 0 for user in unique_users}, "assignments": {}}
        user_counts = {user: 0 for user in unique_users}; assignments = {}; pairs = list(itertools.combinations(unique_users, 2))
        random.shuffle(all_videos)
        logger.info(f"Found {len(all_videos)} videos to distribute to {len(pairs)} pairs.")
        for video_id in all_videos:
            best_pair = min(pairs, key=lambda p: user_counts[p[0]] + user_counts[p[1]])
            assignments[video_id] = best_pair
            user_counts[best_pair[0]] += 1; user_counts[best_pair[1]] += 1
        total_assignments = len(assignments)
        min_count = min(user_counts.values()) if user_counts else 0
        max_count = max(user_counts.values()) if user_counts else 0
        summary = {"total_videos_distributed": total_assignments, "total_unique_users": len(unique_users), "total_pairs_created": len(pairs), "videos_per_user_min": min_count, "videos_per_user_max": max_count, "balance_status": "Excellent" if max_count - min_count <= 1 else "Good", "distribution_method": "Greedy assignment to the least busy pair"}
        logger.info(f"Distribution complete. Final counts: {user_counts}")
        return {"summary": summary, "user_counts": user_counts, "assignments": assignments}

# --- END OF FILE app/retrieval_engine.py ---