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
        self.basic_colors = {'red':(255,0,0), 'green':(0,255,0), 'blue':(0,0,255), 'yellow':(255,255,0), 'cyan':(0,255,255), 'magenta':(255,0,255), 'black':(0,0,0), 'white':(255,255,255), 'gray':(128,128,128), 'orange':(255,165,0), 'brown':(165,42,42), 'pink':(255,192,203), 'purple': (128,0,128)}
        self.color_names = list(self.basic_colors.keys())
        self.color_tree = KDTree(np.array(list(self.basic_colors.values())))
        logger.info("✅ Co-DETR model loaded.")

    def detect(self, image_path: str) -> Tuple[List[str], List[str]]:
        try:
            result = inference_detector(self.model, image_path)
            if isinstance(result, tuple): result = result[0]
            
            frame_cv = cv2.imread(image_path)
            if frame_cv is None: return [], []

            detected_objects, detected_colors = set(), set()
            for class_id, bboxes in enumerate(result):
                if class_id >= len(self.model.CLASSES): continue
                for bbox in bboxes:
                    if bbox[4] < 0.5: continue  # Lọc bỏ các phát hiện có độ tin cậy thấp
                    detected_objects.add(self.model.CLASSES[class_id])
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    crop = frame_cv[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    # Tìm màu chủ đạo
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3)
                    kmeans = KMeans(n_clusters=1, n_init='auto', random_state=0).fit(crop_rgb)
                    _, idx = self.color_tree.query([kmeans.cluster_centers_[0]], k=1)
                    detected_colors.add(self.color_names[idx[0][0]])
            return list(detected_objects), list(detected_colors)
        except Exception as e:
            logger.error(f"Object/color detection failed for {image_path}: {e}", exc_info=True)
            return [], []

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

    def _load_models(self):
        """Tải tất cả các mô hình AI cần thiết vào đúng device."""
        logger.info(f"Loading AI models onto device: '{self.device}'")
        # 1. Tải CLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-H-14', pretrained=settings.CLIP_MODEL_PATH, device=self.device)
        self.clip_model.eval(); self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        logger.info("  - CLIP model loaded.")
        # 2. Tải BEiT-3
        self.beit3_model = BEiT3ForRetrieval(BEiT3Config())
        checkpoint = torch.load(settings.BEIT3_MODEL_PATH, map_location="cpu")
        self.beit3_model.load_state_dict(checkpoint["model"])
        self.beit3_model = self.beit3_model.to(self.device).eval()
        self.beit3_sp_model = spm.SentencePieceProcessor(); self.beit3_sp_model.load(settings.BEIT3_SPM_PATH)
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

    def get_beit3_text_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            text_ids = self.beit3_sp_model.encode_as_ids(text)
            text_padding_mask = [0] * len(text_ids)
            text_ids_tensor = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            text_padding_mask_tensor = torch.tensor(text_padding_mask, dtype=torch.long).unsqueeze(0).to(self.device)
            _, text_emb = self.beit3_model(text_description=text_ids_tensor, text_padding_mask=text_padding_mask_tensor, only_infer=True)
            return text_emb.cpu().numpy()[0]
            
    # --- LOGIC TÌM KIẾM CHÍNH ---
    def search(self, text_query: str, mode: str, object_filters: Optional[List[str]], color_filters: Optional[List[str]],
               ocr_query: Optional[str], asr_query: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        if not self.initialized: raise RuntimeError("Retriever is not initialized.")
        start_time = time.time()
        logger.info(f"--- [STARTING SEARCH] Query: '{text_query}', Mode: {mode.upper()} ---")
        
        candidate_info: Dict[str, Dict[str, Any]] = {}
        num_initial_candidates = top_k * 5 # Lấy số lượng ứng viên ban đầu lớn hơn

        # GĐ1: LẤY ỨNG VIÊN BAN ĐẦU
        if mode in ['hybrid', 'clip']:
            clip_vector = self.get_clip_text_embedding(text_query).tolist()
            clip_candidates = self._search_milvus(settings.CLIP_COLLECTION, clip_vector, num_initial_candidates)
            for hit in clip_candidates:
                score = 1.0 / (1.0 + hit['distance'])
                candidate_info[hit['entity']['keyframe_id']] = {'video_id': hit['entity']['video_id'], 'timestamp': hit['entity']['timestamp'], 'clip_score': score, 'score': score, 'reasons': [f"CLIP match ({score:.3f})"]}
        elif mode == 'beit3':
            beit3_vector = self.get_beit3_text_embedding(text_query).tolist()
            beit3_candidates = self._search_milvus(settings.BEIT3_COLLECTION, beit3_vector, num_initial_candidates)
            for hit in beit3_candidates:
                score = 1.0 / (1.0 + hit['distance'])
                candidate_info[hit['entity']['keyframe_id']] = {'video_id': hit['entity']['video_id'], 'timestamp': hit['entity']['timestamp'], 'beit3_score': score, 'score': score, 'reasons': [f"BEIT-3 match ({score:.3f})"]}

        # GĐ2: TINH CHỈNH (chỉ cho mode hybrid)
        if mode == 'hybrid' and candidate_info: self._hybrid_reranking(candidate_info, text_query)

        # GĐ3: TĂNG ĐIỂM với Object/Color
        if object_filters or color_filters: self._apply_object_color_filters(candidate_info, object_filters, color_filters, top_k)

        # GĐ4: LỌC CỨNG với OCR/ASR
        if ocr_query or asr_query:
            ocr_kf_ids, asr_video_ids = self._search_es(ocr_query, asr_query)
            candidate_info = self._apply_text_filters(candidate_info, ocr_kf_ids, asr_video_ids)
        
        # GĐ5: XẾP HẠNG VÀ TRẢ VỀ
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        final_results = self._format_results(sorted_results[:top_k])
        
        search_time = time.time() - start_time
        logger.info(f"--- [SEARCH FINISHED] Found {len(final_results)} results in {search_time:.2f}s ---")
        return final_results

    # --- CÁC HÀM HELPER CHO LOGIC TÌM KIẾM ---
    def _search_milvus(self, collection_name: str, vector: List[float], top_k: int) -> List[Dict]:
        collection = self.db_manager.get_collection(collection_name)
        if not collection: return []

        search_results = collection.search(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["keyframe_id", "video_id", "timestamp"]
        )[0]

        # Convert Hit objects to dictionaries
        hits = []
        for hit in search_results:
            hit_dict = {
                'id': hit.id,
                'distance': hit.distance,
                'entity': {
                    'keyframe_id': hit.entity.get('keyframe_id'),
                    'video_id': hit.entity.get('video_id'),
                    'timestamp': hit.entity.get('timestamp')
                }
            }
            hits.append(hit_dict)
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
    
    def _apply_object_color_filters(self, candidate_info: Dict, object_filters: Optional[List], color_filters: Optional[List], top_k: int):
        if object_filters:
            for obj in object_filters:
                obj_hits = self._search_milvus(settings.OBJECT_COLLECTION, self.get_clip_text_embedding(obj).tolist(), top_k * 10)
                for hit in obj_hits:
                    kf_id = '_'.join(hit['id'].split('_')[:-2])
                    if kf_id in candidate_info: candidate_info[kf_id]['score'] += 0.1; candidate_info[kf_id]['reasons'].append(f"Object match: '{obj}'")
        if color_filters:
            for color in color_filters:
                color_hits = self._search_milvus(settings.COLOR_COLLECTION, self.get_clip_text_embedding(color).tolist(), top_k * 10)
                for hit in color_hits:
                    kf_id = '_'.join(hit['id'].split('_')[:-2])
                    if kf_id in candidate_info: candidate_info[kf_id]['score'] += 0.05; candidate_info[kf_id]['reasons'].append(f"Color match: '{color}'")

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
        """Tìm kiếm trong Elasticsearch - Fixed version"""
        ocr_kf_ids, asr_video_ids = None, None
        es_client = self.db_manager.es_client
        if not es_client: 
            return ocr_kf_ids, asr_video_ids
        
        # OCR Search
        if ocr_query:
            ocr_kf_ids = set()
            try:
                res = es_client.search(
                    index=settings.OCR_INDEX, 
                    body={"query": {"match": {"text": ocr_query}}, "_source": ["keyframe_id"]}, 
                    size=10000
                )
                
                # Fixed: Only access res['hits']['hits'], not res['hits']['hits']['hits']
                for hit in res['hits']['hits']:
                    try:
                        # Try different ways to get keyframe_id
                        keyframe_id = None
                        
                        # Method 1: Standard _source
                        if '_source' in hit and 'keyframe_id' in hit['_source']:
                            keyframe_id = hit['_source']['keyframe_id']
                        
                        # Method 2: Check if 'entity' exists (your original structure)
                        elif 'entity' in hit:
                            if isinstance(hit['entity'], dict):
                                if 'entity' in hit['entity'] and 'keyframe_id' in hit['entity']['entity']:
                                    keyframe_id = hit['entity']['entity']['keyframe_id']
                                elif 'keyframe_id' in hit['entity']:
                                    keyframe_id = hit['entity']['keyframe_id']
                        
                        # Method 3: Direct access
                        elif 'keyframe_id' in hit:
                            keyframe_id = hit['keyframe_id']
                        
                        if keyframe_id:
                            ocr_kf_ids.add(keyframe_id)
                        else:
                            # Debug: print hit structure if keyframe_id not found
                            print(f"DEBUG - OCR hit structure: {list(hit.keys())}")
                            if '_source' in hit:
                                print(f"DEBUG - _source keys: {list(hit['_source'].keys())}")
                            
                    except Exception as e:
                        print(f"Error processing OCR hit: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"OCR search failed: {e}")
                ocr_kf_ids = set()
        
        # ASR Search  
        if asr_query:
            asr_video_ids = set()
            try:
                res = es_client.search(
                    index=settings.ASR_INDEX, 
                    body={"query": {"match": {"text": asr_query}}, "_source": ["video_id"]}, 
                    size=10000
                )
                
                # Fixed: Only access res['hits']['hits'], not res['hits']['hits']['hits']
                for hit in res['hits']['hits']:
                    try:
                        # Try different ways to get video_id
                        video_id = None
                        
                        # Method 1: Standard _source
                        if '_source' in hit and 'video_id' in hit['_source']:
                            video_id = hit['_source']['video_id']
                        
                        # Method 2: Check if 'entity' exists (your original structure)
                        elif 'entity' in hit:
                            if isinstance(hit['entity'], dict):
                                if 'entity' in hit['entity'] and 'video_id' in hit['entity']['entity']:
                                    video_id = hit['entity']['entity']['video_id']
                                elif 'video_id' in hit['entity']:
                                    video_id = hit['entity']['video_id']
                        
                        # Method 3: Direct access
                        elif 'video_id' in hit:
                            video_id = hit['video_id']
                        
                        if video_id:
                            asr_video_ids.add(video_id)
                        else:
                            # Debug: print hit structure if video_id not found
                            print(f"DEBUG - ASR hit structure: {list(hit.keys())}")
                            if '_source' in hit:
                                print(f"DEBUG - _source keys: {list(hit['_source'].keys())}")
                            
                    except Exception as e:
                        print(f"Error processing ASR hit: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"ASR search failed: {e}")
                asr_video_ids = set()
        
        return ocr_kf_ids, asr_video_ids

    def _format_results(self, sorted_candidates: List[Tuple[str, Dict]]) -> List[Dict]:
        return [{
            "keyframe_id": kf_id, "video_id": info.get('video_id', ''), "timestamp": info.get('timestamp', 0.0),
            "score": round(info.get('score', 0.0), 4), "reasons": info.get('reasons', []),
            "metadata": {"rank": rank + 1, "clip_score": round(info.get('clip_score', 0.0), 4), "beit3_score": round(info.get('beit3_score', 0.0), 4)}
        } for rank, (kf_id, info) in enumerate(sorted_candidates)]
        
    # --- CÁC PHƯƠNG THỨC TIỆN ÍCH ĐƯỢC GỌI TỪ API ---
    def detect_objects_in_image(self, image_path: str) -> Tuple[List[str], List[str]]:
        if not self.object_detector: raise RuntimeError("Object detector is not initialized.")
        return self.object_detector.detect(image_path)
    def check_milvus_connection(self) -> Dict[str, Any]: return self.db_manager.check_milvus_connection()
    def check_elasticsearch_connection(self) -> Dict[str, Any]: return self.db_manager.check_elasticsearch_connection()
# --- END OF FILE app/retrieval_engine.py ---