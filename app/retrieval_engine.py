# video-retrieval-backend/app/retrieval_engine.py

import logging
import time
import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import torch
from PIL import Image

# Thêm đường dẫn tới các repo phụ thuộc
sys.path.append('/app/Co_DETR')
sys.path.append('/app/unilm/beit3')

# Import từ các thư viện ML
import open_clip
import sentencepiece as spm
from torchvision import transforms

# Import từ các repo
from modeling_finetune import BEiT3ForRetrieval

# Import từ các module của ứng dụng
from .config import settings
from .database import db_manager

logger = logging.getLogger(__name__)

# --- Cấu hình cho BEiT-3 (giữ nguyên từ script gốc) ---
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

class HybridRetriever:
    """
    Hybrid retrieval engine kết hợp các model AI và tìm kiếm đa phương thức.
    """
    
    def __init__(self):
        self.db_manager = db_manager
        self.initialized = False
        self.device = settings.DEVICE
        
        # Biến giữ các model
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        
        self.beit3_model = None
        self.beit3_preprocess = None
        self.beit3_sp_model = None
        
    async def initialize(self):
        """Khởi tạo retriever: kết nối DB và tải model."""
        if self.initialized:
            return
            
        logger.info("Initializing Hybrid Retriever...")
        
        # Kết nối databases (lấy từ db_manager đã được init trong lifespan)
        if not self.db_manager.milvus_connected or not self.db_manager.elasticsearch_connected:
            raise RuntimeError("Database connections were not established. Check lifespan startup.")

        # Tải các model AI
        self._load_models()
        
        self.initialized = True
        logger.info("✅ Hybrid Retriever initialized successfully")

    def _load_models(self):
        """Tải các mô hình AI cần thiết."""
        logger.info("Loading AI models...")
        # Sử dụng device được tính toán động thay vì từ settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # --- 1. Tải mô hình CLIP ---
        logger.info("Loading CLIP model (ViT-H-14)...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-H-14', pretrained=settings.CLIP_MODEL_PATH, device=self.device
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        logger.info("✅ CLIP model loaded.")

        # --- 2. Tải mô hình BEiT-3 ---
        logger.info("Loading BEiT-3 model...")
        
        beit3_config = BEiT3Config()
        self.beit3_model = BEiT3ForRetrieval(beit3_config)
        checkpoint = torch.load(settings.BEIT3_MODEL_PATH, map_location="cpu")
        self.beit3_model.load_state_dict(checkpoint["model"])
        self.beit3_model = self.beit3_model.to(self.device).eval()
        
        self.beit3_sp_model = spm.SentencePieceProcessor()
        self.beit3_sp_model.load(settings.BEIT3_SPM_PATH)
        
        self.beit3_preprocess = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        logger.info("✅ BEiT-3 model loaded.")
        
    # --- CÁC HÀM MÃ HÓA (EMBEDDING) ---
    def get_clip_text_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.clip_tokenizer([text]).to(self.device)
            text_emb = self.clip_model.encode_text(tokens).cpu().numpy()[0]
            text_emb /= np.linalg.norm(text_emb, axis=0)
        return text_emb

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
            
    # --- CÁC HÀM TÌM KIẾM ---
    def search(
        self,
        text_query: str,
        mode: str,
        object_filters: Optional[List[str]],
        color_filters: Optional[List[str]],
        ocr_query: Optional[str],
        asr_query: Optional[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        if not self.initialized:
            raise RuntimeError("Retriever is not initialized.")
            
        start_time = time.time()
        logger.info(f"--- [STARTING SEARCH] Query: '{text_query}', Mode: {mode.upper()} ---")
        
        candidate_info: Dict[str, Dict[str, Any]] = {}
        num_initial_candidates = top_k * 5 # Lấy số lượng ứng viên ban đầu lớn hơn

        # GIAI ĐOẠN 1: LẤY ỨNG VIÊN BAN ĐẦU (TỪ CLIP HOẶC BEIT-3)
        if mode in ['hybrid', 'clip']:
            logger.info("1. Broad Semantic Filtering with CLIP...")
            clip_vector = self.get_clip_text_embedding(text_query).tolist()
            clip_candidates = self._search_milvus(settings.CLIP_COLLECTION, clip_vector, num_initial_candidates)
            for hit in clip_candidates:
                kf_id = hit['entity']['keyframe_id']
                score = 1.0 / (1.0 + hit['distance'])
                candidate_info[kf_id] = {'video_id': hit['entity']['video_id'], 'timestamp': hit['entity']['timestamp'], 'clip_score': score, 'score': score, 'reasons': [f"CLIP match (score: {score:.3f})"]}
        elif mode == 'beit3':
            logger.info("1. Searching with BEIT-3 only...")
            beit3_vector = self.get_beit3_text_embedding(text_query).tolist()
            beit3_candidates = self._search_milvus(settings.BEIT3_COLLECTION, beit3_vector, num_initial_candidates)
            for hit in beit3_candidates:
                kf_id = hit['entity']['keyframe_id']
                score = 1.0 / (1.0 + hit['distance'])
                candidate_info[kf_id] = {'video_id': hit['entity']['video_id'], 'timestamp': hit['entity']['timestamp'], 'beit3_score': score, 'score': score, 'reasons': [f"BEIT-3 match (score: {score:.3f})"]}

        # GIAI ĐOẠN 2: TINH CHỈNH BẰNG BEIT-3 (CHỈ CHO MODE HYBRID)
        if mode == 'hybrid' and candidate_info:
            logger.info("2. Refining results with BEIT-3...")
            self._hybrid_reranking(candidate_info, text_query)

        # GIAI ĐOẠN 3: TĂNG ĐIỂM (BOOSTING) VỚI OBJECT/COLOR
        if object_filters or color_filters:
            logger.info("3. Boosting scores with object/color filters...")
            self._apply_object_color_filters(candidate_info, object_filters, color_filters, top_k)

        # GIAI ĐOẠN 4: LỌC CỨNG (HARD FILTERING) VỚI OCR/ASR
        if ocr_query or asr_query:
            logger.info("4. Hard filtering with OpenSearch (OCR/ASR)...")
            ocr_kf_ids, asr_video_ids = self._search_es(ocr_query, asr_query)
            candidate_info = self._apply_text_filters(candidate_info, ocr_kf_ids, asr_video_ids)
        
        # GIAI ĐOẠN 5: XẾP HẠNG VÀ TRẢ VỀ
        logger.info("5. Sorting and returning final results.")
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        final_results = self._format_results(sorted_results[:top_k])
        
        search_time = time.time() - start_time
        logger.info(f"--- [SEARCH FINISHED] Found {len(final_results)} results in {search_time:.2f}s. ---")
        return final_results

    # --- CÁC HÀM HELPER CHO LOGIC TÌM KIẾM ---
    def _search_milvus(self, collection_name: str, vector: List[float], top_k: int) -> List[Dict]:
        """Thực hiện tìm kiếm trên một collection Milvus."""
        collection = self.db_manager.get_collection(collection_name)
        if not collection:
            logger.warning(f"Collection '{collection_name}' not found in DB manager.")
            return []
            
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        output_fields = ["keyframe_id", "video_id", "timestamp"]
        
        results = collection.search(
            data=[vector], anns_field="vector", param=search_params,
            limit=top_k, output_fields=output_fields
        )
        
        return [{'id': hit.id, 'distance': hit.distance, 'entity': hit.entity.to_dict()} for hit in results[0]]

    def _hybrid_reranking(self, candidate_info: Dict[str, Dict], text_query: str):
        """Xếp hạng lại các ứng viên bằng BEIT-3."""
        beit3_collection = self.db_manager.get_collection(settings.BEIT3_COLLECTION)
        if not beit3_collection:
            logger.warning("BEIT-3 collection not found for reranking. Skipping.")
            return

        candidate_kf_ids = list(candidate_info.keys())
        # Tránh lỗi query rỗng
        if not candidate_kf_ids:
            return
            
        expr = f'keyframe_id in {candidate_kf_ids}'
        
        try:
            beit3_vectors_of_candidates = beit3_collection.query(expr=expr, output_fields=["keyframe_id", "vector"])
            beit3_vector_map = {item['keyframe_id']: item['vector'] for item in beit3_vectors_of_candidates}
            
            beit3_query_vector = np.array(self.get_beit3_text_embedding(text_query))
            
            for kf_id, info in candidate_info.items():
                if kf_id in beit3_vector_map:
                    dist = np.linalg.norm(beit3_query_vector - np.array(beit3_vector_map[kf_id]))
                    beit3_score = 1.0 / (1.0 + dist)
                    combined_score = (0.4 * info.get('clip_score', 0)) + (0.6 * beit3_score)
                    info['score'] = combined_score
                    info['beit3_score'] = beit3_score
                    info['reasons'].append(f"BEIT-3 refine (score: {beit3_score:.3f})")
                else:
                    info['score'] *= 0.8 # Hạ điểm nếu không có vector BEIT-3
                    info['reasons'].append("BEIT-3 vector not found for refinement")
        except Exception as e:
            logger.error(f"Error during BEIT-3 reranking: {e}. Skipping.")
    
    def _apply_object_color_filters(self, candidate_info: Dict, object_filters: Optional[List], color_filters: Optional[List], top_k: int):
        """Tăng điểm cho các ứng viên khớp với bộ lọc object/color."""
        if object_filters:
            for obj in object_filters:
                obj_vector = self.get_clip_text_embedding(obj).tolist()
                obj_hits = self._search_milvus(settings.OBJECT_COLLECTION, obj_vector, top_k * 10)
                for hit in obj_hits:
                    # PK của object/color là: {keyframe_id}_{object/color_name}_{index}
                    kf_id = '_'.join(hit['id'].split('_')[:-2])
                    if kf_id in candidate_info:
                        candidate_info[kf_id]['score'] += 0.1
                        candidate_info[kf_id]['reasons'].append(f"Object match: '{obj}'")
        
        if color_filters:
            for color in color_filters:
                color_vector = self.get_clip_text_embedding(color).tolist()
                color_hits = self._search_milvus(settings.COLOR_COLLECTION, color_vector, top_k * 10)
                for hit in color_hits:
                    kf_id = '_'.join(hit['id'].split('_')[:-2])
                    if kf_id in candidate_info:
                        candidate_info[kf_id]['score'] += 0.05
                        candidate_info[kf_id]['reasons'].append(f"Color match: '{color}'")

    def _apply_text_filters(self, candidate_info: Dict, ocr_kf_ids: Optional[Set[str]], asr_video_ids: Optional[Set[str]]) -> Dict:
        """
        Lọc và tăng điểm dựa trên kết quả từ Elasticsearch.
        Hàm này chỉ giữ lại những ứng viên khớp với bộ lọc đang được kích hoạt.
        """
        # Nếu không có bộ lọc nào được áp dụng (cả hai set ID đều rỗng hoặc None), trả về y nguyên
        if not ocr_kf_ids and not asr_video_ids:
            return candidate_info

        final_candidates = {}
        for kf_id, info in candidate_info.items():
            is_match = False
            
            # Kiểm tra khớp OCR (chỉ khi bộ lọc OCR được kích hoạt)
            if ocr_kf_ids is not None and kf_id in ocr_kf_ids:
                info['score'] += 0.5
                info['reasons'].append("OCR text match")
                is_match = True
            
            # Kiểm tra khớp ASR (chỉ khi bộ lọc ASR được kích hoạt)
            if asr_video_ids is not None and info['video_id'] in asr_video_ids:
                info['score'] += 0.3
                info['reasons'].append("ASR transcript match")
                is_match = True

            # Chỉ giữ lại ứng viên nếu nó khớp với ít nhất một bộ lọc được kích hoạt
            if is_match:
                final_candidates[kf_id] = info
        
        logger.info(f"   - Remaining candidates after ES filtering: {len(final_candidates)}")
        return final_candidates

    def _search_es(self, ocr_query: Optional[str], asr_query: Optional[str]) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """Thực hiện tìm kiếm trên Elasticsearch để lấy ID cho việc lọc."""
        ocr_kf_ids, asr_video_ids = None, None
        es_client = self.db_manager.es_client
        if not es_client: 
            return ocr_kf_ids, asr_video_ids

        if ocr_query:
            ocr_kf_ids = set()
            query_body = {"query": {"match": {"text": ocr_query}}, "_source": ["keyframe_id"]}
            try:
                res = es_client.search(index=settings.OCR_INDEX, body=query_body, size=10000)
                for hit in res['hits']['hits']: 
                    ocr_kf_ids.add(hit['_source']['keyframe_id'])
            except Exception as e:
                logger.warning(f"Failed to search OCR index: {e}")

        if asr_query:
            asr_video_ids = set()
            query_body = {"query": {"match": {"text": asr_query}}, "_source": ["video_id"]}
            try:
                res = es_client.search(index=settings.ASR_INDEX, body=query_body, size=10000)
                for hit in res['hits']['hits']: 
                    asr_video_ids.add(hit['_source']['video_id'])
            except Exception as e:
                logger.warning(f"Failed to search ASR index: {e}")
        
        return ocr_kf_ids, asr_video_ids

    def _format_results(self, sorted_candidates: List[Tuple[str, Dict]]) -> List[Dict]:
        """Định dạng kết quả cuối cùng theo model SearchResult."""
        final_results = []
        for rank, (kf_id, info) in enumerate(sorted_candidates):
            result = {
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
            }
            final_results.append(result)
        return final_results
        
    # --- CÁC PHƯƠNG THỨC TIỆN ÍCH CHO API (PROXY METHODS) ---
    def check_milvus_connection(self) -> Dict[str, Any]:
        """Proxy method để kiểm tra kết nối Milvus từ db_manager."""
        return self.db_manager.check_milvus_connection()

    def check_elasticsearch_connection(self) -> Dict[str, Any]:
        """Proxy method để kiểm tra kết nối Elasticsearch từ db_manager."""
        return self.db_manager.check_elasticsearch_connection()
    
    def get_collections_info(self) -> Dict[str, Any]:
        """Lấy thông tin các collection từ Milvus."""
        milvus_status = self.db_manager.check_milvus_connection()
        if milvus_status.get("status") == "connected":
            return milvus_status.get("collections", {})
        return {}