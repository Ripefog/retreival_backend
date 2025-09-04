# --- START OF FILE app/retrieval_engine.py ---

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional

# Import các component đã refactor
from .ai_models import EmbeddingManager
from .search import MilvusClient, ObjectColorFilter, TextProcessor
from .config import settings
from .database import db_manager

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Main orchestrator for hybrid retrieval - Lightweight coordinator
    
    Đây là class chính điều phối tất cả các component:
    - EmbeddingManager: Quản lý CLIP & BEiT-3 models
    - MilvusClient: Xử lý vector search
    - ObjectColorFilter: Lọc object và color
    - TextProcessor: Xử lý OCR và ASR
    """

    def __init__(self):
        self.db_manager = db_manager
        self.device = settings.DEVICE
        self.initialized = False
        
        # Khởi tạo các component chuyên biệt
        self.embedding_manager = EmbeddingManager(self.device)
        self.milvus_client = MilvusClient(self.db_manager)
        self.object_filter = ObjectColorFilter(self.embedding_manager, self.milvus_client)
        self.text_processor = TextProcessor(self.db_manager)

    async def initialize(self):
        """Khởi tạo retriever: kết nối DB và tải model một cách an toàn."""
        if self.initialized:
            return
            
        logger.info("Initializing Hybrid Retriever engine...")
        
        if not self.db_manager.milvus_connected or not self.db_manager.elasticsearch_connected:
            raise RuntimeError("Database connections must be established before initializing the retriever.")
        
        # Load AI models
        self.embedding_manager.load_models()
        
        # Load Milvus collections
        await self.db_manager._load_milvus_collections()
        
        self.initialized = True
        logger.info("✅ Hybrid Retriever initialized successfully.")

    async def search(self, text_query: str, mode: str, user_query: str, object_filters: Optional[Dict],
                     color_filters: Optional[List], ocr_query: Optional[str], asr_query: Optional[str],
                     top_k: int) -> List[Dict[str, Any]]:
        """
        Main search orchestration - delegates to specialized components
        
        Đây chỉ là orchestrator, logic chính được chia vào các specialized classes:
        - Vector search: MilvusClient
        - Object/Color filtering: ObjectColorFilter  
        - Text processing: TextProcessor
        - Embedding generation: EmbeddingManager
        """
        if not self.initialized:
            raise RuntimeError("Retriever is not initialized.")

        start_time = time.time()
        candidate_info: Dict[str, Dict[str, Any]] = {}
        num_initial_candidates = top_k

        # BƯỚC 1: LẤY ỨNG VIÊN BAN ĐẦU (Vector Search)
        tasks = []

        if mode in ['hybrid', 'clip']:
            clip_vector = self.embedding_manager.get_clip_text_embedding(text_query).tolist()
            tasks.append(self.milvus_client.search_vectors_async(
                settings.CLIP_COLLECTION, clip_vector, num_initial_candidates, None, user_query, 'clip'
            ))

        if mode == 'beit3':
            beit3_vector = self.embedding_manager.get_beit3_text_embedding(text_query).tolist()
            tasks.append(self.milvus_client.search_vectors_async(
                settings.BEIT3_COLLECTION, beit3_vector, num_initial_candidates, None, user_query, 'beit3'
            ))

        # Parallel execution of vector searches
        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process search results
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Search task {i} failed: {result}")
                continue

            search_type = 'clip' if (mode in ['hybrid', 'clip'] and i == 0) else 'beit3'
            self.milvus_client.process_search_results(result, candidate_info, search_type)

        # BƯỚC 2: TINH CHỈNH với Hybrid Reranking
        refinement_tasks = []
        if mode == 'hybrid' and candidate_info:
            refinement_tasks.append(
                self.milvus_client.hybrid_reranking(candidate_info, text_query, self.embedding_manager)
            )

        # BƯỚC 3: TĂNG ĐIỂM với Object/Color Filters
        if object_filters or color_filters:
            refinement_tasks.append(
                self.object_filter.apply_object_color_filters(
                    candidate_info, object_filters, color_filters, top_k
                )
            )

        # BƯỚC 4: LỌC VÀ TĂNG ĐIỂM với OCR
        if ocr_query:
            refinement_tasks.append(
                self.text_processor.apply_ocr_filter(candidate_info, ocr_query)
            )

        # BƯỚC 5: LỌC VÀ TĂNG ĐIỂM với ASR
        if asr_query:
            refinement_tasks.append(
                self.text_processor.apply_asr_filter(candidate_info, asr_query)
            )

        # Execute all refinement steps in parallel
        if refinement_tasks:
            await asyncio.gather(*refinement_tasks, return_exceptions=True)

        # BƯỚC 6: XẾP HẠNG VÀ TRẢ VỀ
        sorted_results = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        final_results = self._format_results(sorted_results[:top_k])

        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.2f}s with {len(final_results)} results")
        return final_results

    def _format_results(self, sorted_candidates: List[tuple]) -> List[Dict]:
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

    # --- METHODS ĐỂ MAINTAIN BACKWARD COMPATIBILITY ---
    def check_milvus_connection(self) -> Dict[str, Any]:
        return self.db_manager.check_milvus_connection()

    def check_elasticsearch_connection(self) -> Dict[str, Any]:
        return self.db_manager.check_elasticsearch_connection()

# --- END OF FILE app/retrieval_engine.py ---
