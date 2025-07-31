import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import Collection
from opensearchpy import OpenSearch

from .config import settings
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retrieval engine kết hợp CLIP và BEIT-3 models
    với khả năng tìm kiếm đa phương thức
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.initialized = False
        
    async def initialize(self):
        """Khởi tạo retriever và kết nối databases"""
        try:
            logger.info("Initializing Hybrid Retriever...")
            
            # Kết nối databases
            milvus_ok = await self.db_manager.connect_milvus()
            es_ok = await self.db_manager.connect_elasticsearch()
            
            if not milvus_ok:
                logger.error("Failed to connect to Milvus")
                return False
                
            if not es_ok:
                logger.warning("Failed to connect to Elasticsearch - OCR/ASR features disabled")
            
            self.initialized = True
            logger.info("✅ Hybrid Retriever initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hybrid Retriever: {e}")
            return False
    
    def check_milvus_connection(self) -> Dict[str, Any]:
        """Kiểm tra kết nối Milvus"""
        try:
            if not self.initialized:
                return {"status": "not_initialized", "connected": False}
            
            collections_info = {}
            for name, collection in self.db_manager.collections.items():
                collections_info[name] = {
                    "num_entities": collection.num_entities,
                    "loaded": collection.is_empty is False
                }
            
            return {
                "status": "connected",
                "connected": True,
                "collections": collections_info
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e)
            }
    
    def check_elasticsearch_connection(self) -> Dict[str, Any]:
        """Kiểm tra kết nối Elasticsearch"""
        try:
            if not self.initialized or not self.db_manager.elasticsearch_connected:
                return {"status": "not_connected", "connected": False}
            
            # Kiểm tra các indices
            indices_info = {}
            for index_name in [settings.METADATA_INDEX, settings.OCR_INDEX, settings.ASR_INDEX]:
                try:
                    stats = self.db_manager.es_client.indices.stats(index=index_name)
                    indices_info[index_name] = {
                        "exists": True,
                        "doc_count": stats['indices'][index_name]['total']['docs']['count']
                    }
                except:
                    indices_info[index_name] = {"exists": False, "doc_count": 0}
            
            return {
                "status": "connected",
                "connected": True,
                "indices": indices_info
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e)
            }
    
    def get_collections_info(self) -> Dict[str, Any]:
        """Lấy thông tin về các collections"""
        try:
            collections_info = {}
            for name, collection in self.db_manager.collections.items():
                collections_info[name] = {
                    "num_entities": collection.num_entities,
                    "schema": str(collection.schema),
                    "loaded": collection.is_empty is False
                }
            return {"collections": collections_info}
        except Exception as e:
            logger.error(f"Failed to get collections info: {e}")
            return {"collections": {}, "error": str(e)}
    
    def search(
        self,
        text_query: str,
        mode: str = "hybrid",
        object_filters: Optional[List[str]] = None,
        color_filters: Optional[List[str]] = None,
        ocr_query: Optional[str] = None,
        asr_query: Optional[str] = None,
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm video với hybrid approach
        
        Args:
            text_query: Query text chính
            mode: Chế độ tìm kiếm (hybrid, clip, beit3)
            object_filters: Lọc theo objects
            color_filters: Lọc theo màu sắc
            ocr_query: Query OCR text
            asr_query: Query ASR transcript
            top_k: Số lượng kết quả trả về
            
        Returns:
            List các kết quả tìm kiếm
        """
        start_time = time.time()
        
        try:
            if not self.initialized:
                raise Exception("Retriever not initialized")
            
            logger.info(f"Starting search: {text_query} (mode: {mode})")
            
            # Validate mode
            if mode not in ["hybrid", "clip", "beit3"]:
                raise ValueError(f"Invalid mode: {mode}")
            
            # GIAI ĐOẠN 1: Lấy candidates ban đầu
            candidates = self._get_initial_candidates(text_query, mode, top_k * 2)
            
            # GIAI ĐOẠN 2: Áp dụng filters
            if object_filters or color_filters:
                candidates = self._apply_object_color_filters(candidates, object_filters, color_filters)
            
            # GIAI ĐOẠN 3: Áp dụng OCR/ASR filters
            if ocr_query or asr_query:
                candidates = self._apply_text_filters(candidates, ocr_query, asr_query)
            
            # GIAI ĐOẠN 4: Reranking với hybrid approach
            if mode == "hybrid" and len(candidates) > 0:
                candidates = self._hybrid_reranking(candidates, text_query)
            
            # GIAI ĐOẠN 5: Format kết quả
            results = self._format_results(candidates[:top_k])
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise e
    
    def _get_initial_candidates(self, text_query: str, mode: str, top_k: int) -> List[Dict[str, Any]]:
        """Lấy candidates ban đầu từ vector search"""
        try:
            if mode == "clip":
                collection_name = settings.CLIP_COLLECTION
            elif mode == "beit3":
                collection_name = settings.BEIT3_COLLECTION
            else:  # hybrid - sử dụng CLIP cho initial search
                collection_name = settings.CLIP_COLLECTION
            
            collection = self.db_manager.get_collection(collection_name)
            if not collection:
                logger.warning(f"Collection {collection_name} not found")
                return []
            
            # TODO: Implement text embedding và vector search
            # Đây là placeholder - cần implement thực tế
            logger.info(f"Getting initial candidates from {collection_name}")
            
            # Mock results cho demo
            candidates = []
            for i in range(min(top_k, 50)):  # Giới hạn 50 kết quả mock
                candidates.append({
                    "keyframe_id": f"frame_{i:06d}",
                    "video_id": f"video_{i//10:03d}",
                    "timestamp": float(i * 2.5),
                    "score": 1.0 - (i * 0.01),
                    "collection": collection_name,
                    "reasons": [f"Matched query: {text_query}"]
                })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to get initial candidates: {e}")
            return []
    
    def _apply_object_color_filters(self, candidates: List[Dict], object_filters: Optional[List[str]], color_filters: Optional[List[str]]) -> List[Dict]:
        """Áp dụng object và color filters"""
        if not object_filters and not color_filters:
            return candidates
        
        try:
            logger.info(f"Applying filters: objects={object_filters}, colors={color_filters}")
            
            # TODO: Implement actual filtering logic
            # Đây là placeholder - cần implement thực tế với object/color collections
            
            filtered_candidates = []
            for candidate in candidates:
                # Mock filtering logic
                should_include = True
                reasons = candidate.get("reasons", [])
                
                if object_filters:
                    # Mock object matching
                    if "person" in object_filters and "person" in candidate.get("keyframe_id", ""):
                        reasons.append("Object filter: person")
                    else:
                        should_include = False
                
                if color_filters and should_include:
                    # Mock color matching
                    if "red" in color_filters and "red" in candidate.get("keyframe_id", ""):
                        reasons.append("Color filter: red")
                    else:
                        should_include = False
                
                if should_include:
                    candidate["reasons"] = reasons
                    filtered_candidates.append(candidate)
            
            logger.info(f"Filtered from {len(candidates)} to {len(filtered_candidates)} candidates")
            return filtered_candidates
            
        except Exception as e:
            logger.error(f"Failed to apply object/color filters: {e}")
            return candidates
    
    def _apply_text_filters(self, candidates: List[Dict], ocr_query: Optional[str], asr_query: Optional[str]) -> List[Dict]:
        """Áp dụng OCR và ASR text filters"""
        if not ocr_query and not asr_query:
            return candidates
        
        try:
            logger.info(f"Applying text filters: OCR={ocr_query}, ASR={asr_query}")
            
            if not self.db_manager.elasticsearch_connected:
                logger.warning("Elasticsearch not connected - skipping text filters")
                return candidates
            
            # TODO: Implement actual text search với Elasticsearch
            # Đây là placeholder - cần implement thực tế
            
            filtered_candidates = []
            for candidate in candidates:
                should_include = True
                reasons = candidate.get("reasons", [])
                
                # Mock text matching
                if ocr_query and ocr_query.lower() in candidate.get("keyframe_id", "").lower():
                    reasons.append(f"OCR match: {ocr_query}")
                
                if asr_query and asr_query.lower() in candidate.get("video_id", "").lower():
                    reasons.append(f"ASR match: {asr_query}")
                
                if should_include:
                    candidate["reasons"] = reasons
                    filtered_candidates.append(candidate)
            
            logger.info(f"Text filtered from {len(candidates)} to {len(filtered_candidates)} candidates")
            return filtered_candidates
            
        except Exception as e:
            logger.error(f"Failed to apply text filters: {e}")
            return candidates
    
    def _hybrid_reranking(self, candidates: List[Dict], text_query: str) -> List[Dict]:
        """Reranking với hybrid approach (CLIP + BEIT-3)"""
        try:
            logger.info("Applying hybrid reranking...")
            
            # TODO: Implement actual hybrid reranking
            # Đây là placeholder - cần implement thực tế
            
            # Mock reranking - sắp xếp theo score
            reranked_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
            
            # Thêm hybrid reason
            for candidate in reranked_candidates:
                candidate["reasons"].append("Hybrid reranking applied")
            
            logger.info("Hybrid reranking completed")
            return reranked_candidates
            
        except Exception as e:
            logger.error(f"Failed to apply hybrid reranking: {e}")
            return candidates
    
    def _format_results(self, candidates: List[Dict]) -> List[Dict[str, Any]]:
        """Format kết quả cuối cùng"""
        try:
            formatted_results = []
            
            for i, candidate in enumerate(candidates):
                formatted_result = {
                    "keyframe_id": candidate.get("keyframe_id", ""),
                    "video_id": candidate.get("video_id", ""),
                    "timestamp": candidate.get("timestamp", 0.0),
                    "score": candidate.get("score", 0.0),
                    "reasons": candidate.get("reasons", []),
                    "metadata": {
                        "rank": i + 1,
                        "collection": candidate.get("collection", ""),
                        "confidence": candidate.get("score", 0.0)
                    }
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to format results: {e}")
            return [] 