# --- START OF FILE app/search/milvus_client.py ---

import logging
import asyncio
import numpy as np
from typing import List, Dict, Optional, Any

from ..utils.data_parsers import DataParsers
from ..config import settings

logger = logging.getLogger(__name__)


class MilvusClient:
    """Handles all Milvus operations"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def build_expr(self, expr: Optional[str] = None, user_query: str = "") -> Optional[str]:
        """Build Milvus expression for filtering"""
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
    
    async def search_vectors(self, collection_name: str, vector: List[float], top_k: int,
                           expr: Optional[str] = None, user_query: str = "") -> List[Dict]:
        """Search vectors in Milvus collection"""
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

            if collection_name in (settings.CLIP_COLLECTION, settings.BEIT3_COLLECTION):
                raw_kf = hit_data.get("keyframe_id", "")
                if isinstance(raw_kf, str):
                    pos = raw_kf.lower().find(".jpg")
                    kf_clean = raw_kf[:pos + 4] if pos != -1 else raw_kf
                else:
                    kf_clean = raw_kf

                vid, kf_normalized = DataParsers.parse_video_id_from_kf(kf_clean)

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
    
    async def search_vectors_async(self, collection_name: str, vector: List[float], top_k: int,
                                 expr: Optional[str] = None, user_query: str = "",
                                 search_type: str = "") -> List[Dict]:
        """Async wrapper for vector search to enable parallel execution."""
        return await self.search_vectors(collection_name, vector, top_k, expr, user_query)
    
    async def batch_search_objects(self, all_object_ids: List[int], obj_vector: List[float]) -> Dict[int, Dict]:
        """
        Single batch query for all object IDs instead of individual queries.
        Returns dict mapping object_id -> search result data
        """
        if not all_object_ids:
            return {}

        # Remove duplicates while preserving order
        unique_obj_ids = list(dict.fromkeys(all_object_ids))

        try:
            expr = f"object_id in [{','.join(map(str, unique_obj_ids))}]"
            limit = max(len(unique_obj_ids), 1)
            obj_hits = await self.search_vectors(settings.OBJECT_COLLECTION, obj_vector, limit, expr=expr)

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
    
    def process_search_results(self, search_results: List[Dict], candidate_info: Dict[str, Dict[str, Any]],
                              search_type: str):
        """Process search results and populate candidate_info."""
        for hit in search_results:
            kf_id = hit["entity"]["keyframe_id"]
            vid, kf_id = DataParsers.parse_video_id_from_kf(kf_id)
            score = hit['distance']
            obj_ids = DataParsers.split_csv_ints(hit['entity']['object_ids'])
            lab6 = DataParsers.parse_lab_colors18(hit['entity']['lab_colors'])

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
    
    async def hybrid_reranking(self, candidate_info: Dict[str, Dict], text_query: str, embedding_manager):
        """Perform hybrid reranking using BEiT-3 embeddings."""
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
                vid, normalized_kf_id = DataParsers.parse_video_id_from_kf(db_kf_id)
                beit3_vector_map[normalized_kf_id] = item['vector']

            # Debug: log found vs missing keyframes
            found_kfs = set(beit3_vector_map.keys())
            missing_kfs = set(candidate_kf_ids) - found_kfs
            if missing_kfs:
                logger.warning(f"BEIT-3 vectors missing for {len(missing_kfs)}/{len(candidate_kf_ids)} keyframes")
            else:
                logger.info(f"Found BEIT-3 vectors for all {len(candidate_kf_ids)} keyframes")

            beit3_query_vector = np.array(embedding_manager.get_beit3_text_embedding(text_query))

            # Vectorized distance computation
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

# --- END OF FILE app/search/milvus_client.py ---
