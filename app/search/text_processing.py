# --- START OF FILE app/search/text_processing.py ---

import logging
from typing import Dict, Optional
from rapidfuzz import fuzz

from ..utils.data_parsers import DataParsers
from ..config import settings

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles OCR and ASR text processing"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def _normalize_text(self, s: str) -> str:
        """Normalize text for fuzzy matching."""
        return " ".join((s or "").lower().split())
    
    async def apply_ocr_filter(self, candidate_info: Dict, ocr_query: str):
        """Apply OCR filtering with fuzzy matching."""
        if not ocr_query or not candidate_info:
            logger.info(f"OCR filter skipped: ocr_query='{ocr_query}', candidates={len(candidate_info) if candidate_info else 0}")
            return

        es_client = self.db_manager.es_client
        if not es_client:
            logger.error("Elasticsearch client không khả dụng. Bỏ qua bộ lọc OCR.")
            return

        logger.info(f"Starting OCR filter with query: '{ocr_query}' for {len(candidate_info)} candidates")

        kf_ids_to_fetch = list(candidate_info.keys())
        ocr_texts_from_es = {}

        # Query Elasticsearch để lấy OCR text cho các keyframes
        try:
            if kf_ids_to_fetch:
                query_body = {
                    "query": {
                        "terms": {
                            "keyframe_id": kf_ids_to_fetch
                        }
                    },
                    "size": len(kf_ids_to_fetch)
                }

                response = es_client.search(
                    index=settings.OCR_INDEX,
                    body=query_body
                )

                for hit in response['hits']['hits']:
                    kf_id = hit['_source']['keyframe_id']
                    
                    # Thử các tên trường khác nhau
                    possible_ocr_fields = ['text', 'ocr_text', 'ocr', 'content', 'ocr_content', 'extracted_text']
                    ocr_text = ''
                    
                    for field in possible_ocr_fields:
                        if field in hit['_source']:
                            ocr_text = hit['_source'].get(field, '')
                            break
                    
                    if ocr_text:
                        ocr_texts_from_es[kf_id] = ocr_text

        except Exception as e:
            logger.error(f"Failed to query OCR texts from Elasticsearch: {e}", exc_info=True)
            return

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

    async def apply_asr_filter(self, candidate_info: Dict, asr_query: str):
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

        # Bước 1: Chuẩn bị query cho Elasticsearch
        time_window_sec = 15.0  # ±15s = 30s total
        es_should_clauses = []
        candidate_map = {}

        for kf_id, info in candidate_info.items():
            timestamp = info.get("timestamp")
            if timestamp is None:
                continue

            video_id, _ = DataParsers.parse_video_id_from_kf(kf_id)
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

        # Bước 2: Lấy dữ liệu ASR từ Elasticsearch
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

            # Bước 3: Ghép nối ASR text cho từng keyframe
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

        # Bước 4: So sánh và cộng điểm
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

# --- END OF FILE app/search/text_processing.py ---
