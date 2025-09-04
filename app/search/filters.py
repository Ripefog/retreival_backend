# --- START OF FILE app/search/filters.py ---

import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import linear_sum_assignment

from ..utils.color_utils import ColorUtils
from ..utils.data_parsers import DataParsers

logger = logging.getLogger(__name__)


class ObjectColorFilter:
    """Handles object and color filtering logic"""
    
    def __init__(self, embedding_manager, milvus_client):
        self.embedding_manager = embedding_manager
        self.milvus_client = milvus_client
    
    async def apply_object_color_filters(
        self,
        candidate_info: Dict[str, Dict[str, Any]],
        object_filters: Optional[Dict[str, List]],
        color_filters: Optional[List],
        top_k: int
    ):
        """Apply object and color filters to enhance candidate scores"""
        
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
                return ColorUtils.rgb_to_lab((int(L), int(A), int(B)))
            return (float(L), float(A), float(B))  # assume already LAB

        def _sim_from_delta(d: float, sigma: float = 20.0) -> float:
            """Similarity from ΔE: exp(-(ΔE/σ)²)."""
            return np.exp(-(d / sigma) ** 2)

        # ===== OBJECT FILTERS =====
        if object_filters:
            norm_object_filters = self._normalize_object_filters(object_filters)

            for obj_label, queries in norm_object_filters.items():
                obj_vector = self.embedding_manager.get_clip_text_embedding(obj_label).tolist()

                # Collect all object IDs from all candidates for batch query
                all_object_ids = []
                candidate_objects = {}  # kf_id -> list of object_ids

                for kf_id, info in candidate_info.items():
                    obj_ids = info.get("object_ids") or []
                    if obj_ids:
                        candidate_objects[kf_id] = obj_ids
                        all_object_ids.extend(obj_ids)

                if not all_object_ids:
                    continue

                # Single batch query instead of per-candidate queries
                batch_results = await self.milvus_client.batch_search_objects(all_object_ids, obj_vector)

                # Process each candidate
                for kf_id, obj_ids in candidate_objects.items():
                    # Get results for this candidate's objects
                    obj_hits = [batch_results[obj_id] for obj_id in obj_ids if obj_id in batch_results]

                    if not obj_hits:
                        continue

                    # Prepare queries
                    Q = []
                    for (q_color, q_bbox) in queries:
                        q_lab = _ensure_lab(q_color) if q_color is not None else None
                        q_bb = tuple(q_bbox) if q_bbox is not None else None
                        Q.append((q_lab, q_bb))

                    m = len(Q)
                    if m == 0:
                        continue

                    # Parse hits
                    O_vec_sim = []
                    O_color_lab = []
                    O_bbox = []

                    for h in obj_hits:
                        ent = h.get("entity", {})
                        d = float(h.get("distance", 0.0))
                        s_vec = 1.0 / (1.0 + d)
                        O_vec_sim.append(s_vec)

                        cl = DataParsers.split_csv_floats(ent.get("color_lab"))
                        bl = DataParsers.split_csv_floats(ent.get("bbox_xyxy"))
                        O_color_lab.append(cl if len(cl) == 3 else None)
                        O_bbox.append(tuple(bl) if len(bl) == 4 else None)

                    n = len(O_vec_sim)
                    if n == 0:
                        continue

                    # Vectorized similarity matrix computation
                    S = np.zeros((m, n))

                    for i in range(m):
                        q_lab, q_bb = Q[i]
                        use_vec = True
                        use_color = (q_lab is not None)
                        use_bbox = (q_bb is not None)

                        # Re-normalize weights
                        w_sum = 0.0
                        wv = W_VEC if use_vec else 0.0
                        wc = W_COLOR if use_color else 0.0
                        wb = W_BBOX if use_bbox else 0.0
                        w_sum = wv + wc + wb

                        if w_sum == 0:
                            continue

                        wv /= w_sum
                        wc /= w_sum
                        wb /= w_sum

                        # Vector similarity (vectorized)
                        vec_sim = np.array(O_vec_sim) * wv
                        S[i, :] += vec_sim

                        # Color similarity (vectorized if possible)
                        if use_color:
                            valid_colors = [(j, tuple(O_color_lab[j])) for j in range(n) if
                                            O_color_lab[j] is not None]
                            if valid_colors:
                                indices, colors = zip(*valid_colors)
                                # Use vectorized color distance computation
                                distances = ColorUtils.vectorized_color_distances([q_lab], list(colors))[0]

                                for idx_in_valid, j in enumerate(indices):
                                    de = distances[idx_in_valid]
                                    if de <= MAX_DELTA_E:
                                        s_col = _sim_from_delta(de, SIGMA_COLOR)
                                        S[i, j] += wc * s_col

                        # Bbox similarity
                        if use_bbox:
                            for j in range(n):
                                p_bb = O_bbox[j]
                                if p_bb is not None:
                                    iou = float(ColorUtils.compare_bbox(q_bb, p_bb))
                                    if iou >= MIN_IOU:
                                        S[i, j] += wb * iou

                    # Hungarian algorithm
                    cost_matrix = 1.0 - S
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)

                    # Aggregate score
                    sim_sum = 0.0
                    covered = 0

                    for row_idx, col_idx in zip(row_indices, col_indices):
                        if row_idx < m and col_idx < n:
                            sij = S[row_idx, col_idx]
                            sim_sum += sij
                            if sij >= TAU_S:
                                covered += 1

                    S_match = (sim_sum / m) if m > 0 else 0.0
                    C = (covered / m) if m > 0 else 0.0
                    S_obj = ALPHA * S_match + BETA * C
                    boost = W_OBJ * S_obj

                    if boost > 0:
                        candidate_info[kf_id]["score"] += boost
                        candidate_info[kf_id].setdefault("reasons", []).append(
                            f"Object match: '{obj_label}' +{boost:.3f} (S={S_obj:.3f}, cov={C:.2f})"
                        )

        # ===== COLOR FILTERS =====
        if color_filters:
            queries_lab = []
            for qc in color_filters:
                if qc is not None:
                    queries_lab.append(ColorUtils.rgb_to_lab(tuple(qc)))

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

                    # Vectorized distance matrix computation
                    distance_matrix = ColorUtils.vectorized_color_distances(queries_lab, palette)
                    similarity_matrix = np.exp(-(distance_matrix / 20.0) ** 2)
                    cost_matrix = 1.0 - similarity_matrix

                    # Hungarian algorithm
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
                    min_distances = np.min(distance_matrix, axis=1)
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
        """Normalize and validate object filters."""
        norm: Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]] = {}

        for obj, items in object_filters.items():
            fixed: List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]] = []

            for it in items:
                if (not isinstance(it, (list, tuple))) or len(it) != 2:
                    continue

                lab, bbox = it[0], it[1]
                if not (isinstance(lab, (list, tuple)) and len(lab) == 3):
                    continue
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue

                lab_t = (float(lab[0]), float(lab[1]), float(lab[2]))
                bbox_t = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                fixed.append((lab_t, bbox_t))

            if fixed:
                norm[obj] = fixed

        return norm

# --- END OF FILE app/search/filters.py ---
