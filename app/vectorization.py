# --- START OF FILE app/vectorization.py ---

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import torch

# Try to import optimized libraries
try:
    import colorspacious
    HAS_COLORSPACIOUS = True
except ImportError:
    HAS_COLORSPACIOUS = False

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

class VectorizedOperations:
    """
    Highly optimized vectorized operations for color matching, similarity calculations,
    and mathematical operations used in the retrieval engine.
    """
    
    def __init__(self):
        self.setup_optimization_flags()
    
    def setup_optimization_flags(self):
        """Setup optimization flags based on available libraries"""
        self.use_colorspacious = HAS_COLORSPACIOUS
        self.use_numba = HAS_NUMBA
        
        if HAS_COLORSPACIOUS:
            logger.info("✅ Using colorspacious for accurate CIEDE2000 color distance")
        else:
            logger.warning("⚠️ colorspacious not available, using fast Euclidean approximation")
            
        if HAS_NUMBA:
            logger.info("✅ Using Numba JIT compilation for accelerated operations")
        else:
            logger.info("ℹ️ Numba not available, using pure NumPy operations")
    
    def vectorized_color_distances(self, 
                                 colors1: Union[List[Tuple[float, float, float]], np.ndarray], 
                                 colors2: Union[List[Tuple[float, float, float]], np.ndarray]) -> np.ndarray:
        """
        OPTIMIZED: Vectorized color distance calculation.
        Returns distance matrix of shape (len(colors1), len(colors2))
        
        Args:
            colors1: Query colors in LAB format
            colors2: Palette colors in LAB format
            
        Returns:
            Distance matrix with CIEDE2000 or Euclidean distances
        """
        if not colors1 or not colors2:
            return np.array([[]])
        
        # Convert to numpy arrays for vectorized operations
        c1_array = np.asarray(colors1, dtype=np.float32)  # shape: (m, 3)
        c2_array = np.asarray(colors2, dtype=np.float32)  # shape: (n, 3)
        
        if c1_array.ndim == 1:
            c1_array = c1_array.reshape(1, -1)
        if c2_array.ndim == 1:
            c2_array = c2_array.reshape(1, -1)
        
        if self.use_colorspacious:
            return self._accurate_color_distances(c1_array, c2_array)
        else:
            return self._fast_euclidean_distances(c1_array, c2_array)
    
    def _accurate_color_distances(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """Accurate CIEDE2000 color distances using colorspacious"""
        m, n = colors1.shape[0], colors2.shape[0]
        distances = np.zeros((m, n), dtype=np.float32)
        
        # Vectorized colorspacious computation
        for i in range(m):
            for j in range(n):
                try:
                    distances[i, j] = colorspacious.deltaE(
                        colors1[i], colors2[j], input_space="CIELab"
                    )
                except:
                    # Fallback to Euclidean if colorspacious fails
                    distances[i, j] = np.linalg.norm(colors1[i] - colors2[j])
        
        return distances
    
    def _fast_euclidean_distances(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """Fast Euclidean distance in LAB space (good approximation)"""
        # Broadcast to compute all pairwise distances at once
        c1_expanded = colors1[:, np.newaxis, :]  # shape: (m, 1, 3)
        c2_expanded = colors2[np.newaxis, :, :]  # shape: (1, n, 3)
        
        # Vectorized Euclidean distance computation
        distances = np.sqrt(np.sum((c1_expanded - c2_expanded) ** 2, axis=2))
        return distances.astype(np.float32)
    
    def vectorized_cosine_similarity(self, 
                                   vectors1: Union[List[np.ndarray], np.ndarray],
                                   vectors2: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        OPTIMIZED: Vectorized cosine similarity computation.
        
        Args:
            vectors1: Query vectors (m, d)
            vectors2: Database vectors (n, d) 
            
        Returns:
            Similarity matrix (m, n)
        """
        # Convert to numpy arrays
        v1 = np.asarray(vectors1, dtype=np.float32)
        v2 = np.asarray(vectors2, dtype=np.float32)
        
        if v1.ndim == 1:
            v1 = v1.reshape(1, -1)
        if v2.ndim == 1:
            v2 = v2.reshape(1, -1)
        
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
        
        # Vectorized dot product
        similarities = np.dot(v1_norm, v2_norm.T)
        return similarities
    
    def vectorized_l2_distances(self,
                               vectors1: Union[List[np.ndarray], np.ndarray],
                               vectors2: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        OPTIMIZED: Vectorized L2 distance computation.
        
        Args:
            vectors1: Query vectors (m, d)
            vectors2: Database vectors (n, d)
            
        Returns:
            Distance matrix (m, n)
        """
        v1 = np.asarray(vectors1, dtype=np.float32)
        v2 = np.asarray(vectors2, dtype=np.float32)
        
        if v1.ndim == 1:
            v1 = v1.reshape(1, -1)
        if v2.ndim == 1:
            v2 = v2.reshape(1, -1)
        
        # Vectorized L2 distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        v1_sq = np.sum(v1**2, axis=1, keepdims=True)  # (m, 1)
        v2_sq = np.sum(v2**2, axis=1, keepdims=False)  # (n,)
        dot_product = np.dot(v1, v2.T)  # (m, n)
        
        distances = v1_sq + v2_sq - 2 * dot_product
        distances = np.sqrt(np.maximum(distances, 0))  # Avoid negative values due to numerical errors
        
        return distances
    
    def vectorized_similarity_scoring(self, distances: np.ndarray, sigma: float = 20.0) -> np.ndarray:
        """
        OPTIMIZED: Convert distances to similarity scores using Gaussian kernel.
        
        Args:
            distances: Distance matrix (m, n)
            sigma: Gaussian kernel parameter
            
        Returns:
            Similarity matrix (m, n) with values in [0, 1]
        """
        return np.exp(-(distances / sigma) ** 2)
    
    def optimized_hungarian_assignment(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        OPTIMIZED: Hungarian algorithm using scipy's optimized C++ implementation.
        
        Args:
            cost_matrix: Cost matrix (m, n)
            
        Returns:
            Tuple of (row_indices, col_indices) for optimal assignment
        """
        # Ensure the cost matrix is finite and non-negative
        cost_matrix = np.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=0)
        
        # Use scipy's optimized Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        return row_indices, col_indices
    
    def batch_bbox_iou(self, bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """
        OPTIMIZED: Vectorized IoU computation for bounding boxes.
        
        Args:
            bboxes1: Bounding boxes (m, 4) in format [x1, y1, x2, y2]
            bboxes2: Bounding boxes (n, 4) in format [x1, y1, x2, y2]
            
        Returns:
            IoU matrix (m, n)
        """
        bboxes1 = np.asarray(bboxes1, dtype=np.float32)
        bboxes2 = np.asarray(bboxes2, dtype=np.float32)
        
        if bboxes1.ndim == 1:
            bboxes1 = bboxes1.reshape(1, -1)
        if bboxes2.ndim == 1:
            bboxes2 = bboxes2.reshape(1, -1)
        
        m, n = bboxes1.shape[0], bboxes2.shape[0]
        
        # Expand dimensions for broadcasting
        b1 = bboxes1[:, None, :]  # (m, 1, 4)
        b2 = bboxes2[None, :, :]  # (1, n, 4)
        
        # Calculate intersection coordinates
        x1 = np.maximum(b1[:, :, 0], b2[:, :, 0])  # (m, n)
        y1 = np.maximum(b1[:, :, 1], b2[:, :, 1])  # (m, n)
        x2 = np.minimum(b1[:, :, 2], b2[:, :, 2])  # (m, n)
        y2 = np.minimum(b1[:, :, 3], b2[:, :, 3])  # (m, n)
        
        # Calculate intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate areas of both bounding boxes
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])  # (m,)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])  # (n,)
        
        # Calculate union
        union = area1[:, None] + area2[None, :] - intersection  # (m, n)
        
        # Calculate IoU, avoid division by zero
        iou = intersection / (union + 1e-8)
        
        return iou
    
    def vectorized_score_fusion(self, scores_dict: Dict[str, np.ndarray], 
                               weights: Dict[str, float]) -> np.ndarray:
        """
        OPTIMIZED: Vectorized score fusion for multiple modalities.
        
        Args:
            scores_dict: Dictionary of score arrays {modality: scores}
            weights: Dictionary of weights {modality: weight}
            
        Returns:
            Fused scores array
        """
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return np.zeros_like(next(iter(scores_dict.values())))
        
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Vectorized weighted sum
        fused_scores = np.zeros_like(next(iter(scores_dict.values())))
        
        for modality, scores in scores_dict.items():
            if modality in normalized_weights:
                fused_scores += normalized_weights[modality] * scores
        
        return fused_scores
    
    def batch_color_conversion_rgb_to_lab(self, rgb_colors: np.ndarray) -> np.ndarray:
        """
        OPTIMIZED: Batch RGB to LAB color conversion.
        
        Args:
            rgb_colors: RGB colors array (n, 3) with values in [0, 255]
            
        Returns:
            LAB colors array (n, 3)
        """
        rgb_colors = np.asarray(rgb_colors, dtype=np.float32) / 255.0
        
        # Apply gamma correction
        rgb_linear = np.where(rgb_colors <= 0.04045,
                             rgb_colors / 12.92,
                             np.power((rgb_colors + 0.055) / 1.055, 2.4))
        
        # Convert to XYZ using sRGB matrix
        xyz_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)
        
        xyz = np.dot(rgb_linear, xyz_matrix.T)
        
        # Normalize by D65 white point
        xyz_n = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
        
        # Apply LAB conversion
        f_xyz = np.where(xyz_n > 0.008856,
                        np.power(xyz_n, 1/3),
                        (7.787 * xyz_n) + (16/116))
        
        L = (116 * f_xyz[:, 1]) - 16
        a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
        b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])
        
        lab = np.column_stack([L, a, b])
        return lab
    
    def parallel_embedding_similarity(self, query_embedding: np.ndarray,
                                    candidate_embeddings: np.ndarray,
                                    similarity_type: str = 'cosine') -> np.ndarray:
        """
        OPTIMIZED: Parallel computation of embedding similarities.
        
        Args:
            query_embedding: Single query embedding (d,)
            candidate_embeddings: Multiple candidate embeddings (n, d)
            similarity_type: 'cosine' or 'l2'
            
        Returns:
            Similarity scores (n,)
        """
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if similarity_type == 'cosine':
            similarities = self.vectorized_cosine_similarity(
                query_embedding, candidate_embeddings
            )[0]  # Take first row since query is single vector
        elif similarity_type == 'l2':
            distances = self.vectorized_l2_distances(
                query_embedding, candidate_embeddings
            )[0]  # Take first row
            # Convert L2 distances to similarities
            similarities = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        return similarities


# Numba-accelerated functions (if available)
if HAS_NUMBA:
    @jit(nopython=True, parallel=True)
    def fast_euclidean_distance_matrix(colors1, colors2):
        """Ultra-fast Euclidean distance computation with Numba JIT"""
        m, n = colors1.shape[0], colors2.shape[0]
        distances = np.zeros((m, n), dtype=np.float32)
        
        for i in prange(m):
            for j in prange(n):
                diff = colors1[i] - colors2[j]
                distances[i, j] = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        
        return distances
    
    @jit(nopython=True, parallel=True)
    def fast_cosine_similarity_matrix(vectors1, vectors2):
        """Ultra-fast cosine similarity with Numba JIT"""
        m, n = vectors1.shape[0], vectors2.shape[0]
        similarities = np.zeros((m, n), dtype=np.float32)
        
        for i in prange(m):
            v1_norm = np.sqrt(np.sum(vectors1[i]**2))
            for j in prange(n):
                v2_norm = np.sqrt(np.sum(vectors2[j]**2))
                dot_product = np.sum(vectors1[i] * vectors2[j])
                similarities[i, j] = dot_product / (v1_norm * v2_norm + 1e-8)
        
        return similarities
else:
    # Fallback implementations
    def fast_euclidean_distance_matrix(colors1, colors2):
        return VectorizedOperations()._fast_euclidean_distances(colors1, colors2)
    
    def fast_cosine_similarity_matrix(vectors1, vectors2):
        return VectorizedOperations().vectorized_cosine_similarity(vectors1, vectors2)


# Global instance for easy import
vectorized_ops = VectorizedOperations()

# --- END OF FILE app/vectorization.py ---