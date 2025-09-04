# --- START OF FILE app/utils/color_utils.py ---

import numpy as np
from typing import List, Tuple
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

try:
    import colorspacious
    HAS_COLORSPACIOUS = True
except ImportError:
    HAS_COLORSPACIOUS = False


class ColorUtils:
    """Color conversion and comparison utilities"""
    
    @staticmethod
    def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB tuple to CIELAB tuple."""
        rgb_obj = sRGBColor(*rgb, is_upscaled=True)
        lab_obj = convert_color(rgb_obj, LabColor)
        return (lab_obj.lab_l, lab_obj.lab_a, lab_obj.lab_b)
    
    @staticmethod
    def vectorized_color_distances(colors1: List[Tuple[float, float, float]],
                                  colors2: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        OPTIMIZED: Vectorized color distance calculation using NumPy.
        Returns distance matrix of shape (len(colors1), len(colors2))
        """
        if not colors1 or not colors2:
            return np.array([[]])

        # Convert to numpy arrays for vectorized operations
        c1_array = np.array(colors1)  # shape: (m, 3)
        c2_array = np.array(colors2)  # shape: (n, 3)

        if HAS_COLORSPACIOUS:
            # Use optimized colorspacious for accurate CIEDE2000
            distances = np.zeros((len(colors1), len(colors2)))
            for i, color1 in enumerate(colors1):
                for j, color2 in enumerate(colors2):
                    distances[i, j] = colorspacious.deltaE(color1, color2, input_space="CIELab")
            return distances
        else:
            # Fallback: Euclidean distance in LAB space (much faster, reasonably accurate)
            # Broadcast to compute all pairwise distances at once
            c1_expanded = c1_array[:, np.newaxis, :]  # shape: (m, 1, 3)
            c2_expanded = c2_array[np.newaxis, :, :]  # shape: (1, n, 3)

            # Euclidean distance in LAB space
            distances = np.sqrt(np.sum((c1_expanded - c2_expanded) ** 2, axis=2))
            return distances
    
    @staticmethod
    def compare_bbox(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
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

# --- END OF FILE app/utils/color_utils.py ---
