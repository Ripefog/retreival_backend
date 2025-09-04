# --- START OF FILE app/utils/data_parsers.py ---

import os
from typing import List, Optional, Tuple


class DataParsers:
    """Utility functions for parsing data"""
    
    @staticmethod
    def parse_video_id_from_kf(kf: str) -> Tuple[str, str]:
        """
        Nhận keyframe: L02_L02_V002_1130.04s.jpg hoặc L02_V002_1130.04s.jpg
        Trả về: (video_id, kf_id) dạng ('L02_V002', 'L02_V002_1130.04s.jpg')
        """
        name = os.path.splitext(os.path.basename(kf))[0]
        parts = name.split("_")

        # bỏ trùng L02_L02 -> giữ 1
        unique_parts = []
        for p in parts:
            if not unique_parts or unique_parts[-1] != p:
                unique_parts.append(p)

        # timestamp là phần cuối
        timestamp = unique_parts[-1]
        # lấy Lxx và Vxxx
        l_code = None
        v_code = None
        for p in unique_parts:
            if p.upper().startswith("L") and p[1:].isdigit():
                l_code = p.upper()
            if p.upper().startswith("K") and p[1:].isdigit():
                l_code = p.upper()
            if p.upper().startswith("V") and p[1:].isdigit():
                v_code = p.upper()

        video_id = f"{l_code}_{v_code}"
        kf_id = f"{video_id}_{timestamp}.jpg"
        return video_id, kf_id
    
    @staticmethod
    def split_csv_ints(s: Optional[str]) -> List[int]:
        """'1,2,3' -> [1,2,3]; bỏ qua phần tử rỗng/lỗi."""
        if not s:
            return []
        out = []
        for p in s.split(","):
            p = p.strip()
            if not p:
                continue
            try:
                out.append(int(p))
            except ValueError:
                # phòng trường hợp '12.0'
                try:
                    out.append(int(float(p)))
                except ValueError:
                    pass
        return out
    
    @staticmethod
    def split_csv_floats(s: Optional[str]) -> List[float]:
        """'1.2,3,4.5' -> [1.2,3.0,4.5]; bỏ qua phần tử rỗng/lỗi."""
        if not s:
            return []
        out = []
        for p in s.split(","):
            p = p.strip()
            if p == "":
                continue
            try:
                out.append(float(p))
            except ValueError:
                pass
        return out
    
    @staticmethod
    def parse_lab_colors18(s: Optional[str]) -> List[Tuple[float, float, float]]:
        """
        Nhận chuỗi 18 số (L,a,b * 6 màu), trả về list 6 tuple (L,a,b).
        Nếu thiếu thì pad 0.0; nếu thừa thì cắt bớt.
        """
        vals = DataParsers.split_csv_floats(s)
        if len(vals) < 18:
            vals += [0.0] * (18 - len(vals))
        elif len(vals) > 18:
            vals = vals[:18]

        lab6 = []
        i = 0
        while i + 2 < 18:
            lab6.append((vals[i], vals[i + 1], vals[i + 2]))
            i += 3
        return lab6

# --- END OF FILE app/utils/data_parsers.py ---
