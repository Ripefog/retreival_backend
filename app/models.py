# --- START OF FILE app/models.py ---

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

# --- Enums for controlled vocabularies ---
class SearchMode(str, Enum):
    HYBRID = "hybrid"
    CLIP = "clip"
    BEIT3 = "beit3"

# --- API Request Models ---
class SearchRequest(BaseModel):
    text_query: str = Field(..., description="Câu truy vấn ngữ nghĩa bằng ngôn ngữ tự nhiên để tìm kiếm video.")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Chế độ tìm kiếm để sử dụng.")
    object_filters: Optional[List[str]] = Field(default=None, description="Danh sách các nhãn đối tượng để tăng điểm ưu tiên.")
    color_filters: Optional[List[str]] = Field(default=None, description="Danh sách các màu sắc để tăng điểm ưu tiên.")
    ocr_query: Optional[str] = Field(default=None, description="Từ khóa để lọc các keyframe có chứa văn bản này (OCR).")
    asr_query: Optional[str] = Field(default=None, description="Từ khóa để lọc các video có chứa lời thoại này (ASR).")
    top_k: int = Field(default=20, ge=1, le=200, description="Số lượng kết quả hàng đầu để trả về.")

# --- API Response Models ---
class SearchResultMetadata(BaseModel):
    rank: int = Field(..., description="Thứ hạng của kết quả.")
    clip_score: Optional[float] = Field(default=None, description="Điểm tương đồng từ mô hình CLIP (nếu có).")
    beit3_score: Optional[float] = Field(default=None, description="Điểm tương đồng từ mô hình BEiT-3 (nếu có).")

class SearchResult(BaseModel):
    keyframe_id: str = Field(..., description="ID định danh duy nhất của keyframe.")
    video_id: str = Field(..., description="ID của video chứa keyframe này.")
    timestamp: float = Field(..., description="Thời điểm (giây) của keyframe trong video.")
    score: float = Field(..., description="Điểm relevancy tổng hợp của kết quả, càng cao càng tốt.")
    reasons: List[str] = Field(default=[], description="Giải thích ngắn gọn tại sao kết quả này được trả về.")
    metadata: SearchResultMetadata = Field(..., description="Metadata bổ sung về điểm số và xếp hạng.")

class SearchResponse(BaseModel):
    query: str = Field(..., description="Câu truy vấn gốc đã được xử lý.")
    mode: SearchMode = Field(..., description="Chế độ tìm kiếm đã được sử dụng.")
    results: List[SearchResult] = Field(..., description="Danh sách các kết quả tìm kiếm được.")
    total_results: int = Field(..., description="Tổng số kết quả được trả về.")

class ImageObjectsResponse(BaseModel):
    objects: List[str] = Field(..., description="Danh sách các nhãn đối tượng duy nhất được phát hiện trong ảnh.")
    colors: List[str] = Field(..., description="Danh sách các màu sắc chính duy nhất được phát hiện trong ảnh.")

# --- Examples for Swagger UI documentation ---
search_examples = {
    "simple_hybrid": {
        "summary": "1. Tìm kiếm Hybrid đơn giản",
        "description": "Một tìm kiếm cơ bản cho **'một người ngồi ở bàn'** sử dụng chế độ `hybrid`. Đây là trường hợp sử dụng phổ biến nhất.",
        "value": {"text_query": "a person sitting at a desk", "mode": "hybrid", "top_k": 5},
    },
    "complex_filters": {
        "summary": "2. Tìm kiếm với bộ lọc Object & Color",
        "description": "Tìm **'người đàn ông mặc đồ màu xanh'**. Hệ thống sẽ ưu tiên các kết quả có đối tượng 'person', 'man' và màu 'blue'.",
        "value": {"text_query": "a man wearing something blue", "mode": "hybrid", "object_filters": ["person", "man"], "color_filters": ["blue"], "top_k": 5},
    },
    "full_filters": {
        "summary": "3. Tìm kiếm phức hợp với tất cả bộ lọc",
        "description": "Một truy vấn phức hợp để tìm **'một người dẫn chương trình trên sân khấu'**, kết hợp bộ lọc đối tượng, màu sắc, văn bản trong ảnh (OCR) và lời thoại (ASR).",
        "value": {"text_query": "a presenter on stage", "mode": "hybrid", "object_filters": ["person"], "color_filters": ["red"], "ocr_query": "VIỆT NAM", "asr_query": "kinh tế", "top_k": 5},
    },
    "vietnamese_query": {
        "summary": "4. Tìm kiếm bằng Tiếng Việt",
        "description": "Ví dụ về một truy vấn tìm kiếm bằng Tiếng Việt: **'nữ biên tập viên mặc áo hồng'**.",
        "value": {"text_query": "nữ biên tập viên mặc áo hồng đang dẫn chương trình thời sự", "mode": "hybrid", "color_filters": ["pink", "red"], "top_k": 5},
    },
}

compare_examples = {
    "default": {
        "summary": "So sánh các chế độ tìm kiếm",
        "description": "So sánh hiệu quả giữa các chế độ `hybrid`, `clip`, và `beit3` trên cùng một truy vấn.",
        "value": {"text_query": "a news anchor in a studio", "top_k": 5},
    }
}
# --- END OF FILE app/models.py ---