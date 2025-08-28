# --- START OF FILE app/models.py ---

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set, Union

ObjectItem = Union[
    Tuple[Tuple[float, float, float], Tuple[int, int, int, int]],  # ((LAB),(BBOX))
    Tuple[float, float, float],                                    # LAB
    Tuple[int, int, int, int]                                      # BBOX
]


# --- Enums for controlled vocabularies ---
class SearchMode(str, Enum):
    HYBRID = "hybrid"
    CLIP = "clip"
    BEIT3 = "beit3"

# --- API Request Models ---
class SearchRequest(BaseModel):
    text_query: str = Field(..., description="Câu truy vấn ngữ nghĩa bằng ngôn ngữ tự nhiên để tìm kiếm video.")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Chế độ tìm kiếm để sử dụng.")
    user_query: Optional[str] = Field(default=None, description="Tên người dùng để lọc kết quả")
    object_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""Từ điển: tên đối tượng -> danh sách constraints để tăng điểm ưu tiên. 
        
        Supported formats:
        - Empty array: object-only search -> {"person": []}
        - Color only: [R,G,B] or [L,a,b] -> {"person": [[255,0,0]]}  
        - BBox only: [x1,y1,x2,y2] -> {"person": [[100,100,200,200]]}
        - Full constraint: [[color], [bbox]] -> {"person": [[[255,0,0], [100,100,200,200]]]}
        - Dict format: {"person": [{"color": [255,0,0], "bbox": [100,100,200,200]}]}
        - Mixed: {"person": [[], [255,0,0], [[255,0,0], [100,100,200,200]]]}
        
        Color auto-detection: RGB (0-255) hoặc LAB space."""
    )

    # màu RGB hoặc LAB (auto-detect)
    color_filters: Optional[List[Any]] = Field(
        default=None,
        description="Danh sách các màu để tăng điểm ưu tiên. Có thể là RGB [R,G,B] (0-255) hoặc LAB [L,a,b]. Hệ thống tự động phát hiện và chuyển đổi."
    )
    ocr_query: Optional[str] = Field(default=None, description="Từ khóa để lọc các keyframe có chứa văn bản này (OCR).")
    asr_query: Optional[str] = Field(default=None, description="Từ khóa để lọc các video có chứa lời thoại này (ASR).")
    top_k: int = Field(default=20, ge=1, le=2000, description="Số lượng kết quả hàng đầu để trả về.")
    #exact_match: bool = Field(default=False, description="Chế độ exact matching: ưu tiên kết quả có object/color filters khớp chính xác.")

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
    objects: Dict[str, List[Tuple[Tuple[float, float, float], Tuple[int, int, int, int]]]] = Field(..., description="Danh sách các nhãn đối tượng duy nhất được phát hiện trong ảnh.")
    colors: List[Tuple[float, float, float]] = Field(..., description="Danh sách các màu sắc chính duy nhất được phát hiện trong ảnh.")

# --- Examples for Swagger UI documentation ---
# --- Examples for Swagger UI documentation ---
search_examples = {
    "simple_hybrid": {
        "summary": "1. Tìm kiếm Hybrid đơn giản",
        "description": "Một tìm kiếm cơ bản cho 'một người ngồi ở bàn' sử dụng chế độ hybrid.",
        "value": {
            "text_query": "a person sitting at a desk",
            "mode": "hybrid",
            "user_query": "Minh Tâm",
            "top_k": 5
        },
    },
    "object_only_search": {
        "summary": "2. Object-only search (NEW FORMAT)",
        "description": "Tìm keyframes có object 'person' và 'chair' không quan tâm màu sắc hay vị trí.",
        "value": {
            "text_query": "A man is speaking in front of a microphone on HTV7",
            "mode": "hybrid",
            "user_query": "",
            "object_filters": {
                "person": [],
                "chair": []
            },
            "top_k": 20
        },
    },
    "color_constraint": {
        "summary": "3. Color-only constraint (NEW FORMAT)",
        "description": "Tìm object 'person' có màu đỏ và 'car' có màu xanh (RGB format).",
        "value": {
            "text_query": "person in red shirt near blue car",
            "mode": "hybrid",
            "object_filters": {
                "person": [[255, 0, 0]],
                "car": [[0, 0, 255]]
            },
            "top_k": 10
        },
    },
    "bbox_constraint": {
        "summary": "4. Spatial constraint (NEW FORMAT)",  
        "description": "Tìm object 'person' trong vùng trung tâm màn hình.",
        "value": {
            "text_query": "person in the center of the frame",
            "mode": "hybrid",
            "object_filters": {
                "person": [[400, 200, 800, 600]]
            },
            "top_k": 10
        },
    },
    "mixed_constraints": {
        "summary": "5. Mixed constraints (NEW FORMAT)",
        "description": "Kết hợp nhiều loại constraint cho cùng object.",
        "value": {
            "text_query": "presenter on stage",
            "mode": "hybrid",
            "object_filters": {
                "person": [
                    [],
                    [255, 0, 0],
                    [[100, 100, 500, 500]],
                    [[255, 255, 255], [200, 150, 600, 400]]
                ]
            },
            "top_k": 15
        },
    },
    "dict_format": {
        "summary": "6. Dictionary format (NEW FORMAT)",
        "description": "Sử dụng dictionary format rõ ràng hơn.",
        "value": {
            "text_query": "news anchor in studio",
            "mode": "hybrid",
            "object_filters": {
                "person": [
                    {"color": [128, 128, 128], "bbox": [100, 50, 600, 400]}
                ],
                "chair": [
                    {"color": [139, 69, 19]}
                ]
            },
            "top_k": 10
        },
    },
    "backward_compatibility": {
        "summary": "7. Backward compatibility (OLD FORMAT)",
        "description": "Format cũ vẫn hoạt động: ((LAB), (BBOX)).",
        "value": {
            "text_query": "a presenter on stage",
            "mode": "hybrid",
            "user_query": "Minh Tâm",
            "object_filters": {
                "person": [
                    [[70.0, 0.0, 0.0], [10, 20, 300, 400]]
                ],
                "presenter": [
                    [[65.0, 5.0, 5.0], [200, 100, 800, 700]]
                ]
            },
            "color_filters": [
                [55.0, 65.0, 25.0]
            ],
            "ocr_query": "VIỆT NAM",
            "asr_query": "kinh tế",
            "top_k": 5
        },
    },
}


compare_examples = {
    "default": {
        "summary": "So sánh các chế độ tìm kiếm",
        "description": "So sánh hiệu quả giữa các chế độ `hybrid`, `clip`, và `beit3` trên cùng một truy vấn.",
        "value": {"text_query": "a news anchor in a studio", "top_k": 5},
    }
}

# --- Temporal Search Models ---
class TemporalQuery(BaseModel):
    step: int = Field(..., description="Thứ tự bước trong sequence (1, 2, 3...)")
    text_query: str = Field(..., description="Câu truy vấn cho bước này")

class TemporalSearchRequest(BaseModel):
    sequential_queries: List[TemporalQuery] = Field(..., description="Danh sách các query theo thứ tự thời gian")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Chế độ tìm kiếm")
    object_filters: Optional[Dict[str, Any]] = Field(default=None, description="Object filters áp dụng cho tất cả queries")
    color_filters: Optional[List[Any]] = Field(default=None, description="Color filters áp dụng cho tất cả queries")
    ocr_query: Optional[str] = Field(default=None, description="OCR filter áp dụng cho tất cả queries")
    asr_query: Optional[str] = Field(default=None, description="ASR filter áp dụng cho tất cả queries")
    top_k_per_query: int = Field(default=50, ge=1, le=200, description="Số kết quả top cho mỗi query riêng lẻ")
    max_time_gap: float = Field(default=300.0, ge=1.0, description="Khoảng cách thời gian tối đa giữa các bước (giây)")
    min_time_gap: float = Field(default=1.0, ge=0.1, description="Khoảng cách thời gian tối thiểu giữa các bước (giây)")
    top_sequences: int = Field(default=10, ge=1, le=50, description="Số sequence tốt nhất trả về")

class TemporalStep(BaseModel):
    step: int = Field(..., description="Số thứ tự bước")
    keyframe_id: str = Field(..., description="ID của keyframe")
    video_id: str = Field(..., description="ID của video")
    timestamp: float = Field(..., description="Thời điểm trong video (giây)")
    score: float = Field(..., description="Điểm relevancy của bước này")
    text_query: str = Field(..., description="Query gốc cho bước này")

class TemporalSequence(BaseModel):
    sequence_id: int = Field(..., description="ID định danh của sequence")
    video_id: str = Field(..., description="ID của video chứa sequence")
    steps: List[TemporalStep] = Field(..., description="Danh sách các bước trong sequence")
    sequence_score: float = Field(..., description="Điểm tổng hợp của toàn bộ sequence")
    temporal_consistency: float = Field(..., description="Điểm nhất quán về thời gian")
    time_gaps: List[float] = Field(..., description="Khoảng cách thời gian giữa các bước liên tiếp")
    total_duration: float = Field(..., description="Tổng thời lượng của sequence")

class TemporalSearchResponse(BaseModel):
    sequential_queries: List[TemporalQuery] = Field(..., description="Danh sách queries đã xử lý")
    query_results: List[List[Dict[str, Any]]] = Field(..., description="Kết quả riêng lẻ của từng query")
    temporal_sequences: List[TemporalSequence] = Field(..., description="Danh sách sequences được tìm thấy")
    total_sequences: int = Field(..., description="Tổng số sequences hợp lệ")
    processing_time: float = Field(..., description="Thời gian xử lý (giây)")

# Temporal search examples
temporal_examples = {
    "cooking_sequence": {
        "summary": "Chuỗi nấu ăn: Cho cá vào tô → Trộn bột → Nhấc đũa",
        "description": "Tìm kiếm chuỗi hành động nấu ăn theo thứ tự thời gian",
        "value": {
            "sequential_queries": [
                {
                    "step": 1,
                    "text_query": "Người đầu bếp cho cá vào một tô màu trắng"
                },
                {
                    "step": 2, 
                    "text_query": "Người đầu bếp đổ bột vào một tô cá để chiên"
                },
                {
                    "step": 3,
                    "text_query": "Người đầu bếp dùng đũa để kiểm tra độ nóng của dầu"
                }
            ],
            "mode": "hybrid",
            "top_k_per_query": 30,
            "max_time_gap": 180,
            "min_time_gap": 3,
            "top_sequences": 5
        }
    },
    "news_sequence": {
        "summary": "Chuỗi tin tức: MC giới thiệu → Phóng viên báo cáo → Kết thúc",
        "description": "Tìm kiếm chuỗi hành động trong chương trình tin tức",
        "value": {
            "sequential_queries": [
                {
                    "step": 1,
                    "text_query": "MC giới thiệu chủ đề tin tức"
                },
                {
                    "step": 2,
                    "text_query": "Phóng viên đưa tin tại hiện trường"
                },
                {
                    "step": 3,
                    "text_query": "MC kết thúc và chuyển sang tin khác"
                }
            ],
            "mode": "hybrid",
            "top_k_per_query": 20,
            "max_time_gap": 300,
            "min_time_gap": 5,
            "top_sequences": 8
        }
    }
}
# --- END OF FILE app/models.py ---