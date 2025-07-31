from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class SearchMode(str, Enum):
    HYBRID = "hybrid"
    CLIP = "clip"
    BEIT3 = "beit3"

class SearchRequest(BaseModel):
    text_query: str = Field(..., description="Query text để tìm kiếm")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Chế độ tìm kiếm")
    object_filters: Optional[List[str]] = Field(default=None, description="Danh sách object để filter")
    color_filters: Optional[List[str]] = Field(default=None, description="Danh sách màu sắc để filter")
    ocr_query: Optional[str] = Field(default=None, description="Query OCR text")
    asr_query: Optional[str] = Field(default=None, description="Query ASR transcript")
    top_k: int = Field(default=100, ge=1, le=1000, description="Số lượng kết quả trả về")

class SearchResult(BaseModel):
    keyframe_id: str = Field(..., description="ID của keyframe")
    video_id: str = Field(..., description="ID của video")
    timestamp: float = Field(..., description="Thời gian trong video (giây)")
    score: float = Field(..., description="Điểm số relevance")
    reasons: List[str] = Field(default=[], description="Lý do match")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata bổ sung")

class SearchResponse(BaseModel):
    query: str = Field(..., description="Query gốc")
    mode: SearchMode = Field(..., description="Chế độ tìm kiếm được sử dụng")
    results: List[SearchResult] = Field(..., description="Danh sách kết quả")
    total_results: int = Field(..., description="Tổng số kết quả")
    search_time: Optional[float] = Field(default=None, description="Thời gian tìm kiếm (giây)")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Trạng thái hệ thống")
    milvus: Dict[str, Any] = Field(..., description="Trạng thái Milvus")
    elasticsearch: Dict[str, Any] = Field(..., description="Trạng thái Elasticsearch")
    retriever: str = Field(..., description="Trạng thái retriever")

class CollectionInfo(BaseModel):
    name: str = Field(..., description="Tên collection")
    num_entities: int = Field(..., description="Số lượng entities")
    schema: str = Field(..., description="Schema của collection")

class CollectionsResponse(BaseModel):
    collections: Dict[str, CollectionInfo] = Field(..., description="Thông tin các collections")

class ComparisonRequest(BaseModel):
    text_query: str = Field(..., description="Query text để so sánh")
    object_filters: Optional[List[str]] = Field(default=None, description="Object filters")
    color_filters: Optional[List[str]] = Field(default=None, description="Color filters")
    ocr_query: Optional[str] = Field(default=None, description="OCR query")
    asr_query: Optional[str] = Field(default=None, description="ASR query")
    top_k: int = Field(default=10, ge=1, le=100, description="Số lượng kết quả")

class ComparisonResponse(BaseModel):
    query: str = Field(..., description="Query gốc")
    comparison: Dict[str, Dict[str, Any]] = Field(..., description="Kết quả so sánh các modes")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Mô tả lỗi")
    detail: Optional[str] = Field(default=None, description="Chi tiết lỗi")
    timestamp: str = Field(..., description="Thời gian xảy ra lỗi") 