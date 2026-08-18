from pydantic import BaseModel, Field


class GroundingDetection(BaseModel):
    label: str = Field(..., description="Detected label from GroundingDINO")
    score: float = Field(..., description="Confidence score for the detection")
    bbox: list[float] = Field(..., description="Bounding box [x0, y0, x1, y1]")


class GroundingResult(BaseModel):
    keyframe_id: str = Field(..., description="Unique image/frame ID")
    grounding_score: float = Field(..., description="Score produced by GroundingDINO")
    retrieval_score: float | None = Field(
        default=None,
        description="Original retrieval score",
    )
    final_score: float = Field(..., description="Fused ranking score")
    detections: list[GroundingDetection] = Field(..., description="Detected objects for the image")


class RerankingImage(BaseModel):
    keyframe_id: str = Field(..., description="Unique image/frame ID")
    retrieval_score: float | None = Field(
        default=None,
        description="Score returned by the initial retrieval stage",
    )


class RerankingSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)

    frames: list[RerankingImage] = Field(
        ...,
        min_length=1,
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
    )


class RerankingSearchResponse(BaseModel):
    query: str
    total_candidates: int
    returned_results: int
    results: list[GroundingResult]

class RerankingKeyframe(BaseModel):
    keyframe_id: str

    image_path: str

    retrieval_score: float | None = None


class YOLOERerankRequest(BaseModel):
    query: str

    classes: list[str]

    keyframes: list[
        RerankingKeyframe
    ]

    top_k: int = 50


class YOLOERerankResponse(BaseModel):
    query: str

    classes: list[str]

    results: list[
        GroundingResult
    ]

    
# class DetectionResponse(BaseModel):
#     label: str
#     score: float
#     bbox: list[float]


# class RerankingResultResponse(BaseModel):
#     image_id: str
#     image_path: str
#     grounding_score: float
#     retrieval_score: float | None
#     final_score: float
#     detections: list[DetectionResponse]

