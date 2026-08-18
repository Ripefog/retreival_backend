# --- START OF FILE app/main.py ---

import os
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager

# Setup detailed logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from pydantic import ValidationError
import uvicorn
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .groudingDINO import GroundingDINO
from .YOLOE26 import YOLOE26LReranker

from .config import settings
from .database import init_database, close_database
from .retrieval_engine import HybridRetriever
from .models import (
    SearchRequest,
    SearchResponse,
    ImageObjectsResponse,
    search_examples,
    compare_examples,
    # Add temporal search models
    TemporalSearchRequest,
    TemporalSearchResponse,
    temporal_examples
)
from .temporal_search import TemporalSearchEngine
from typing import List, Dict, Any, Optional, Tuple, Set


# schema 
from .schemas.reranking import (
    RerankingSearchRequest,
    RerankingSearchResponse,
    RerankingImage,
    GroundingDetection,
    GroundingResult,
)

from .services.keyframe_repository import keyframe_repository


# Cấu hình logging cơ bản cho ứng dụng
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Khai báo biến global cho retriever engine
from typing import Optional

retriever: Optional[HybridRetriever] = None

# Add global temporal engine variables
temporal_engine: Optional[TemporalSearchEngine] = None
grounding_reranker: Optional[GroundingDINO] = None
yoloe_reranker: Optional[YOLOE26LReranker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng: khởi tạo tài nguyên khi bắt đầu và giải phóng khi kết thúc.
    """
    global retriever, temporal_engine, grounding_reranker, yoloe_reranker
    logger.info("--- Application Startup ---")

    # 1. Kết nối cơ sở dữ liệu
    await init_database()

    # 2. Khởi tạo và tải các mô hình AI
    retriever = HybridRetriever()
    await retriever.initialize()

    # 3. Khởi tạo temporal search engine
    temporal_engine = TemporalSearchEngine(retriever)

    # 4. Khởi tạo GroundingDINO reranker
    #grounding_reranker = GroundingDINO()

    # 5. Khởi tạo YOLOE-26L reranker cho query-driven class extraction
    try:
        yoloe_reranker = YOLOE26LReranker(device=settings.DEVICE)
        logger.info("✅ YOLOE-26L reranker initialized for natural language query parsing.")
    except Exception as exc:
        logger.warning(f"YOLOE-26L reranker initialization failed: {exc}")
        yoloe_reranker = None

    logger.info("✅ Application startup complete. Ready to accept requests.")

    yield  # Ứng dụng chạy ở đây

    # --- Application Shutdown ---
    logger.info("--- Application Shutdown ---")
    await close_database()
    logger.info("✅ Application shutdown complete.")


# Khởi tạo ứng dụng FastAPI với lifespan
app = FastAPI(
    title="Hybrid Video Retrieval API",
    description="Một API mạnh mẽ để tìm kiếm video đa phương thức, sử dụng MetaCLIP 2, BEiT-3, Co-DETR và cơ sở dữ liệu vector/text (Milvus, Elasticsearch).",
    version="1.1.0",
    lifespan=lifespan,
    contact={
        "name": "AI Team",
        "url": "https://example.com",
        "email": "ai-team@example.com",
    },
)

# Cấu hình CORS để cho phép truy cập từ các domain khác (ví dụ: frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, nên giới hạn lại: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === API Endpoints ===

@app.get("/", tags=["General"])
async def root():
    """Endpoint gốc để kiểm tra API có đang hoạt động hay không."""
    return {"message": "Welcome to the Hybrid Video Retrieval API. Visit /docs for interactive documentation."}


@app.get("/health", tags=["General"])
async def health_check():
    """Kiểm tra 'sức khỏe' toàn diện của hệ thống, bao gồm các kết nối DB và trạng thái của retriever."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever service is not available.")

    milvus_status = retriever.check_milvus_connection()
    es_status = retriever.check_elasticsearch_connection()

    is_healthy = milvus_status.get("status") == "connected"

    if not is_healthy:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "milvus": milvus_status,
                "elasticsearch": es_status,
            }
        )

    return {
        "status": "healthy",
        "milvus": milvus_status,
        "elasticsearch": es_status,
    }


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_videos(request: SearchRequest = Body(..., examples=search_examples)):
    """
    **Endpoint chính để thực hiện tìm kiếm video đa phương thức.**

    Cung cấp một truy vấn văn bản và các bộ lọc tùy chọn để tìm các keyframe video phù hợp.
    """
    if not retriever: raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        logger.info(f"Raw request data: {request}")
        logger.info(f"Request dict: {request.model_dump()}")
        logger.info(f"Received search request: query='{request.text_query}', mode='{request.mode.value}', num_query={request.num_query}")
        
        # DEBUG: Log object filters specifically
        if request.object_filters:
            logger.info(f"🔍 Object filters received: {request.object_filters}")
            for obj_name, filter_spec in request.object_filters.items():
                logger.info(f"  {obj_name}: {filter_spec}")
                if isinstance(filter_spec, dict):
                    logger.info(f"    Keys: {list(filter_spec.keys())}")
                    if 'constraints' in filter_spec:
                        logger.info(f"    Constraint count: {len(filter_spec['constraints'])}")
        else:
            logger.info("🔍 No object filters in request")
            
        results = await retriever.search(
            text_query=request.text_query,
            mode=request.mode.value,
            user_query=request.user_query,
            object_filters=request.object_filters,
            color_filters=request.color_filters,
            ocr_query=request.ocr_query,
            asr_query=request.asr_query,
            top_k=request.top_k,
            num_query=request.num_query,  # Sử dụng num_query từ request
        )
        return SearchResponse(query=request.text_query, mode=request.mode, results=results, total_results=len(results))
    except Exception as e:
        logger.error(f"Search failed for query '{request.text_query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during the search process.")


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


@app.post("/search/compare", tags=["Search"])
async def compare_search_modes(request: SearchRequest = Body(..., examples=compare_examples)):
    """
    **So sánh kết quả giữa các chế độ tìm kiếm (`hybrid`, `metaclip2`, `beit3`)** trên cùng một truy vấn.

    Rất hữu ích cho việc đánh giá và gỡ lỗi.
    """
    if not retriever: raise HTTPException(status_code=503, detail="Retriever not initialized")
    comparison_results = {}
    modes_to_compare = ["hybrid", "metaclip2", "beit3"]
    for mode in modes_to_compare:
        results = await retriever.search(
            text_query=request.text_query, mode=mode, user_query=request.user_query,
            object_filters=request.object_filters,
            color_filters=request.color_filters, ocr_query=request.ocr_query,
            asr_query=request.asr_query, top_k=request.top_k
        )
        comparison_results[mode] = {"results": results, "total_results": len(results)}
    return {"query": request.text_query, "comparison": comparison_results}


@app.get("/optimization/stats", tags=["Optimization"])
async def get_optimization_stats():
    """Get performance optimization statistics and cache metrics."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever service is not available.")

    stats = retriever.get_optimization_stats()
    return {
        "status": "optimized",
        "optimization_stats": stats,
        "message": "Performance optimizations active: caching, vectorization, precomputed embeddings"
    }


@app.post("/optimization/cache/clear", tags=["Optimization"])
async def clear_optimization_cache(cache_type: str = "all"):
    """Clear optimization caches (all, embeddings, search, objects, colors)."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever service is not available.")

    valid_types = ["all", "embeddings", "search", "objects", "colors"]
    if cache_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid cache_type. Must be one of: {valid_types}")

    retriever.clear_cache(cache_type)
    return {
        "status": "success",
        "message": f"Cache cleared: {cache_type}",
        "cache_type": cache_type
    }


@app.post("/search/temporal", response_model=TemporalSearchResponse, tags=["Search"])
async def temporal_search(request: TemporalSearchRequest = Body(..., examples=temporal_examples)):
    """
    **Tìm kiếm chuỗi hành động theo thứ tự thời gian (Temporal Sequential Search)**

    Thực hiện tìm kiếm các chuỗi hành động liên tiếp trong cùng một video theo thứ tự thời gian.
    Ví dụ: tìm chuỗi "đầu bếp cho cá vào tô" → "trộn bột" → "nhấc đũa ra khỏi dầu".
    """
    if not temporal_engine:
        raise HTTPException(status_code=503, detail="Temporal search engine not initialized")

    try:
        logger.info(f"Received temporal search request with {len(request.sequential_queries)} queries")
        result = await temporal_engine.temporal_search(request)
        return result

    except ValueError as e:
        logger.error(f"Temporal search validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Temporal search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during temporal search")



@app.post(
    "/search/rerank",
    response_model=RerankingSearchResponse,
    tags=["Reranking"],
)
async def rerank_search(
    request: RerankingSearchRequest,
):
    if not yoloe_reranker:
        raise HTTPException(
            status_code=503,
            detail="YOLOE-26L reranker is not initialized",
        )

    try:
        query_classes = []
        if hasattr(request, "classes") and request.classes:
            query_classes = request.classes
        else:
            query_classes = yoloe_reranker.extract_classes_from_query(request.query)

        logger.info(
            f"Received YOLOE reranking request "
            f"query='{request.query}' "
            f"classes={query_classes} "
            f"top_k={request.top_k} "
            f"frames={len(request.frames)}"
        )

        if not query_classes:
            raise HTTPException(
                status_code=400,
                detail="YOLOE classes cannot be empty",
            )

        items: list[dict[str, Any]] = []

        for frame in request.frames:

            # --------------------------------------------------
            # 1. Convert keyframe_id -> local image path
            # --------------------------------------------------

            try:
                image_path = (
                    keyframe_repository.get_image_path(
                        frame.keyframe_id
                    )
                )

            except (
                ValueError,
                FileNotFoundError,
            ) as e:

                logger.error(
                    f"Failed to resolve keyframe "
                    f"'{frame.keyframe_id}': {e}"
                )

                raise HTTPException(
                    status_code=404,
                    detail=str(e),
                )

            # --------------------------------------------------
            # 2. Load image
            # --------------------------------------------------

            try:
                pil_image = (
                    Image.open(
                        image_path
                    ).convert("RGB")
                )

            except Exception as e:

                logger.error(
                    f"Failed to open image "
                    f"'{image_path}': {e}"
                )

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unable to read keyframe "
                        f"'{frame.keyframe_id}'"
                    ),
                )

            # --------------------------------------------------
            # 3. Prepare YOLOE input
            # --------------------------------------------------

            items.append(
                {
                    "keyframe_id": (
                        frame.keyframe_id
                    ),
                    "image": pil_image,
                    "retrieval_score": (
                        frame.retrieval_score
                    ),
                }
            )

        # ------------------------------------------------------
        # 4. YOLOE-26L reranking
        # ------------------------------------------------------

        # query_classes = [
        #     "person",
        #     "motorcycle",
        #     "helmet",
        #     "construction barrier",
        #     "construction debris",
        # ]


        yoloe_results = (
            yoloe_reranker.rerank(
                query_classes=query_classes,
                images=items,
                top_k=request.top_k,
            )
        )

        # ------------------------------------------------------
        # 5. Convert internal result -> API response
        # ------------------------------------------------------

        results: list[
            GroundingResult
        ] = []

        for result in yoloe_results:

            results.append(
                GroundingResult(
                    keyframe_id=(
                        result.keyframe_id
                    ),
                    grounding_score=(
                        result.grounding_score
                    ),
                    retrieval_score=(
                        result.retrieval_score
                    ),
                    final_score=(
                        result.final_score
                    ),
                    detections=[
                        GroundingDetection(
                            label=det.label,
                            score=det.score,
                            bbox=det.bbox,
                        )
                        for det in result.detections
                    ],
                )
            )

        # ------------------------------------------------------
        # 6. Return response
        # ------------------------------------------------------

        return RerankingSearchResponse(
            query=request.query,
            total_candidates=len(
                request.frames
            ),
            returned_results=len(
                results
            ),
            results=results,
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.error(
            f"YOLOE reranking search failed: {e}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred "
                "during YOLOE reranking search."
            ),
        )

# @app.post(
#     "/search/rerank",
#     response_model=RerankingSearchResponse,
#     tags=["Reranking"],
# )
# async def rerank_search(
#     request: RerankingSearchRequest,
# ):
#     if not grounding_reranker:
#         raise HTTPException(
#             status_code=503,
#             detail="Grounding DINO reranker is not initialized",
#         )

#     try:
#         logger.info(
#             f"Received reranking request "
#             f"query='{request.query}' "
#             f"top_k={request.top_k} "
#             f"frames={len(request.frames)}"
#         )

#         items: list[dict[str, Any]] = []

#         for frame in request.frames:

#             # --------------------------------------------------
#             # 1. Convert keyframe_id -> local image path
#             # --------------------------------------------------
#             try:
#                 image_path = keyframe_repository.get_image_path(
#                     frame.keyframe_id
#                 )
#             except (ValueError, FileNotFoundError) as e:
#                 logger.error(
#                     f"Failed to resolve keyframe "
#                     f"'{frame.keyframe_id}': {e}"
#                 )

#                 raise HTTPException(
#                     status_code=404,
#                     detail=str(e),
#                 )

#             # --------------------------------------------------
#             # 2. Load image
#             # --------------------------------------------------
#             try:
#                 pil_image = Image.open(image_path).convert("RGB")
#             except Exception as e:
#                 logger.error(
#                     f"Failed to open image "
#                     f"'{image_path}': {e}"
#                 )

#                 raise HTTPException(
#                     status_code=400,
#                     detail=(
#                         f"Unable to read keyframe "
#                         f"'{frame.keyframe_id}'"
#                     ),
#                 )

#             # --------------------------------------------------
#             # 3. Prepare input for GroundingDINO
#             # --------------------------------------------------
#             items.append(
#                 {
#                     "keyframe_id": frame.keyframe_id,
#                     "image": pil_image,
#                     "retrieval_score": frame.retrieval_score,
#                 }
#             )

#         # ------------------------------------------------------
#         # 4. GroundingDINO reranking
#         # ------------------------------------------------------
#         grounding_results = grounding_reranker.rerank(
#             query=request.query,
#             images=items,
#             top_k=request.top_k,
#         )

#         # ------------------------------------------------------
#         # 5. Convert internal result -> API response
#         # ------------------------------------------------------
#         results: list[GroundingResult] = []

#         for result in grounding_results:

#             results.append(
#                 GroundingResult(
#                     keyframe_id=result.keyframe_id,
#                     grounding_score=result.grounding_score,
#                     retrieval_score=result.retrieval_score,
#                     final_score=result.final_score,
#                     detections=[
#                         GroundingDetection(
#                             label=det.label,
#                             score=det.score,
#                             bbox=det.bbox,
#                         )
#                         for det in result.detections
#                     ],
#                 )
#             )

#         # ------------------------------------------------------
#         # 6. Return response
#         # ------------------------------------------------------
#         return RerankingSearchResponse(
#             query=request.query,
#             total_candidates=len(request.frames),
#             returned_results=len(results),
#             results=results,
#         )

#     except HTTPException:
#         raise

#     except Exception as e:
#         logger.error(
#             f"Reranking search failed: {e}",
#             exc_info=True,
#         )

#         raise HTTPException(
#             status_code=500,
#             detail="An internal error occurred during reranking search.",
#         )

if __name__ == "__main__":
    # Chạy server Uvicorn khi thực thi file này trực tiếp
    # Hữu ích cho việc phát triển và gỡ lỗi cục bộ
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )
# --- END OF FILE app/main.py ---
