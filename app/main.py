# --- START OF FILE app/main.py ---

import os
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager
from pydantic import ValidationError
import uvicorn
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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

# Cấu hình logging cơ bản cho ứng dụng
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Khai báo biến global cho retriever engine
from typing import Optional

retriever: Optional[HybridRetriever] = None

# Add global temporal engine variable
temporal_engine: Optional[TemporalSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng: khởi tạo tài nguyên khi bắt đầu và giải phóng khi kết thúc.
    """
    global retriever, temporal_engine
    logger.info("--- Application Startup ---")

    # 1. Kết nối cơ sở dữ liệu
    await init_database()

    # 2. Khởi tạo và tải các mô hình AI
    retriever = HybridRetriever()
    await retriever.initialize()

    # 3. Khởi tạo temporal search engine
    temporal_engine = TemporalSearchEngine(retriever)

    logger.info("✅ Application startup complete. Ready to accept requests.")

    yield  # Ứng dụng chạy ở đây

    # --- Application Shutdown ---
    logger.info("--- Application Shutdown ---")
    await close_database()
    logger.info("✅ Application shutdown complete.")


# Khởi tạo ứng dụng FastAPI với lifespan
app = FastAPI(
    title="Hybrid Video Retrieval API",
    description="Một API mạnh mẽ để tìm kiếm video đa phương thức, sử dụng kết hợp các mô hình AI (CLIP, BEiT-3, Co-DETR) và cơ sở dữ liệu vector/text (Milvus, Elasticsearch).",
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

    is_healthy = (
            milvus_status.get("status") == "connected" and
            es_status.get("status") == "connected"
    )

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
        logger.info(f"Received search request: query='{request.text_query}', mode='{request.mode.value}'")
        results = await retriever.search(
            text_query=request.text_query,
            mode=request.mode.value,
            user_query=request.user_query,
            object_filters=request.object_filters,
            color_filters=request.color_filters,
            ocr_query=request.ocr_query,
            asr_query=request.asr_query,
            top_k=request.top_k,
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
    **So sánh kết quả giữa các chế độ tìm kiếm (`hybrid`, `clip`, `beit3`)** trên cùng một truy vấn.

    Rất hữu ích cho việc đánh giá và gỡ lỗi.
    """
    if not retriever: raise HTTPException(status_code=503, detail="Retriever not initialized")
    comparison_results = {}
    modes_to_compare = ["hybrid", "clip", "beit3"]
    for mode in modes_to_compare:
        results = retriever.search(
            text_query=request.text_query, mode=mode, object_filters=request.object_filters,
            color_filters=request.color_filters, ocr_query=request.ocr_query,
            asr_query=request.asr_query, top_k=request.top_k
        )
        comparison_results[mode] = {"results": results, "total_results": len(results)}
    return {"query": request.text_query, "comparison": comparison_results}


@app.post("/process/image-objects", response_model=ImageObjectsResponse, tags=["Processing"])
async def process_image_for_objects(file: UploadFile = File(..., description="File ảnh để phân tích.")):
    """
    **Tải lên một ảnh và nhận diện các đối tượng/màu sắc có trong đó.**

    Sử dụng mô hình Co-DETR. Endpoint này mô phỏng một bước trong pipeline đánh chỉ mục dữ liệu.
    """
    if not retriever or not retriever.object_detector:
        raise HTTPException(status_code=503, detail="Object Detector is not available.")

    # Lưu file tải lên vào một file tạm thời vì Co-DETR yêu cầu đường dẫn file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        logger.info(f"Processing image for objects: {file.filename}")
        objects, colors = retriever.detect_objects_in_image(tmp_path)
        return ImageObjectsResponse(objects=objects, colors=colors)
    finally:
        # Đảm bảo dọn dẹp file tạm sau khi xử lý xong
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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