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
    compare_examples
)
from typing import List, Dict, Any, Optional

# Cấu hình logging cơ bản cho ứng dụng
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Khai báo biến global cho retriever engine
retriever: Optional[HybridRetriever] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng: khởi tạo tài nguyên khi bắt đầu và giải phóng khi kết thúc.
    """
    global retriever
    logger.info("--- Application Startup ---")
    
    # 1. Kết nối cơ sở dữ liệu
    await init_database()
    
    # 2. Khởi tạo và tải các mô hình AI
    retriever = HybridRetriever()
    await retriever.initialize()
    
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# === General Endpoints ===

@app.get("/", tags=["General"])
async def root():
    """Endpoint gốc để kiểm tra API có đang hoạt động hay không."""
    return {"message": "Welcome to the Hybrid Video Retrieval API. Visit /docs for interactive documentation."}

@app.get("/health", tags=["General"])
async def health_check():
    """Kiểm tra 'sức khỏe' toàn diện của hệ thống, bao gồm các kết nối DB và trạng thái của retriever."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever service is not available.")
    
    milvus_status = db_manager.check_milvus_connection()
    es_status = db_manager.check_elasticsearch_connection()
    
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

# === Search Endpoints ===

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_videos(request: SearchRequest = Body(..., examples=search_examples)):
    """
    **Endpoint chính để thực hiện tìm kiếm video đa phương thức.**

    Cung cấp một truy vấn văn bản và các bộ lọc tùy chọn để tìm các keyframe video phù hợp.
    """
    if not retriever: raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        logger.info(f"Received search request: query='{request.text_query}', mode='{request.mode.value}'")
        results = retriever.search(
            text_query=request.text_query,
            mode=request.mode.value,
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

# === Processing Endpoints ===

@app.post("/process/image-objects", response_model=ImageObjectsResponse, tags=["Processing"])
async def process_image_for_objects(file: UploadFile = File(..., description="File ảnh để phân tích.")):
    """
    **Tải lên một ảnh và nhận diện các đối tượng/màu sắc có trong đó.**

    Sử dụng mô hình Co-DETR. Endpoint này mô phỏng một bước trong pipeline đánh chỉ mục dữ liệu.
    """
    if not retriever or not retriever.object_detector:
        raise HTTPException(status_code=503, detail="Object Detector is not available.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        logger.info(f"Processing image for objects: {file.filename}")
        objects, colors = retriever.detect_objects_in_image(tmp_path)
        return ImageObjectsResponse(objects=objects, colors=colors)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# === Admin Endpoints ===

@app.post("/admin/distribute-tasks", 
          tags=["Admin"], 
          summary="Phân chia video cho người dùng",
          response_description="Kết quả phân chia chi tiết bao gồm thống kê và danh sách gán cụ thể.")
async def distribute_tasks_to_workers(user_list_payload: Dict[str, List[str]] = Body(
    ..., 
    examples={
        "default": {
            "summary": "Phân chia cho 5 người dùng",
            "value": { "user_list": ["worker_alpha", "worker_beta", "worker_gamma", "worker_delta", "worker_epsilon"] }
        },
        "minimum_users": {
            "summary": "Phân chia cho 2 người dùng",
            "value": { "user_list": ["user_1", "user_2"] }
        }
    }
)):
    """
    **Phân chia tất cả video trong hệ thống cho một nhóm người dùng (workers).**

    API này thực hiện các công việc sau:
    1. Lấy danh sách tất cả các video ID duy nhất từ cơ sở dữ liệu.
    2. Nhận vào một danh sách các `user_id`.
    3. Tạo tất cả các cặp người dùng có thể có.
    4. Gán mỗi video cho một cặp người dùng, sử dụng thuật toán tham lam (greedy) để 
       đảm bảo khối lượng công việc của mỗi người dùng là cân bằng nhất có thể.

    - **Input**: Một JSON object chứa key `user_list` là một mảng các chuỗi.
    - **Output**: Một JSON object chứa:
        - `summary`: Tóm tắt kết quả phân phối.
        - `user_counts`: Thống kê số video mỗi người dùng được gán.
        - `assignments`: Chi tiết việc gán mỗi video ID cho cặp người dùng nào.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever service is not initialized.")
    
    user_list = user_list_payload.get("user_list")
    if not user_list or not isinstance(user_list, list):
         raise HTTPException(
            status_code=422, # Unprocessable Entity
            detail="Payload must be a JSON object with a key 'user_list' containing a list of strings."
        )

    try:
        logger.info(f"Received request to distribute tasks for users: {user_list}")
        result = retriever.distribute_videos_to_workers(user_list)
        return result
    except ValueError as e:
        # Lỗi do người dùng nhập vào (ví dụ: ít hơn 2 user)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Các lỗi hệ thống khác
        logger.error(f"An unexpected error occurred during task distribution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during the task distribution process.")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )
# --- END OF FILE app/main.py ---