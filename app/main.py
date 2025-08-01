# app/main.py

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager
import logging

from .config import settings
from .database import init_database, close_database
from .retrieval_engine import HybridRetriever
from .models import SearchRequest, SearchResponse

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- [MỚI] ĐỊNH NGHĨA CÁC VÍ DỤ ĐẦU VÀO CHO API ---
search_examples = {
    "simple_hybrid": {
        "summary": "1. Tìm kiếm Hybrid đơn giản",
        "description": "Một tìm kiếm cơ bản cho **'một người ngồi ở bàn'** sử dụng chế độ `hybrid` mặc định. Đây là trường hợp sử dụng phổ biến nhất.",
        "value": {
            "text_query": "a person sitting at a desk",
            "mode": "hybrid",
            "top_k": 5
        }
    },
    "complex_filters": {
        "summary": "2. Tìm kiếm với bộ lọc Object & Color",
        "description": "Tìm **'người đàn ông mặc đồ màu xanh'**. Hệ thống sẽ ưu tiên (tăng điểm) cho các kết quả có chứa đối tượng 'person', 'man' và màu 'blue'.",
        "value": {
            "text_query": "a man wearing something blue",
            "mode": "hybrid",
            "object_filters": ["person", "man"],
            "color_filters": ["blue"],
            "top_k": 5
        }
    },
    "full_filters": {
        "summary": "3. Tìm kiếm phức hợp với tất cả bộ lọc",
        "description": "Một truy vấn phức hợp để tìm **'một người dẫn chương trình trên sân khấu'**, kết hợp bộ lọc đối tượng, màu sắc, văn bản trong ảnh (OCR) và lời thoại (ASR).",
        "value": {
            "text_query": "a presenter on stage",
            "mode": "hybrid",
            "object_filters": ["person"],
            "color_filters": ["red"],
            "ocr_query": "VIỆT NAM",
            "asr_query": "kinh tế",
            "top_k": 5
        }
    },
    "vietnamese_query": {
        "summary": "4. Tìm kiếm bằng Tiếng Việt",
        "description": "Ví dụ về một truy vấn tìm kiếm bằng Tiếng Việt: **'nữ biên tập viên mặc áo hồng'**.",
        "value": {
            "text_query": "nữ biên tập viên mặc áo hồng đang dẫn chương trình thời sự",
            "mode": "hybrid",
            "color_filters": ["pink", "red"],
            "top_k": 5
        }
    },
    "clip_only_mode": {
        "summary": "5. Tìm kiếm chỉ bằng chế độ CLIP",
        "description": "Thực hiện tìm kiếm chỉ bằng mô hình CLIP, bỏ qua bước tinh chỉnh của BEiT-3. Thường nhanh hơn nhưng có thể kém chính xác hơn `hybrid`.",
        "value": {
            "text_query": "a news anchor in a studio",
            "mode": "clip",
            "top_k": 5
        }
    }
}


# Global variables
retriever: Optional[HybridRetriever] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global retriever
    logger.info("Initializing application startup...")
    await init_database()
    
    logger.info("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()
    await retriever.initialize()
    logger.info("✅ Backend startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down backend...")
    await close_database()
    logger.info("✅ Backend shutdown complete.")


# Tạo FastAPI app
app = FastAPI(
    title="Video Retrieval Backend API",
    description="API cho hệ thống tìm kiếm video đa phương thức. Sử dụng các mô hình AI tiên tiến như CLIP và BEiT-3 để tìm kiếm theo ngữ nghĩa, đối tượng, màu sắc, OCR và ASR.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Video Retrieval Backend API is running. Visit /docs for documentation."}

@app.get("/health")
async def health_check():
    """Kiểm tra 'sức khỏe' của hệ thống, bao gồm kết nối đến Milvus và Elasticsearch."""
    try:
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        milvus_status = retriever.check_milvus_connection()
        es_status = retriever.check_elasticsearch_connection()
        
        if milvus_status.get('status') != 'connected' or es_status.get('status') != 'connected':
             raise HTTPException(status_code=503, detail="One or more database connections are down.")

        return {
            "status": "healthy",
            "milvus": milvus_status,
            "elasticsearch": es_status,
            "retriever": "initialized"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_videos(
    request: SearchRequest = Body(..., examples=search_examples)
):
    """
    **Endpoint chính để tìm kiếm video.**

    Cho phép tìm kiếm theo nhiều tiêu chí:
    - `text_query`: Câu truy vấn ngữ nghĩa (bắt buộc).
    - `mode`: Chế độ tìm kiếm ('hybrid', 'clip', 'beit3').
    - `object_filters`: Lọc và ưu tiên các đối tượng có trong ảnh.
    - `color_filters`: Lọc và ưu tiên các màu sắc có trong ảnh.
    - `ocr_query`: Lọc theo văn bản xuất hiện trong ảnh.
    - `asr_query`: Lọc theo lời thoại trong video.
    """
    try:
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        logger.info(f"Processing search request: {request.text_query}")
        
        results = retriever.search(
            text_query=request.text_query,
            mode=request.mode.value,
            object_filters=request.object_filters,
            color_filters=request.color_filters,
            ocr_query=request.ocr_query,
            asr_query=request.asr_query,
            top_k=request.top_k
        )
        
        return SearchResponse(
            query=request.text_query,
            mode=request.mode,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Search failed for query '{request.text_query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during search.")

@app.post("/search/compare", tags=["Search"])
async def compare_search_modes(
    request: SearchRequest = Body(..., examples=search_examples)
):
    """
    **So sánh kết quả tìm kiếm giữa các chế độ (`hybrid`, `clip`, `beit3`).**

    Endpoint này hữu ích để đánh giá hiệu quả của các mô hình khác nhau trên cùng một truy vấn.
    """
    try:
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        comparison_results = {}
        modes_to_compare = ["hybrid", "clip", "beit3"]
        
        for mode in modes_to_compare:
            mode_results = retriever.search(
                text_query=request.text_query,
                mode=mode,
                object_filters=request.object_filters,
                color_filters=request.color_filters,
                ocr_query=request.ocr_query,
                asr_query=request.asr_query,
                top_k=request.top_k
            )
            comparison_results[mode] = {
                "results": mode_results,
                "total_results": len(mode_results)
            }
        
        return {
            "query": request.text_query,
            "comparison": comparison_results
        }
        
    except Exception as e:
        logger.error(f"Comparison search failed for query '{request.text_query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during comparison search.")

@app.get("/search/modes", tags=["Information"])
async def get_search_modes():
    """Lấy danh sách các chế độ tìm kiếm có sẵn và mô tả của chúng."""
    return {
        "modes": ["hybrid", "clip", "beit3"],
        "descriptions": {
            "hybrid": "Đề xuất: Kết hợp CLIP (lọc rộng) và BEiT-3 (tinh chỉnh) cho kết quả cân bằng và chính xác nhất.",
            "clip": "Chỉ sử dụng mô hình CLIP. Nhanh và hiệu quả cho các truy vấn ngữ nghĩa chung.",
            "beit3": "Chỉ sử dụng mô hình BEiT-3. Tốt cho các truy vấn chi tiết và phức tạp nhưng có thể chậm hơn."
        }
    }

@app.get("/collections", tags=["Information"])
async def get_collections_info():
    """Lấy thông tin về các collection đã được tải trong Milvus (số lượng, schema)."""
    try:
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        collections_info = retriever.get_collections_info()
        return {"collections": collections_info}
        
    except Exception as e:
        logger.error(f"Failed to get collections info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get collections info.")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # Sửa lại đường dẫn để uvicorn có thể tìm thấy app
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )