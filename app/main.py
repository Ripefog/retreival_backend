from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager
import logging

from .config import *
from .database import init_database
from .retrieval_engine import HybridRetriever
from .models import SearchRequest, SearchResponse

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
retriever: Optional[HybridRetriever] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global retriever
    logger.info("Initializing database connections...")
    await init_database()
    
    logger.info("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()
    await retriever.initialize()
    logger.info("Backend startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down backend...")

# Tạo FastAPI app
app = FastAPI(
    title="Video Retrieval Backend",
    description="Backend cho hệ thống tìm kiếm video đa phương thức",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể cấu hình cụ thể cho production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Video Retrieval Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        # Kiểm tra kết nối database
        milvus_status = retriever.check_milvus_connection()
        es_status = retriever.check_elasticsearch_connection()
        
        return {
            "status": "healthy",
            "milvus": milvus_status,
            "elasticsearch": es_status,
            "retriever": "initialized"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_videos(request: SearchRequest):
    """Tìm kiếm video với query text và các filter"""
    try:
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        logger.info(f"Processing search request: {request.text_query}")
        
        results = retriever.search(
            text_query=request.text_query,
            mode=request.mode,
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
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/modes")
async def get_search_modes():
    """Lấy danh sách các chế độ tìm kiếm có sẵn"""
    return {
        "modes": ["hybrid", "clip", "beit3"],
        "descriptions": {
            "hybrid": "Kết hợp CLIP và BEIT-3 cho kết quả tốt nhất",
            "clip": "Chỉ sử dụng CLIP model",
            "beit3": "Chỉ sử dụng BEIT-3 model"
        }
    }

@app.get("/collections")
async def get_collections():
    """Lấy thông tin về các collection trong Milvus"""
    try:
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        collections_info = retriever.get_collections_info()
        return collections_info
        
    except Exception as e:
        logger.error(f"Failed to get collections info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collections: {str(e)}")

@app.post("/search/compare")
async def compare_search_modes(request: SearchRequest):
    """So sánh kết quả tìm kiếm giữa các chế độ khác nhau"""
    try:
        if retriever is None:
            raise HTTPException(status_code=503, detail="Retriever not initialized")
        
        results = {}
        modes = ["hybrid", "clip", "beit3"]
        
        for mode in modes:
            mode_results = retriever.search(
                text_query=request.text_query,
                mode=mode,
                object_filters=request.object_filters,
                color_filters=request.color_filters,
                ocr_query=request.ocr_query,
                asr_query=request.asr_query,
                top_k=request.top_k
            )
            results[mode] = {
                "results": mode_results,
                "total_results": len(mode_results)
            }
        
        return {
            "query": request.text_query,
            "comparison": results
        }
        
    except Exception as e:
        logger.error(f"Comparison search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    ) 