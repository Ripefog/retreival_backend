# --- START OF FILE app/database.py ---

import logging
from typing import Dict, Any, Optional
from pymilvus import connections, Collection, utility
from opensearchpy import OpenSearch, NotFoundError
from .config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Quản lý tập trung các kết nối đến Milvus và Elasticsearch.
    Được thiết kế để là một singleton instance trong ứng dụng.
    """
    def __init__(self):
        self.milvus_connected: bool = False
        self.elasticsearch_connected: bool = False
        self.collections: Dict[str, Collection] = {}
        self.es_client: Optional[OpenSearch] = None
    
    async def connect_all(self):
        """Kết nối đến tất cả các cơ sở dữ liệu được cấu hình."""
        logger.info("Establishing database connections...")
        await self.connect_milvus()
        await self.connect_elasticsearch()

        if not self.milvus_connected or not self.elasticsearch_connected:
            raise RuntimeError("Failed to establish one or more database connections.")
        
        logger.info("✅ All database connections established successfully.")

    async def connect_milvus(self):
        """Thiết lập kết nối đến máy chủ Milvus."""
        connections.disconnect("default")
        try:
            params = settings.get_milvus_connection_params()
            logger.info(f"Connecting to Milvus at {params['host']}:{params['port']} (User: {params.get('user', 'N/A')})...")
            # logger.info(f"Connecting to Milvus at cloud")

            connections.connect(**params)

            if connections.has_connection(settings.MILVUS_ALIAS):
                self.milvus_connected = True
                logger.info("✅ Milvus connected.")
                await self._load_milvus_collections()
            else:
                logger.error("❌ Milvus connection could not be established.")
                self.milvus_connected = False

        except Exception as e:
            logger.error(f"❌ Milvus connection failed with an exception: {e}", exc_info=True)
            self.milvus_connected = False


    async def connect_elasticsearch(self):
        """Thiết lập kết nối đến máy chủ Elasticsearch/OpenSearch."""
        try:
            params = settings.get_elasticsearch_connection_params()
            logger.info(f"Connecting to Elasticsearch at {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT} (User: {settings.ELASTICSEARCH_USERNAME})...")

            self.es_client = OpenSearch(**params)
            
            if self.es_client.ping():
                self.elasticsearch_connected = True
                logger.info("✅ Elasticsearch connected.")
            else:
                logger.error("❌ Elasticsearch connection ping failed.")
                self.elasticsearch_connected = False
                
        except Exception as e:
            logger.error(f"❌ Elasticsearch connection failed with an exception: {e}", exc_info=True)
            self.elasticsearch_connected = False

    
    async def _load_milvus_collections(self):
        """Tải các collection từ Milvus vào bộ nhớ và chuẩn bị chúng cho việc tìm kiếm."""
        logger.info("Loading Milvus collections...")
        try:
            collection_names = [
                settings.CLIP_COLLECTION,
                settings.BEIT3_COLLECTION,
                settings.OBJECT_COLLECTION,
            ]
            
            for name in collection_names:
                if utility.has_collection(name, using=settings.MILVUS_ALIAS):
                    collection = Collection(name, using=settings.MILVUS_ALIAS)
                    collection.load()
                    self.collections[name] = collection
                    logger.info(f"  - Loaded and prepared collection: '{name}' ({collection.num_entities} entities)")
                else:
                    logger.warning(f"  - ⚠️ Collection not found in Milvus: '{name}'")
                    
        except Exception as e:
            logger.error(f"Failed to load Milvus collections: {e}", exc_info=True)
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """Lấy một collection đã được tải theo tên."""
        return self.collections.get(collection_name)
    
    def check_milvus_connection(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái kết nối Milvus và thông tin các collection."""
        if not self.milvus_connected: return {"status": "disconnected"}
        try:
            if connections.has_connection(settings.MILVUS_ALIAS):
                return {
                    "status": "connected",
                    "host": f"{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                    "collections": {
                        name: {"num_entities": col.num_entities}
                        for name, col in self.collections.items()
                    }
                }
            return {"status": "disconnected", "error": "Connection lost"}
        except Exception as e: return {"status": "error", "detail": str(e)}
    
    def check_elasticsearch_connection(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái kết nối Elasticsearch và thông tin các index."""
        if not self.elasticsearch_connected or not self.es_client: return {"status": "disconnected"}
        try:
            if self.es_client.ping():
                indices = [settings.OCR_INDEX, settings.ASR_INDEX, settings.METADATA_INDEX]
                indices_info = {}
                for index in indices:
                    try:
                        exists = self.es_client.indices.exists(index=index)
                        doc_count = self.es_client.count(index=index)['count'] if exists else 0
                        indices_info[index] = {"exists": exists, "doc_count": doc_count}
                    except Exception: indices_info[index] = {"exists": False, "doc_count": 0}
                return {
                    "status": "connected",
                    "host": f"{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}",
                    "indices": indices_info
                }
            return {"status": "disconnected", "error": "Ping failed"}
        except Exception as e: return {"status": "error", "detail": str(e)}
    
    async def disconnect_all(self):
        """Đóng tất cả các kết nối cơ sở dữ liệu một cách an toàn."""
        logger.info("Closing database connections...")
        if self.milvus_connected:
            connections.disconnect(settings.MILVUS_ALIAS)
            self.milvus_connected = False
            logger.info("Disconnected from Milvus.")
        if self.elasticsearch_connected and self.es_client:
            self.es_client.close()
            self.elasticsearch_connected = False
            logger.info("Disconnected from Elasticsearch.")

# Tạo một instance duy nhất (singleton-like) để sử dụng trong toàn bộ ứng dụng
db_manager = DatabaseManager()

# Các hàm tiện ích để sử dụng trong `lifespan` của FastAPI
async def init_database():
    await db_manager.connect_all()

async def close_database():
    await db_manager.disconnect_all()
# --- END OF FILE app/database.py ---