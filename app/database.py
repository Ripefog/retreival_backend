import asyncio
import logging
from typing import Dict, Any
from pymilvus import connections, Collection, utility
from opensearchpy import OpenSearch, NotFoundError
from .config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.milvus_connected = False
        self.elasticsearch_connected = False
        self.collections: Dict[str, Collection] = {}
        self.es_client: OpenSearch = None
    
    async def connect_milvus(self) -> bool:
        """Kết nối đến Milvus thông qua ngrok"""
        try:
            logger.info(f"Connecting to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
            
            # Kết nối Milvus
            connections.connect(
                alias=settings.MILVUS_ALIAS,
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            
            # Kiểm tra kết nối
            if connections.has_connection(settings.MILVUS_ALIAS):
                self.milvus_connected = True
                logger.info("✅ Milvus connected successfully")
                
                # Load các collection
                await self._load_milvus_collections()
                return True
            else:
                logger.error("❌ Failed to connect to Milvus")
                return False
                
        except Exception as e:
            logger.error(f"❌ Milvus connection failed: {e}")
            return False
    
    async def connect_elasticsearch(self) -> bool:
        """Kết nối đến Elasticsearch thông qua ngrok"""
        try:
            logger.info(f"Connecting to Elasticsearch at {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}")
            
            # Cấu hình kết nối
            hosts = [{'host': settings.ELASTICSEARCH_HOST, 'port': settings.ELASTICSEARCH_PORT}]
            
            # Authentication nếu có
            http_auth = None
            if settings.ELASTICSEARCH_USERNAME and settings.ELASTICSEARCH_PASSWORD:
                http_auth = (settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD)
            
            # Tạo client
            self.es_client = OpenSearch(
                hosts=hosts,
                http_auth=http_auth,
                use_ssl=settings.ELASTICSEARCH_USE_SSL,
                verify_certs=True,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
            )
            
            # Kiểm tra kết nối
            if self.es_client.ping():
                self.elasticsearch_connected = True
                logger.info("✅ Elasticsearch connected successfully")
                return True
            else:
                logger.error("❌ Failed to connect to Elasticsearch")
                return False
                
        except Exception as e:
            logger.error(f"❌ Elasticsearch connection failed: {e}")
            return False
    
    async def _load_milvus_collections(self):
        """Load các collection từ Milvus"""
        try:
            collection_names = [
                settings.CLIP_COLLECTION,
                settings.BEIT3_COLLECTION,
                settings.OBJECT_COLLECTION,
                settings.COLOR_COLLECTION
            ]
            
            for collection_name in collection_names:
                if utility.has_collection(collection_name):
                    collection = Collection(collection_name, using=settings.MILVUS_ALIAS)
                    collection.load()
                    self.collections[collection_name] = collection
                    logger.info(f"✅ Loaded collection: {collection_name}")
                else:
                    logger.warning(f"⚠️ Collection not found: {collection_name}")
                    
        except Exception as e:
            logger.error(f"Failed to load collections: {e}")
    
    def get_collection(self, collection_name: str) -> Collection:
        """Lấy collection theo tên"""
        return self.collections.get(collection_name)
    
    def get_all_collections(self) -> Dict[str, Collection]:
        """Lấy tất cả collections"""
        return self.collections
    
    def check_milvus_connection(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái kết nối Milvus"""
        try:
            if not self.milvus_connected:
                return {"status": "disconnected", "error": "Not connected"}
            
            # Kiểm tra kết nối
            if connections.has_connection(settings.MILVUS_ALIAS):
                # Lấy thông tin collections
                collections_info = {}
                for name, collection in self.collections.items():
                    collections_info[name] = {
                        "num_entities": collection.num_entities,
                        "schema": str(collection.schema)
                    }
                
                return {
                    "status": "connected",
                    "host": f"{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                    "collections": collections_info
                }
            else:
                return {"status": "disconnected", "error": "Connection lost"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_elasticsearch_connection(self) -> Dict[str, Any]:
        """Kiểm tra trạng thái kết nối Elasticsearch"""
        try:
            if not self.elasticsearch_connected or not self.es_client:
                return {"status": "disconnected", "error": "Not connected"}
            
            # Kiểm tra ping
            if self.es_client.ping():
                # Lấy thông tin cluster
                cluster_info = self.es_client.info()
                
                # Kiểm tra các index
                indices_info = {}
                for index_name in [settings.METADATA_INDEX, settings.OCR_INDEX, settings.ASR_INDEX]:
                    try:
                        index_stats = self.es_client.indices.stats(index=index_name)
                        indices_info[index_name] = {
                            "exists": True,
                            "doc_count": index_stats['indices'][index_name]['total']['docs']['count']
                        }
                    except NotFoundError:
                        indices_info[index_name] = {"exists": False}
                
                return {
                    "status": "connected",
                    "host": f"{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}",
                    "cluster": cluster_info.get('cluster_name', 'unknown'),
                    "indices": indices_info
                }
            else:
                return {"status": "disconnected", "error": "Ping failed"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def disconnect(self):
        """Đóng kết nối database"""
        try:
            # Đóng kết nối Milvus
            if self.milvus_connected:
                connections.disconnect(settings.MILVUS_ALIAS)
                self.milvus_connected = False
                logger.info("Disconnected from Milvus")
            
            # Đóng kết nối Elasticsearch
            if self.elasticsearch_connected and self.es_client:
                self.es_client.close()
                self.elasticsearch_connected = False
                logger.info("Disconnected from Elasticsearch")
                
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

# Global database manager instance
db_manager = DatabaseManager()

async def init_database():
    """Khởi tạo kết nối database"""
    logger.info("Initializing database connections...")
    
    # Kết nối Milvus
    milvus_success = await db_manager.connect_milvus()
    
    # Kết nối Elasticsearch
    es_success = await db_manager.connect_elasticsearch()
    
    if not milvus_success:
        logger.error("Failed to connect to Milvus")
        raise Exception("Milvus connection failed")
    
    if not es_success:
        logger.error("Failed to connect to Elasticsearch")
        raise Exception("Elasticsearch connection failed")
    
    logger.info("✅ All database connections established")

async def close_database():
    """Đóng kết nối database"""
    await db_manager.disconnect() 