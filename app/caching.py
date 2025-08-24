# --- START OF FILE app/caching.py ---

import redis
import pickle
import hashlib
import logging
import json
from functools import wraps
from typing import Any, Optional, Dict, List, Union
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Comprehensive caching system for embeddings, search results, and computed values.
    Supports both Redis (distributed) and in-memory (LRU) caching.
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, 
                 memory_cache_size=1000, enable_redis=True):
        self.memory_cache_size = memory_cache_size
        self.enable_redis = enable_redis
        
        # Initialize Redis connection
        self.redis_client = None
        if enable_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, 
                    port=redis_port, 
                    db=redis_db,
                    decode_responses=False  # Keep binary for pickle
                )
                # Test connection
                self.redis_client.ping()
                logger.info("✅ Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}. Using memory cache only.")
                self.redis_client = None
        
        # In-memory LRU caches for different data types
        self._embedding_cache = {}  # {text_hash: (embedding, timestamp)}
        self._search_cache = {}     # {query_hash: (results, timestamp)}
        self._object_cache = {}     # {object_name: embedding}
        self._color_cache = {}      # {color_tuple: lab_values}
        
        # Cache statistics
        self.stats = {
            'embedding_hits': 0,
            'embedding_misses': 0,
            'search_hits': 0,
            'search_misses': 0,
            'redis_hits': 0,
            'redis_misses': 0
        }
        
        # Cache TTL settings (seconds)
        self.ttl_settings = {
            'embeddings': 3600 * 24,     # 24 hours - stable
            'search_results': 3600,      # 1 hour - may change with data updates
            'object_embeddings': 3600 * 24 * 7,  # 1 week - very stable
            'color_conversions': 3600 * 24 * 30  # 1 month - never changes
        }
    
    def _generate_hash(self, data: Union[str, Dict, List]) -> str:
        """Generate consistent hash for cache keys"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_memory_cache(self, cache_dict: Dict, max_size: int):
        """Remove oldest entries when cache exceeds max size"""
        if len(cache_dict) > max_size:
            # Sort by timestamp and remove oldest
            sorted_items = sorted(cache_dict.items(), 
                                key=lambda x: x[1][1] if isinstance(x[1], tuple) else 0)
            # Remove oldest 20% when cache is full
            remove_count = len(cache_dict) - int(max_size * 0.8)
            for i in range(remove_count):
                if i < len(sorted_items):
                    del cache_dict[sorted_items[i][0]]
    
    def cache_embedding(self, text: str, embedding: np.ndarray, cache_type='clip'):
        """Cache text embedding with automatic cleanup"""
        cache_key = f"{cache_type}:{self._generate_hash(text)}"
        timestamp = datetime.now().timestamp()
        
        # Store in memory cache
        self._embedding_cache[cache_key] = (embedding.copy(), timestamp)
        self._cleanup_memory_cache(self._embedding_cache, self.memory_cache_size)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                serialized = pickle.dumps((embedding, timestamp))
                self.redis_client.setex(
                    cache_key, 
                    self.ttl_settings['embeddings'], 
                    serialized
                )
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
    
    def get_cached_embedding(self, text: str, cache_type='clip') -> Optional[np.ndarray]:
        """Retrieve cached embedding"""
        cache_key = f"{cache_type}:{self._generate_hash(text)}"
        
        # Check memory cache first (fastest)
        if cache_key in self._embedding_cache:
            embedding, timestamp = self._embedding_cache[cache_key]
            # Check if not expired
            if datetime.now().timestamp() - timestamp < self.ttl_settings['embeddings']:
                self.stats['embedding_hits'] += 1
                return embedding.copy()
            else:
                del self._embedding_cache[cache_key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    embedding, timestamp = pickle.loads(cached_data)
                    # Update memory cache
                    self._embedding_cache[cache_key] = (embedding.copy(), timestamp)
                    self.stats['redis_hits'] += 1
                    self.stats['embedding_hits'] += 1
                    return embedding.copy()
                else:
                    self.stats['redis_misses'] += 1
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
        
        self.stats['embedding_misses'] += 1
        return None
    
    def cache_search_results(self, query_params: Dict, results: List[Dict]):
        """Cache search results"""
        cache_key = f"search:{self._generate_hash(query_params)}"
        timestamp = datetime.now().timestamp()
        
        # Store in memory cache
        self._search_cache[cache_key] = (results.copy(), timestamp)
        self._cleanup_memory_cache(self._search_cache, self.memory_cache_size // 2)
        
        # Store in Redis
        if self.redis_client:
            try:
                serialized = pickle.dumps((results, timestamp))
                self.redis_client.setex(
                    cache_key,
                    self.ttl_settings['search_results'],
                    serialized
                )
            except Exception as e:
                logger.warning(f"Redis search cache write failed: {e}")
    
    def get_cached_search_results(self, query_params: Dict) -> Optional[List[Dict]]:
        """Retrieve cached search results"""
        cache_key = f"search:{self._generate_hash(query_params)}"
        
        # Check memory cache
        if cache_key in self._search_cache:
            results, timestamp = self._search_cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.ttl_settings['search_results']:
                self.stats['search_hits'] += 1
                return results.copy()
            else:
                del self._search_cache[cache_key]
        
        # Check Redis
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    results, timestamp = pickle.loads(cached_data)
                    self._search_cache[cache_key] = (results.copy(), timestamp)
                    self.stats['search_hits'] += 1
                    return results.copy()
            except Exception as e:
                logger.warning(f"Redis search cache read failed: {e}")
        
        self.stats['search_misses'] += 1
        return None
    
    def cache_object_embedding(self, object_name: str, embedding: np.ndarray):
        """Cache object embeddings (very stable, long TTL)"""
        cache_key = f"object:{object_name.lower()}"
        self._object_cache[cache_key] = embedding.copy()
        
        if self.redis_client:
            try:
                serialized = pickle.dumps(embedding)
                self.redis_client.setex(
                    cache_key,
                    self.ttl_settings['object_embeddings'],
                    serialized
                )
            except Exception as e:
                logger.warning(f"Redis object cache write failed: {e}")
    
    def get_cached_object_embedding(self, object_name: str) -> Optional[np.ndarray]:
        """Get cached object embedding"""
        cache_key = f"object:{object_name.lower()}"
        
        # Check memory first
        if cache_key in self._object_cache:
            return self._object_cache[cache_key].copy()
        
        # Check Redis
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    embedding = pickle.loads(cached_data)
                    self._object_cache[cache_key] = embedding.copy()
                    return embedding.copy()
            except Exception as e:
                logger.warning(f"Redis object cache read failed: {e}")
        
        return None
    
    def cache_color_conversion(self, rgb: tuple, lab: tuple):
        """Cache RGB to LAB color conversions"""
        cache_key = f"color:{rgb}"
        self._color_cache[cache_key] = lab
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.ttl_settings['color_conversions'],
                    pickle.dumps(lab)
                )
            except Exception as e:
                logger.warning(f"Redis color cache write failed: {e}")
    
    def get_cached_color_conversion(self, rgb: tuple) -> Optional[tuple]:
        """Get cached color conversion"""
        cache_key = f"color:{rgb}"
        
        if cache_key in self._color_cache:
            return self._color_cache[cache_key]
        
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    lab = pickle.loads(cached_data)
                    self._color_cache[cache_key] = lab
                    return lab
            except Exception as e:
                logger.warning(f"Redis color cache read failed: {e}")
        
        return None
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_embedding_requests = self.stats['embedding_hits'] + self.stats['embedding_misses']
        total_search_requests = self.stats['search_hits'] + self.stats['search_misses']
        
        return {
            'embedding_hit_rate': (
                self.stats['embedding_hits'] / max(total_embedding_requests, 1) * 100
            ),
            'search_hit_rate': (
                self.stats['search_hits'] / max(total_search_requests, 1) * 100
            ),
            'memory_cache_sizes': {
                'embeddings': len(self._embedding_cache),
                'search_results': len(self._search_cache),
                'object_embeddings': len(self._object_cache),
                'color_conversions': len(self._color_cache)
            },
            'redis_connected': self.redis_client is not None,
            **self.stats
        }
    
    def clear_cache(self, cache_type: str = 'all'):
        """Clear specified cache type"""
        if cache_type in ['all', 'embeddings']:
            self._embedding_cache.clear()
        if cache_type in ['all', 'search']:
            self._search_cache.clear()
        if cache_type in ['all', 'objects']:
            self._object_cache.clear()
        if cache_type in ['all', 'colors']:
            self._color_cache.clear()
        
        if self.redis_client and cache_type == 'all':
            try:
                self.redis_client.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning(f"Redis cache clear failed: {e}")


# Decorators for easy caching
def cached_embedding(cache_manager: CacheManager, cache_type: str = 'clip'):
    """Decorator to automatically cache embeddings"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, text: str, *args, **kwargs):
            # Try to get from cache first
            cached_result = cache_manager.get_cached_embedding(text, cache_type)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            result = func(self, text, *args, **kwargs)
            if isinstance(result, np.ndarray):
                cache_manager.cache_embedding(text, result, cache_type)
            
            return result
        return wrapper
    return decorator


def cached_search_results(cache_manager: CacheManager):
    """Decorator to automatically cache search results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Create cache key from all parameters
            cache_params = {
                'args': args,
                'kwargs': {k: v for k, v in kwargs.items() if k != 'self'}
            }
            
            # Try to get from cache
            cached_result = cache_manager.get_cached_search_results(cache_params)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            result = await func(self, *args, **kwargs)
            if isinstance(result, list):
                cache_manager.cache_search_results(cache_params, result)
            
            return result
        return wrapper
    return decorator

# --- END OF FILE app/caching.py ---