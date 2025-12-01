# DEPENDENCIES
import time
import pickle
from typing import Any
from typing import Dict
from typing import Optional
from pathlib import Path  
from collections import OrderedDict
from config.settings import get_settings
from config.logging_config import get_logger


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class LRUCache:
    """
    Least Recently Used cache with TTL support
    """
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size   = max_size
        self.ttl        = ttl
        self.cache      = OrderedDict()
        self.timestamps = {}

    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache
        """
        if key not in self.cache:
            return None
        
        # Check TTL
        if self._is_expired(key):
            self.delete(key)
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        return self.cache[key]
    

    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache
        """
        # Remove if exists
        if key in self.cache:
            self.cache.pop(key)
        
        # Evict if needed
        elif (len(self.cache) >= self.max_size):
            oldest_key = next(iter(self.cache))
            self.delete(oldest_key)
        
        # Add new item
        self.cache[key]      = value
        self.timestamps[key] = time.time()
    

    def delete(self, key: str) -> bool:
        """
        Delete item from cache
        """
        if key in self.cache:
            self.cache.pop(key)
            self.timestamps.pop(key, None)
            return True
        
        return False
    

    def clear(self) -> None:
        """
        Clear entire cache
        """
        self.cache.clear()
        self.timestamps.clear()
    

    def _is_expired(self, key: str) -> bool:
        """
        Check if item has expired
        """
        if key not in self.timestamps:
            return True
        
        return ((time.time() - self.timestamps[key]) > self.ttl)
    

    def size(self) -> int:
        """
        Get current cache size
        """
        return len(self.cache)
    

    def keys(self) -> list:
        """
        Get all cache keys
        """
        return list(self.cache.keys())


class EmbeddingCache:
    """
    Specialized cache for embeddings with serialization support
    """
    def __init__(self, max_size: int = 1000, ttl: int = 86400, auto_persist: bool = True, persist_path: Optional[str] = None): 
        self.cache        = LRUCache(max_size = max_size, 
                                     ttl      = ttl,
                                    )
        self.hits         = 0
        self.misses       = 0

        self.auto_persist = auto_persist
        self.persist_path = persist_path or "cache/embeddings.pkl"

        # Ensure cache directory exists
        cache_dir = Path(self.persist_path).parent
        cache_dir.mkdir(parents = True, exist_ok = True) 

        # Load cache on startup if exists
        if (auto_persist and Path(self.persist_path).exists()):
            self.load_from_file(self.persist_path)
    

    def get_embedding(self, text: str) -> Optional[list]:
        """
        Get embedding for text
        """
        key    = self._generate_key(text)
        result = self.cache.get(key)
        
        if result is not None:
            self.hits += 1

        else:
            self.misses += 1
        
        return result
    

    def set_embedding(self, text: str, embedding: list) -> None:
        """
        Set embedding for text
        """
        key = self._generate_key(text)

        self.cache.set(key, embedding)
    

    def _generate_key(self, text: str) -> str:
        """
        Generate cache key from text
        """
        return f"emb_{hash(text) & 0xFFFFFFFF}"
    

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        total    = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if (total > 0) else 0
        
        return {"hits"     : self.hits,
                "misses"   : self.misses,
                "hit_rate" : hit_rate,
                "size"     : self.cache.size(),
                "max_size" : self.cache.max_size,
               }
    

    def save_to_file(self, file_path: str) -> bool:
        """
        Save cache to file
        """
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents = True, exist_ok = True) 
            
            with open(file_path, 'wb') as f:
                pickle.dump({'cache'      : self.cache.cache,
                             'timestamps' : self.cache.timestamps,
                             'hits'       : self.hits,
                             'misses'     : self.misses,
                            }, 
                            f
                           )
            return True

        except Exception as e:
            logger.error(f"Failed to save cache: {repr(e)}")
            return False
    

    def load_from_file(self, file_path: str) -> bool:
        """
        Load cache from file
        """
        try:
            with open(file_path, 'rb') as f:
                data                  = pickle.load(f)
                self.cache.cache      = data['cache']
                self.cache.timestamps = data['timestamps']
                self.hits             = data.get('hits', 0)
                self.misses           = data.get('misses', 0)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load cache: {repr(e)}")
            return False

        
    def __del__(self):
        """
        Auto-save cache on destruction
        """
        if self.auto_persist:
            self.save_to_file(self.persist_path)


# Global cache instances
embedding_cache = EmbeddingCache(max_size = settings.CACHE_MAX_SIZE,
                                 ttl      = settings.CACHE_TTL,
                                )

# Convenience functions
def get_embedding_cache() -> EmbeddingCache:
    """
    Get global embedding cache instance
    """
    return embedding_cache


def clear_embedding_cache() -> None:
    """
    Clear embedding cache
    """
    embedding_cache.cache.clear()
    embedding_cache.hits   = 0
    embedding_cache.misses = 0