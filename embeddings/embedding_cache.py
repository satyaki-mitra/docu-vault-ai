"""
Embedding Cache
In-memory caching for embeddings to avoid recomputation
"""

from typing import Optional, List, Dict
import hashlib
import time
from collections import OrderedDict
import numpy as np

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingCache:
    """
    LRU cache for embeddings.
    Stores embeddings with text hash as key for fast retrieval.
    """
    
    def __init__(
        self,
        max_size: Optional[int] = None,
        ttl_seconds: Optional[int] = None
    ):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            ttl_seconds: Time-to-live for cache entries (None = no expiry)
        """
        self.max_size = max_size or settings.CACHE_MAX_SIZE
        self.ttl_seconds = ttl_seconds or settings.CACHE_TTL
        
        # OrderedDict for LRU behavior
        self._cache: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        self.logger = logger
        self.logger.info(
            f"EmbeddingCache initialized: max_size={self.max_size}, "
            f"ttl={self.ttl_seconds}s"
        )
    
    def _hash_text(self, text: str) -> str:
        """
        Generate hash for text.
        
        Args:
            text: Input text
        
        Returns:
            Hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """
        Check if cache entry is expired.
        
        Args:
            timestamp: Entry timestamp
        
        Returns:
            True if expired
        """
        if self.ttl_seconds is None:
            return False
        
        return (time.time() - timestamp) > self.ttl_seconds
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to lookup
        
        Returns:
            Cached embedding or None
        """
        key = self._hash_text(text)
        
        if key in self._cache:
            embedding, timestamp = self._cache[key]
            
            # Check expiry
            if self._is_expired(timestamp):
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            
            return embedding
        
        self._misses += 1
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """
        Store embedding in cache.
        
        Args:
            text: Text key
            embedding: Embedding to cache
        """
        key = self._hash_text(text)
        
        # Remove if exists (for LRU reordering)
        if key in self._cache:
            del self._cache[key]
        
        # Add to cache
        self._cache[key] = (embedding, time.time())
        
        # Evict oldest if over size limit
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # Remove oldest
    
    def get_batch(self, texts: List[str]) -> tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Get multiple embeddings from cache.
        
        Args:
            texts: List of texts
        
        Returns:
            Tuple of (embeddings_or_none, missing_indices)
        """
        results = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text)
            results.append(embedding)
            
            if embedding is None:
                missing_indices.append(i)
        
        return results, missing_indices
    
    def set_batch(self, texts: List[str], embeddings: np.ndarray):
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of texts
            embeddings: Array of embeddings
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def invalidate(self, text: str) -> bool:
        """
        Remove specific entry from cache.
        
        Args:
            text: Text to invalidate
        
        Returns:
            True if entry was found and removed
        """
        key = self._hash_text(text)
        
        if key in self._cache:
            del self._cache[key]
            return True
        
        return False
    
    def clear(self):
        """Clear entire cache"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self.logger.info("Cache cleared")
    
    def get_statistics(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }
    
    def resize(self, new_max_size: int):
        """
        Resize cache.
        
        Args:
            new_max_size: New maximum size
        """
        self.max_size = new_max_size
        
        # Evict oldest entries if needed
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        
        self.logger.info(f"Cache resized to {new_max_size}")
    
    def prune_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        if self.ttl_seconds is None:
            return 0
        
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if self._is_expired(timestamp)
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self.logger.info(f"Pruned {len(expired_keys)} expired entries")
        
        return len(expired_keys)
    
    def get_memory_usage(self) -> dict:
        """
        Estimate memory usage.
        
        Returns:
            Memory usage dictionary
        """
        if not self._cache:
            return {
                "total_bytes": 0,
                "total_mb": 0,
                "avg_bytes_per_entry": 0,
            }
        
        # Sample first entry to estimate
        sample_embedding = next(iter(self._cache.values()))[0]
        bytes_per_embedding = sample_embedding.nbytes
        
        total_bytes = len(self._cache) * bytes_per_embedding
        total_mb = total_bytes / (1024 * 1024)
        
        return {
            "total_bytes": total_bytes,
            "total_mb": total_mb,
            "avg_bytes_per_entry": bytes_per_embedding,
            "num_entries": len(self._cache),
        }
    
    def __len__(self) -> int:
        """Get number of cached entries"""
        return len(self._cache)
    
    def __contains__(self, text: str) -> bool:
        """Check if text is in cache"""
        key = self._hash_text(text)
        return key in self._cache


class CachedEmbedder:
    """
    Wrapper that adds caching to an embedder.
    """
    
    def __init__(
        self,
        embedder,
        cache: Optional[EmbeddingCache] = None
    ):
        """
        Initialize cached embedder.
        
        Args:
            embedder: Base embedder (BGEEmbedder)
            cache: Cache instance (creates new if None)
        """
        self.embedder = embedder
        self.cache = cache or EmbeddingCache()
        self.logger = logger
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text with caching.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        # Check cache
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        
        # Generate embedding
        embedding = self.embedder.embed_text(text)
        
        # Store in cache
        self.cache.set(text, embedding)
        
        return embedding
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed multiple texts with caching.
        
        Args:
            texts: List of texts
            show_progress: Show progress bar
        
        Returns:
            Array of embeddings
        """
        # Check cache for all texts
        cached_results, missing_indices = self.cache.get_batch(texts)
        
        if not missing_indices:
            # All cached
            return np.array([emb for emb in cached_results])
        
        # Generate embeddings for missing texts
        missing_texts = [texts[i] for i in missing_indices]
        new_embeddings = self.embedder.embed_texts(
            missing_texts,
            show_progress=show_progress
        )
        
        # Update cache
        self.cache.set_batch(missing_texts, new_embeddings)
        
        # Combine cached and new embeddings
        final_embeddings = []
        new_idx = 0
        
        for i, cached in enumerate(cached_results):
            if cached is not None:
                final_embeddings.append(cached)
            else:
                final_embeddings.append(new_embeddings[new_idx])
                new_idx += 1
        
        return np.array(final_embeddings)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return self.cache.get_statistics()


# Global cache instance
_global_cache = None


def get_global_cache() -> EmbeddingCache:
    """
    Get global cache instance.
    
    Returns:
        EmbeddingCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCache()
    return _global_cache


if __name__ == "__main__":
    # Test embedding cache
    print("=== Embedding Cache Tests ===\n")
    
    cache = EmbeddingCache(max_size=10, ttl_seconds=None)
    
    # Test 1: Basic set/get
    print("Test 1: Basic cache operations")
    test_text = "This is a test sentence."
    test_embedding = np.random.rand(384)  # Fake embedding
    
    cache.set(test_text, test_embedding)
    retrieved = cache.get(test_text)
    
    print(f"  Set embedding: shape={test_embedding.shape}")
    print(f"  Retrieved: shape={retrieved.shape}")
    print(f"  Match: {np.allclose(test_embedding, retrieved)}")
    print()
    
    # Test 2: Cache hit/miss
    print("Test 2: Cache hit/miss")
    _ = cache.get(test_text)  # Hit
    _ = cache.get("Not in cache")  # Miss
    
    stats = cache.get_statistics()
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")
    print()
    
    # Test 3: Batch operations
    print("Test 3: Batch operations")
    texts = [f"Sentence {i}" for i in range(5)]
    embeddings = np.random.rand(5, 384)
    
    cache.set_batch(texts, embeddings)
    cached, missing = cache.get_batch(texts)
    
    print(f"  Cached {len(texts)} texts")
    print(f"  Retrieved: {len([c for c in cached if c is not None])}")
    print(f"  Missing: {len(missing)}")
    print()
    
    # Test 4: LRU eviction
    print("Test 4: LRU eviction")
    for i in range(15):
        cache.set(f"Text {i}", np.random.rand(384))
    
    stats = cache.get_statistics()
    print(f"  Added 15 items to cache with max_size=10")
    print(f"  Current size: {stats['size']}")
    print()
    
    # Test 5: Memory usage
    print("Test 5: Memory usage")
    memory = cache.get_memory_usage()
    print(f"  Total memory: {memory['total_mb']:.2f} MB")
    print(f"  Entries: {memory['num_entries']}")
    print(f"  Avg per entry: {memory['avg_bytes_per_entry']:,} bytes")
    print()
    
    # Test 6: Clear cache
    print("Test 6: Clear cache")
    cache.clear()
    stats = cache.get_statistics()
    print(f"  Size after clear: {stats['size']}")
    print()
    
    print("✓ Embedding cache module created successfully!")