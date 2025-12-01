# DEPENDENCIES
import numpy as np
from typing import List
from typing import Optional
from numpy.typing import NDArray
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.cache_manager import EmbeddingCache as BaseEmbeddingCache


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class EmbeddingCache:
    """
    Embedding cache with numpy array support and statistics: Wraps the base cache with embedding-specific features
    """
    def __init__(self, max_size: int = None, ttl: int = None):
        """
        Initialize embedding cache
        
        Arguments:
        ----------
            max_size { int } : Maximum cache size
            
            ttl      { int } : Time to live in seconds
        """
        self.logger               = logger
        self.max_size             = max_size or settings.CACHE_MAX_SIZE
        self.ttl                  = ttl or settings.CACHE_TTL
        
        # Initialize base cache
        self.base_cache           = BaseEmbeddingCache(max_size = self.max_size,
                                                       ttl      = self.ttl,
                                                      )
        
        # Enhanced statistics
        self.hits                 = 0
        self.misses               = 0
        self.embeddings_generated = 0
        
        self.logger.info(f"Initialized EmbeddingCache: max_size={self.max_size}, ttl={self.ttl}")
    

    def get_embedding(self, text: str) -> Optional[NDArray]:
        """
        Get embedding from cache
        
        Arguments:
        ----------
            text { str }   : Input text
        
        Returns:
        --------
               { NDArray } : Cached embedding or None
        """
        cached = self.base_cache.get_embedding(text)
        
        if cached is not None:
            self.hits += 1
            
            # Convert list back to numpy array
            return np.array(cached)
        
        else:
            self.misses += 1
            return None
    

    def set_embedding(self, text: str, embedding: NDArray):
        """
        Store embedding in cache
        
        Arguments:
        ----------
            text      { str }     : Input text

            embedding { NDArray } : Embedding vector
        """
        # Convert numpy array to list for serialization
        embedding_list             = embedding.tolist()
        
        self.base_cache.set_embedding(text, embedding_list)

        self.embeddings_generated += 1
    

    def batch_get_embeddings(self, texts: List[str]) -> tuple[List[Optional[NDArray]], List[str]]:
        """
        Get multiple embeddings from cache
        
        Arguments:
        ----------
            texts { list } : List of texts
        
        Returns:
        --------
             { tuple }     : Tuple of (cached_embeddings, missing_texts)
        """
        cached_embeddings = list()
        missing_texts     = list()
        
        for text in texts:
            embedding = self.get_embedding(text)
            
            if embedding is not None:
                cached_embeddings.append(embedding)
            
            else:
                missing_texts.append(text)
                cached_embeddings.append(None)
        
        return cached_embeddings, missing_texts
    

    def batch_set_embeddings(self, texts: List[str], embeddings: List[NDArray]):
        """
        Store multiple embeddings in cache
        
        Arguments:
        ----------
            texts      { list } : List of texts

            embeddings { list } : List of embedding vectors
        """
        if (len(texts) != len(embeddings)):
            raise ValueError("Texts and embeddings must have same length")
        
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, embedding)
    

    def get_cached_embeddings(self, texts: List[str], embed_function: callable, batch_size: Optional[int] = None) -> List[NDArray]:
        """
        Smart embedding getter: uses cache for existing, generates for missing
        
        Arguments:
        ----------
            texts          { list }     : List of texts

            embed_function { callable } : Function to generate embeddings for missing texts
            
            batch_size     { int }      : Batch size for generation
        
        Returns:
        --------
                       { list }         : List of embeddings
        """
        # Get cached embeddings
        cached_embeddings, missing_texts = self.batch_get_embeddings(texts = texts)
        
        if not missing_texts:
            self.logger.debug(f"All {len(texts)} embeddings found in cache")
            return cached_embeddings
        
        # Generate missing embeddings
        self.logger.info(f"Generating {len(missing_texts)} embeddings ({(len(missing_texts)/len(texts))*100:.1f}% cache miss)")
        
        missing_embeddings               = embed_function(missing_texts, batch_size = batch_size)
        
        # Store new embeddings in cache
        self.batch_set_embeddings(missing_texts, missing_embeddings)
        
        # Combine results
        result_embeddings = list()
        missing_idx       = 0
        
        for emb in cached_embeddings:
            if emb is not None:
                result_embeddings.append(emb)
            
            else:
                result_embeddings.append(missing_embeddings[missing_idx])
                missing_idx += 1
        
        return result_embeddings
    

    def clear(self):
        """
        Clear entire cache
        """
        self.base_cache.clear()
        self.hits                  = 0
        self.misses                = 0
        self.embeddings_generated  = 0
        
        self.logger.info("Cleared embedding cache")
    

    def get_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
        --------
            { dict }    : Statistics dictionary
        """
        base_stats     = self.base_cache.get_stats()
        
        total_requests = self.hits + self.misses
        hit_rate       = (self.hits / total_requests * 100) if (total_requests > 0) else 0
        
        stats = {**base_stats,
                 "hits"                  : self.hits,
                 "misses"                : self.misses,
                 "hit_rate_percentage"   : hit_rate,
                 "embeddings_generated"  : self.embeddings_generated,
                 "cache_size"            : self.base_cache.cache.size(),
                 "max_size"              : self.max_size,
                }
        
        return stats
    

    def save_to_file(self, file_path: str) -> bool:
        """
        Save cache to file
        
        Arguments:
        ----------
            file_path { str } : Path to save file
        
        Returns:
        --------
               { bool }       : True if successful
        """
        return self.base_cache.save_to_file(file_path)
    

    def load_from_file(self, file_path: str) -> bool:
        """
        Load cache from file
        
        Arguments:
        ----------
            file_path { str } : Path to load file
        
        Returns:
        --------
               { bool }       : True if successful
        """
        return self.base_cache.load_from_file(file_path)
    

    def warm_cache(self, texts: List[str], embed_function: callable, batch_size: Optional[int] = None):
        """
        Pre-populate cache with embeddings
        
        Arguments:
        ----------
            texts            { list }   : List of texts to warm cache with

            embed_function { callable } : Embedding generation function
            
            batch_size       { int }    : Batch size
        """
        # Check which texts are not in cache
        _, missing_texts = self.batch_get_embeddings(texts = texts)
        
        if not missing_texts:
            self.logger.info("Cache already warm for all texts")
            return
        
        self.logger.info(f"Warming cache with {len(missing_texts)} embeddings")
        
        # Generate and cache embeddings
        embeddings = embed_function(missing_texts, batch_size = batch_size)

        self.batch_set_embeddings(missing_texts, embeddings)
        
        self.logger.info(f"Cache warming complete: added {len(missing_texts)} embeddings")


# Global embedding cache instance
_embedding_cache = None


def get_embedding_cache() -> EmbeddingCache:
    """
    Get global embedding cache instance
    
    Returns:
    --------
        { EmbeddingCache } : EmbeddingCache instance
    """
    global _embedding_cache

    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    
    return _embedding_cache


def cache_embeddings(texts: List[str], embeddings: List[NDArray]):
    """
    Convenience function to cache embeddings
    
    Arguments:
    ----------
        texts      { list } : List of texts

        embeddings { list } : List of embeddings
    """
    cache = get_embedding_cache()

    cache.batch_set_embeddings(texts, embeddings)


def get_cached_embeddings(texts: List[str], embed_function: callable, **kwargs) -> List[NDArray]:
    """
    Convenience function to get cached embeddings
    
    Arguments:
    ----------
        texts          { list } : List of texts

        embed_function { callable } : Embedding function
        
        **kwargs                   : Additional arguments
    
    Returns:
    --------
                 { list }          : List of embeddings
    """
    cache = get_embedding_cache()

    return cache.get_cached_embeddings(texts, embed_function, **kwargs)