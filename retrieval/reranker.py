# DEPENDENCIES
from typing import List
from typing import Optional
from config.models import ChunkWithScore
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import RerankingError
from sentence_transformers import CrossEncoder


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class Reranker:
    """
    Cross-encoder reranking for retrieval results: Provides more accurate relevance scoring using cross-encoder models
    Optionally enabled for improved accuracy at cost of latency
    """
    def __init__(self, model_name: Optional[str] = None, enable_reranking: Optional[bool] = None):
        """
        Initialize reranker
        
        Arguments:
        ----------
            model_name       { str }  : Cross-encoder model name
            
            enable_reranking { bool } : Whether reranking is enabled
        """
        self.logger           = logger
        self.model_name       = model_name or settings.RERANKER_MODEL
        self.enable_reranking = enable_reranking if (enable_reranking is not None) else settings.ENABLE_RERANKING
        self.model            = None
        
        # Statistics
        self.rerank_count     = 0
        
        # Load model if reranking is enabled
        if self.enable_reranking:
            self._load_model()
        
        self.logger.info(f"Initialized Reranker: enabled={self.enable_reranking}, model={self.model_name}")


    def _load_model(self):
        """
        Load cross-encoder model
        """
        try:
            self.logger.info(f"Loading cross-encoder model: {self.model_name}")
            
            self.model = CrossEncoder(self.model_name)
            
            self.logger.info("Cross-encoder model loaded successfully")
            
        except ImportError:
            self.logger.error("sentence-transformers not available for cross-encoder")
            
            self.model            = None
            self.enable_reranking = False
        
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {repr(e)}")
            
            self.model            = None
            self.enable_reranking = False


    @handle_errors(error_type = RerankingError, log_error = True, reraise = False)
    def rerank(self, query: str, chunks_with_scores: List[ChunkWithScore], top_k: Optional[int] = None) -> List[ChunkWithScore]:
        """
        Rerank retrieved chunks using cross-encoder
        
        Arguments:
        ----------
            query              { str }  : Original query
            
            chunks_with_scores { list } : Initial retrieval results
            
            top_k              { int }  : Number of top results to return (default: all)
        
        Returns:
        --------
                        { list }        : Reranked ChunkWithScore objects
        """
        if not self.enable_reranking or self.model is None:
            self.logger.debug("Reranking disabled, returning original results")
            return chunks_with_scores
        
        if not chunks_with_scores:
            return []
        
        if not query or not query.strip():
            self.logger.warning("Empty query provided for reranking")
            return chunks_with_scores
        
        self.logger.debug(f"Reranking {len(chunks_with_scores)} chunks")
        
        try:
            # Prepare query-document pairs
            pairs    = [(query, cws.chunk.text) for cws in chunks_with_scores]
            
            # Get cross-encoder scores
            scores   = self.model.predict(pairs)
            
            # Update scores and rerank
            reranked = list()
            
            for i, (cws, new_score) in enumerate(zip(chunks_with_scores, scores)):
                # Create new ChunkWithScore with updated score
                reranked_cws = ChunkWithScore(chunk            = cws.chunk,
                                              score            = float(new_score),
                                              rank             = i + 1,  # Will be updated after sorting
                                              retrieval_method = 'reranked',
                                             )

                reranked.append(reranked_cws)
            
            # Sort by new scores (descending)
            reranked.sort(key     = lambda x: x.score, 
                          reverse = True,
                         )
            
            # Update ranks
            for rank, cws in enumerate(reranked, 1):
                cws.rank = rank
            
            # Return top_k if specified
            if top_k:
                reranked = reranked[:top_k]
            
            self.rerank_count += 1
            
            self.logger.info(f"Reranked {len(reranked)} chunks using cross-encoder")
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {repr(e)}, returning original results")
            return chunks_with_scores


    def rerank_with_scores(self, query: str, texts: List[str]) -> List[tuple]:
        """
        Rerank texts and return with scores
        
        Arguments:
        ----------
            query { str }  : Query string
            
            texts { list } : List of text strings
        
        Returns:
        --------
            { list }      : List of (text, score) tuples sorted by score
        """
        if not self.enable_reranking or self.model is None:
            self.logger.warning("Reranking not available")
            return [(text, 0.0) for text in texts]
        
        try:
            # Prepare pairs
            pairs   = [(query, text) for text in texts]
            
            # Get scores
            scores  = self.model.predict(pairs)
            
            # Combine and sort
            results = list(zip(texts, scores))
            
            results.sort(key     = lambda x: x[1], 
                         reverse = True,
                        )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Reranking with scores failed: {repr(e)}")
            return [(text, 0.0) for text in texts]


    def get_reranker_stats(self) -> dict:
        """
        Get reranker statistics
        
        Returns:
        --------
            { dict }    : Reranker statistics
        """
        return {"enabled"       : self.enable_reranking,
                "model_name"    : self.model_name,
                "model_loaded"  : self.model is not None,
                "rerank_count"  : self.rerank_count,
               }


    def is_available(self) -> bool:
        """
        Check if reranking is available
        
        Returns:
        --------
            { bool }    : True if reranking is available
        """
        return self.enable_reranking and (self.model is not None)
    
    
# Global reranker instance
_reranker = None
    
    
def get_reranker() -> Reranker:
    """
    Get global reranker instance
    
    Returns:
    --------
        { Reranker } : Reranker instance
    """
    global _reranker

    if _reranker is None:
        _reranker = Reranker()

    return _reranker
    
    
def rerank_results(query: str, chunks_with_scores: List[ChunkWithScore], **kwargs) -> List[ChunkWithScore]:
    """
    Convenience function for reranking
    
    Arguments:
    ----------
        query              { str }  : Query string
        
        chunks_with_scores { list } : Results to rerank
        
        **kwargs                    : Additional arguments

    Returns:
    --------
                    { list }        : Reranked results
    """
    reranker = get_reranker()

    return reranker.rerank(query, chunks_with_scores, **kwargs)