# DEPENDENCIES
import time
from typing import Dict
from typing import List
from typing import Optional
from config.settings import get_settings
from config.models import ChunkWithScore
from config.models import RetrievalRequest
from config.models import RetrievalResponse
from retrieval.reranker import get_reranker
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import HybridRetrievalError
from retrieval.vector_search import get_vector_search
from retrieval.keyword_search import get_keyword_search
from retrieval.context_assembler import get_context_assembler


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining vector and keyword search: Implements Reciprocal Rank Fusion (RRF) for optimal results
    Combines semantic understanding (vector) with exact matching (BM25)
    """
    def __init__(self, vector_weight: Optional[float] = None, bm25_weight: Optional[float] = None):
        """
        Initialize hybrid retriever
        
        Arguments:
        ----------
            vector_weight { float } : Weight for vector search (default from settings)
            
            bm25_weight   { float } : Weight for BM25 search (default from settings)
        """
        self.logger            = logger
        self.vector_weight     = vector_weight or settings.VECTOR_WEIGHT
        self.bm25_weight       = bm25_weight or settings.BM25_WEIGHT
        
        # Validate weights sum to 1.0
        if (abs(self.vector_weight + self.bm25_weight - 1.0) > 0.01):
            self.logger.warning(f"Weights don't sum to 1.0: vector={self.vector_weight}, bm25={self.bm25_weight}")
        
        # Initialize components
        self.vector_search     = get_vector_search()
        self.keyword_search    = get_keyword_search()
        self.reranker          = get_reranker()
        self.context_assembler = get_context_assembler()
        
        # Statistics
        self.retrieval_count   = 0
        
        self.logger.info(f"Initialized HybridRetriever: vector_weight={self.vector_weight}, bm25_weight={self.bm25_weight}")


    @handle_errors(error_type = HybridRetrievalError, log_error = True, reraise = True)
    def retrieve(self, query: str, top_k: int = 10, enable_reranking: Optional[bool] = None, use_vector: bool = True, use_bm25: bool = True, min_score: float = 0.0) -> List[ChunkWithScore]:
        """
        Perform hybrid retrieval combining vector and keyword search
        
        Arguments:
        ----------
            query            { str }   : Search query
            
            top_k            { int }   : Number of results to return
            
            enable_reranking { bool }  : Enable reranking (default from settings)
            
            use_vector       { bool }  : Use vector search
            
            use_bm25         { bool }  : Use BM25 search
            
            min_score        { float } : Minimum score threshold
        
        Returns:
        --------
                    { list }        : List of ChunkWithScore objects
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided to hybrid retriever")
            return []
        
        self.logger.info(f"Performing hybrid retrieval: '{query}' (top_k={top_k})")
        
        enable_reranking = enable_reranking if (enable_reranking is not None) else settings.ENABLE_RERANKING
        
        # Retrieve more candidates for fusion
        candidate_k      = top_k * 2
        
        vector_results   = list()
        bm25_results     = list()
        
        # Vector search
        if use_vector:
            self.logger.debug("Performing vector search")
            vector_results = self.vector_search.search(query     = query, 
                                                       top_k     = candidate_k,
                                                       min_score = min_score,
                                                      )
        
        # Keyword search
        if use_bm25:
            self.logger.debug("Performing BM25 search")
            bm25_results = self.keyword_search.search(query     = query, 
                                                      top_k     = candidate_k, 
                                                      min_score = min_score,
                                                     )
        
        # If only one method is used, return those results
        if (not use_vector):
            combined_results = bm25_results
        
        elif (not use_bm25):
            combined_results = vector_results
        
        else:
            # Normalize scores before fusion
            vector_results_normalized = self._normalize_scores(vector_results, "vector")
            bm25_results_normalized   = self._normalize_scores(bm25_results, "bm25")
            
            self.logger.debug(f"Normalized scores: vector={len(vector_results_normalized)}, bm25={len(bm25_results_normalized)}")
            
            # Hybrid fusion using RRF
            combined_results = self._reciprocal_rank_fusion(vector_results = vector_results_normalized,
                                                            bm25_results   = bm25_results_normalized,
                                                            top_k          = top_k * 2,  # Get more for reranking
                                                           )
        
        # Rerank if enabled
        if enable_reranking and self.reranker.is_available():
            self.logger.debug("Reranking results")

            combined_results = self.reranker.rerank(query              = query,
                                                    chunks_with_scores = combined_results,
                                                    top_k              = top_k,
                                                   )
        else:
            # Just take top_k
            combined_results = combined_results[:top_k]
        
        self.retrieval_count += 1
        
        self.logger.info(f"Hybrid retrieval returned {len(combined_results)} results")
        
        return combined_results


    def _normalize_scores(self, results: List[ChunkWithScore], method: str) -> List[ChunkWithScore]:
        """
        Normalize scores to [0, 1] range for fair comparison
        
        Arguments:
        ----------
            results { list } : Results to normalize

            method  { str }  : Method name for logging ('vector' or 'bm25')
        
        Returns:
        --------
                { list }     : Normalized results
        """
        if not results:
            return []
        
        # Extract scores
        scores      = [r.score for r in results]
        
        if not scores:
            return results
        
        min_score   = min(scores)
        max_score   = max(scores)
        score_range = max_score - min_score
        
        # Avoid division by zero
        if (score_range < 1e-6):
            self.logger.debug(f"{method}: All scores equal, setting to 0.5")
            normalized = []
            
            for r in results:
                normalized.append(ChunkWithScore(chunk            = r.chunk,
                                                 score            = 0.5,
                                                 rank             = r.rank,
                                                 retrieval_method = r.retrieval_method,
                                                )
                                 )

            return normalized
        
        # Min-max normalization to [0, 1]
        normalized = list()

        for r in results:
            normalized_score = (r.score - min_score) / score_range
            normalized.append(ChunkWithScore(chunk            = r.chunk,
                                             score            = normalized_score,
                                             rank             = r.rank,
                                             retrieval_method = r.retrieval_method,
                                            )
                             )
        
        self.logger.debug(f"{method} scores normalized: [{min_score:.3f}, {max_score:.3f}] → [0.0, 1.0]")
        
        return normalized


    def _reciprocal_rank_fusion(self, vector_results: List[ChunkWithScore], bm25_results: List[ChunkWithScore], top_k: int = 20, k: int = 60) -> List[ChunkWithScore]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)
        
        Arguments:
        ----------
            vector_results { list } : Vector search results (normalized)
            
            bm25_results   { list } : BM25 search results (normalized)
            
            top_k          { int }  : Number of results to return
            
            k              { int }  : RRF constant (default: 60)
        
        Returns:
        --------
                    { list }        : Fused ChunkWithScore objects
        """
        if not vector_results and not bm25_results:
            self.logger.warning("RRF called with no results from either method")
            return []
        
        # RRF formula: score(d) = Σ (weight / (k + rank_i(d)))
        rrf_scores = dict()
        chunk_map  = dict()
        
        # Track contribution from each method
        vector_contributions = 0
        bm25_contributions   = 0
        
        # Process vector results
        for rank, cws in enumerate(vector_results, 1):
            chunk_id              = cws.chunk.chunk_id
            contribution          = self.vector_weight / (k + rank)
            rrf_scores[chunk_id]  = rrf_scores.get(chunk_id, 0) + contribution
            chunk_map[chunk_id]   = cws.chunk
            vector_contributions += contribution
        
        # Process BM25 results
        for rank, cws in enumerate(bm25_results, 1):
            chunk_id             = cws.chunk.chunk_id
            contribution         = self.bm25_weight / (k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + contribution
            chunk_map[chunk_id]  = cws.chunk
            bm25_contributions  += contribution
        
        # Log fusion statistics
        self.logger.debug(f"RRF Fusion Statistics:")
        self.logger.debug(f"- Vector results: {len(vector_results)}, total contribution: {vector_contributions:.3f}")
        self.logger.debug(f"- BM25 results: {len(bm25_results)}, total contribution: {bm25_contributions:.3f}")
        self.logger.debug(f"- Unique chunks: {len(rrf_scores)}")
        
        # Sort by RRF score (descending)
        sorted_chunks = sorted(rrf_scores.items(), key = lambda x: x[1], reverse = True)
        
        if not sorted_chunks:
            self.logger.warning("RRF produced no results")
            return []
        
        # Normalize RRF scores to [0, 1] for consistency
        max_rrf_score = sorted_chunks[0][1] if sorted_chunks else 1.0
        
        # Create ChunkWithScore objects
        fused_results = list()
        
        for rank, (chunk_id, score) in enumerate(sorted_chunks[:top_k], 1):
            # Normalize score
            normalized_score  = score / max_rrf_score if (max_rrf_score > 0) else score
            
            chunk_with_scores = ChunkWithScore(chunk            = chunk_map[chunk_id],
                                               score            = normalized_score,
                                               rank             = rank,
                                               retrieval_method = 'hybrid_rrf',
                                             )
                                             
            fused_results.append(chunk_with_scores)
        
        self.logger.debug(f"RRF fusion produced {len(fused_results)} results (top score: {fused_results[0].score:.3f})")
        
        return fused_results


    def retrieve_with_context(self, query: str, top_k: int = 5, enable_reranking: Optional[bool] = None, include_citations: bool = True) -> dict:
        """
        Retrieve and assemble context ready for LLM
        
        Arguments:
        ----------
            query             { str }  : Search query
            
            top_k             { int }  : Number of results to retrieve
            
            enable_reranking  { bool } : Enable reranking
            
            include_citations { bool } : Include citations in context
        
        Returns:
        --------
                    { dict }        : Dictionary with context and metadata
        """
        start_time = time.time()
        
        # Retrieve chunks
        chunks     = self.retrieve(query              = query,
                                top_k              = top_k,
                                enable_reranking   = enable_reranking,
                                )
        
        if not chunks:
            return {"context"          : "",
                    "chunks"           : [],
                    "retrieval_time"   : time.time() - start_time,
                    "num_chunks"       : 0,
                   }
        
        # Assemble context
        context        = self.context_assembler.assemble_context(chunks            = chunks,
                                                                 query             = query,
                                                                 include_citations = include_citations,
                                                                )
        
        # Get statistics
        stats          = self.context_assembler.get_context_statistics(context, chunks)
        
        retrieval_time = time.time() - start_time
        
        return {"context"        : context,
                "chunks"         : chunks,
                "retrieval_time" : retrieval_time,
                "num_chunks"     : len(chunks),
                "stats"          : stats,
               }


    def process_retrieval_request(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Process a structured retrieval request
        
        Arguments:
        ----------
            request { RetrievalRequest } : Retrieval request object
        
        Returns:
        --------
            { RetrievalResponse }        : Retrieval response object
        """
        start_time       = time.time()
        
        # Extract parameters
        query            = request.query
        top_k            = request.top_k or settings.TOP_K_RETRIEVE
        enable_reranking = request.enable_reranking if (request.enable_reranking is not None) else settings.ENABLE_RERANKING
        use_vector       = request.use_vector
        use_bm25         = request.use_bm25
        min_score        = request.min_score or 0.0
        
        # Perform retrieval
        chunks           = self.retrieve(query            = query,
                                         top_k            = top_k,
                                         enable_reranking = enable_reranking,
                                         use_vector       = use_vector,
                                         use_bm25         = use_bm25,
                                         min_score        = min_score,
                                        )
        
        # Convert to ms
        retrieval_time   = (time.time() - start_time) * 1000  
        
        # Create response
        response         = RetrievalResponse(chunks            = chunks,
                                             retrieval_time_ms = retrieval_time,
                                             num_candidates    = len(chunks),
                                            )
        
        return response


    def get_retrieval_stats(self) -> dict:
        """
        Get retrieval statistics
        
        Returns:
        --------
            { dict }    : Retrieval statistics
        """
        return {"retrieval_count"   : self.retrieval_count,
                "vector_weight"     : self.vector_weight,
                "bm25_weight"       : self.bm25_weight,
                "vector_stats"      : self.vector_search.get_search_statistics(),
                "keyword_stats"     : self.keyword_search.get_search_statistics(),
                "reranker_stats"    : self.reranker.get_reranker_stats(),
            }



# Global hybrid retriever instance
_hybrid_retriever = None

def get_hybrid_retriever() -> HybridRetriever:
    """
    Get global hybrid retriever instance
    
    Returns:
    --------
        { HybridRetriever } : HybridRetriever instance
    """
    global _hybrid_retriever

    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()

    return _hybrid_retriever


def retrieve_hybrid(query: str, top_k: int = 10, **kwargs) -> List[ChunkWithScore]:
    """
    Convenience function for hybrid retrieval
    
    Arguments:
    ----------
        query  { str } : Search query
        
        top_k  { int } : Number of results
        
        **kwargs       : Additional arguments

    Returns:
    --------
            { list }   : ChunkWithScore results
    """
    retriever = get_hybrid_retriever()

    return retriever.retrieve(query, top_k, **kwargs)