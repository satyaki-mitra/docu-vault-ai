# DEPENDENCIES
import numpy as np
from typing import List
from typing import Optional
from config.models import ChunkWithScore
from config.models import DocumentChunk
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from embeddings.bge_embedder import get_embedder
from utils.error_handler import VectorSearchError
from vector_store.faiss_manager import get_faiss_manager
from vector_store.metadata_store import get_metadata_store


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class VectorSearch:
    """
    FAISS-based vector similarity search: Uses existing FAISSManager from vector_store module
    Performs semantic search using embedding similarity
    """
    def __init__(self):
        """
        Initialize vector search
        """
        self.logger         = logger
        self.faiss_manager  = get_faiss_manager()
        self.embedder       = get_embedder()
        self.metadata_store = get_metadata_store()
        
        # Search statistics
        self.search_count   = 0
        self.total_results  = 0
        
        self.logger.info("Initialized VectorSearch")
    

    @handle_errors(error_type = VectorSearchError, log_error = True, reraise = True)
    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> List[ChunkWithScore]:
        """
        Perform vector similarity search
        
        Arguments:
        ----------
            query     { str }   : Search query
            
            top_k     { int }   : Number of results to return
            
            min_score { float } : Minimum similarity score threshold
        
        Returns:
        --------
               { list }         : List of ChunkWithScore objects
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided to vector search")
            return []
        
        self.logger.debug(f"Performing vector search: '{query}' (top_k={top_k})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(text      = query, 
                                                       normalize = True,
                                                      )
            
            # Search FAISS index (returns List[Tuple[str, float]] = [(chunk_id, score), ...])
            faiss_results   = self.faiss_manager.search(query_embedding = query_embedding, 
                                                        top_k           = top_k,
                                                       )
            
            if not faiss_results:
                self.logger.info(f"No results found for query: '{query}'")
                return []
            
            # Convert to ChunkWithScore objects
            chunks_with_scores = list()
            
            for rank, (chunk_id, score) in enumerate(faiss_results, 1):
                # Filter by minimum score
                if (score < min_score):
                    continue
                
                # Get chunk metadata
                chunk_metadata = self.metadata_store.get_chunk_metadata(chunk_id)
                
                if not chunk_metadata:
                    self.logger.warning(f"Chunk metadata not found for: {chunk_id}")
                    continue
                
                # Create DocumentChunk
                chunk = self._metadata_to_chunk(chunk_metadata)
                
                # Create ChunkWithScore
                cws   = ChunkWithScore(chunk            = chunk,
                                       score            = score,
                                       rank             = rank,
                                       retrieval_method = 'vector',
                                      )
                
                chunks_with_scores.append(cws)
            
            # Update statistics
            self.search_count  += 1
            self.total_results += len(chunks_with_scores)
            
            self.logger.info(f"Vector search returned {len(chunks_with_scores)} results")
            
            return chunks_with_scores
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {repr(e)}")
            raise VectorSearchError(f"Vector search failed: {repr(e)}")
    

    def search_with_embedding(self, query_embedding: np.ndarray, top_k: int = 10, min_score: float = 0.0) -> List[ChunkWithScore]:
        """
        Search using pre-computed query embedding
        
        Arguments:
        ----------
            query_embedding { np.ndarray } : Query embedding vector
            
            top_k           { int }        : Number of results
            
            min_score       { float }      : Minimum score threshold
        
        Returns:
        --------
                     { list }              : List of ChunkWithScore objects
        """
        self.logger.debug(f"Performing vector search with pre-computed embedding (top_k={top_k})")
        
        try:
            # Search FAISS index
            faiss_results      = self.faiss_manager.search(query_embedding=query_embedding, top_k=top_k)
            
            # Convert to ChunkWithScore objects
            chunks_with_scores = list()
            
            for rank, (chunk_id, score) in enumerate(faiss_results, 1):
                if (score < min_score):
                    continue
                
                chunk_metadata = self.metadata_store.get_chunk_metadata(chunk_id)
                
                if not chunk_metadata:
                    continue
                
                chunk = self._metadata_to_chunk(chunk_metadata)
                cws   = ChunkWithScore(chunk            = chunk,
                                       score            = score,
                                       rank             = rank,
                                       retrieval_method = 'vector',
                                      )
                
                chunks_with_scores.append(cws)
            
            self.search_count  += 1
            self.total_results += len(chunks_with_scores)
            
            return chunks_with_scores
            
        except Exception as e:
            self.logger.error(f"Vector search with embedding failed: {repr(e)}")
            raise VectorSearchError(f"Vector search with embedding failed: {repr(e)}")
    

    def _metadata_to_chunk(self, metadata: dict) -> DocumentChunk:
        """
        Convert metadata dictionary to DocumentChunk object
        
        Arguments:
        ----------
            metadata { dict } : Chunk metadata from store
        
        Returns:
        --------
            { DocumentChunk } : DocumentChunk object
        """
        return DocumentChunk(chunk_id      = metadata['chunk_id'],
                             document_id   = metadata['document_id'],
                             text          = metadata['text'],
                             embedding     = metadata.get('embedding'),
                             chunk_index   = metadata['chunk_index'],
                             start_char    = metadata['start_char'],
                             end_char      = metadata['end_char'],
                             page_number   = metadata.get('page_number'),
                             section_title = metadata.get('section_title'),
                             token_count   = metadata['token_count'],
                             metadata      = metadata.get('metadata', {}),
                            )
    

    def search_with_filters(self, query: str, top_k: int = 10, document_ids: Optional[List[str]] = None, 
                           min_score: float = 0.0) -> List[ChunkWithScore]:
        """
        Search with document filters
        
        Arguments:
        ----------
            query        { str }   : Search query
            
            top_k        { int }   : Number of results
            
            document_ids { list }  : Filter by specific documents
            
            min_score    { float } : Minimum score threshold
        
        Returns:
        --------
                  { list }        : Filtered ChunkWithScore objects
        """
        # Get more results for filtering
        results = self.search(query     = query, 
                              top_k     = top_k * 2, 
                              min_score = min_score,
                             )
        
        # Filter by document IDs if provided
        if document_ids:
            results = [r for r in results if r.chunk.document_id in document_ids]
        
        # Return top_k after filtering
        return results[:top_k]
    

    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[ChunkWithScore]]:
        """
        Perform batch vector search for multiple queries
        
        Arguments:
        ----------
            queries { list } : List of query strings
            
            top_k   { int }  : Number of results per query
        
        Returns:
        --------
                 { list }    : List of result lists
        """
        self.logger.info(f"Performing batch vector search for {len(queries)} queries")
        
        results = list()
        
        for query in queries:
            query_results = self.search(query, top_k)
            
            results.append(query_results)
        
        return results
    

    def get_search_statistics(self) -> dict:
        """
        Get vector search statistics
        
        Returns:
        --------
            { dict }    : Search statistics
        """
        avg_results = (self.total_results / self.search_count) if (self.search_count > 0) else 0
        
        return {"search_count"           : self.search_count,
                "total_results"          : self.total_results,
                "avg_results_per_query"  : avg_results,
                "faiss_index_stats"      : self.faiss_manager.get_index_stats(),
                "embedding_model"        : self.embedder.model_name,
                "embedding_dimension"    : self.embedder.embedding_dim,
               }


# Global vector search instance
_vector_search = None


def get_vector_search() -> VectorSearch:
    """
    Get global vector search instance
    
    Returns:
    --------
        { VectorSearch } : VectorSearch instance
    """
    global _vector_search

    if _vector_search is None:
        _vector_search = VectorSearch()
    
    return _vector_search


def search_vectors(query: str, top_k: int = 10, **kwargs) -> List[ChunkWithScore]:
    """
    Convenience function for vector search
    
    Arguments:
    ----------
        query  { str } : Search query

        top_k  { int } : Number of results
        
        **kwargs       : Additional arguments
    
    Returns:
    --------
            { list }   : ChunkWithScore results
    """
    searcher = get_vector_search()
    
    return searcher.search(query, top_k, **kwargs)