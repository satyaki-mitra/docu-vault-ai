# DEPENDENCIES
from typing import List
from typing import Optional
from config.models import DocumentChunk
from config.settings import get_settings
from config.models import ChunkWithScore
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from vector_store.bm25_index import get_bm25_index
from utils.error_handler import KeywordSearchError
from vector_store.metadata_store import get_metadata_store


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class KeywordSearch:
    """
    BM25 keyword search wrapper: Uses existing BM25Index from vector_store module
    Retrieves chunks from MetadataStore and returns ChunkWithScore objects
    """
    def __init__(self):
        """
        Initialize keyword search
        """
        self.logger         = logger
        self.bm25_index     = get_bm25_index()
        self.metadata_store = get_metadata_store()
        
        # Search statistics
        self.search_count   = 0
        self.total_results  = 0
        
        self.logger.info("Initialized KeywordSearch")
    

    @handle_errors(error_type = KeywordSearchError, log_error = True, reraise = True)
    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> List[ChunkWithScore]:
        """
        Perform keyword search using BM25
        
        Arguments:
        ----------
            query     { str }   : Search query
            
            top_k     { int }   : Number of results to return
            
            min_score { float } : Minimum BM25 score threshold
        
        Returns:
        --------
               { list }         : List of ChunkWithScore objects
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided to keyword search")
            return []
        
        self.logger.debug(f"Performing keyword search: '{query}' (top_k={top_k})")
        
        try:
            # Search BM25 index (returns List[Tuple[str, float]] = [(chunk_id, score), ...])
            bm25_results = self.bm25_index.search(query = query, 
                                                  top_k = top_k,
                                                 )
            
            if not bm25_results:
                self.logger.info(f"No results found for query: '{query}'")
                return []
            
            # Convert to ChunkWithScore objects
            chunks_with_scores = list()
            
            for rank, (chunk_id, score) in enumerate(bm25_results, 1):
                # Filter by minimum score
                if (score < min_score):
                    continue
                
                # Get chunk metadata
                chunk_metadata = self.metadata_store.get_chunk_metadata(chunk_id)
                
                if not chunk_metadata:
                    self.logger.warning(f"Chunk metadata not found for: {chunk_id}")
                    continue
                
                # Create DocumentChunk
                chunk = self._metadata_to_chunk(metadata = chunk_metadata)
                
                # Create ChunkWithScore
                cws   = ChunkWithScore(chunk            = chunk,
                                       score            = score,
                                       rank             = rank,
                                       retrieval_method = 'bm25',
                                      )
                
                chunks_with_scores.append(cws)
            
            # Update statistics
            self.search_count  += 1
            self.total_results += len(chunks_with_scores)
            
            self.logger.info(f"Keyword search returned {len(chunks_with_scores)} results")
            
            return chunks_with_scores
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {repr(e)}")
            raise KeywordSearchError(f"Keyword search failed: {repr(e)}")
    

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
    

    def search_with_filters(self, query: str, top_k: int = 10, document_ids: Optional[List[str]] = None, min_score: float = 0.0) -> List[ChunkWithScore]:
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
                  { list }         : Filtered ChunkWithScore objects
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
    

    def get_search_statistics(self) -> dict:
        """
        Get keyword search statistics
        
        Returns:
        --------
            { dict }    : Search statistics
        """
        avg_results = (self.total_results / self.search_count) if (self.search_count > 0) else 0
        
        return {"search_count"           : self.search_count,
                "total_results"          : self.total_results,
                "avg_results_per_query"  : avg_results,
                "bm25_index_stats"       : self.bm25_index.get_index_stats(),
               }


# Global keyword search instance
_keyword_search = None


def get_keyword_search() -> KeywordSearch:
    """
    Get global keyword search instance
    
    Returns:
    --------
        { KeywordSearch } : KeywordSearch instance
    """
    global _keyword_search

    if _keyword_search is None:
        _keyword_search = KeywordSearch()
    
    return _keyword_search


def search_keywords(query: str, top_k: int = 10, **kwargs) -> List[ChunkWithScore]:
    """
    Convenience function for keyword search
    
    Arguments:
    ----------
        query  { str } : Search query
        
        top_k  { int } : Number of results
        
        **kwargs       : Additional arguments
    
    Returns:
    --------
            { list }   : ChunkWithScore results
    """
    searcher = get_keyword_search()
    
    return searcher.search(query, top_k, **kwargs)