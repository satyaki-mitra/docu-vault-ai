# DEPENDENCIES
import time
import pickle
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
from typing import Tuple
from typing import Optional
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import IndexingError
from vector_store.index_persister import get_index_persister


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True

except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available, BM25 indexing disabled")


class BM25Index:
    """
    BM25 keyword search index: Provides traditional keyword-based search as complement to vector search: Implements probabilistic relevance scoring for text retrieval
    """
    def __init__(self):
        """
        Initialize BM25 index
        """
        self.logger = logger
        
        if not BM25_AVAILABLE:
            self.logger.warning("BM25 indexing disabled - install rank_bm25")
            self.bm25      = None
            self.chunk_ids = []
            return
        
        # BM25 configuration (from project document)
        self.k1             = settings.BM25_K1      # Term saturation parameter
        self.b              = settings.BM25_B       # Length normalization parameter
        
        # Index components
        self.bm25           = None
        self.chunk_ids      = list()
        self.vocabulary     = set()
        self.index_metadata = dict()
        
        # Initialize persister
        self.persister      = get_index_persister()
        
        # Statistics
        self.search_count   = 0
        self.build_count    = 0
        
        self.logger.info(f"Initialized BM25Index: k1={self.k1}, b={self.b}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def build_index(self, texts: List[str], chunk_ids: List[str], rebuild: bool = False) -> dict:
        """
        Build BM25 index from texts
        
        Arguments:
        ----------
            texts     { list } : List of text documents
            
            chunk_ids { list } : Corresponding chunk IDs
            
            rebuild   { bool } : Whether to rebuild existing index
        
        Returns:
        --------
               { dict }        : Build statistics
        """
        if not BM25_AVAILABLE:
            raise IndexingError("BM25 not available - install rank_bm25 package")
        
        if (len(texts) != len(chunk_ids)):
            raise IndexingError(f"Texts count {len(texts)} doesn't match chunk IDs count {len(chunk_ids)}")
        
        if (self.bm25 is not None) and (not rebuild):
            self.logger.info("BM25 index already exists, use rebuild=True to rebuild")
            return self.get_index_stats()
        
        self.logger.info(f"Building BM25 index for {len(texts)} documents")
        
        start_time        = time.time()
        
        # Tokenize texts
        tokenized_texts   = [self._tokenize_text(text) for text in texts]
        
        # Build BM25 index
        self.bm25         = BM25Okapi(tokenized_texts, 
                                      k1 = self.k1, 
                                      b  = self.b,
                                     )
        
        # Update instance state
        self.chunk_ids    = chunk_ids
        self.build_count += 1
        
        # Build vocabulary
        self.vocabulary   = set()

        for tokens in tokenized_texts:
            self.vocabulary.update(tokens)
        
        build_time        = time.time() - start_time
        
        # Save to disk
        metadata          = {"build_time"      : build_time,
                             "document_count"  : len(chunk_ids),
                             "vocabulary_size" : len(self.vocabulary),
                             "parameters"      : {"k1": self.k1, "b": self.b},
                            }
          
        self.persister.save_bm25_index(self.bm25, chunk_ids, metadata)
        
        stats = {"documents"            : len(chunk_ids),
                 "build_time_seconds"   : build_time,
                 "vocabulary_size"      : len(self.vocabulary),
                 "documents_per_second" : len(chunk_ids) / build_time if build_time > 0 else 0,
                 "parameters"           : {"k1": self.k1, "b": self.b},
                }
        
        self.logger.info(f"BM25 index built: {len(chunk_ids)} documents in {build_time:.2f}s")
        
        return stats
    

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
               { list }  : List of tokens
        """
        # Simple tokenization: lowercase, split, remove short tokens
        tokens          = text.lower().split()
        
        # Filter tokens
        filtered_tokens = list()
        
        for token in tokens:
            # Remove punctuation and short tokens
            token = ''.join(char for char in token if char.isalnum())
            
            # Reasonable token length
            if ((len(token) >= 2) and (len(token) <= 50)):  
                filtered_tokens.append(token)
        
        return filtered_tokens
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def add_to_index(self, texts: List[str], chunk_ids: List[str]) -> dict:
        """
        Add new documents to existing index
        
        Arguments:
        ----------
            texts     { list } : New text documents
            
            chunk_ids { list } : New chunk IDs
        
        Returns:
        --------
               { dict }        : Add operation statistics
        """
        if not BM25_AVAILABLE:
            raise IndexingError("BM25 not available")
        
        if self.bm25 is None:
            raise IndexingError("No BM25 index exists. Build index first.")
        
        if (len(texts) != len(chunk_ids)):
            raise IndexingError(f"Texts count {len(texts)} doesn't match chunk IDs count {len(chunk_ids)}")
        
        self.logger.info(f"Adding {len(chunk_ids)} documents to BM25 index")
        
        start_time          = time.time()
        
        # Tokenize new texts
        new_tokenized_texts = [self._tokenize_text(text) for text in texts]
        
        # Update BM25 index (this is a limitation of BM25Okapi - we need to rebuild)
        all_texts           = [self.bm25.corpus[i] for i in range(len(self.bm25.corpus))] + new_tokenized_texts
        all_chunk_ids       = self.chunk_ids + chunk_ids
        
        # Rebuild index with all documents
        self.bm25           = BM25Okapi(all_texts, k1 = self.k1, b = self.b)
        self.chunk_ids      = all_chunk_ids
        
        # Update vocabulary
        for tokens in new_tokenized_texts:
            self.vocabulary.update(tokens)
        
        add_time = time.time() - start_time
        
        # Save updated index
        metadata = {"added_documents" : len(chunk_ids),
                    "total_documents" : len(self.chunk_ids),
                    "vocabulary_size" : len(self.vocabulary),
                    "add_time"        : add_time,
                   }
        
        self.persister.save_bm25_index(self.bm25, self.chunk_ids, metadata)
        
        stats = {"added"            : len(chunk_ids),
                 "new_total"        : len(self.chunk_ids),
                 "add_time_seconds" : add_time,
                 "vocabulary_size"  : len(self.vocabulary),
                }
        
        self.logger.info(f"Added {len(chunk_ids)} documents to BM25 index in {add_time:.2f}s")
        
        return stats
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for relevant documents using BM25
        
        Arguments:
        ----------
            query  { str } : Search query string
            
            top_k  { int } : Number of results to return
        
        Returns:
        --------
               { list }    : List of (chunk_id, score) tuples
        """
        if not BM25_AVAILABLE:
            raise IndexingError("BM25 not available")
        
        if self.bm25 is None:
            # Try to load from disk
            self._load_index_from_disk()
            
            if self.bm25 is None:
                raise IndexingError("No BM25 index available for search")
        
        if not query or not query.strip():
            return []
        
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores      = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), 
                             key     = lambda i: scores[i], 
                             reverse = True,
                            )[:top_k]
        
        # Convert to results
        results     = list()
        
        for idx in top_indices:
            if ((scores[idx] > 0) and (idx < len(self.chunk_ids))):
                chunk_id = self.chunk_ids[idx]

                results.append((chunk_id, float(scores[idx])))
        
        self.search_count += 1
        
        self.logger.debug(f"BM25 search returned {len(results)} results for query: '{query}'")
        
        return results
    

    def _load_index_from_disk(self):
        """
        Load index from disk if available
        """
        try:
            index, chunk_ids, metadata = self.persister.load_bm25_index()
            
            if index is not None:
                self.bm25           = index
                self.chunk_ids      = chunk_ids
                self.index_metadata = metadata
                
                # Rebuild vocabulary
                self.vocabulary     = set()

                if hasattr(self.bm25, 'corpus'):
                    for tokens in self.bm25.corpus:
                        self.vocabulary.update(tokens)
                
                self.logger.info(f"Loaded BM25 index from disk: {len(chunk_ids)} documents")
            
        except Exception as e:
            self.logger.warning(f"Could not load BM25 index from disk: {e}")
    

    def is_index_built(self) -> bool:
        """
        Check if index is built and ready
        
        Returns:
        --------
            { bool }    : True if index is built
        """
        if self.bm25 is not None:
            return True
        
        # Check if index exists on disk
        return self.persister.index_files_exist()
    

    def get_index_stats(self) -> dict:
        """
        Get BM25 index statistics
        
        Returns:
        --------
            { dict }    : Index statistics
        """
        if not BM25_AVAILABLE:
            return {"available": False}
        
        if self.bm25 is None:
            self._load_index_from_disk()
            
            if self.bm25 is None:
                return {"built": False}
        
        stats = {"built"           : True,
                 "document_count"  : len(self.chunk_ids),
                 "vocabulary_size" : len(self.vocabulary),
                 "search_count"    : self.search_count,
                 "build_count"     : self.build_count,
                 "parameters"      : {"k1": self.k1, "b": self.b},
                }
        
        # Add metadata
        stats.update(self.index_metadata)
        
        return stats
    

    def optimize_index(self) -> dict:
        """
        Optimize BM25 index
        
        Returns:
        --------
            { dict }    : Optimization results
        """
        if not BM25_AVAILABLE or self.bm25 is None:
            return {"optimized" : False, 
                    "message"   : "No BM25 index to optimize",
                   }
        
        self.logger.info("Optimizing BM25 index")
        
        # BM25 optimization is limited, but we can adjust parameters
        return {"optimized"       : True,
                "parameters"      : {"k1" : self.k1, "b" : self.b},
                "vocabulary_size" : len(self.vocabulary),
                "message"         : "BM25 index optimization completed",
               }
    

    def clear_index(self):
        """
        Clear the index
        """
        self.bm25           = None
        self.chunk_ids      = list()
        self.vocabulary     = set()
        self.index_metadata = dict()
        self.search_count   = 0
        
        self.logger.info("BM25 index cleared")
    

    def get_index_size(self) -> dict:
        """
        Get index size information
        
        Returns:
        --------
            { dict }    : Size information
        """
        if not BM25_AVAILABLE or self.bm25 is None:
            return {"memory_mb": 0, "disk_mb": 0}
        
        # Estimate memory usage (rough)
        vocab_size    = len(self.vocabulary)
        doc_count     = len(self.chunk_ids)
        memory_bytes  = (vocab_size * 50) + (doc_count * 100)  # Rough estimates
        
        # Get disk size from persister
        files_info    = self.persister.get_index_files_info()
        bm25_files    = {k: v for k, v in files_info.items() if 'bm25' in k}
        disk_size     = sum(info.get('size_mb', 0) for info in bm25_files.values())
        
        return {"memory_mb"       : memory_bytes / (1024 * 1024),
                "disk_mb"         : disk_size,
                "document_count"  : doc_count,
                "vocabulary_size" : vocab_size,
                "files"           : bm25_files,
               }
    

    def get_term_stats(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific term
        
        Arguments:
        ----------
            term { str } : Term to analyze
        
        Returns:
        --------
               { dict }  : Term statistics or None
        """
        if not BM25_AVAILABLE or self.bm25 is None:
            return None
        
        term = term.lower()
        
        if term not in self.vocabulary:
            return None
        
        # Calculate term frequency across documents
        term_freq = 0
        doc_freq  = 0
        
        if hasattr(self.bm25, 'corpus'):
            for tokens in self.bm25.corpus:
                if term in tokens:
                    doc_freq  += 1
                    term_freq += tokens.count(term)
        
        return {"term"          : term,
                "term_frequency": term_freq,
                "document_frequency": doc_freq,
                "in_vocabulary" : True,
               }


# Global BM25 index instance
_bm25_index = None


def get_bm25_index() -> BM25Index:
    """
    Get global BM25 index instance
    
    Returns:
    --------
        { BM25Index } : BM25Index instance
    """
    global _bm25_index

    if _bm25_index is None:
        _bm25_index = BM25Index()
    
    return _bm25_index


def search_with_bm25(query: str, top_k: int = 10, **kwargs) -> List[Tuple[str, float]]:
    """
    Convenience function for BM25 search
    
    Arguments:
    ----------
        query { str } : Search query
        
        top_k { int } : Number of results
        
        **kwargs      : Additional arguments
    
    Returns:
    --------
          { list }    : Search results
    """
    index = get_bm25_index()

    return index.search(query, top_k, **kwargs)
    