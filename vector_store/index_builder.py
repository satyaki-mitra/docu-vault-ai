# DEPENDENCIES
import time
import faiss
import numpy as np
from typing import List
from pathlib import Path
from typing import Optional
from config.models import DocumentChunk
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import IndexingError
from vector_store.bm25_index import BM25Index
from vector_store.faiss_manager import FAISSManager
from vector_store.metadata_store import MetadataStore


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class IndexBuilder:
    """
    Main index builder orchestrator: Builds and manages both vector and keyword indexes
    Coordinates FAISS vector index, BM25 keyword index, and metadata storage
    """
    def __init__(self, vector_store_dir: Optional[Path] = None):
        """
        Initialize index builder
        
        Arguments:
        ----------
            vector_store_dir { Path } : Directory for index storage
        """
        self.logger               = logger
        self.vector_store_dir     = Path(vector_store_dir or settings.VECTOR_STORE_DIR)
        
        # Initialize component managers
        self.faiss_manager        = FAISSManager(vector_store_dir = self.vector_store_dir)
        self.bm25_index           = BM25Index()
        self.metadata_store       = MetadataStore()
        
        # Index statistics
        self.total_chunks_indexed = 0
        self.last_build_time      = None
        
        self.logger.info(f"Initialized IndexBuilder: store_dir={self.vector_store_dir}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def build_indexes(self, chunks: List[DocumentChunk], rebuild: bool = False) -> dict:
        """
        Build both vector and keyword indexes from document chunks - FIXED VERSION
        
        Arguments:
        ----------
            chunks  { list } : List of DocumentChunk objects with embeddings
            
            rebuild { bool } : Whether to rebuild existing indexes
        
        Returns:
        --------
               { dict }      : Build statistics
        """
        if not chunks:
            raise IndexingError("No chunks provided for indexing")
        
        # Validate chunks have embeddings
        chunks_with_embeddings     = [c for c in chunks if (c.embedding is not None)]
        
        if (len(chunks_with_embeddings) != len(chunks)):
            self.logger.warning(f"{len(chunks) - len(chunks_with_embeddings)} chunks missing embeddings")
        
        if not chunks_with_embeddings:
            raise IndexingError("No chunks with embeddings found")
        
        self.logger.info(f"Building indexes for {len(chunks_with_embeddings)} chunks (rebuild={rebuild})")
        
        start_time                 = time.time()
        
        # Extract data for indexing
        embeddings                 = self._extract_embeddings(chunks = chunks_with_embeddings)
        texts                      = [chunk.text for chunk in chunks_with_embeddings]
        chunk_ids                  = [chunk.chunk_id for chunk in chunks_with_embeddings]
        
        # Build vector index (FAISS)
        self.logger.info("Building FAISS vector index...")

        faiss_stats                = self.faiss_manager.build_index(embeddings = embeddings,
                                                                    chunk_ids  = chunk_ids,
                                                                    rebuild    = rebuild,
                                                                   )
        
        
        # Build keyword index (BM25)
        self.logger.info("Building BM25 keyword index...")
        bm25_stats                 = self.bm25_index.build_index(texts     = texts,
                                                                 chunk_ids = chunk_ids,
                                                                 rebuild   = rebuild,
                                                                )
        
        # Store metadata
        self.logger.info("Storing chunk metadata...")
        metadata_stats             = self.metadata_store.store_chunks(chunks  = chunks_with_embeddings,
                                                                      rebuild = rebuild,
                                                                     )
        
        # Update statistics
        self.total_chunks_indexed += len(chunks_with_embeddings)
        self.last_build_time       = time.time()
        
        build_time                 = time.time() - start_time
        
        stats                      = {"total_chunks"       : len(chunks_with_embeddings),
                                      "build_time_seconds" : build_time,
                                      "chunks_per_second"  : len(chunks_with_embeddings) / build_time if build_time > 0 else 0,
                                      "faiss"              : faiss_stats,
                                      "bm25"               : bm25_stats,
                                      "metadata"           : metadata_stats,
                                      "vector_dimension"   : embeddings.shape[1] if (len(embeddings) > 0) else 0,
                                     }
        
        self.logger.info(f"Index building completed: {len(chunks_with_embeddings)} chunks in {build_time:.2f}s")
        self.logger.info(f"FAISS index: {faiss_stats.get('vectors', 0)} vectors")
        self.logger.info(f"BM25 index: {bm25_stats.get('documents', 0)} documents")
        self.logger.info(f"Metadata: {metadata_stats.get('stored_chunks', 0)} chunks stored")
        
        return stats
    

    def _extract_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Extract embeddings from chunks as numpy array
        
        Arguments:
        ----------
            chunks { list } : List of DocumentChunk objects
        
        Returns:
        --------
               { np.ndarray } : Embeddings matrix
        """
        embeddings = list()
        
        for chunk in chunks:
            if (chunk.embedding is not None):
                embeddings.append(chunk.embedding)
        
        if not embeddings:
            raise IndexingError("No embeddings found in chunks")
        
        return np.array(embeddings).astype('float32')
    

    def get_index_stats(self) -> dict:
        """
        Get comprehensive index statistics
        
        Returns:
        --------
            { dict }    : Index statistics
        """
        faiss_stats    = self.faiss_manager.get_index_stats()
        bm25_stats     = self.bm25_index.get_index_stats()
        metadata_stats = self.metadata_store.get_stats()
        
        # Also check VectorSearch stats
        try:
            vector_search = get_vector_search()
            vector_stats  = vector_search.get_index_stats()
        
        except Exception as e:
            vector_stats = {"error": str(e)}
        
        stats = {"total_chunks_indexed" : self.total_chunks_indexed,
                 "last_build_time"      : self.last_build_time,
                 "faiss"                : faiss_stats,
                 "bm25"                 : bm25_stats,
                 "metadata"             : metadata_stats,
                 "index_directory"      : str(self.vector_store_dir),
                }
        
        return stats
    

    def is_index_built(self) -> bool:
        """
        Check if indexes are built and ready
        
        Returns:
        --------
            { bool }    : True if indexes are built
        """
        faiss_ready    = self.faiss_manager.is_index_built()
        bm25_ready     = self.bm25_index.is_index_built()
        metadata_ready = self.metadata_store.is_ready()
        
        return faiss_ready and bm25_ready and metadata_ready
    

    def optimize_indexes(self) -> dict:
        """
        Optimize indexes for better performance
        
        Returns:
        --------
            { dict }    : Optimization results
        """
        self.logger.info("Optimizing indexes")
        
        faiss_optimization = self.faiss_manager.optimize_index()
        bm25_optimization  = self.bm25_index.optimize_index()
        
        optimization_stats = {"faiss"   : faiss_optimization,
                              "bm25"    : bm25_optimization,
                              "message" : "Index optimization completed",
                             }
        
        return optimization_stats
    

    def clear_indexes(self):
        """
        Clear all indexes
        """
        self.logger.warning("Clearing all indexes")
        
        self.faiss_manager.clear_index()
        self.bm25_index.clear_index()
        self.metadata_store.clear()
        
        self.total_chunks_indexed = 0
    

    def get_index_size(self) -> dict:
        """
        Get index sizes in memory and disk
        
        Returns:
        --------
            { dict }    : Size information
        """
        faiss_size    = self.faiss_manager.get_index_size()
        bm25_size     = self.bm25_index.get_index_size()
        metadata_size = self.metadata_store.get_size()
        
        total_memory = (faiss_size.get("memory_mb", 0) + bm25_size.get("memory_mb", 0) + metadata_size.get("memory_mb", 0))
        
        total_disk   = (faiss_size.get("disk_mb", 0) + bm25_size.get("disk_mb", 0) + metadata_size.get("disk_mb", 0))
        
        return {"total_memory_mb" : total_memory,
                "total_disk_mb"   : total_disk,
                "faiss"           : faiss_size,
                "bm25"            : bm25_size,
                "metadata"        : metadata_size,
               }


# Global index builder instance
_index_builder = None


def get_index_builder(vector_store_dir: Optional[Path] = None) -> IndexBuilder:
    """
    Get global index builder instance
    
    Arguments:
    ----------
        vector_store_dir { Path } : Vector store directory
    
    Returns:
    --------
        { IndexBuilder }          : IndexBuilder instance
    """
    global _index_builder
    
    if _index_builder is None:
        _index_builder = IndexBuilder(vector_store_dir)
    
    return _index_builder


def build_indexes(chunks: List[DocumentChunk], **kwargs) -> dict:
    """
    Convenience function to build indexes
    
    Arguments:
    ----------
        chunks { list } : List of DocumentChunk objects

        **kwargs        : Additional arguments
    
    Returns:
    --------
             { dict }   : Build statistics
    """
    builder = get_index_builder()
    
    return builder.build_indexes(chunks, **kwargs)