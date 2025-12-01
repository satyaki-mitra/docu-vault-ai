# DEPENDENCIES
import time
import faiss
import numpy as np
from typing import List
from typing import Tuple
from pathlib import Path
from typing import Optional
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import IndexingError
from vector_store.index_persister import get_index_persister


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class FAISSManager:
    """
    FAISS vector index management: Handles creation, optimization, and search operations for vector similarity search
    Supports multiple index types based on dataset size (Flat, IVF, HNSW)
    """
    def __init__(self, vector_store_dir: Optional[Path] = None):
        """
        Initialize FAISS manager
        
        Arguments:
        ----------
            vector_store_dir { Path } : Directory for index storage
        """
        self.logger           = logger
        self.vector_store_dir = Path(vector_store_dir or settings.VECTOR_STORE_DIR)
        
        # FAISS configuration
        self.faiss_nprobe     = settings.FAISS_NPROBE
        self.embedding_dim    = settings.EMBEDDING_DIMENSION
        
        # Index components
        self.index            = None
        self.chunk_ids        = list()
        self.index_metadata   = dict()
        
        # Initialize persister
        self.persister        = get_index_persister(self.vector_store_dir)
        
        # Statistics
        self.search_count     = 0
        self.build_count      = 0
        
        self.logger.info(f"Initialized FAISSManager: dim={self.embedding_dim}, nprobe={self.faiss_nprobe}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def build_index(self, embeddings: np.ndarray, chunk_ids: List[str], rebuild: bool = False) -> dict:
        """
        Build FAISS index from embeddings
        
        Arguments:
        ----------
            embeddings { np.ndarray } : Embedding vectors (n x dim)
            
            chunk_ids  { list }       : Corresponding chunk IDs
            
            rebuild    { bool }       : Whether to rebuild existing index
        
        Returns:
        --------
               { dict }               : Build statistics
        """
        if (embeddings.shape[0] != len(chunk_ids)):
            raise IndexingError(f"Embeddings count {embeddings.shape[0]} doesn't match chunk IDs count {len(chunk_ids)}")
        
        if (embeddings.shape[1] != self.embedding_dim):
            self.logger.warning(f"Embedding dimension {embeddings.shape[1]} doesn't match expected {self.embedding_dim}")
        
        if (self.index is not None) and (not rebuild):
            self.logger.info("Index already exists, use rebuild=True to rebuild")
            return self.get_index_stats()
        
        self.logger.info(f"Building FAISS index for {len(chunk_ids)} vectors")
        
        start_time = time.time()
        
        # Select appropriate index type based on dataset size
        index = self._create_optimal_index(embeddings.shape[0])
        
        # Train index if needed
        if hasattr(index, 'train') and index.ntotal == 0:
            self.logger.info("Training FAISS index")
            index.train(embeddings)
        
        # Add vectors to index
        index.add(embeddings)
        
        # Configure search parameters
        if hasattr(index, 'nprobe'):
            index.nprobe = self.faiss_nprobe
        
        # Update instance state
        self.index          = index
        self.chunk_ids      = chunk_ids
        self.build_count   += 1
        
        build_time = time.time() - start_time
        
        # Save to disk
        metadata = {"build_time"    : build_time,
                    "vector_count"  : len(chunk_ids),
                    "embedding_dim" : embeddings.shape[1],
                    "index_type"    : type(index).__name__,
                   }
        
        self.persister.save_faiss_index(index, chunk_ids, metadata)
        
        stats = {"vectors"          : len(chunk_ids),
                 "build_time_seconds": build_time,
                 "index_type"       : type(index).__name__,
                 "embedding_dim"    : embeddings.shape[1],
                 "vectors_per_second": len(chunk_ids) / build_time if build_time > 0 else 0,
                }
        
        self.logger.info(f"FAISS index built: {len(chunk_ids)} vectors in {build_time:.2f}s")
        
        return stats
    

    def _create_optimal_index(self, vector_count: int) -> faiss.Index:
        """
        Create optimal FAISS index type based on dataset size
        
        Arguments:
        ----------
            vector_count { int } : Number of vectors to index
        
        Returns:
        --------
            { faiss.Index }      : FAISS index instance
        """
        # Based on project document specifications:
        # - Small datasets (<100K): IndexFlatL2 (exact search)
        # - Medium datasets (100K-1M): IndexIVFFlat (balanced)
        # - Large datasets (>1M): IndexHNSW (approximate, fast)
        
        if (vector_count < 100000):
            # Exact search for small datasets
            index = faiss.IndexFlatL2(self.embedding_dim)
            self.logger.info(f"Using IndexFlatL2 for {vector_count} vectors (exact search)")
        
        elif (vector_count < 1000000):
            # Balanced approach for medium datasets
            nlist     = min(4096, vector_count // 39)  # FAISS recommendation
            
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index     = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            self.logger.info(f"Using IndexIVFFlat for {vector_count} vectors (nlist={nlist})")
        
        else:
            # Approximate search for large datasets
            M     = 32  # Connections per node
            index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            
            self.logger.info(f"Using IndexHNSW for {vector_count} vectors (M={M})")
        
        return index
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def add_to_index(self, embeddings: np.ndarray, chunk_ids: List[str]) -> dict:
        """
        Add new vectors to existing index
        
        Arguments:
        ----------
            embeddings { np.ndarray } : New embedding vectors
            
            chunk_ids  { list }       : New chunk IDs
        
        Returns:
        --------
               { dict }               : Add operation statistics
        """
        if self.index is None:
            raise IndexingError("No index exists. Build index first.")
        
        if (embeddings.shape[0] != len(chunk_ids)):
            raise IndexingError(f"Embeddings count {embeddings.shape[0]} doesn't match chunk IDs count {len(chunk_ids)}")
        
        self.logger.info(f"Adding {len(chunk_ids)} vectors to FAISS index")
        
        start_time = time.time()
        
        # Add to index
        self.index.add(embeddings)
        
        # Update chunk IDs
        self.chunk_ids.extend(chunk_ids)
        
        add_time   = time.time() - start_time
        
        # Save updated index
        metadata   = {"added_vectors"  : len(chunk_ids),
                      "total_vectors"  : len(self.chunk_ids),
                      "add_time"       : add_time,
                     }
        
        self.persister.save_faiss_index(self.index, self.chunk_ids, metadata)
        
        stats      = {"added"              : len(chunk_ids),
                      "new_total"          : len(self.chunk_ids),
                      "add_time_seconds"   : add_time,
                      "vectors_per_second" : len(chunk_ids) / add_time if add_time > 0 else 0,
                     }
        
        self.logger.info(f"Added {len(chunk_ids)} vectors to FAISS index in {add_time:.2f}s")
        
        return stats
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar vectors
        
        Arguments:
        ----------
            query_embedding { np.ndarray } : Query embedding vector
            
            top_k           { int }        : Number of results to return
        
        Returns:
        --------
                       { list }            : List of (chunk_id, score) tuples
        """
        if self.index is None:
            # Try to load from disk
            self._load_index_from_disk()
            
            if self.index is None:
                raise IndexingError("No FAISS index available for search")
        
        if (query_embedding.ndim != 1) or (query_embedding.shape[0] != self.embedding_dim):
            raise IndexingError(f"Query embedding must be 1D vector of dimension {self.embedding_dim}")
        
        # Reshape for FAISS (1 x dim)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Convert to results
        results         = list()
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if ((idx < len(self.chunk_ids)) and (idx >= 0)):
                chunk_id         = self.chunk_ids[idx]
                
                # Convert L2 distance to similarity score (higher = more similar)
                similarity_score = 1.0 / (1.0 + score) if score > 0 else 1.0
                
                results.append((chunk_id, float(similarity_score)))
        
        self.search_count += 1
        
        self.logger.debug(f"FAISS search returned {len(results)} results")
        
        return results
    

    def _load_index_from_disk(self):
        """
        Load index from disk if available
        """
        try:
            index, chunk_ids, metadata = self.persister.load_faiss_index()
            
            if index is not None:
                self.index          = index
                self.chunk_ids      = chunk_ids
                self.index_metadata = metadata
                
                # Configure search parameters
                if hasattr(self.index, 'nprobe'):
                    self.index.nprobe = self.faiss_nprobe
                
                self.logger.info(f"Loaded FAISS index from disk: {len(chunk_ids)} vectors")
            
        except Exception as e:
            self.logger.warning(f"Could not load FAISS index from disk: {e}")
    

    def is_index_built(self) -> bool:
        """
        Check if index is built and ready
        
        Returns:
        --------
            { bool }    : True if index is built
        """
        if self.index is not None:
            return True
        
        # Check if index exists on disk
        return self.persister.index_files_exist()
    

    def get_index_stats(self) -> dict:
        """
        Get FAISS index statistics
        
        Returns:
        --------
            { dict }    : Index statistics
        """
        if self.index is None:
            self._load_index_from_disk()
            
            if self.index is None:
                return {"built": False}
        
        stats = {"built"           : True,
                 "vector_count"    : len(self.chunk_ids),
                 "embedding_dim"   : self.embedding_dim,
                 "index_type"      : type(self.index).__name__,
                 "search_count"    : self.search_count,
                 "build_count"     : self.build_count,
                 "faiss_nprobe"    : self.faiss_nprobe,
                }
        
        # Add index-specific stats
        if hasattr(self.index, 'ntotal'):
            stats["index_ntotal"] = self.index.ntotal
        
        if hasattr(self.index, 'nlist'):
            stats["index_nlist"] = self.index.nlist
        
        # Add metadata
        stats.update(self.index_metadata)
        
        return stats
    

    def optimize_index(self) -> dict:
        """
        Optimize index for better performance
        
        Returns:
        --------
            { dict }    : Optimization results
        """
        if self.index is None:
            return {"optimized": False, "message": "No index to optimize"}
        
        self.logger.info("Optimizing FAISS index")
        
        # For IVF indexes, we can adjust nprobe
        if hasattr(self.index, 'nprobe'):
            old_nprobe = self.index.nprobe
            new_nprobe = min(old_nprobe * 2, 50)  # Increase but cap
            
            if new_nprobe != old_nprobe:
                self.index.nprobe = new_nprobe
                self.faiss_nprobe = new_nprobe
                
                self.logger.info(f"Adjusted nprobe: {old_nprobe} -> {new_nprobe}")
        
        return {"optimized"    : True,
                "faiss_nprobe" : self.faiss_nprobe,
                "message"      : "FAISS index optimization completed",
               }
    

    def clear_index(self):
        """
        Clear the index
        """
        self.index          = None
        self.chunk_ids      = []
        self.index_metadata = {}
        self.search_count   = 0
        
        self.logger.info("FAISS index cleared")
    

    def get_index_size(self) -> dict:
        """
        Get index size information
        
        Returns:
        --------
            { dict }    : Size information
        """
        if self.index is None:
            return {"memory_mb": 0, "disk_mb": 0}
        
        # Estimate memory usage
        vector_count = len(self.chunk_ids)
        memory_bytes = vector_count * self.embedding_dim * 4  # float32 = 4 bytes
        
        # Get disk size from persister
        files_info = self.persister.get_index_files_info()
        faiss_files = {k: v for k, v in files_info.items() if 'faiss' in k}
        disk_size   = sum(info.get('size_mb', 0) for info in faiss_files.values())
        
        return {"memory_mb"    : memory_bytes / (1024 * 1024),
                "disk_mb"      : disk_size,
                "vector_count" : vector_count,
                "files"        : faiss_files,
               }


# Global FAISS manager instance
_faiss_manager = None


def get_faiss_manager(vector_store_dir: Optional[Path] = None) -> FAISSManager:
    """
    Get global FAISS manager instance
    
    Arguments:
    ----------
        vector_store_dir { Path } : Vector store directory
    
    Returns:
    --------
        { FAISSManager }          : FAISSManager instance
    """
    global _faiss_manager

    if _faiss_manager is None:
        _faiss_manager = FAISSManager(vector_store_dir)
    
    return _faiss_manager


def search_similar_vectors(query_embedding: np.ndarray, top_k: int = 10, **kwargs) -> List[Tuple[str, float]]:
    """
    Convenience function for vector search
    
    Arguments:
    ----------
        query_embedding { np.ndarray } : Query embedding
        
        top_k           { int }        : Number of results
        
        **kwargs                       : Additional arguments
    
    Returns:
    --------
                       { list }        : Search results
    """
    manager = get_faiss_manager()

    return manager.search(query_embedding, top_k, **kwargs)