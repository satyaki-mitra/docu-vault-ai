"""
FAISS Manager
Manages FAISS vector index for similarity search
"""

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import faiss
import pickle

from config.logging_config import get_logger
from config.settings import get_settings
from config.models import DocumentChunk

logger = get_logger(__name__)
settings = get_settings()


class FAISSManager:
    """
    Manages FAISS vector index with automatic index selection.
    Supports multiple index types based on dataset size.
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "auto",
        metric: str = "l2"
    ):
        """
        Initialize FAISS manager.
        
        Args:
            dimension: Embedding dimension
            index_type: Index type (auto, flat, ivf, hnsw)
            metric: Distance metric (l2 or ip for inner product)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        self.index: Optional[faiss.Index] = None
        self.id_map: List[str] = []  # Maps FAISS ID to chunk ID
        
        self.logger = logger
        self.logger.info(
            f"FAISSManager initialized: dim={dimension}, "
            f"type={index_type}, metric={metric}"
        )
    
    def _create_index(self, num_vectors: int = 0) -> faiss.Index:
        """
        Create appropriate FAISS index based on size and type.
        
        Args:
            num_vectors: Expected number of vectors
        
        Returns:
            FAISS index
        """
        if self.index_type == "auto":
            # Auto-select based on size
            if num_vectors < 100_000:
                index_type = "flat"
            elif num_vectors < 1_000_000:
                index_type = "ivf"
            else:
                index_type = "hnsw"
        else:
            index_type = self.index_type
        
        self.logger.info(f"Creating {index_type.upper()} index")
        
        if index_type == "flat":
            # Flat index (exact search, slower but accurate)
            if self.metric == "l2":
                index = faiss.IndexFlatL2(self.dimension)
            else:
                index = faiss.IndexFlatIP(self.dimension)
        
        elif index_type == "ivf":
            # IVF index (approximate search, faster)
            nlist = min(int(np.sqrt(num_vectors)), 1000)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            
            if self.metric == "l2":
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        elif index_type == "hnsw":
            # HNSW index (graph-based, very fast)
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 = M parameter
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return index
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        chunk_ids: List[str]
    ):
        """
        Add vectors to index.
        
        Args:
            vectors: Array of vectors (shape: [N, dimension])
            chunk_ids: List of chunk IDs corresponding to vectors
        """
        if vectors.shape[0] != len(chunk_ids):
            raise ValueError("Number of vectors must match number of chunk IDs")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Create index if doesn't exist
        if self.index is None:
            self.index = self._create_index(num_vectors=vectors.shape[0])
            
            # Train IVF index if needed
            if isinstance(self.index, faiss.IndexIVFFlat):
                self.logger.info("Training IVF index...")
                self.index.train(vectors)
        
        # Ensure vectors are float32 and C-contiguous
        vectors = np.ascontiguousarray(vectors.astype('float32'))
        
        # Add to index
        start_id = len(self.id_map)
        self.index.add(vectors)
        self.id_map.extend(chunk_ids)
        
        self.logger.info(
            f"Added {len(vectors)} vectors (total: {self.index.ntotal})"
        )
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_vector: Query vector (1D array)
            k: Number of results to return
            nprobe: Number of clusters to search (IVF only)
        
        Returns:
            Tuple of (chunk_ids, distances)
        """
        if self.index is None or self.index.ntotal == 0:
            self.logger.warning("Index is empty")
            return [], []
        
        # Ensure query is 2D float32
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype('float32')
        
        # Set nprobe for IVF index
        if isinstance(self.index, faiss.IndexIVFFlat) and nprobe:
            self.index.nprobe = nprobe
        elif isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = settings.FAISS_NPROBE
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # Map indices to chunk IDs
        chunk_ids = [
            self.id_map[idx] 
            for idx in indices[0] 
            if 0 <= idx < len(self.id_map)
        ]
        distances_list = distances[0].tolist()
        
        return chunk_ids, distances_list
    
    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for multiple queries.
        
        Args:
            query_vectors: Array of query vectors (shape: [N, dimension])
            k: Number of results per query
            nprobe: Number of clusters to search (IVF only)
        
        Returns:
            Tuple of (chunk_ids_list, distances_list)
        """
        if self.index is None or self.index.ntotal == 0:
            return [[]] * len(query_vectors), [[]] * len(query_vectors)
        
        query_vectors = np.ascontiguousarray(query_vectors.astype('float32'))
        
        # Set nprobe for IVF index
        if isinstance(self.index, faiss.IndexIVFFlat) and nprobe:
            self.index.nprobe = nprobe
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vectors, k)
        
        # Map indices to chunk IDs for each query
        all_chunk_ids = []
        all_distances = []
        
        for query_indices, query_distances in zip(indices, distances):
            chunk_ids = [
                self.id_map[idx] 
                for idx in query_indices 
                if 0 <= idx < len(self.id_map)
            ]
            all_chunk_ids.append(chunk_ids)
            all_distances.append(query_distances.tolist())
        
        return all_chunk_ids, all_distances
    
    def remove_vectors(self, chunk_ids: List[str]):
        """
        Remove vectors from index (requires IDMap wrapper).
        Note: Standard FAISS indices don't support removal.
        This is a placeholder for future implementation.
        
        Args:
            chunk_ids: Chunk IDs to remove
        """
        self.logger.warning(
            "Vector removal not yet implemented for standard FAISS indices"
        )
        # To implement: use IndexIDMap wrapper
    
    def save(self, directory: Path):
        """
        Save index and ID mapping to disk.
        
        Args:
            directory: Directory to save files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        if self.index is None:
            self.logger.warning("No index to save")
            return
        
        # Save FAISS index
        index_path = directory / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save ID mapping
        id_map_path = directory / "id_map.pkl"
        with open(id_map_path, 'wb') as f:
            pickle.dump(self.id_map, f)
        
        self.logger.info(f"Saved index to {directory}")
    
    def load(self, directory: Path):
        """
        Load index and ID mapping from disk.
        
        Args:
            directory: Directory containing saved files
        """
        directory = Path(directory)
        
        # Load FAISS index
        index_path = directory / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load ID mapping
        id_map_path = directory / "id_map.pkl"
        if not id_map_path.exists():
            raise FileNotFoundError(f"ID map file not found: {id_map_path}")
        
        with open(id_map_path, 'rb') as f:
            self.id_map = pickle.load(f)
        
        self.logger.info(
            f"Loaded index from {directory} ({self.index.ntotal} vectors)"
        )
    
    def get_statistics(self) -> dict:
        """
        Get index statistics.
        
        Returns:
            Statistics dictionary
        """
        if self.index is None:
            return {
                "num_vectors": 0,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "is_trained": False,
            }
        
        stats = {
            "num_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "metric": self.metric,
        }
        
        if isinstance(self.index, faiss.IndexIVFFlat):
            stats["is_trained"] = self.index.is_trained
            stats["nlist"] = self.index.nlist
            stats["nprobe"] = self.index.nprobe
        
        return stats
    
    def reset(self):
        """Reset index and ID map"""
        self.index = None
        self.id_map = []
        self.logger.info("Index reset")


if __name__ == "__main__":
    # Test FAISS manager
    print("=== FAISS Manager Tests ===\n")
    
    # Test 1: Create index and add vectors
    print("Test 1: Create and add vectors")
    manager = FAISSManager(dimension=384, index_type="flat")
    
    # Create test vectors
    num_vectors = 100
    test_vectors = np.random.rand(num_vectors, 384).astype('float32')
    chunk_ids = [f"chunk_{i}" for i in range(num_vectors)]
    
    manager.add_vectors(test_vectors, chunk_ids)
    print(f"  Added {num_vectors} vectors")
    
    stats = manager.get_statistics()
    print(f"  Index stats: {stats}")
    print()
    
    # Test 2: Search
    print("Test 2: Search")
    query = np.random.rand(384).astype('float32')
    results, distances = manager.search(query, k=5)
    
    print(f"  Found {len(results)} results")
    for i, (chunk_id, dist) in enumerate(zip(results, distances)):
        print(f"    {i+1}. {chunk_id} (distance: {dist:.4f})")
    print()
    
    # Test 3: Batch search
    print("Test 3: Batch search")
    queries = np.random.rand(3, 384).astype('float32')
    batch_results, batch_distances = manager.batch_search(queries, k=3)
    
    print(f"  Searched {len(queries)} queries")
    for i, (results, distances) in enumerate(zip(batch_results, batch_distances)):
        print(f"  Query {i}: {len(results)} results")
    print()
    
    # Test 4: Save and load
    print("Test 4: Save and load")
    test_dir = Path("test_faiss_index")
    manager.save(test_dir)
    print(f"  Saved to {test_dir}")
    
    # Create new manager and load
    manager2 = FAISSManager(dimension=384)
    manager2.load(test_dir)
    stats2 = manager2.get_statistics()
    print(f"  Loaded: {stats2['num_vectors']} vectors")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print()
    
    print("✓ FAISS manager module created successfully!")