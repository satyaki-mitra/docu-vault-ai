"""
Index Builder
Coordinates building FAISS, BM25, and metadata indices
"""

from typing import List, Optional
from pathlib import Path
import time
import numpy as np

from config.logging_config import get_logger
from config.settings import get_settings
from config.models import DocumentChunk, DocumentMetadata, ProcessingStatus

from embeddings import get_embedder, BatchProcessor
from vector_store.faiss_manager import FAISSManager
from vector_store.bm25_index import BM25Index
from vector_store.metadata_store import MetadataStore

logger = get_logger(__name__)
settings = get_settings()


class IndexBuilder:
    """
    Coordinates building and managing all indices.
    Handles FAISS vector index, BM25 keyword index, and metadata store.
    """
    
    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        vector_store_dir: Optional[Path] = None,
        metadata_db_path: Optional[Path] = None
    ):
        """
        Initialize index builder.
        
        Args:
            embedding_dim: Embedding dimension (auto-detected if None)
            vector_store_dir: Directory for vector indices
            metadata_db_path: Path to metadata database
        """
        self.logger = logger
        
        # Initialize embedder
        self.embedder = get_embedder()
        self.batch_processor = BatchProcessor(self.embedder)
        
        # Get embedding dimension
        self.embedding_dim = embedding_dim or self.embedder.get_embedding_dimension()
        
        # Initialize indices
        self.faiss_manager = FAISSManager(
            dimension=self.embedding_dim,
            index_type="auto"
        )
        self.bm25_index = BM25Index()
        self.metadata_store = MetadataStore(db_path=metadata_db_path)
        
        # Storage paths
        self.vector_store_dir = Path(vector_store_dir or settings.VECTOR_STORE_DIR)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"IndexBuilder initialized: dim={self.embedding_dim}, "
            f"store_dir={self.vector_store_dir}"
        )
    
    def build_indices(
        self,
        chunks: List[DocumentChunk],
        metadata: DocumentMetadata,
        show_progress: bool = True
    ):
        """
        Build all indices for document chunks.
        
        Args:
            chunks: List of document chunks
            metadata: Document metadata
            show_progress: Show progress during processing
        """
        if not chunks:
            self.logger.warning("No chunks to index")
            return
        
        self.logger.info(f"Building indices for {len(chunks)} chunks")
        start_time = time.time()
        
        try:
            # Update document status
            self.metadata_store.update_document_status(
                metadata.document_id,
                ProcessingStatus.PROCESSING
            )
            
            # 1. Generate embeddings
            self.logger.info("Generating embeddings...")
            embedded_chunks = self.batch_processor.process_chunks(
                chunks,
                callback=None
            )
            
            # Extract embeddings and IDs
            embeddings = np.array([
                chunk.embedding for chunk in embedded_chunks
            ], dtype='float32')
            chunk_ids = [chunk.chunk_id for chunk in embedded_chunks]
            
            # 2. Build FAISS index
            self.logger.info("Building FAISS index...")
            self.faiss_manager.add_vectors(embeddings, chunk_ids)
            
            # 3. Build BM25 index
            self.logger.info("Building BM25 index...")
            texts = [chunk.text for chunk in embedded_chunks]
            self.bm25_index.add_documents(texts, chunk_ids)
            
            # 4. Store metadata
            self.logger.info("Storing metadata...")
            self.metadata_store.add_chunks_batch(embedded_chunks)
            
            # Update document metadata
            processing_time = time.time() - start_time
            metadata.num_chunks = len(chunks)
            metadata.processing_time_seconds = processing_time
            metadata.processed_date = None  # Will be set by update_status
            
            self.metadata_store.add_document(metadata)
            self.metadata_store.update_document_status(
                metadata.document_id,
                ProcessingStatus.COMPLETED
            )
            
            self.logger.info(
                f"Indices built successfully in {processing_time:.2f}s"
            )
        
        except Exception as e:
            self.logger.error(f"Failed to build indices: {e}")
            self.metadata_store.update_document_status(
                metadata.document_id,
                ProcessingStatus.FAILED,
                error_message=str(e)
            )
            raise
    
    def add_document(
        self,
        chunks: List[DocumentChunk],
        metadata: DocumentMetadata,
        show_progress: bool = True
    ):
        """
        Add a document to all indices.
        Alias for build_indices for clarity.
        
        Args:
            chunks: Document chunks
            metadata: Document metadata
            show_progress: Show progress
        """
        self.build_indices(chunks, metadata, show_progress)
    
    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None
    ) -> List[DocumentChunk]:
        """
        Perform hybrid search (vector + BM25).
        
        Args:
            query: Search query
            k: Number of results
            vector_weight: Weight for vector search (default from settings)
            bm25_weight: Weight for BM25 search (default from settings)
        
        Returns:
            List of DocumentChunk objects with scores
        """
        vector_weight = vector_weight or settings.VECTOR_WEIGHT
        bm25_weight = bm25_weight or settings.BM25_WEIGHT
        
        # 1. Vector search
        query_embedding = self.embedder.embed_query(query)
        vector_chunk_ids, vector_distances = self.faiss_manager.search(
            query_embedding,
            k=k * 2  # Retrieve more for fusion
        )
        
        # 2. BM25 search
        bm25_chunk_ids, bm25_scores = self.bm25_index.search(
            query,
            k=k * 2
        )
        
        # 3. Reciprocal Rank Fusion
        fused_scores = self._reciprocal_rank_fusion(
            vector_results=list(zip(vector_chunk_ids, vector_distances)),
            bm25_results=list(zip(bm25_chunk_ids, bm25_scores)),
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            k_param=60
        )
        
        # Get top k results
        top_chunk_ids = list(fused_scores.keys())[:k]
        
        # Retrieve chunks from metadata store
        chunks = self.metadata_store.get_chunks_batch(top_chunk_ids)
        
        # Add scores to chunks
        for chunk in chunks:
            if chunk and chunk.chunk_id in fused_scores:
                chunk.metadata = chunk.metadata or {}
                chunk.metadata["hybrid_score"] = fused_scores[chunk.chunk_id]
        
        return [c for c in chunks if c is not None]
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[tuple],
        bm25_results: List[tuple],
        vector_weight: float,
        bm25_weight: float,
        k_param: int = 60
    ) -> dict:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            vector_results: List of (chunk_id, distance) tuples
            bm25_results: List of (chunk_id, score) tuples
            vector_weight: Weight for vector results
            bm25_weight: Weight for BM25 results
            k_param: RRF parameter (typically 60)
        
        Returns:
            Dictionary of {chunk_id: score}
        """
        scores = {}
        
        # Process vector results
        for rank, (chunk_id, _) in enumerate(vector_results, 1):
            rrf_score = vector_weight / (k_param + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
        
        # Process BM25 results
        for rank, (chunk_id, _) in enumerate(bm25_results, 1):
            rrf_score = bm25_weight / (k_param + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
        
        # Sort by score
        sorted_scores = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_scores
    
    def save_indices(self):
        """Save all indices to disk"""
        self.logger.info("Saving indices...")
        
        # Save FAISS
        faiss_dir = self.vector_store_dir / "faiss"
        self.faiss_manager.save(faiss_dir)
        
        # Save BM25
        bm25_dir = self.vector_store_dir / "bm25"
        self.bm25_index.save(bm25_dir)
        
        self.logger.info(f"Indices saved to {self.vector_store_dir}")
    
    def load_indices(self):
        """Load all indices from disk"""
        self.logger.info("Loading indices...")
        
        # Load FAISS
        faiss_dir = self.vector_store_dir / "faiss"
        if faiss_dir.exists():
            self.faiss_manager.load(faiss_dir)
        else:
            self.logger.warning(f"FAISS index not found at {faiss_dir}")
        
        # Load BM25
        bm25_dir = self.vector_store_dir / "bm25"
        if bm25_dir.exists():
            self.bm25_index.load(bm25_dir)
        else:
            self.logger.warning(f"BM25 index not found at {bm25_dir}")
        
        self.logger.info("Indices loaded")
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "faiss": self.faiss_manager.get_statistics(),
            "bm25": self.bm25_index.get_statistics(),
            "metadata": self.metadata_store.get_statistics(),
        }
    
    def reset_indices(self):
        """Reset all indices"""
        self.faiss_manager.reset()
        self.bm25_index.reset()
        self.logger.info("All indices reset")


# Global index builder instance
_builder = None


def get_index_builder() -> IndexBuilder:
    """
    Get global IndexBuilder instance.
    
    Returns:
        IndexBuilder instance
    """
    global _builder
    if _builder is None:
        _builder = IndexBuilder()
    return _builder


if __name__ == "__main__":
    # Test index builder
    print("=== Index Builder Tests ===\n")
    
    from config.models import DocumentType
    
    # Create test data
    print("Test 1: Create indices")
    builder = IndexBuilder()
    
    # Create test chunks
    test_chunks = []
    for i in range(10):
        chunk = DocumentChunk(
            chunk_id=f"chunk_test_{i}",
            document_id="doc_test_001",
            text=f"This is test chunk number {i} about machine learning and AI.",
            chunk_index=i,
            start_char=i*100,
            end_char=(i+1)*100,
            token_count=15
        )
        test_chunks.append(chunk)
    
    # Create metadata
    metadata = DocumentMetadata(
        document_id="doc_test_001",
        filename="test.txt",
        document_type=DocumentType.TXT,
        file_size_bytes=5000
    )
    
    # Build indices
    print("  Building indices...")
    builder.build_indices(test_chunks, metadata, show_progress=False)
    print("  ✓ Indices built")
    print()
    
    # Test 2: Search
    print("Test 2: Hybrid search")
    query = "machine learning"
    results = builder.search_hybrid(query, k=5)
    
    print(f"  Query: '{query}'")
    print(f"  Found {len(results)} results:")
    for i, chunk in enumerate(results, 1):
        score = chunk.metadata.get("hybrid_score", 0)
        print(f"    {i}. {chunk.chunk_id} (score: {score:.4f})")
        print(f"       {chunk.text[:60]}...")
    print()
    
    # Test 3: Statistics
    print("Test 3: Get statistics")
    stats = builder.get_statistics()
    print(f"  FAISS: {stats['faiss']['num_vectors']} vectors")
    print(f"  BM25: {stats['bm25']['num_documents']} documents")
    print(f"  Metadata: {stats['metadata']['total_chunks']} chunks")
    print()
    
    # Test 4: Save and load
    print("Test 4: Save and load indices")
    test_dir = Path("test_vector_store")
    builder.vector_store_dir = test_dir
    builder.save_indices()
    print(f"  Saved to {test_dir}")
    
    # Create new builder and load
    builder2 = IndexBuilder(vector_store_dir=test_dir)
    builder2.load_indices()
    stats2 = builder2.get_statistics()
    print(f"  Loaded: {stats2['faiss']['num_vectors']} vectors")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    Path("test_metadata.db").unlink(missing_ok=True)
    print()
    
    print("✓ Index builder module created successfully!")