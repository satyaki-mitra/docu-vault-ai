"""
BGE Embedder
Generate embeddings using BGE (BAAI General Embedding) models
"""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from config.logging_config import get_logger
from config.settings import get_settings
from config.models import DocumentChunk, EmbeddingRequest, EmbeddingResponse
from embeddings.model_loader import get_model_loader

logger = get_logger(__name__)
settings = get_settings()


class BGEEmbedder:
    """
    BGE embedding generator.
    Optimized for generating embeddings for RAG applications.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: Optional[int] = None
    ):
        """
        Initialize BGE embedder.
        
        Args:
            model_name: Model name (default from settings)
            device: Device (cuda, mps, cpu)
            normalize_embeddings: Normalize to unit length
            batch_size: Batch size for encoding (default from settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        self.logger = logger
        
        # Load model
        self.loader = get_model_loader()
        self.model = self.loader.load_model(
            model_name=self.model_name,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings
        )
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.logger.info(
            f"BGEEmbedder initialized: model={self.model_name}, "
            f"dim={self.embedding_dim}, batch_size={self.batch_size}"
        )
    
    def embed_text(
        self,
        text: str,
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            normalize: Override normalization setting
        
        Returns:
            Embedding vector (numpy array)
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize if normalize is not None else self.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            return embedding
        
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size (default: from init)
            show_progress: Show progress bar
            normalize: Override normalization setting
        
        Returns:
            Array of embeddings (shape: [num_texts, embedding_dim])
        """
        if not texts:
            self.logger.warning("Empty text list provided")
            return np.empty((0, self.embedding_dim))
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(texts) if text and text.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            self.logger.warning("All texts are empty")
            return np.zeros((len(texts), self.embedding_dim))
        
        batch_size = batch_size or self.batch_size
        
        try:
            self.logger.debug(f"Generating embeddings for {len(valid_texts)} texts")
            
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize if normalize is not None else self.normalize_embeddings,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Handle empty texts by inserting zero vectors
            if len(valid_indices) < len(texts):
                full_embeddings = np.zeros((len(texts), self.embedding_dim))
                full_embeddings[valid_indices] = embeddings
                embeddings = full_embeddings
            
            self.logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            
            return embeddings
        
        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks and update them in-place.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size
            show_progress: Show progress bar
        
        Returns:
            Updated chunks with embeddings
        """
        if not chunks:
            return chunks
        
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        
        self.logger.info(f"Successfully embedded {len(chunks)} chunks")
        
        return chunks
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        Some models use different embeddings for queries vs documents.
        
        Args:
            query: Search query
        
        Returns:
            Query embedding
        """
        # For BGE models, we can add "Represent this sentence for searching relevant passages: "
        # but the model handles this internally, so we just encode normally
        return self.embed_text(query)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
        
        Returns:
            Similarity score (0-1 if normalized, -1 to 1 otherwise)
        """
        # If embeddings are normalized, dot product = cosine similarity
        if self.normalize_embeddings:
            return float(np.dot(embedding1, embedding2))
        else:
            # Compute cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def compute_similarities(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarities between query and multiple documents.
        
        Args:
            query_embedding: Query embedding (1D array)
            document_embeddings: Document embeddings (2D array)
        
        Returns:
            Array of similarity scores
        """
        if self.normalize_embeddings:
            # Simple dot product for normalized vectors
            similarities = np.dot(document_embeddings, query_embedding)
        else:
            # Compute cosine similarities
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(document_embeddings, axis=1)
            
            # Avoid division by zero
            doc_norms[doc_norms == 0] = 1e-10
            
            similarities = np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm)
        
        return similarities
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Input text
        
        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within model's max sequence length.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens (default: model's max)
        
        Returns:
            Truncated text
        """
        max_tokens = max_tokens or self.model.max_seq_length
        
        # Estimate current tokens
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Truncate to approximate character count
        chars_per_token = len(text) / estimated_tokens
        max_chars = int(max_tokens * chars_per_token)
        
        truncated = text[:max_chars]
        
        self.logger.debug(
            f"Truncated text from {estimated_tokens} to ~{max_tokens} tokens"
        )
        
        return truncated
    
    def create_embedding_request(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> EmbeddingRequest:
        """
        Create embedding request object.
        
        Args:
            texts: Texts to embed
            batch_size: Batch size
            normalize: Normalize embeddings
        
        Returns:
            EmbeddingRequest object
        """
        return EmbeddingRequest(
            texts=texts,
            batch_size=batch_size or self.batch_size,
            normalize=normalize
        )
    
    def process_embedding_request(
        self,
        request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """
        Process embedding request.
        
        Args:
            request: EmbeddingRequest object
        
        Returns:
            EmbeddingResponse object
        """
        import time
        
        start_time = time.time()
        
        embeddings = self.embed_texts(
            texts=request.texts,
            batch_size=request.batch_size,
            normalize=request.normalize
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            dimension=self.embedding_dim,
            num_embeddings=len(embeddings),
            processing_time_ms=processing_time
        )


# Global embedder instance
_embedder = None


def get_embedder(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> BGEEmbedder:
    """
    Get global BGEEmbedder instance.
    
    Args:
        model_name: Model name
        device: Device
    
    Returns:
        BGEEmbedder instance
    """
    global _embedder
    if _embedder is None or (model_name and model_name != _embedder.model_name):
        _embedder = BGEEmbedder(model_name=model_name, device=device)
    return _embedder


def embed_text(text: str) -> np.ndarray:
    """
    Convenience function to embed single text.
    
    Args:
        text: Input text
    
    Returns:
        Embedding vector
    """
    embedder = get_embedder()
    return embedder.embed_text(text)


def embed_texts(texts: List[str], show_progress: bool = False) -> np.ndarray:
    """
    Convenience function to embed multiple texts.
    
    Args:
        texts: Input texts
        show_progress: Show progress bar
    
    Returns:
        Embeddings array
    """
    embedder = get_embedder()
    return embedder.embed_texts(texts, show_progress=show_progress)


if __name__ == "__main__":
    # Test BGE embedder
    print("=== BGE Embedder Tests ===\n")
    
    try:
        embedder = BGEEmbedder()
        
        # Test 1: Single text embedding
        print("Test 1: Single text embedding")
        text = "This is a test sentence for embedding."
        embedding = embedder.embed_text(text)
        print(f"  Text: {text}")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"  First 5 values: {embedding[:5]}")
        print()
        
        # Test 2: Batch embedding
        print("Test 2: Batch embedding")
        texts = [
            "Artificial intelligence is transforming technology.",
            "Machine learning enables computers to learn from data.",
            "Natural language processing helps machines understand text."
        ]
        embeddings = embedder.embed_texts(texts)
        print(f"  Embedded {len(texts)} texts")
        print(f"  Embeddings shape: {embeddings.shape}")
        print()
        
        # Test 3: Similarity computation
        print("Test 3: Similarity computation")
        query = "AI and machine learning"
        query_emb = embedder.embed_query(query)
        
        similarities = embedder.compute_similarities(query_emb, embeddings)
        print(f"  Query: {query}")
        for i, (text, sim) in enumerate(zip(texts, similarities)):
            print(f"  Doc {i} (sim={sim:.4f}): {text[:50]}...")
        print()
        
        # Test 4: DocumentChunk embedding
        print("Test 4: DocumentChunk embedding")
        from config.models import DocumentChunk
        
        chunks = [
            DocumentChunk(
                chunk_id=f"chunk_{i}",
                document_id="test_doc",
                text=text,
                chunk_index=i,
                start_char=0,
                end_char=len(text),
                token_count=len(text.split())
            )
            for i, text in enumerate(texts)
        ]
        
        embedded_chunks = embedder.embed_chunks(chunks)
        print(f"  Embedded {len(embedded_chunks)} chunks")
        print(f"  First chunk has embedding: {embedded_chunks[0].embedding is not None}")
        print(f"  Embedding length: {len(embedded_chunks[0].embedding)}")
        print()
        
        # Test 5: Convenience functions
        print("Test 5: Convenience functions")
        test_embedding = embed_text("Quick test")
        print(f"  Convenience embed shape: {test_embedding.shape}")
        print()
        
        # Test 6: Truncation
        print("Test 6: Text truncation")
        long_text = "This is a test sentence. " * 1000
        truncated = embedder.truncate_text(long_text, max_tokens=100)
        print(f"  Original length: {len(long_text)} chars")
        print(f"  Truncated length: {len(truncated)} chars")
        print()
        
        print("✓ BGE embedder module created successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This module requires sentence-transformers library")
        print("Install with: pip install sentence-transformers")