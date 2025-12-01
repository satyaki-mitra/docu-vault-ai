# DEPENDENCIES
import time
import numpy as np
from typing import List
from typing import Optional
from numpy.typing import NDArray
from config.models import DocumentChunk
from config.settings import get_settings
from config.models import EmbeddingRequest
from config.models import EmbeddingResponse
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import EmbeddingError
from embeddings.model_loader import get_model_loader
from embeddings.batch_processor import BatchProcessor


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class BGEEmbedder:
    """
    BGE (BAAI General Embedding) model wrapper: Optimized for BAAI/bge models with proper normalization and batching
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize BGE embedder
        
        Arguments:
        ----------
            model_name { str } : BGE model name (default from settings)
            
            device     { str } : Device to run on
        """
        self.logger          = logger
        self.model_name      = model_name or settings.EMBEDDING_MODEL
        self.device          = device
        
        # Initialize components
        self.model_loader    = get_model_loader()
        self.batch_processor = BatchProcessor()
        
        # Load model
        self.model           = self.model_loader.load_model(model_name = self.model_name, 
                                                            device     = self.device,
                                                           )
        
        # Get model info
        self.embedding_dim   = self.model.get_sentence_embedding_dimension()
        self.supports_batch  = True
        
        self.logger.info(f"Initialized BGEEmbedder: model={self.model_name}, dim={self.embedding_dim}, device={self.model.device}")
    

    @handle_errors(error_type = EmbeddingError, log_error = True, reraise = True)
    def embed_text(self, text: str, normalize: bool = True) -> NDArray:
        """
        Embed single text string
        
        Arguments:
        ----------
            text      { str } : Input text

            normalize { bool } : Normalize embeddings to unit length
        
        Returns:
        --------
               { NDArray }     : Embedding vector
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        try:
            # Encode single text
            embedding = self.model.encode([text],
                                          normalize_embeddings = normalize,
                                          show_progress_bar    = False,
                                         )
            
            # Return single vector
            return embedding[0]  
            
        except Exception as e:
            self.logger.error(f"Failed to embed text: {repr(e)}")
            raise EmbeddingError(f"Text embedding failed: {repr(e)}")
    

    @handle_errors(error_type = EmbeddingError, log_error = True, reraise = True)
    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None, normalize: bool = True) -> List[NDArray]:
        """
        Embed multiple texts with batching
        
        Arguments:
        ----------
            texts      { list } : List of text strings

            batch_size { int }  : Batch size (default from settings)
            
            normalize  { bool } : Normalize embeddings
        
        Returns:
        --------
                 { list }       : List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if (len(valid_texts) != len(texts)):
            self.logger.warning(f"Filtered {len(texts) - len(valid_texts)} empty texts")
        
        if not valid_texts:
            return []
        
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        try:
            # Use batch processing for efficiency
            embeddings = self.batch_processor.process_embeddings_batch(model      = self.model,
                                                                       texts      = valid_texts,
                                                                       batch_size = batch_size,
                                                                       normalize  = normalize,
                                                                      )
            
            self.logger.debug(f"Generated {len(embeddings)} embeddings for {len(texts)} texts")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Batch embedding failed: {repr(e)}")
            raise EmbeddingError(f"Batch embedding failed: {repr(e)}")
    

    @handle_errors(error_type = EmbeddingError, log_error = True, reraise = True)
    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: Optional[int] = None, normalize: bool = True) -> List[DocumentChunk]:
        """
        Embed document chunks and update them with embeddings
        
        Arguments:
        ----------
            chunks     { list } : List of DocumentChunk objects

            batch_size { int }  : Batch size
            
            normalize  { bool } : Normalize embeddings
        
        Returns:
        --------
                 { list }       : Chunks with embeddings added
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts      = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts       = texts,
                                      batch_size  = batch_size,
                                      normalize   = normalize,
                                     )
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            # Convert numpy to list for serialization
            chunk.embedding = embedding.tolist()  
        
        self.logger.info(f"Embedded {len(chunks)} document chunks")
        
        return chunks
    

    def process_embedding_request(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Process embedding request from API
        
        Arguments:
        ----------
            request { EmbeddingRequest } : Embedding request
        
        Returns:
        --------
            { EmbeddingResponse }        : Embedding response
        """
        start_time      = time.time()
        
        # Generate embeddings
        embeddings      = self.embed_texts(texts      = request.texts,
                                           batch_size = request.batch_size,
                                           normalize  = request.normalize,
                                          )
        
        # Convert to milliseconds
        processing_time = (time.time() - start_time) * 1000  
        
        # Convert to list for serialization
        embedding_list  = [emb.tolist() for emb in embeddings]
        
        response        = EmbeddingResponse(embeddings         = embedding_list,
                                            dimension          = self.embedding_dim,
                                            num_embeddings     = len(embeddings),
                                            processing_time_ms = processing_time,
                                           )
         
        return response
    

    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension
        
        Returns:
        --------
            { int }    : Embedding vector dimension
        """
        return self.embedding_dim
    

    def cosine_similarity(self, emb1: NDArray, emb2: NDArray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Arguments:
        ----------
            emb1 { NDArray } : First embedding

            emb2 { NDArray } : Second embedding
        
        Returns:
        --------
               { float }     : Cosine similarity (-1 to 1)
        """
        # Ensure embeddings are normalized
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        return float(np.dot(emb1_norm, emb2_norm))
    

    def validate_embedding(self, embedding: NDArray) -> bool:
        """
        Validate embedding vector
        
        Arguments:
        ----------
            embedding { NDArray } : Embedding vector
        
        Returns:
        --------
                 { bool }         : True if valid
        """
        if (embedding is None):
            return False
        
        if (not isinstance(embedding, np.ndarray)):
            return False
        
        if (embedding.shape != (self.embedding_dim,)):
            return False
        
        if (np.all(embedding == 0)):
            return False
        
        if (np.any(np.isnan(embedding))):
            return False
        
        return True
    

    def get_model_info(self) -> dict:
        """
        Get embedder information
        
        Returns:
        --------
            { dict }    : Embedder information
        """
        return {"model_name"        : self.model_name,
                "embedding_dim"     : self.embedding_dim,
                "device"            : str(self.model.device),
                "supports_batch"    : self.supports_batch,
                "normalize_default" : True,
               }


# Global embedder instance
_embedder = None


def get_embedder(model_name: Optional[str] = None, device: Optional[str] = None) -> BGEEmbedder:
    """
    Get global embedder instance
    
    Arguments:
    ----------
        model_name { str } : Model name
        
        device     { str } : Device
    
    Returns:
    --------
        { BGEEmbedder }    : BGEEmbedder instance
    """
    global _embedder

    if _embedder is None:
        _embedder = BGEEmbedder(model_name, device)
    
    return _embedder


def embed_texts(texts: List[str], **kwargs) -> List[NDArray]:
    """
    Convenience function to embed texts
    
    Arguments:
    ----------
        texts { list } : List of texts
        
        **kwargs       : Additional arguments for BGEEmbedder
    
    Returns:
    --------
             { list }  : List of embeddings
    """
    embedder = get_embedder()

    return embedder.embed_texts(texts, **kwargs)


def embed_chunks(chunks: List[DocumentChunk], **kwargs) -> List[DocumentChunk]:
    """
    Convenience function to embed document chunks
    
    Arguments:
    ----------
        chunks { list } : List of DocumentChunk objects
        
        **kwargs        : Additional arguments
    
    Returns:
    --------
              { list }  : Chunks with embeddings
    """
    embedder = get_embedder()

    return embedder.embed_chunks(chunks, **kwargs)