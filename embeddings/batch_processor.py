# DEPENDENCIES
import numpy as np
from typing import List
from typing import Optional
from numpy.typing import NDArray
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import EmbeddingError
from chunking.token_counter import get_token_counter
from sentence_transformers import SentenceTransformer
from utils.helpers import BatchProcessor as BaseBatchProcessor


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class BatchProcessor:
    """
    Efficient batch processing for embeddings: Handles large batches with memory optimization and progress tracking
    """
    def __init__(self):
        self.logger           = logger
        self.base_processor   = BaseBatchProcessor()
        
        # Batch processing statistics
        self.total_batches    = 0
        self.total_texts      = 0
        self.failed_batches   = 0
    

    @handle_errors(error_type = EmbeddingError, log_error = True, reraise = True)
    def process_embeddings_batch(self, model: SentenceTransformer, texts: List[str], batch_size: Optional[int] = None, normalize: bool = True, **kwargs) -> List[NDArray]:
        """
        Process embeddings in optimized batches
        
        Arguments:
        ----------
            model      { SentenceTransformer } : Embedding model

            texts             { list }         : List of texts to embed
            
            batch_size        { int }          : Batch size (default from settings)
            
            normalize         { bool }         : Normalize embeddings
            
            **kwargs                           : Additional model.encode parameters
        
        Returns:
        --------
                      { list }                 : List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        self.logger.debug(f"Processing {len(texts)} texts in batches of {batch_size}")
        
        try:
            # Use model's built-in batching with optimization
            embeddings          = model.encode(texts,
                                               batch_size           = batch_size,
                                               normalize_embeddings = normalize,
                                               show_progress_bar    = False,
                                               convert_to_numpy     = True,
                                               **kwargs
                                              )
            
            # Update statistics
            self.total_batches += ((len(texts) + batch_size - 1) // batch_size)
            self.total_texts   += len(texts)
            
            self.logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            
            # Convert to list of arrays
            return list(embeddings)  
            
        except Exception as e:
            self.failed_batches += 1
            self.logger.error(f"Batch embedding failed: {repr(e)}")
            raise EmbeddingError(f"Batch processing failed: {repr(e)}")
    

    def process_embeddings_with_fallback(self, model: SentenceTransformer, texts: List[str], batch_size: Optional[int] = None, normalize: bool = True) -> List[NDArray]:
        """
        Process embeddings with automatic batch size reduction on failure
        
        Arguments:
        ----------
            model      { SentenceTransformer } : Embedding model

            texts      { list }                : List of texts
            
            batch_size { int }                 : Initial batch size
            
            normalize  { bool }                : Normalize embeddings
        
        Returns:
        --------
                 { list }                      : List of embeddings
        """
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        try:
            return self.process_embeddings_batch(model      = model,
                                                 texts      = texts,
                                                 batch_size = batch_size,
                                                 normalize  = normalize,
                                                )
        
        except (MemoryError, RuntimeError) as e:
            self.logger.warning(f"Batch size {batch_size} failed, reducing to {batch_size // 2}")
            
            # Reduce batch size and retry
            return self.process_embeddings_batch(model      = model,
                                                 texts      = texts,
                                                 batch_size = batch_size // 2,
                                                 normalize  = normalize,
                                                )
    

    def split_into_optimal_batches(self, texts: List[str], target_batch_size: int, max_batch_size: int = 1000) -> List[List[str]]:
        """
        Split texts into optimal batches considering token counts
        
        Arguments:
        ----------
            texts            { list } : List of texts

            target_batch_size { int } : Target batch size in texts
            
            max_batch_size    { int } : Maximum batch size to allow
        
        Returns:
        --------
                       { list }       : List of text batches
        """
        if not texts:
            return []
        
        token_counter  = get_token_counter()
        batches        = list()
        current_batch  = list()
        current_tokens = 0
        
        # Estimate tokens per text (average of first 10 or all if less)
        sample_size    = min(10, len(texts))
        sample_tokens  = [token_counter.count_tokens(text) for text in texts[:sample_size]]
        avg_tokens     = sum(sample_tokens) / len(sample_tokens) if sample_tokens else 100
        
        # Target tokens per batch (approximate)
        target_tokens  = target_batch_size * avg_tokens
        
        for text in texts:
            text_tokens = token_counter.count_tokens(text)
            
            # If single text is too large, put it in its own batch
            if (text_tokens > (target_tokens * 0.8)):
                if current_batch:
                    batches.append(current_batch)
                    current_batch  = list()
                    current_tokens = 0
                
                batches.append([text])
                continue
            
            # Check if adding this text would exceed limits
            if (((current_tokens + text_tokens) > target_tokens) and current_batch) or (len(current_batch) >= max_batch_size):
                batches.append(current_batch)
                current_batch  = list()
                current_tokens = 0
            
            current_batch.append(text)
            current_tokens += text_tokens
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        self.logger.debug(f"Split {len(texts)} texts into {len(batches)} optimal batches")
        
        return batches
    

    def process_batches_with_progress(self, model: SentenceTransformer, texts: List[str], batch_size: Optional[int] = None, progress_callback: Optional[callable] = None, **kwargs) -> List[NDArray]:
        """
        Process batches with progress reporting
        
        Arguments:
        ----------
            model            { SentenceTransformer } : Embedding model

            texts            { list }                : List of texts
            
            batch_size       { int }                 : Batch size
            
            progress_callback { callable }           : Callback for progress updates
            
            **kwargs                                 : Additional parameters
        
        Returns:
        --------
                         { list }                    : List of embeddings
        """
        if not texts:
            return []
        
        batch_size     = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        # Split into batches
        batches        = self.split_into_optimal_batches(texts             = texts, 
                                                         target_batch_size = batch_size,
                                                        )
        
        all_embeddings = list()
        
        for i, batch_texts in enumerate(batches):
            if progress_callback:
                progress = (i / len(batches)) * 100
                progress_callback(progress, f"Processing batch {i + 1}/{len(batches)}")
            
            try:
                batch_embeddings = self.process_embeddings_batch(model      = model,
                                                                 texts      = batch_texts,
                                                                 batch_size = len(batch_texts),
                                                                 **kwargs
                                                                )
                
                all_embeddings.extend(batch_embeddings)
                
                self.logger.debug(f"Processed batch {i + 1}/{len(batches)}: {len(batch_texts)} texts")
            
            except Exception as e:
                self.logger.error(f"Failed to process batch {i + 1}: {repr(e)}")
                
                # Add None placeholders for failed batch
                all_embeddings.extend([None] * len(batch_texts))
        
        if progress_callback:
            progress_callback(100, "Embedding complete")
        
        return all_embeddings
    

    def validate_embeddings_batch(self, embeddings: List[NDArray], expected_count: int) -> bool:
        """
        Validate a batch of embeddings
        
        Arguments:
        ----------
            embeddings     { list } : List of embedding vectors

            expected_count { int }  : Expected number of embeddings
        
        Returns:
        --------
                   { bool }         : True if valid
        """
        if (len(embeddings) != expected_count):
            self.logger.error(f"Embedding count mismatch: expected {expected_count}, got {len(embeddings)}")
            return False
        
        valid_count = 0
        
        for i, emb in enumerate(embeddings):
            if emb is None:
                self.logger.warning(f"None embedding at index {i}")
                continue
            
            if not isinstance(emb, np.ndarray):
                self.logger.warning(f"Invalid embedding type at index {i}: {type(emb)}")
                continue
            
            if (emb.ndim != 1):
                self.logger.warning(f"Invalid embedding dimension at index {i}: {emb.ndim}")
                continue
            
            if np.any(np.isnan(emb)):
                self.logger.warning(f"NaN values in embedding at index {i}")
                continue
            
            valid_count += 1
        
        validity_ratio = valid_count / expected_count
        
        if (validity_ratio < 0.9):
            self.logger.warning(f"Low embedding validity: {valid_count}/{expected_count} ({validity_ratio:.1%})")
            return False
        
        return True
    

    def get_processing_stats(self) -> dict:
        """
        Get batch processing statistics
        
        Returns:
        --------
            { dict }    : Statistics dictionary
        """
        success_rate = ((self.total_batches - self.failed_batches) / self.total_batches * 100) if (self.total_batches > 0) else 100
        
        stats        = {"total_batches"    : self.total_batches,
                        "total_texts"      : self.total_texts,
                        "failed_batches"   : self.failed_batches,
                        "success_rate"     : success_rate,
                        "avg_batch_size"   : self.total_texts / self.total_batches if (self.total_batches > 0) else 0,
                       }
        
        return stats
    

    def reset_stats(self):
        """
        Reset processing statistics
        """
        self.total_batches  = 0
        self.total_texts    = 0
        self.failed_batches = 0
        
        self.logger.debug("Reset batch processing statistics")


# Global batch processor instance
_batch_processor = None


def get_batch_processor() -> BatchProcessor:
    """
    Get global batch processor instance
    
    Returns:
    --------
        { BatchProcessor } : BatchProcessor instance
    """
    global _batch_processor

    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    
    return _batch_processor


def process_embeddings_batch(model: SentenceTransformer, texts: List[str], **kwargs) -> List[NDArray]:
    """
    Convenience function for batch embedding
    
    Arguments:
    ----------
        model { SentenceTransformer } : Embedding model

        texts { list }                : List of texts
        
        **kwargs                      : Additional arguments
    
    Returns:
    --------
             { list }                 : List of embeddings
    """
    processor = get_batch_processor()

    return processor.process_embeddings_batch(model, texts, **kwargs)