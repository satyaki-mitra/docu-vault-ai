"""
Batch Processor
Efficient batch processing for embedding generation
"""

from typing import List, Optional, Callable, Generator
import time
from dataclasses import dataclass
import numpy as np

from config.logging_config import get_logger
from config.settings import get_settings
from config.models import DocumentChunk
from embeddings.bge_embedder import BGEEmbedder, get_embedder

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class BatchStats:
    """Statistics for batch processing"""
    total_items: int
    processed_items: int
    failed_items: int
    total_time_seconds: float
    avg_time_per_batch: float
    items_per_second: float


class BatchProcessor:
    """
    Efficient batch processor for embeddings.
    Handles large datasets with progress tracking and error recovery.
    """
    
    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            embedder: BGE embedder instance (default: global)
            batch_size: Batch size (default: from settings)
            show_progress: Show progress during processing
        """
        self.embedder = embedder or get_embedder()
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.show_progress = show_progress
        self.logger = logger
        
        self.logger.info(f"BatchProcessor initialized: batch_size={self.batch_size}")
    
    def process_texts(
        self,
        texts: List[str],
        callback: Optional[Callable[[int, np.ndarray], None]] = None
    ) -> np.ndarray:
        """
        Process texts in batches.
        
        Args:
            texts: List of texts to embed
            callback: Optional callback(batch_idx, embeddings) after each batch
        
        Returns:
            Array of all embeddings
        """
        if not texts:
            return np.empty((0, self.embedder.embedding_dim))
        
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        self.logger.info(
            f"Processing {len(texts)} texts in {total_batches} batches"
        )
        
        all_embeddings = []
        start_time = time.time()
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            # Generate embeddings for batch
            try:
                batch_embeddings = self.embedder.embed_texts(
                    batch_texts,
                    show_progress=False
                )
                all_embeddings.append(batch_embeddings)
                
                # Call callback if provided
                if callback:
                    callback(batch_idx, batch_embeddings)
                
                # Log progress
                if self.show_progress:
                    progress = (batch_idx + 1) / total_batches * 100
                    elapsed = time.time() - start_time
                    rate = (batch_end) / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(
                        f"Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) | "
                        f"Rate: {rate:.1f} texts/s"
                    )
            
            except Exception as e:
                self.logger.error(f"Failed to process batch {batch_idx}: {e}")
                # Add zero embeddings for failed batch
                zero_embeddings = np.zeros(
                    (len(batch_texts), self.embedder.embedding_dim)
                )
                all_embeddings.append(zero_embeddings)
        
        # Concatenate all batches
        final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.embedder.embedding_dim))
        
        total_time = time.time() - start_time
        self.logger.info(
            f"Completed: {len(texts)} texts in {total_time:.2f}s "
            f"({len(texts) / total_time:.1f} texts/s)"
        )
        
        return final_embeddings
    
    def process_chunks(
        self,
        chunks: List[DocumentChunk],
        callback: Optional[Callable[[int, List[DocumentChunk]], None]] = None
    ) -> List[DocumentChunk]:
        """
        Process document chunks in batches.
        
        Args:
            chunks: List of chunks to embed
            callback: Optional callback(batch_idx, chunks) after each batch
        
        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks
        
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        self.logger.info(
            f"Processing {len(chunks)} chunks in {total_batches} batches"
        )
        
        start_time = time.time()
        processed_chunks = []
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            try:
                # Extract texts
                texts = [chunk.text for chunk in batch_chunks]
                
                # Generate embeddings
                embeddings = self.embedder.embed_texts(
                    texts,
                    show_progress=False
                )
                
                # Update chunks
                for chunk, embedding in zip(batch_chunks, embeddings):
                    chunk.embedding = embedding.tolist()
                
                processed_chunks.extend(batch_chunks)
                
                # Call callback
                if callback:
                    callback(batch_idx, batch_chunks)
                
                # Log progress
                if self.show_progress:
                    progress = (batch_idx + 1) / total_batches * 100
                    elapsed = time.time() - start_time
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(
                        f"Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) | "
                        f"Rate: {rate:.1f} chunks/s"
                    )
            
            except Exception as e:
                self.logger.error(f"Failed to process chunk batch {batch_idx}: {e}")
                # Add chunks without embeddings
                processed_chunks.extend(batch_chunks)
        
        total_time = time.time() - start_time
        self.logger.info(
            f"Completed: {len(chunks)} chunks in {total_time:.2f}s "
            f"({len(chunks) / total_time:.1f} chunks/s)"
        )
        
        return processed_chunks
    
    def process_generator(
        self,
        text_generator: Generator[str, None, None],
        max_items: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Process texts from a generator in batches.
        Yields embeddings as they are generated.
        
        Args:
            text_generator: Generator yielding texts
            max_items: Maximum items to process (None = all)
        
        Yields:
            Embedding arrays for each batch
        """
        batch_texts = []
        processed_count = 0
        
        for text in text_generator:
            batch_texts.append(text)
            processed_count += 1
            
            # Process batch when full
            if len(batch_texts) >= self.batch_size:
                try:
                    embeddings = self.embedder.embed_texts(
                        batch_texts,
                        show_progress=False
                    )
                    yield embeddings
                except Exception as e:
                    self.logger.error(f"Failed to process generator batch: {e}")
                    yield np.zeros((len(batch_texts), self.embedder.embedding_dim))
                
                batch_texts = []
            
            # Check max items
            if max_items and processed_count >= max_items:
                break
        
        # Process remaining texts
        if batch_texts:
            try:
                embeddings = self.embedder.embed_texts(
                    batch_texts,
                    show_progress=False
                )
                yield embeddings
            except Exception as e:
                self.logger.error(f"Failed to process final batch: {e}")
                yield np.zeros((len(batch_texts), self.embedder.embedding_dim))
    
    def process_with_retry(
        self,
        texts: List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> np.ndarray:
        """
        Process texts with automatic retry on failure.
        
        Args:
            texts: Texts to embed
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
        
        Returns:
            Embeddings array
        """
        for attempt in range(max_retries):
            try:
                return self.process_texts(texts)
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.logger.error("All retry attempts failed")
                    raise
    
    def get_statistics(
        self,
        total_items: int,
        processed_items: int,
        failed_items: int,
        total_time: float
    ) -> BatchStats:
        """
        Calculate processing statistics.
        
        Args:
            total_items: Total items to process
            processed_items: Successfully processed items
            failed_items: Failed items
            total_time: Total processing time (seconds)
        
        Returns:
            BatchStats object
        """
        num_batches = (total_items + self.batch_size - 1) // self.batch_size
        
        return BatchStats(
            total_items=total_items,
            processed_items=processed_items,
            failed_items=failed_items,
            total_time_seconds=total_time,
            avg_time_per_batch=total_time / num_batches if num_batches > 0 else 0,
            items_per_second=processed_items / total_time if total_time > 0 else 0
        )
    
    def estimate_time(self, num_items: int, items_per_second: float = 50) -> float:
        """
        Estimate processing time.
        
        Args:
            num_items: Number of items to process
            items_per_second: Processing rate (default: 50)
        
        Returns:
            Estimated time in seconds
        """
        return num_items / items_per_second
    
    def optimize_batch_size(
        self,
        sample_texts: List[str],
        target_time_per_batch: float = 1.0
    ) -> int:
        """
        Find optimal batch size for given texts.
        
        Args:
            sample_texts: Sample texts to test
            target_time_per_batch: Target time per batch (seconds)
        
        Returns:
            Recommended batch size
        """
        if len(sample_texts) < 10:
            self.logger.warning("Need at least 10 sample texts for optimization")
            return self.batch_size
        
        test_sizes = [8, 16, 32, 64, 128]
        results = []
        
        for size in test_sizes:
            if size > len(sample_texts):
                break
            
            test_batch = sample_texts[:size]
            
            start_time = time.time()
            try:
                _ = self.embedder.embed_texts(test_batch, show_progress=False)
                elapsed = time.time() - start_time
                results.append((size, elapsed))
            except Exception as e:
                self.logger.warning(f"Failed to test batch size {size}: {e}")
                continue
        
        if not results:
            return self.batch_size
        
        # Find size closest to target time
        best_size = min(
            results,
            key=lambda x: abs(x[1] - target_time_per_batch)
        )[0]
        
        self.logger.info(
            f"Recommended batch size: {best_size} "
            f"(target: {target_time_per_batch}s/batch)"
        )
        
        return best_size


if __name__ == "__main__":
    # Test batch processor
    print("=== Batch Processor Tests ===\n")
    
    try:
        processor = BatchProcessor(batch_size=8, show_progress=True)
        
        # Test 1: Process texts
        print("Test 1: Process texts in batches")
        test_texts = [
            f"This is test sentence number {i} for batch processing."
            for i in range(25)
        ]
        
        embeddings = processor.process_texts(test_texts)
        print(f"  Processed {len(test_texts)} texts")
        print(f"  Embeddings shape: {embeddings.shape}")
        print()
        
        # Test 2: Process chunks
        print("Test 2: Process DocumentChunks")
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
            for i, text in enumerate(test_texts[:15])
        ]
        
        processed_chunks = processor.process_chunks(chunks)
        embedded_count = sum(1 for c in processed_chunks if c.embedding is not None)
        print(f"  Processed {len(processed_chunks)} chunks")
        print(f"  Chunks with embeddings: {embedded_count}")
        print()
        
        # Test 3: Callback function
        print("Test 3: Process with callback")
        batch_count = [0]
        
        def callback(batch_idx, embeddings):
            batch_count[0] += 1
            print(f"  Callback: Batch {batch_idx} processed")
        
        embeddings = processor.process_texts(test_texts[:10], callback=callback)
        print(f"  Callback called {batch_count[0]} times")
        print()
        
        # Test 4: Statistics
        print("Test 4: Processing statistics")
        stats = processor.get_statistics(
            total_items=25,
            processed_items=25,
            failed_items=0,
            total_time=2.5
        )
        print(f"  Items per second: {stats.items_per_second:.1f}")
        print(f"  Avg time per batch: {stats.avg_time_per_batch:.3f}s")
        print()
        
        # Test 5: Time estimation
        print("Test 5: Time estimation")
        estimated = processor.estimate_time(1000, items_per_second=50)
        print(f"  Estimated time for 1000 items: {estimated:.1f}s")
        print()
        
        print("✓ Batch processor module created successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This module requires sentence-transformers library")