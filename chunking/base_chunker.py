"""
Base Chunker
Abstract base class defining chunking interface
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from config.logging_config import get_logger
from config.models import DocumentChunk, DocumentMetadata, ChunkingStrategy

logger = get_logger(__name__)


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    Implements Template Method pattern for consistent chunking pipeline.
    """
    
    def __init__(self, strategy_name: ChunkingStrategy):
        """
        Initialize base chunker.
        
        Args:
            strategy_name: Name of the chunking strategy
        """
        self.strategy_name = strategy_name
        self.logger = logger
    
    @abstractmethod
    def chunk_text(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> List[DocumentChunk]:
        """
        Chunk text into smaller pieces.
        Must be implemented by subclasses.
        
        Args:
            text: Input text to chunk
            metadata: Document metadata
        
        Returns:
            List of DocumentChunk objects
        """
        pass
    
    def chunk_document(
        self,
        text: str,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """
        Chunk document with full metadata.
        Template method that calls chunk_text and adds metadata.
        
        Args:
            text: Document text
            metadata: Document metadata
        
        Returns:
            List of DocumentChunk objects with metadata
        """
        self.logger.info(
            f"Chunking document {metadata.document_id} using {self.strategy_name}"
        )
        
        # Validate input
        if not text or not text.strip():
            self.logger.warning(f"Empty text for document {metadata.document_id}")
            return []
        
        # Perform chunking
        chunks = self.chunk_text(text, metadata)
        
        # Update metadata
        metadata.num_chunks = len(chunks)
        metadata.chunking_strategy = self.strategy_name
        
        self.logger.info(
            f"Created {len(chunks)} chunks for {metadata.document_id}"
        )
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        document_id: str,
        start_char: int,
        end_char: int,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> DocumentChunk:
        """
        Create a DocumentChunk object with proper formatting.
        
        Args:
            text: Chunk text
            chunk_index: Index of chunk in document
            document_id: Parent document ID
            start_char: Start character position
            end_char: End character position
            page_number: Page number (if applicable)
            section_title: Section heading (if applicable)
            metadata: Additional metadata
        
        Returns:
            DocumentChunk object
        """
        from chunking.token_counter import count_tokens
        
        # Generate unique chunk ID
        chunk_id = f"chunk_{document_id}_{chunk_index}"
        
        # Count tokens
        token_count = count_tokens(text)
        
        # Create chunk
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            page_number=page_number,
            section_title=section_title,
            token_count=token_count,
            metadata=metadata or {}
        )
        
        return chunk
    
    def _extract_page_number(self, text: str, full_text: str) -> Optional[int]:
        """
        Try to extract page number from text.
        Looks for [PAGE N] markers inserted during parsing.
        
        Args:
            text: Chunk text
            full_text: Full document text
        
        Returns:
            Page number or None
        """
        import re
        
        # Look for page markers
        page_match = re.search(r'\[PAGE (\d+)\]', text)
        if page_match:
            return int(page_match.group(1))
        
        # Alternative: try to determine from position in full text
        if full_text:
            position = full_text.find(text[:100])  # Use first 100 chars
            if position >= 0:
                # Count page markers before this position
                text_before = full_text[:position]
                page_matches = re.findall(r'\[PAGE (\d+)\]', text_before)
                if page_matches:
                    return int(page_matches[-1])
        
        return None
    
    def _clean_chunk_text(self, text: str) -> str:
        """
        Clean chunk text by removing markers and extra whitespace.
        
        Args:
            text: Raw chunk text
        
        Returns:
            Cleaned text
        """
        import re
        
        # Remove page markers
        text = re.sub(r'\[PAGE \d+\]', '', text)
        
        # Remove other common markers
        text = re.sub(r'\[HEADER\]|\[FOOTER\]|\[TABLE \d+\]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def validate_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Validate chunk list for consistency.
        
        Args:
            chunks: List of chunks to validate
        
        Returns:
            True if valid
        """
        if not chunks:
            return True
        
        # Check all chunks have the same document_id
        doc_ids = {chunk.document_id for chunk in chunks}
        if len(doc_ids) > 1:
            self.logger.error(f"Chunks have multiple document IDs: {doc_ids}")
            return False
        
        # Check chunk indices are sequential
        indices = [chunk.chunk_index for chunk in chunks]
        expected_indices = list(range(len(chunks)))
        if indices != expected_indices:
            self.logger.warning(f"Non-sequential chunk indices: {indices}")
        
        # Check for empty chunks
        empty_chunks = [c.chunk_index for c in chunks if not c.text.strip()]
        if empty_chunks:
            self.logger.warning(f"Empty chunks at indices: {empty_chunks}")
        
        # Check token counts
        zero_token_chunks = [c.chunk_index for c in chunks if c.token_count == 0]
        if zero_token_chunks:
            self.logger.warning(f"Zero-token chunks at indices: {zero_token_chunks}")
        
        return True
    
    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """
        Calculate statistics for chunk list.
        
        Args:
            chunks: List of chunks
        
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "total_tokens": 0,
                "avg_tokens_per_chunk": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "total_chars": 0,
                "avg_chars_per_chunk": 0,
            }
        
        token_counts = [c.token_count for c in chunks]
        char_counts = [len(c.text) for c in chunks]
        
        stats = {
            "num_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_chars": sum(char_counts),
            "avg_chars_per_chunk": sum(char_counts) / len(chunks),
            "strategy": self.strategy_name.value,
        }
        
        return stats
    
    def merge_chunks(
        self,
        chunks: List[DocumentChunk],
        max_tokens: int
    ) -> List[DocumentChunk]:
        """
        Merge small chunks up to max_tokens.
        Useful for optimizing chunk sizes.
        
        Args:
            chunks: List of chunks to merge
            max_tokens: Maximum tokens per merged chunk
        
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
        
        merged = []
        current_texts = []
        current_tokens = 0
        document_id = chunks[0].document_id
        
        for chunk in chunks:
            if current_tokens + chunk.token_count <= max_tokens:
                current_texts.append(chunk.text)
                current_tokens += chunk.token_count
            else:
                # Save current merged chunk
                if current_texts:
                    merged_text = " ".join(current_texts)
                    merged_chunk = self._create_chunk(
                        text=merged_text,
                        chunk_index=len(merged),
                        document_id=document_id,
                        start_char=0,  # Position tracking lost in merge
                        end_char=len(merged_text)
                    )
                    merged.append(merged_chunk)
                
                # Start new chunk
                current_texts = [chunk.text]
                current_tokens = chunk.token_count
        
        # Add final merged chunk
        if current_texts:
            merged_text = " ".join(current_texts)
            merged_chunk = self._create_chunk(
                text=merged_text,
                chunk_index=len(merged),
                document_id=document_id,
                start_char=0,
                end_char=len(merged_text)
            )
            merged.append(merged_chunk)
        
        self.logger.info(f"Merged {len(chunks)} chunks into {len(merged)}")
        
        return merged
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(strategy={self.strategy_name})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return self.__str__()


class ChunkerConfig:
    """
    Configuration for chunking strategies.
    Provides a way to pass parameters to chunkers.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        respect_boundaries: bool = True,
        min_chunk_size: int = 100,
        **kwargs
    ):
        """
        Initialize chunker configuration.
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
            respect_boundaries: Respect sentence/paragraph boundaries
            min_chunk_size: Minimum chunk size in tokens
            **kwargs: Additional strategy-specific parameters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_boundaries = respect_boundaries
        self.min_chunk_size = min_chunk_size
        self.extra = kwargs
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "respect_boundaries": self.respect_boundaries,
            "min_chunk_size": self.min_chunk_size,
            **self.extra
        }
    
    def __repr__(self) -> str:
        return f"ChunkerConfig({self.to_dict()})"


if __name__ == "__main__":
    # Test base chunker (through a simple implementation)
    print("=== Base Chunker Tests ===\n")
    
    from config.models import DocumentMetadata, DocumentType
    
    # Create a simple test chunker
    class SimpleChunker(BaseChunker):
        def chunk_text(self, text, metadata=None):
            # Simple implementation: split by sentences
            sentences = text.split('. ')
            chunks = []
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    chunk = self._create_chunk(
                        text=sentence.strip(),
                        chunk_index=i,
                        document_id=metadata.document_id if metadata else "test",
                        start_char=0,
                        end_char=len(sentence)
                    )
                    chunks.append(chunk)
            return chunks
    
    # Create test metadata
    metadata = DocumentMetadata(
        document_id="test_doc_123",
        filename="test.txt",
        document_type=DocumentType.TXT,
        file_size_bytes=1000
    )
    
    # Test chunking
    chunker = SimpleChunker(ChunkingStrategy.FIXED)
    test_text = "This is sentence one. This is sentence two. This is sentence three. And this is the last one."
    
    print("Test 1: Basic chunking")
    chunks = chunker.chunk_document(test_text, metadata)
    print(f"  Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i}: {chunk.token_count} tokens - {chunk.text}")
    print()
    
    # Test validation
    print("Test 2: Chunk validation")
    is_valid = chunker.validate_chunks(chunks)
    print(f"  Valid: {is_valid}")
    print()
    
    # Test statistics
    print("Test 3: Chunk statistics")
    stats = chunker.get_chunk_statistics(chunks)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test configuration
    print("Test 4: Chunker configuration")
    config = ChunkerConfig(chunk_size=512, overlap=50, custom_param="value")
    print(f"  Config: {config}")
    print(f"  As dict: {config.to_dict()}")
    print()
    
    print("✓ Base chunker module created successfully!")