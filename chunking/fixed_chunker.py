"""
Fixed-Size Chunker
Splits text into fixed-size chunks with overlap
"""

from typing import List, Optional
import re

from config.logging_config import get_logger
from config.models import DocumentChunk, DocumentMetadata, ChunkingStrategy
from config.settings import get_settings
from chunking.base_chunker import BaseChunker, ChunkerConfig
from chunking.token_counter import TokenCounter

logger = get_logger(__name__)
settings = get_settings()


class FixedChunker(BaseChunker):
    """
    Fixed-size chunking strategy.
    Splits text into chunks of approximately equal token count with overlap.
    
    Best for:
    - Small to medium documents (<50K tokens)
    - Homogeneous content
    - When simplicity is preferred
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        respect_sentence_boundaries: bool = True,
        min_chunk_size: int = 100
    ):
        """
        Initialize fixed chunker.
        
        Args:
            chunk_size: Target tokens per chunk (default from settings)
            overlap: Overlap tokens between chunks (default from settings)
            respect_sentence_boundaries: Try to break at sentence boundaries
            min_chunk_size: Minimum chunk size in tokens
        """
        super().__init__(ChunkingStrategy.FIXED)
        
        self.chunk_size = chunk_size or settings.FIXED_CHUNK_SIZE
        self.overlap = overlap or settings.FIXED_CHUNK_OVERLAP
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.min_chunk_size = min_chunk_size
        
        # Initialize token counter
        self.token_counter = TokenCounter()
        
        # Validate parameters
        if self.overlap >= self.chunk_size:
            raise ValueError(f"Overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})")
        
        self.logger.info(
            f"Initialized FixedChunker: chunk_size={self.chunk_size}, "
            f"overlap={self.overlap}, respect_boundaries={self.respect_sentence_boundaries}"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> List[DocumentChunk]:
        """
        Chunk text into fixed-size pieces.
        
        Args:
            text: Input text
            metadata: Document metadata
        
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        document_id = metadata.document_id if metadata else "unknown"
        
        # Split into sentences if respecting boundaries
        if self.respect_sentence_boundaries:
            chunks = self._chunk_with_sentence_boundaries(text, document_id)
        else:
            chunks = self._chunk_without_boundaries(text, document_id)
        
        # Clean and validate
        chunks = [c for c in chunks if c.token_count >= self.min_chunk_size]
        
        self.logger.debug(f"Created {len(chunks)} fixed-size chunks")
        
        return chunks
    
    def _chunk_with_sentence_boundaries(
        self,
        text: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Chunk text respecting sentence boundaries.
        
        Args:
            text: Input text
            document_id: Document ID
        
        Returns:
            List of chunks
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_sentences = []
        current_tokens = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            # If single sentence exceeds chunk_size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if any
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunk = self._create_chunk(
                        text=self._clean_chunk_text(chunk_text),
                        chunk_index=len(chunks),
                        document_id=document_id,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text)
                    )
                    chunks.append(chunk)
                    current_sentences = []
                    current_tokens = 0
                    start_char += len(chunk_text)
                
                # Split long sentence and add as separate chunks
                long_sentence_chunks = self._split_long_sentence(
                    sentence,
                    document_id,
                    start_index=len(chunks),
                    start_char=start_char
                )
                chunks.extend(long_sentence_chunks)
                start_char += len(sentence)
                continue
            
            # Check if adding this sentence exceeds chunk_size
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunk = self._create_chunk(
                    text=self._clean_chunk_text(chunk_text),
                    chunk_index=len(chunks),
                    document_id=document_id,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text)
                )
                chunks.append(chunk)
                
                # Handle overlap
                if self.overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = self._get_overlap_sentences(
                        current_sentences,
                        self.overlap
                    )
                    current_sentences = overlap_sentences + [sentence]
                    current_tokens = sum(
                        self.token_counter.count_tokens(s) for s in current_sentences
                    )
                else:
                    current_sentences = [sentence]
                    current_tokens = sentence_tokens
                
                start_char += len(chunk_text)
            else:
                # Add sentence to current chunk
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = self._create_chunk(
                text=self._clean_chunk_text(chunk_text),
                chunk_index=len(chunks),
                document_id=document_id,
                start_char=start_char,
                end_char=start_char + len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_without_boundaries(
        self,
        text: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Chunk text without respecting boundaries (pure token-based).
        
        Args:
            text: Input text
            document_id: Document ID
        
        Returns:
            List of chunks
        """
        # Use token counter's split method
        chunk_texts = self.token_counter.split_into_token_chunks(
            text,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunk_texts):
            chunk = self._create_chunk(
                text=self._clean_chunk_text(chunk_text),
                chunk_index=i,
                document_id=document_id,
                start_char=current_pos,
                end_char=current_pos + len(chunk_text)
            )
            chunks.append(chunk)
            current_pos += len(chunk_text) - (len(chunk_text) * self.overlap // self.chunk_size)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        # Handle common abbreviations
        # Protect them temporarily
        protected = text
        abbreviations = [
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.', 'Prof.',
            'Inc.', 'Ltd.', 'Corp.', 'Co.', 'vs.', 'etc.', 'e.g.', 'i.e.',
            'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'U.S.', 'U.K.'
        ]
        
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence boundaries
        # Pattern: period/question/exclamation followed by space and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, protected)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter empty
        sentences = [s for s in sentences if s]
        
        return sentences
    
    def _split_long_sentence(
        self,
        sentence: str,
        document_id: str,
        start_index: int,
        start_char: int
    ) -> List[DocumentChunk]:
        """
        Split a sentence that's longer than chunk_size.
        
        Args:
            sentence: Long sentence
            document_id: Document ID
            start_index: Starting chunk index
            start_char: Starting character position
        
        Returns:
            List of chunks
        """
        # Split by commas, semicolons, or just by tokens
        parts = re.split(r'[,;]', sentence)
        
        chunks = []
        current_text = []
        current_tokens = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_tokens = self.token_counter.count_tokens(part)
            
            if current_tokens + part_tokens > self.chunk_size and current_text:
                # Save current chunk
                chunk_text = " ".join(current_text)
                chunk = self._create_chunk(
                    text=self._clean_chunk_text(chunk_text),
                    chunk_index=start_index + len(chunks),
                    document_id=document_id,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text)
                )
                chunks.append(chunk)
                start_char += len(chunk_text)
                current_text = []
                current_tokens = 0
            
            current_text.append(part)
            current_tokens += part_tokens
        
        # Add final part
        if current_text:
            chunk_text = " ".join(current_text)
            chunk = self._create_chunk(
                text=self._clean_chunk_text(chunk_text),
                chunk_index=start_index + len(chunks),
                document_id=document_id,
                start_char=start_char,
                end_char=start_char + len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        overlap_tokens: int
    ) -> List[str]:
        """
        Get last few sentences that fit in overlap window.
        
        Args:
            sentences: List of sentences
            overlap_tokens: Target overlap tokens
        
        Returns:
            List of overlap sentences
        """
        overlap = []
        tokens = 0
        
        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if tokens + sentence_tokens <= overlap_tokens:
                overlap.insert(0, sentence)
                tokens += sentence_tokens
            else:
                break
        
        return overlap
    
    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'FixedChunker':
        """
        Create FixedChunker from configuration.
        
        Args:
            config: ChunkerConfig object
        
        Returns:
            FixedChunker instance
        """
        return cls(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            respect_sentence_boundaries=config.respect_boundaries,
            min_chunk_size=config.min_chunk_size
        )


if __name__ == "__main__":
    # Test fixed chunker
    print("=== Fixed Chunker Tests ===\n")
    
    from config.models import DocumentMetadata, DocumentType
    
    # Create test metadata
    metadata = DocumentMetadata(
        document_id="test_doc_123",
        filename="test.txt",
        document_type=DocumentType.TXT,
        file_size_bytes=5000
    )
    
    # Test text
    test_text = """
    Artificial intelligence is transforming how we work and live. Machine learning algorithms 
    can now recognize patterns in data that humans might miss. Deep learning, a subset of 
    machine learning, uses neural networks with multiple layers to process complex information.
    
    Natural language processing enables computers to understand and generate human language.
    This technology powers virtual assistants, translation services, and content analysis tools.
    Computer vision allows machines to interpret visual information from the world around them.
    
    The applications of AI are vast and growing. In healthcare, AI helps diagnose diseases and 
    develop new treatments. In finance, it detects fraud and optimizes trading strategies. 
    In transportation, self-driving cars use AI to navigate roads safely.
    
    However, AI also raises important ethical questions. How do we ensure AI systems are fair 
    and unbiased? How do we protect privacy when AI processes personal data? How do we maintain 
    human oversight over critical decisions? These questions require ongoing attention as the 
    technology continues to advance.
    """
    
    # Test 1: Basic chunking with sentence boundaries
    print("Test 1: Chunking with sentence boundaries")
    chunker = FixedChunker(chunk_size=100, overlap=20, respect_sentence_boundaries=True)
    chunks = chunker.chunk_document(test_text, metadata)
    
    print(f"  Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.token_count} tokens")
        print(f"    Preview: {chunk.text[:100]}...")
        print()
    
    # Test 2: Statistics
    print("Test 2: Chunk statistics")
    stats = chunker.get_chunk_statistics(chunks)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test 3: Without sentence boundaries
    print("Test 3: Chunking without sentence boundaries")
    chunker_no_boundaries = FixedChunker(
        chunk_size=100,
        overlap=20,
        respect_sentence_boundaries=False
    )
    chunks_no_boundaries = chunker_no_boundaries.chunk_document(test_text, metadata)
    print(f"  Created {len(chunks_no_boundaries)} chunks")
    print()
    
    # Test 4: Different chunk sizes
    print("Test 4: Different chunk sizes")
    for size in [50, 100, 200]:
        test_chunker = FixedChunker(chunk_size=size, overlap=10)
        test_chunks = test_chunker.chunk_text(test_text, metadata)
        print(f"  Chunk size {size}: {len(test_chunks)} chunks")
    print()
    
    # Test 5: From config
    print("Test 5: Create from config")
    config = ChunkerConfig(chunk_size=150, overlap=30, respect_boundaries=True)
    chunker_from_config = FixedChunker.from_config(config)
    print(f"  Created chunker: {chunker_from_config}")
    print()
    
    print("✓ Fixed chunker module created successfully!")