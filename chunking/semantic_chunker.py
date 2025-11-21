"""
Semantic Chunker
Splits text based on semantic similarity between sentences
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from config.logging_config import get_logger
from config.models import DocumentChunk, DocumentMetadata, ChunkingStrategy
from config.settings import get_settings
from chunking.base_chunker import BaseChunker, ChunkerConfig
from chunking.token_counter import TokenCounter

logger = get_logger(__name__)
settings = get_settings()


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy.
    Creates chunks based on semantic similarity between sentences.
    Identifies natural topic boundaries using embedding similarity.
    
    Best for:
    - Medium documents (50K-500K tokens)
    - Documents with clear topics/sections
    - When context coherence is critical
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        similarity_threshold: float = None,
        min_chunk_size: int = 100,
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target tokens per chunk (soft limit)
            overlap: Overlap tokens between chunks
            similarity_threshold: Threshold for semantic breakpoints (0-1)
            min_chunk_size: Minimum chunk size in tokens
            embedding_model: Pre-loaded embedding model (optional)
        """
        super().__init__(ChunkingStrategy.SEMANTIC)
        
        self.chunk_size = chunk_size or settings.FIXED_CHUNK_SIZE
        self.overlap = overlap or settings.FIXED_CHUNK_OVERLAP
        self.similarity_threshold = similarity_threshold or settings.SEMANTIC_BREAKPOINT_THRESHOLD
        self.min_chunk_size = min_chunk_size
        
        # Initialize token counter
        self.token_counter = TokenCounter()
        
        # Initialize or use provided embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            try:
                self.logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
                self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                self.logger.info("Embedding model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        
        self.logger.info(
            f"Initialized SemanticChunker: chunk_size={self.chunk_size}, "
            f"threshold={self.similarity_threshold}, model_loaded={self.embedding_model is not None}"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> List[DocumentChunk]:
        """
        Chunk text based on semantic similarity.
        
        Args:
            text: Input text
            metadata: Document metadata
        
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        document_id = metadata.document_id if metadata else "unknown"
        
        # If embedding model not available, fall back to fixed chunking
        if self.embedding_model is None:
            self.logger.warning("Embedding model not available, using sentence-based chunking")
            return self._fallback_chunking(text, document_id)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            # Too few sentences, return as single chunk
            chunk = self._create_chunk(
                text=self._clean_chunk_text(text),
                chunk_index=0,
                document_id=document_id,
                start_char=0,
                end_char=len(text)
            )
            return [chunk]
        
        # Calculate semantic similarities between adjacent sentences
        similarities = self._calculate_similarities(sentences)
        
        # Find breakpoints where similarity drops
        breakpoints = self._find_breakpoints(similarities)
        
        # Create chunks from breakpoints
        chunks = self._create_chunks_from_breakpoints(
            sentences,
            breakpoints,
            document_id
        )
        
        # Filter out chunks that are too small
        chunks = [c for c in chunks if c.token_count >= self.min_chunk_size]
        
        self.logger.debug(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        import re
        
        # Protect abbreviations
        protected = text
        abbreviations = [
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.', 'Prof.',
            'Inc.', 'Ltd.', 'Corp.', 'Co.', 'vs.', 'etc.', 'e.g.', 'i.e.',
            'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'U.S.', 'U.K.'
        ]
        
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, protected)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter empty
        sentences = [s for s in sentences if s]
        
        return sentences
    
    def _calculate_similarities(self, sentences: List[str]) -> List[float]:
        """
        Calculate cosine similarity between adjacent sentences.
        
        Args:
            sentences: List of sentences
        
        Returns:
            List of similarity scores
        """
        if len(sentences) < 2:
            return []
        
        # Generate embeddings for all sentences
        self.logger.debug(f"Generating embeddings for {len(sentences)} sentences")
        embeddings = self.embedding_model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Calculate cosine similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)
        
        return similarities
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        Find breakpoints where semantic similarity drops significantly.
        Uses percentile-based threshold.
        
        Args:
            similarities: List of similarity scores
        
        Returns:
            List of breakpoint indices
        """
        if not similarities:
            return []
        
        # Calculate threshold using percentile
        # Lower similarity = more likely to be a breakpoint
        similarities_array = np.array(similarities)
        threshold = np.percentile(similarities_array, (1 - self.similarity_threshold) * 100)
        
        # Find points where similarity is below threshold
        breakpoints = [0]  # Always start with 0
        
        for i, sim in enumerate(similarities):
            if sim < threshold:
                # This is a potential breakpoint
                # Add the index of the next sentence (after the low similarity)
                breakpoints.append(i + 1)
        
        self.logger.debug(
            f"Found {len(breakpoints)} breakpoints with threshold {threshold:.3f}"
        )
        
        return breakpoints
    
    def _create_chunks_from_breakpoints(
        self,
        sentences: List[str],
        breakpoints: List[int],
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Create chunks from sentences and breakpoints.
        
        Args:
            sentences: List of sentences
            breakpoints: List of breakpoint indices
            document_id: Document ID
        
        Returns:
            List of chunks
        """
        chunks = []
        
        # Ensure breakpoints are sorted and include the end
        breakpoints = sorted(set(breakpoints))
        if breakpoints[-1] != len(sentences):
            breakpoints.append(len(sentences))
        
        current_pos = 0
        
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            
            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            
            if not chunk_sentences:
                continue
            
            # Combine sentences
            chunk_text = " ".join(chunk_sentences)
            
            # Check token count
            token_count = self.token_counter.count_tokens(chunk_text)
            
            # If chunk is too large, split it further
            if token_count > self.chunk_size * 1.5:
                sub_chunks = self._split_large_chunk(
                    chunk_sentences,
                    document_id,
                    len(chunks),
                    current_pos
                )
                chunks.extend(sub_chunks)
            else:
                # Create single chunk
                chunk = self._create_chunk(
                    text=self._clean_chunk_text(chunk_text),
                    chunk_index=len(chunks),
                    document_id=document_id,
                    start_char=current_pos,
                    end_char=current_pos + len(chunk_text),
                    metadata={"sentences": len(chunk_sentences)}
                )
                chunks.append(chunk)
            
            current_pos += len(chunk_text)
        
        return chunks
    
    def _split_large_chunk(
        self,
        sentences: List[str],
        document_id: str,
        start_index: int,
        start_char: int
    ) -> List[DocumentChunk]:
        """
        Split a large chunk into smaller pieces.
        
        Args:
            sentences: List of sentences in the large chunk
            document_id: Document ID
            start_index: Starting chunk index
            start_char: Starting character position
        
        Returns:
            List of sub-chunks
        """
        sub_chunks = []
        current_sentences = []
        current_tokens = 0
        current_pos = start_char
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                # Save current sub-chunk
                chunk_text = " ".join(current_sentences)
                chunk = self._create_chunk(
                    text=self._clean_chunk_text(chunk_text),
                    chunk_index=start_index + len(sub_chunks),
                    document_id=document_id,
                    start_char=current_pos,
                    end_char=current_pos + len(chunk_text)
                )
                sub_chunks.append(chunk)
                
                # Start new sub-chunk with overlap
                if self.overlap > 0:
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
                
                current_pos += len(chunk_text)
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final sub-chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = self._create_chunk(
                text=self._clean_chunk_text(chunk_text),
                chunk_index=start_index + len(sub_chunks),
                document_id=document_id,
                start_char=current_pos,
                end_char=current_pos + len(chunk_text)
            )
            sub_chunks.append(chunk)
        
        return sub_chunks
    
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
        
        for sentence in reversed(sentences):
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if tokens + sentence_tokens <= overlap_tokens:
                overlap.insert(0, sentence)
                tokens += sentence_tokens
            else:
                break
        
        return overlap
    
    def _fallback_chunking(
        self,
        text: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Fallback to sentence-based chunking when embeddings unavailable.
        
        Args:
            text: Input text
            document_id: Document ID
        
        Returns:
            List of chunks
        """
        from chunking.fixed_chunker import FixedChunker
        
        # Use fixed chunker with sentence boundaries
        fallback_chunker = FixedChunker(
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            respect_sentence_boundaries=True
        )
        
        metadata = DocumentMetadata(
            document_id=document_id,
            filename="fallback",
            document_type="txt",
            file_size_bytes=len(text)
        )
        
        return fallback_chunker.chunk_text(text, metadata)
    
    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'SemanticChunker':
        """
        Create SemanticChunker from configuration.
        
        Args:
            config: ChunkerConfig object
        
        Returns:
            SemanticChunker instance
        """
        return cls(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            similarity_threshold=config.extra.get('semantic_threshold', settings.SEMANTIC_BREAKPOINT_THRESHOLD),
            min_chunk_size=config.min_chunk_size
        )


if __name__ == "__main__":
    # Test semantic chunker
    print("=== Semantic Chunker Tests ===\n")
    
    from config.models import DocumentMetadata, DocumentType
    
    # Create test metadata
    metadata = DocumentMetadata(
        document_id="test_doc_semantic",
        filename="test.txt",
        document_type=DocumentType.TXT,
        file_size_bytes=5000
    )
    
    # Test text with clear topic shifts
    test_text = """
    Artificial intelligence is transforming the technology industry. Machine learning algorithms 
    can now process vast amounts of data quickly. Deep learning neural networks have achieved 
    remarkable results in image recognition and natural language processing.
    
    The history of computing dates back to the 1940s. Early computers were massive machines 
    that filled entire rooms. The invention of the transistor revolutionized computing technology.
    Personal computers became popular in the 1980s and changed how people work.
    
    Climate change is one of the most pressing global challenges. Rising temperatures are 
    affecting ecosystems worldwide. Scientists warn that immediate action is needed to 
    reduce greenhouse gas emissions. Renewable energy sources offer promising solutions.
    
    The human brain is incredibly complex. Neurons communicate through electrical and 
    chemical signals. Memory formation involves multiple brain regions working together.
    Neuroscience research continues to reveal new insights about consciousness.
    """
    
    try:
        # Test 1: Basic semantic chunking
        print("Test 1: Semantic chunking with embedding model")
        chunker = SemanticChunker(
            chunk_size=100,
            overlap=20,
            similarity_threshold=0.95
        )
        
        chunks = chunker.chunk_document(test_text, metadata)
        
        print(f"  Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"\n  Chunk {i}: {chunk.token_count} tokens")
            print(f"    Preview: {chunk.text[:150]}...")
        print()
        
        # Test 2: Statistics
        print("Test 2: Chunk statistics")
        stats = chunker.get_chunk_statistics(chunks)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
        
        # Test 3: Different thresholds
        print("Test 3: Different similarity thresholds")
        for threshold in [0.90, 0.95, 0.98]:
            test_chunker = SemanticChunker(
                chunk_size=100,
                similarity_threshold=threshold
            )
            test_chunks = test_chunker.chunk_text(test_text, metadata)
            print(f"  Threshold {threshold}: {len(test_chunks)} chunks")
        print()
        
    except Exception as e:
        print(f"Note: Semantic chunking requires sentence-transformers library")
        print(f"Error: {e}")
        print("\nTesting fallback mode...")
        
        # Test fallback
        chunker = SemanticChunker(chunk_size=100, embedding_model=None)
        chunks = chunker.chunk_text(test_text, metadata)
        print(f"Fallback created {len(chunks)} chunks")
    
    print("\n✓ Semantic chunker module created successfully!")