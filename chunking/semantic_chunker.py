# DEPENDENCIES
import re
import numpy as np
from typing import List
from typing import Optional
from config.models import DocumentChunk
from config.settings import get_settings
from config.models import DocumentMetadata
from config.models import ChunkingStrategy
from config.logging_config import get_logger
from chunking.base_chunker import BaseChunker
from chunking.base_chunker import ChunkerConfig
from chunking.token_counter import TokenCounter
from chunking.overlap_manager import OverlapManager
from sentence_transformers import SentenceTransformer


# Setup Settings and Logging
logger   = get_logger(__name__)
settings = get_settings()


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy: 
    - Creates chunks based on semantic similarity between sentences
    - Identifies natural topic boundaries using embedding similarity
    
    Best for:
    - Medium documents (50K-500K tokens)
    - Documents with clear topics/sections
    - When context coherence is critical
    """
    def __init__(self, chunk_size: int = None, overlap: int = None, similarity_threshold: float = None, min_chunk_size: int = 100, embedding_model: Optional[SentenceTransformer] = None):
        """
        Initialize semantic chunker
        
        Arguments:
        ----------
            chunk_size                   { int }         : Target tokens per chunk (soft limit)
            
            overlap                      { int }         : Overlap tokens between chunks
            
            similarity_threshold        { float }        : Threshold for semantic breakpoints (0-1)
            
            min_chunk_size               { int }         : Minimum chunk size in tokens
            
            embedding_model      { SentenceTransformer } : Pre-loaded embedding model (optional)
        """
        super().__init__(ChunkingStrategy.SEMANTIC)
        
        self.chunk_size           = chunk_size or settings.FIXED_CHUNK_SIZE
        self.overlap              = overlap or settings.FIXED_CHUNK_OVERLAP
        self.similarity_threshold = similarity_threshold or settings.SEMANTIC_BREAKPOINT_THRESHOLD
        self.min_chunk_size       = min_chunk_size
        
        # Initialize token counter and overlap manager
        self.token_counter        = TokenCounter()
        self.overlap_manager      = OverlapManager(overlap_tokens = self.overlap)
        
        # Initialize or use provided embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        
        else:
            try:
                self.logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
                self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

                self.logger.info("Embedding model loaded successfully")
            
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {repr(e)}")
                self.embedding_model = None
        
        self.logger.info(f"Initialized SemanticChunker: chunk_size={self.chunk_size}, threshold={self.similarity_threshold}, model_loaded={self.embedding_model is not None}")
    

    def chunk_text(self, text: str, metadata: Optional[DocumentMetadata] = None) -> List[DocumentChunk]:
        """
        Chunk text based on semantic similarity
        
        Arguments:
        ----------
            text            { str }       : Input text

            metadata { DocumentMetaData } : Document metadata
        
        Returns:
        --------
                     { list }             : List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        document_id = metadata.document_id if metadata else "unknown"
        
        # If embedding model not available, fall back to fixed chunking
        if self.embedding_model is None:
            self.logger.warning("Embedding model not available, using sentence-based chunking")
            return self._fallback_chunking(text        = text, 
                                           document_id = document_id,
                                          )
        
        # Split into sentences
        sentences = self._split_sentences(text = text)
        
        if (len(sentences) < 2):
            # Too few sentences, return as single chunk
            return self._create_single_chunk(text        = text, 
                                             document_id = document_id,
                                            )
        
        # Calculate semantic similarities between adjacent sentences
        similarities = self._calculate_similarities(sentences = sentences)
        
        # Find breakpoints where similarity drops
        breakpoints  = self._find_breakpoints(similarities = similarities)
        
        # Create chunks from breakpoints WITHOUT overlap
        chunks       = self._create_chunks_from_breakpoints(sentences   = sentences, 
                                                            breakpoints = breakpoints, 
                                                            document_id = document_id,
                                                           )
        
        # Filter out chunks that are too small
        chunks       = [c for c in chunks if (c.token_count >= self.min_chunk_size)]
        
        # Use OverlapManager to add proper overlap between semantic chunks
        if ((len(chunks) > 1) and (self.overlap > 0)):
            chunks = self.overlap_manager.add_overlap(chunks         = chunks, 
                                                      overlap_tokens = self.overlap,
                                                     )
        
        self.logger.debug(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        
        return chunks
    

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
            { list }     : List of sentences
        """
        # Protect abbreviations
        protected     = text
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'U.S.', 'U.K.']
        
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences        = re.split(sentence_pattern, protected)
        
        # Restore abbreviations
        sentences        = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter empty
        sentences        = [s for s in sentences if s]
        
        return sentences
    

    def _calculate_similarities(self, sentences: List[str]) -> List[float]:
        """
        Calculate cosine similarity between adjacent sentences
        
        Arguments:
        ----------
            sentences { list } : List of sentences
        
        Returns:
        --------
                 { list }      : List of similarity scores
        """
        if (len(sentences) < 2):
            return []
        
        # Generate embeddings for all sentences
        self.logger.debug(f"Generating embeddings for {len(sentences)} sentences")
        
        embeddings   = self.embedding_model.encode(sentences,
                                                   show_progress_bar = False,
                                                   convert_to_numpy  = True,
                                                  )
        
        # Calculate cosine similarity between adjacent sentences
        similarities = list()

        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(vec1 = embeddings[i], 
                                                 vec2 = embeddings[i + 1],
                                                )

            similarities.append(similarity)
        
        return similarities
    

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Arguments:
        ----------
            vec1 { np.ndarray } : First vector

            vec2 { np.ndarray } : Second vector
        
        Returns:
        --------
                 { float }      : Similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1       = np.linalg.norm(vec1)
        norm2       = np.linalg.norm(vec2)
        
        if ((norm1 == 0) or (norm2 == 0)):
            return 0.0
        
        return dot_product / (norm1 * norm2)
    

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        Find breakpoints where semantic similarity drops significantly: Uses percentile-based threshold
        
        Arguments:
        ----------
            similarities { list } : List of similarity scores
        
        Returns:
        --------
                   { list }       : List of breakpoint indices
        """
        if not similarities:
            return []
        
        # Calculate threshold using percentile: Lower similarity = more likely to be a breakpoint
        similarities_array = np.array(similarities)
        threshold          = np.percentile(similarities_array, (1 - self.similarity_threshold) * 100)
        
        # Find points where similarity is below threshold: always start with 0
        breakpoints        = [0]  
        
        for i, sim in enumerate(similarities):
            if (sim < threshold):
                # This is a potential breakpoint: Add the index of the next sentence (after the low similarity)
                breakpoints.append(i + 1)
        
        self.logger.debug(f"Found {len(breakpoints)} breakpoints with threshold {threshold:.3f}")
        
        return breakpoints
    

    def _create_chunks_from_breakpoints(self, sentences: List[str], breakpoints: List[int], document_id: str) -> List[DocumentChunk]:
        """
        Create chunks from sentences and breakpoints WITHOUT overlap (overlap added later)
        
        Arguments:
        ----------
            sentences   { list } : List of sentences

            breakpoints { list } : List of breakpoint indices
            
            document_id { str }  : Document ID
        
        Returns:
        --------
                 { list }        : List of chunks WITHOUT overlap
        """
        chunks      = list()
        
        # Ensure breakpoints are sorted and include the end
        breakpoints = sorted(set(breakpoints))
        
        if (breakpoints[-1] != len(sentences)):
            breakpoints.append(len(sentences))
        
        current_pos = 0
        
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx   = breakpoints[i + 1]
            
            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx:end_idx]
            
            if not chunk_sentences:
                continue
            
            # Combine sentences
            chunk_text  = " ".join(chunk_sentences)
            
            # Check token count
            token_count = self.token_counter.count_tokens(chunk_text)
            
            # If chunk is too large, split it further WITHOUT overlap
            if (token_count > (self.chunk_size * 1.5)):
                sub_chunks = self._split_large_chunk_simple(chunk_sentences = chunk_sentences,
                                                            document_id     = document_id,
                                                            start_index     = len(chunks),
                                                            start_char      = current_pos,
                                                           )
                chunks.extend(sub_chunks)
            
            else:
                # Create single chunk WITHOUT overlap
                chunk = self._create_chunk(text          = self._clean_chunk_text(chunk_text),
                                           chunk_index   = len(chunks),
                                           document_id   = document_id,
                                           start_char    = current_pos,
                                           end_char      = current_pos + len(chunk_text),
                                           metadata      = {"sentences"      : len(chunk_sentences),
                                                            "semantic_chunk" : True,
                                                           }
                                          )
                chunks.append(chunk)
            
            current_pos += len(chunk_text)
        
        return chunks
    

    def _split_large_chunk_simple(self, chunk_sentences: List[str], document_id: str, start_index: int, start_char: int) -> List[DocumentChunk]:
        """
        Split a large chunk into smaller pieces without overlap (overlap added later by OverlapManager)
        
        Arguments:
        ----------
            chunk_sentences { list } : List of sentences in the large chunk

            document_id     { str }  : Document ID
            
            start_index     { int }  : Starting chunk index
            
            start_char      { int }  : Starting character position
        
        Returns:
        --------
                   { list }          : List of sub-chunks WITHOUT overlap
        """
        sub_chunks        = list()
        current_sentences = list()
        current_tokens    = 0
        current_pos       = start_char
        
        for sentence in chunk_sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            # When chunk gets too big, save it without overlap
            if (((current_tokens + sentence_tokens) > self.chunk_size) and current_sentences):
                chunk_text = " ".join(current_sentences)
                chunk      = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                                chunk_index = start_index + len(sub_chunks),
                                                document_id = document_id,
                                                start_char  = current_pos,
                                                end_char    = current_pos + len(chunk_text),
                                               )
                sub_chunks.append(chunk)
                
                # Start new chunk without manual overlap (OverlapManager handles it later)
                current_sentences = [sentence]
                current_tokens    = sentence_tokens
                current_pos      += len(chunk_text)
            
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk      = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                            chunk_index = start_index + len(sub_chunks),
                                            document_id = document_id,
                                            start_char  = current_pos,
                                            end_char    = current_pos + len(chunk_text),
                                           )
            sub_chunks.append(chunk)
        
        return sub_chunks
    

    def _create_single_chunk(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Create a single chunk for short text
        
        Arguments:
        ----------
            text        { str } : Input text

            document_id { str } : Document ID
        
        Returns:
        --------
                { list }        : Single chunk in a list
        """
        chunk = self._create_chunk(text        = self._clean_chunk_text(text),
                                   chunk_index = 0,
                                   document_id = document_id,
                                   start_char  = 0,
                                   end_char    = len(text),
                                  )
        
        return [chunk]
    

    def _fallback_chunking(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Fallback to sentence-based chunking when embeddings unavailable
        
        Arguments:
        ----------
            text        { str } : Input text

            document_id { str } : Document ID
        
        Returns:
        --------
                { list }        : List of chunks
        """
        from chunking.fixed_chunker import FixedChunker
        
        # Use fixed chunker with sentence boundaries
        fallback_chunker = FixedChunker(chunk_size                  = self.chunk_size,
                                        overlap                     = self.overlap,
                                        respect_sentence_boundaries = True,
                                       )
        
        metadata = DocumentMetadata(document_id     = document_id,
                                    filename        = "fallback",
                                    document_type   = "txt",
                                    file_size_bytes = len(text),
                                   )
        
        return fallback_chunker.chunk_text(text, metadata)
    

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'SemanticChunker':
        """
        Create SemanticChunker from configuration
        
        Arguments:
        ----------
            config { ChunkerConfig } : ChunkerConfig object
        
        Returns:
        --------
            { SemanticChunker }      : SemanticChunker instance
        """
        return cls(chunk_size           = config.chunk_size,
                   overlap              = config.overlap,
                   similarity_threshold = config.extra.get('semantic_threshold', settings.SEMANTIC_BREAKPOINT_THRESHOLD),
                   min_chunk_size       = config.min_chunk_size,
                  )