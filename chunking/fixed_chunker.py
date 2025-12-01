# DEPENDENCIES
import re
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


# Setup Settings and Logging
logger   = get_logger(__name__)
settings = get_settings()


class FixedChunker(BaseChunker):
    """
    Fixed-size chunking strategy : Splits text into chunks of approximately equal token count with overlap
    
    Best for:
    - Small to medium documents (<50K tokens)
    - Homogeneous content
    - When simplicity is preferred
    """
    def __init__(self, chunk_size: int = None, overlap: int = None, respect_sentence_boundaries: bool = True, min_chunk_size: int = 100):
        """
        Initialize fixed chunker
        
        Arguments:
        ----------
            chunk_size                  { int }  : Target tokens per chunk (default from settings)
            
            overlap                     { int }  : Overlap tokens between chunks (default from settings)
            
            respect_sentence_boundaries { bool } : Try to break at sentence boundaries
            
            min_chunk_size              { int }  : Minimum chunk size in tokens
        """
        super().__init__(ChunkingStrategy.FIXED)
        
        self.chunk_size                  = chunk_size or settings.FIXED_CHUNK_SIZE
        self.overlap                     = overlap or settings.FIXED_CHUNK_OVERLAP
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.min_chunk_size              = min_chunk_size
        
        # Initialize token counter and overlap manager
        self.token_counter               = TokenCounter()
        self.overlap_manager             = OverlapManager(overlap_tokens = self.overlap)
        
        # Validate parameters
        if (self.overlap >= self.chunk_size):
            raise ValueError(f"Overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})")
        
        self.logger.info(f"Initialized FixedChunker: chunk_size={self.chunk_size}, overlap={self.overlap}, respect_boundaries={self.respect_sentence_boundaries}")
    

    def chunk_text(self, text: str, metadata: Optional[DocumentMetadata] = None) -> List[DocumentChunk]:
        """
        Chunk text into fixed-size pieces
        
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
        
        # Split into sentences if respecting boundaries
        if self.respect_sentence_boundaries:
            chunks = self._chunk_with_sentence_boundaries(text        = text, 
                                                          document_id = document_id,
                                                         )
        
        else:
            chunks = self._chunk_without_boundaries(text        = text, 
                                                    document_id = document_id,
                                                   )
        
        # Clean and validate
        chunks = [c for c in chunks if (c.token_count >= self.min_chunk_size)]
        
        # Use OverlapManager to add proper overlap
        if ((len(chunks) > 1) and (self.overlap > 0)):
            chunks = self.overlap_manager.add_overlap(chunks         = chunks, 
                                                      overlap_tokens = self.overlap,
                                                     )
        
        self.logger.debug(f"Created {len(chunks)} fixed-size chunks")
        
        return chunks
    

    def _chunk_with_sentence_boundaries(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk text respecting sentence boundaries
        
        Arguments:
        ----------
            text        { str } : Input text

            document_id { str } : Document ID
        
        Returns:
        --------
                { list }        : List of chunks without overlap (overlap added later)
        """
        # Split into sentences
        sentences         = self._split_sentences(text = text)
        
        chunks            = list()
        current_sentences = list()
        current_tokens    = 0
        start_char        = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(text = sentence)
            
            # If single sentence exceeds chunk_size, split it
            if (sentence_tokens > self.chunk_size):
                # Save current chunk if any
                if current_sentences:
                    chunk_text        = " ".join(current_sentences)
                    chunk             = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                                           chunk_index = len(chunks),
                                                           document_id = document_id,
                                                           start_char  = start_char,
                                                           end_char    = start_char + len(chunk_text),
                                                          )
                    chunks.append(chunk)

                    current_sentences = list()
                    current_tokens    = 0
                    start_char       += len(chunk_text)
                
                # Split long sentence and add as separate chunks
                long_sentence_chunks = self._split_long_sentence(sentence    = sentence,
                                                                 document_id = document_id,
                                                                 start_index = len(chunks),
                                                                 start_char  = start_char,
                                                                )
                chunks.extend(long_sentence_chunks)
                start_char          += len(sentence)
                
                continue
            
            # Check if adding this sentence exceeds chunk_size
            if (((current_tokens + sentence_tokens) > self.chunk_size) and current_sentences):
                # Save current chunk WITHOUT overlap (overlap added later)
                chunk_text = " ".join(current_sentences)
                chunk      = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                                chunk_index = len(chunks),
                                                document_id = document_id,
                                                start_char  = start_char,
                                                end_char    = start_char + len(chunk_text),
                                               )
                chunks.append(chunk)
                
                # OverlapManager will handle the overlap here
                current_sentences = [sentence]
                current_tokens    = sentence_tokens
                start_char       += len(chunk_text)

            else:
                # Add sentence to current chunk
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk      = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                            chunk_index = len(chunks),
                                            document_id = document_id,
                                            start_char  = start_char,
                                            end_char    = start_char + len(chunk_text),
                                           )
            chunks.append(chunk)
        
        return chunks
    

    def _chunk_without_boundaries(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk text without respecting boundaries (pure token-based)
        
        Arguments:
        ----------
            text        { str } : Input text

            document_id { str } : Document ID
        
        Returns:
        --------
                  { list }      : List of chunks WITHOUT overlap
        """
        # Use token counter's split method
        chunk_texts = self.token_counter.split_into_token_chunks(text,
                                                                 chunk_size = self.chunk_size,
                                                                 overlap    = 0,
                                                                )
        
        chunks      = list()
        current_pos = 0
        
        for i, chunk_text in enumerate(chunk_texts):
            chunk        = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                              chunk_index = i,
                                              document_id = document_id,
                                              start_char  = current_pos,
                                              end_char    = current_pos + len(chunk_text),
                                             )

            chunks.append(chunk)
            current_pos += len(chunk_text)
        
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
        # Handle common abbreviations: Protect them temporarily
        protected     = text
        abbreviations = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'U.S.', 'U.K.']
        
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence boundaries
        # - Pattern: period/question/exclamation followed by space and capital letter
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences        = re.split(sentence_pattern, protected)
        
        # Restore abbreviations
        sentences        = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter empty
        sentences        = [s for s in sentences if s]
        
        return sentences

    
    def _split_long_sentence(self, sentence: str, document_id: str, start_index: int, start_char: int) -> List[DocumentChunk]:
        """
        Split a sentence that's longer than chunk_size
        
        Arguments:
        ----------
            sentence    { str } : Long sentence
            
            document_id { str } : Document ID
            
            start_index { str } : Starting chunk index
            
            start_char  { int } : Starting character position
        
        Returns:
        --------
                { list }        : List of chunks
        """
        # Split by commas, semicolons, or just by tokens
        parts          = re.split(r'[,;]', sentence)
        
        chunks         = list()
        current_text   = list()
        current_tokens = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            part_tokens = self.token_counter.count_tokens(part)
            
            if (((current_tokens + part_tokens) > self.chunk_size) and current_text):
                # Save current chunk
                chunk_text     = " ".join(current_text)
                chunk          = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                                    chunk_index = start_index + len(chunks),
                                                    document_id = document_id,
                                                    start_char  = start_char,
                                                    end_char    = start_char + len(chunk_text),
                                                   )
                chunks.append(chunk)
                start_char    += len(chunk_text)
                current_text   = []
                current_tokens = 0
            
            current_text.append(part)
            current_tokens += part_tokens
        
        # Add final part
        if current_text:
            chunk_text = " ".join(current_text)
            chunk      = self._create_chunk(text        = self._clean_chunk_text(chunk_text),
                                            chunk_index = start_index + len(chunks),
                                            document_id = document_id,
                                            start_char  = start_char,
                                            end_char    = start_char + len(chunk_text),
                                           )
            chunks.append(chunk)
        
        return chunks
    

    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """
        Get last few sentences that fit in overlap window
        
        Arguments:
        ----------
            sentences      { list } : List of sentences

            overlap_tokens { int }  : Target overlap tokens
        
        Returns:
        --------
                  { list }          : List of overlap sentences
        """
        overlap = list()
        tokens  = 0
        
        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            if ((tokens + sentence_tokens) <= overlap_tokens):
                overlap.insert(0, sentence)
                tokens += sentence_tokens
            
            else:
                break
        
        return overlap

    
    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'FixedChunker':
        """
        Create FixedChunker from configuration
        
        Arguments:
        ----------
            config { ChunkerConfig } : ChunkerConfig object
        
        Returns:
        --------
            FixedChunker instance
        """
        return cls(chunk_size                  = config.chunk_size,
                   overlap                     = config.overlap,
                   respect_sentence_boundaries = config.respect_boundaries,
                   min_chunk_size              = config.min_chunk_size,
                  )