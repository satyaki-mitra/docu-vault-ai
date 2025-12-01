# DEPENDENCIES
import re
from abc import ABC
from typing import List
from pathlib import Path
from typing import Optional
from abc import abstractmethod
from config.models import DocumentChunk
from config.models import DocumentMetadata
from config.models import ChunkingStrategy
from config.logging_config import get_logger
from chunking.token_counter import count_tokens


# Setup Logging
logger = get_logger(__name__)


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies : Implements Template Method pattern for consistent chunking pipeline
    """
    def __init__(self, strategy_name: ChunkingStrategy):
        """
        Initialize base chunker
        
        Arguments:
        ----------
            strategy_name { str } : Name of the chunking strategy
        """
        self.strategy_name = strategy_name
        self.logger        = logger
    

    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[DocumentMetadata] = None) -> List[DocumentChunk]:
        """
        Chunk text into smaller pieces : must be implemented by subclasses
        
        Arguments:
        ----------
            text            { str }       : Input text to chunk

            metadata { DocumentMetaData } : Document metadata
        
        Returns:
        --------
                      { list }            : List of DocumentChunk objects
        """
        pass

    
    def chunk_document(self, text: str, metadata: DocumentMetadata) -> List[DocumentChunk]:
        """
        Chunk document with full metadata: Template method that calls chunk_text and adds metadata
        
        Arguments:
        ----------
            text            { str }       : Document text

            metadata { DocumentMetaData } : Document metadata
        
        
        Returns:
        --------
                       { list }           : List of DocumentChunk objects with metadata
        """
        try:
            self.logger.info(f"Chunking document {metadata.document_id} using {self.strategy_name}")
            
            # Validate input
            if not text or not text.strip():
                self.logger.warning(f"Empty text for document {metadata.document_id}")
                return []
            
            # Perform chunking
            chunks                     = self.chunk_text(text     = text, 
                                                         metadata = metadata,
                                                        )
            
            # Update metadata
            metadata.num_chunks        = len(chunks)
            metadata.chunking_strategy = self.strategy_name
            
            # Validate chunks
            if not self.validate_chunks(chunks):
                self.logger.warning(f"Chunk validation failed for {metadata.document_id}")
            
            self.logger.info(f"Created {len(chunks)} chunks for {metadata.document_id}")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Chunking failed for {metadata.document_id}: {repr(e)}")
            raise

    
    def _create_chunk(self, text: str, chunk_index: int, document_id: str, start_char: int, end_char: int, page_number: Optional[int] = None, 
                      section_title: Optional[str] = None, metadata: Optional[dict] = None) -> DocumentChunk:
        """
        Create a DocumentChunk object with proper formatting
        
        Arguments:
        ----------
            text          { str }  : Chunk text
            
            chunk_index   { int }  : Index of chunk in document
            
            document_id   { str }  : Parent document ID
            
            start_char    { int }  : Start character position
            
            end_char      { int }  : End character position
            
            page_number   { int }  : Page number (if applicable)
            
            section_title { str }  : Section heading (if applicable)
            
            metadata      { dict } : Additional metadata
        
        Returns:
        --------
            { DocumentChunk }      : DocumentChunk object
        """
        # Generate unique chunk ID
        chunk_id    = f"chunk_{document_id}_{chunk_index}"
        
        # Count tokens
        token_count = count_tokens(text)
        
        # Create chunk
        chunk       = DocumentChunk(chunk_id      = chunk_id,
                                    document_id   = document_id,
                                    text          = text,
                                    chunk_index   = chunk_index,
                                    start_char    = start_char,
                                    end_char      = end_char,
                                    page_number   = page_number,
                                    section_title = section_title,
                                    token_count   = token_count,
                                    metadata      = metadata or {},
                                   )
        
        return chunk
    

    def _extract_page_number(self, text: str, full_text: str) -> Optional[int]:
        """
        Try to extract page number from text: Looks for [PAGE N] markers inserted during parsing
        
        Arguments:
        ----------
            text      { str } : Chunk text

            full_text { str } : Full document text
        
        Returns:
        --------
                 { int }      : Page number or None
        """
        # Look for page markers in current chunk
        page_match = re.search(r'\[PAGE (\d+)\]', text)
        
        if page_match:
            return int(page_match.group(1))
        
        # Alternative: try to determine from position in full text
        if full_text:
            # Find the chunk's approximate position
            chunk_start = full_text.find(text[:min(200, len(text))])
            
            if (chunk_start >= 0):
                # Count page markers before this position
                text_before  = full_text[:chunk_start]
                page_matches = re.findall(r'\[PAGE (\d+)\]', text_before)

                if page_matches:
                    return int(page_matches[-1])
        
        return None
    

    def _clean_chunk_text(self, text: str) -> str:
        """
        Clean chunk text by removing markers and extra whitespace
        
        Arguments:
        ----------
            text { str } : Raw chunk text
        
        Returns:
        --------
             { str }     : Cleaned text
        """
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
        Validate chunk list for consistency
        
        Arguments:
        ----------
            chunks { list } : List of chunks to validate
        
        Returns:
        --------
              { bool }      : True if valid
        """
        if not chunks:
            return True
        
        # Check all chunks have the same document_id
        doc_ids = {chunk.document_id for chunk in chunks}
        
        if (len(doc_ids) > 1):
            self.logger.error(f"Chunks have multiple document IDs: {doc_ids}")
            return False
        
        # Check chunk indices are sequential
        indices          = [chunk.chunk_index for chunk in chunks]
        expected_indices = list(range(len(chunks)))

        if (indices != expected_indices):
            self.logger.warning(f"Non-sequential chunk indices: {indices}")
        
        # Check for empty chunks
        empty_chunks = [c.chunk_index for c in chunks if not c.text.strip()]
        
        if empty_chunks:
            self.logger.warning(f"Empty chunks at indices: {empty_chunks}")
        
        # Check token counts
        zero_token_chunks = [c.chunk_index for c in chunks if (c.token_count == 0)]
        
        if zero_token_chunks:
            self.logger.warning(f"Zero-token chunks at indices: {zero_token_chunks}")
        
        return True

    
    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """
        Calculate statistics for chunk list
        
        Arguments:
        ----------
            chunks { list } : List of chunks
        
        Returns:
        --------
              { dict }      : Dictionary with statistics
        """
        if not chunks:
            return {"num_chunks"           : 0,
                    "total_tokens"         : 0,
                    "avg_tokens_per_chunk" : 0,
                    "min_tokens"           : 0,
                    "max_tokens"           : 0,
                    "total_chars"          : 0,
                    "avg_chars_per_chunk"  : 0,
                   }
        
        token_counts = [c.token_count for c in chunks]
        char_counts  = [len(c.text) for c in chunks]
        
        stats        = {"num_chunks"           : len(chunks),
                        "total_tokens"         : sum(token_counts),
                        "avg_tokens_per_chunk" : sum(token_counts) / len(chunks),
                        "min_tokens"           : min(token_counts),
                        "max_tokens"           : max(token_counts),
                        "total_chars"          : sum(char_counts),
                        "avg_chars_per_chunk"  : sum(char_counts) / len(chunks),
                        "strategy"             : self.strategy_name.value,
                       }
        
        return stats

    
    def merge_chunks(self, chunks: List[DocumentChunk], max_tokens: int) -> List[DocumentChunk]:
        """
        Merge small chunks up to max_tokens: useful for optimizing chunk sizes
        
        Arguments:
        ----------
            chunks     { list } : List of chunks to merge
            
            max_tokens { int }  : Maximum tokens per merged chunk
        
        Returns:
        --------
                { list }        : List of merged chunks
        """
        if not chunks:
            return []
        
        merged         = list()
        current_chunks = list()  # Track original chunks for position data
        current_tokens = 0
        document_id    = chunks[0].document_id
        
        for chunk in chunks:
            if (current_tokens + chunk.token_count) <= max_tokens:
                current_chunks.append(chunk)
                current_tokens += chunk.token_count
            
            else:
                # Save current merged chunk
                if current_chunks:
                    merged_text  = " ".join(c.text for c in current_chunks)
                    # Preserve position from first chunk
                    merged_chunk = self._create_chunk(text          = merged_text,
                                                      chunk_index   = len(merged),
                                                      document_id   = document_id,
                                                      start_char    = current_chunks[0].start_char,
                                                      end_char      = current_chunks[-1].end_char,
                                                      page_number   = current_chunks[0].page_number,
                                                      section_title = current_chunks[0].section_title,
                                                     )
                    merged.append(merged_chunk)
                
                # Start new chunk
                current_chunks = [chunk]
                current_tokens = chunk.token_count
        
        # Add final merged chunk
        if current_chunks:
            merged_text  = " ".join(c.text for c in current_chunks)
            merged_chunk = self._create_chunk(text          = merged_text,
                                              chunk_index   = len(merged),
                                              document_id   = document_id,
                                              start_char    = current_chunks[0].start_char,
                                              end_char      = current_chunks[-1].end_char,
                                              page_number   = current_chunks[0].page_number,
                                              section_title = current_chunks[0].section_title,
                                             )
            merged.append(merged_chunk)
        
        self.logger.info(f"Merged {len(chunks)} chunks into {len(merged)}")
        
        return merged


    def __str__(self) -> str:
        """
        String representation
        """
        return f"{self.__class__.__name__}(strategy={self.strategy_name})"
    

    def __repr__(self) -> str:
        """
        Detailed representation
        """
        return self.__str__()


class ChunkerConfig:
    """
    Configuration for chunking strategies : Provides a way to pass parameters to chunkers
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 50, respect_boundaries: bool = True, min_chunk_size: int = 100, **kwargs):
        """
        Initialize chunker configuration
        
        Arguments:
        ----------
            chunk_size          { int }  : Target chunk size in tokens

            overlap             { int }  : Overlap between chunks in tokens
            
            respect_boundaries  { bool } : Respect sentence/paragraph boundaries
            
            min_chunk_size      { int }  : Minimum chunk size in tokens
            
            **kwargs                     : Additional strategy-specific parameters
        """
        self.chunk_size         = chunk_size
        self.overlap            = overlap
        self.respect_boundaries = respect_boundaries
        self.min_chunk_size     = min_chunk_size
        self.extra              = kwargs

    
    def to_dict(self) -> dict:
        """
        Convert to dictionary
        """
        return {"chunk_size"         : self.chunk_size,
                "overlap"            : self.overlap,
                "respect_boundaries" : self.respect_boundaries,
                "min_chunk_size"     : self.min_chunk_size,
                **self.extra
               }
    

    def __repr__(self) -> str:
        return f"ChunkerConfig({self.to_dict()})"