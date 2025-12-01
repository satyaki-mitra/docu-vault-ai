# DEPENDENCIES
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
from chunking.fixed_chunker import FixedChunker


# Setup Settings and Logging
logger   = get_logger(__name__)
settings = get_settings()


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking strategy:
    - Creates parent chunks (large) and child chunks (small)
    - Child chunks for granular search, parent chunks for context
    - Maintains parent-child relationships for context expansion
    
    Best for:
    - Large documents (>500K tokens)
    - Complex documents with nested structure
    - When both granular search and context preservation are needed
    """
    def __init__(self, parent_chunk_size: int = None, child_chunk_size: int = None, overlap: int = None, min_chunk_size: int = 100):
        """
        Initialize hierarchical chunker
        
        Arguments:
        ----------
            parent_chunk_size { int } : Size of parent chunks in tokens
            
            child_chunk_size  { int } : Size of child chunks in tokens
            
            overlap           { int } : Overlap between child chunks
            
            min_chunk_size    { int } : Minimum chunk size in tokens
        """
        super().__init__(ChunkingStrategy.HIERARCHICAL)
        
        self.parent_chunk_size = parent_chunk_size or settings.PARENT_CHUNK_SIZE
        self.child_chunk_size  = child_chunk_size or settings.CHILD_CHUNK_SIZE
        self.overlap           = overlap or settings.FIXED_CHUNK_OVERLAP
        self.min_chunk_size    = min_chunk_size
        
        # Validate parameters
        if (self.child_chunk_size >= self.parent_chunk_size):
            raise ValueError(f"Child chunk size ({self.child_chunk_size}) must be smaller than parent chunk size ({self.parent_chunk_size})")
        
        # Initialize dependencies
        self.token_counter     = TokenCounter()
        self.overlap_manager   = OverlapManager(overlap_tokens = self.overlap)
        self.child_chunker     = FixedChunker(chunk_size                  = self.child_chunk_size,
                                              overlap                     = self.overlap,
                                              respect_sentence_boundaries = True,
                                             )
        
        self.logger.info(f"Initialized HierarchicalChunker: parent_size={self.parent_chunk_size}, child_size={self.child_chunk_size}, overlap={self.overlap}")
    

    def chunk_text(self, text: str, metadata: Optional[DocumentMetadata] = None) -> List[DocumentChunk]:
        """
        Create hierarchical chunks with parent-child relationships
        
        Arguments:
        ----------
            text            { str }       : Input text

            metadata { DocumentMetaData } : Document metadata
        
        Returns:
        --------
                     { list }             : List of DocumentChunk objects (children with parent references)
        """
        if not text or not text.strip():
            return []
        
        document_id      = metadata.document_id if metadata else "unknown"
        
        # Create parent chunks (large context windows)
        parent_chunks    = self._create_parent_chunks(text, document_id)
        
        # For each parent chunk, create child chunks (granular search)
        all_child_chunks = list()
        
        for parent_chunk in parent_chunks:
            child_chunks = self._create_child_chunks(parent_chunk    = parent_chunk,
                                                     parent_text     = text,
                                                     document_id     = document_id,
                                                    )
            all_child_chunks.extend(child_chunks)
        
        # Step 3: Filter small chunks
        all_child_chunks = [c for c in all_child_chunks if (c.token_count >= self.min_chunk_size)]
        
        self.logger.info(f"Created {len(all_child_chunks)} child chunks from {len(parent_chunks)} parent chunks")
        
        return all_child_chunks
    

    def _create_parent_chunks(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Create large parent chunks for context preservation
        
        Arguments:
        ----------
            text        { str } : Input text

            document_id { str } : Document ID
        
        Returns:
        --------
                { list }        : List of parent chunks WITHOUT overlap
        """
        # Use fixed chunking for parents (no overlap between parents)
        parent_chunker = FixedChunker(chunk_size                  = self.parent_chunk_size,
                                      overlap                     = 0,  # No overlap between parents
                                      respect_sentence_boundaries = True,
                                     )
        
        # Create parent chunks
        parent_chunks  = parent_chunker._chunk_with_sentence_boundaries(text        = text,
                                                                        document_id = document_id,
                                                                       )
        
        # Add parent metadata
        for i, chunk in enumerate(parent_chunks):
            chunk.metadata["chunk_type"]      = "parent"
            chunk.metadata["parent_chunk_id"] = chunk.chunk_id
        
        return parent_chunks
    

    def _create_child_chunks(self, parent_chunk: DocumentChunk, parent_text: str, document_id: str) -> List[DocumentChunk]:
        """
        Create child chunks within a parent chunk
        
        Arguments:
        ----------
            parent_chunk { DocumentChunk } : Parent chunk object

            parent_text  { str }           : Full parent text (for position reference)
            
            document_id  { str }           : Document ID
        
        Returns:
        --------
                   { list }                : List of child chunks with parent references
        """
        # Extract the actual text segment from parent_text using parent chunk positions
        parent_segment = parent_text[parent_chunk.start_char:parent_chunk.end_char]
        
        # Create child chunks within this parent segment
        child_chunker  = FixedChunker(chunk_size                  = self.child_chunk_size,
                                      overlap                     = self.overlap,
                                      respect_sentence_boundaries = True,
                                     )
        
        # Create child chunks with proper positioning
        child_chunks   = child_chunker._chunk_with_sentence_boundaries(text        = parent_segment,
                                                                       document_id = document_id,
                                                                      )
        
        # Update child chunks with parent relationship and correct positions
        for i, child_chunk in enumerate(child_chunks):
            # Adjust positions to be relative to full document
            child_chunk.start_char                 += parent_chunk.start_char
            child_chunk.end_char                   += parent_chunk.start_char
            
            # Add parent relationship metadata
            child_chunk.metadata["chunk_type"]      = "child"
            child_chunk.metadata["parent_chunk_id"] = parent_chunk.chunk_id
            child_chunk.metadata["parent_index"]    = i
            
            # Update chunk ID to reflect hierarchy
            child_chunk.chunk_id                    = f"{parent_chunk.chunk_id}_child_{i}"
        
        return child_chunks
    

    def expand_to_parent_context(self, child_chunk: DocumentChunk, all_chunks: List[DocumentChunk]) -> DocumentChunk:
        """
        Expand a child chunk to include full parent context for generation
        
        Arguments:
        ----------
            child_chunk { DocumentChunk } : Child chunk to expand

            all_chunks  { list }          : All chunks from the document
        
        Returns:
        --------
                   { DocumentChunk }      : Expanded chunk with parent context
        """
        # Find the parent chunk
        parent_chunk_id = child_chunk.metadata.get("parent_chunk_id")
        
        if not parent_chunk_id:
            return child_chunk
        
        parent_chunk = next((c for c in all_chunks if c.chunk_id == parent_chunk_id), None)
        
        if not parent_chunk:
            return child_chunk
        
        # Create expanded chunk with parent context
        expanded_text  = f"[PARENT_CONTEXT]\n{parent_chunk.text}\n\n[CHILD_CONTEXT]\n{child_chunk.text}"
        
        expanded_chunk = DocumentChunk(chunk_id        = f"{child_chunk.chunk_id}_expanded",
                                       document_id     = child_chunk.document_id,
                                       text            = expanded_text,
                                       chunk_index     = child_chunk.chunk_index,
                                       start_char      = child_chunk.start_char,
                                       end_char        = child_chunk.end_char,
                                       page_number     = child_chunk.page_number,
                                       section_title   = child_chunk.section_title,
                                       token_count     = self.token_counter.count_tokens(expanded_text),
                                       parent_chunk_id = parent_chunk_id,
                                       child_chunk_ids = [child_chunk.chunk_id],
                                       metadata        = {**child_chunk.metadata, "expanded": True},
                                      )
        
        return expanded_chunk
    

    def get_parent_child_relationships(self, chunks: List[DocumentChunk]) -> dict:
        """
        Extract parent-child relationships from chunks
        
        Arguments:
        ----------
            chunks { list } : List of chunks
        
        Returns:
        --------
              { dict }      : Dictionary mapping parent IDs to child chunks
        """
        relationships = dict()
        
        for chunk in chunks:
            if (chunk.metadata.get("chunk_type") == "parent"):
                relationships[chunk.chunk_id] = {"parent"   : chunk,
                                                 "children" : [],
                                                }
        
        for chunk in chunks:
            parent_id = chunk.metadata.get("parent_chunk_id")
            
            if parent_id and parent_id in relationships:
                relationships[parent_id]["children"].append(chunk)
        
        return relationships
    

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'HierarchicalChunker':
        """
        Create HierarchicalChunker from configuration
        
        Arguments:
        ----------
            config { ChunkerConfig } : ChunkerConfig object
        
        Returns:
        --------
            { HierarchicalChunker }  : HierarchicalChunker instance
        """
        return cls(parent_chunk_size = config.extra.get('parent_size', settings.PARENT_CHUNK_SIZE),
                   child_chunk_size  = config.extra.get('child_size', settings.CHILD_CHUNK_SIZE),
                   overlap           = config.overlap,
                   min_chunk_size    = config.min_chunk_size,
                  )