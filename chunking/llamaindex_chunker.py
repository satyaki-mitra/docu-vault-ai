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
from chunking.semantic_chunker import SemanticChunker
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser


# Setup Settings and Logging
logger   = get_logger(__name__)
settings = get_settings()


class LlamaIndexChunker(BaseChunker):
    """
    LlamaIndex-based semantic chunking strategy:
    - Uses LlamaIndex's advanced semantic splitting algorithms
    - Provides superior boundary detection using embeddings
    - Supports multiple LlamaIndex splitter types
    
    Best for:
    - Documents requiring sophisticated semantic analysis
    - When LlamaIndex ecosystem integration is needed
    - Advanced chunking with embedding-based boundaries
    """
    def __init__(self, chunk_size: int = None, overlap: int = None, splitter_type: str = "semantic", min_chunk_size: int = 100):
        """
        Initialize LlamaIndex chunker
        
        Arguments:
        ----------
            chunk_size     { int }  : Target tokens per chunk
            
            overlap        { int }  : Overlap tokens between chunks
            
            splitter_type  { str }  : Type of LlamaIndex splitter ("semantic", "sentence", "token")
            
            min_chunk_size { int }  : Minimum chunk size in tokens
        """
        # Use SEMANTIC since it's semantic-based
        super().__init__(ChunkingStrategy.SEMANTIC)  
        
        self.chunk_size     = chunk_size or settings.FIXED_CHUNK_SIZE
        self.overlap        = overlap or settings.FIXED_CHUNK_OVERLAP
        self.splitter_type  = splitter_type
        self.min_chunk_size = min_chunk_size
        
        # Initialize token counter
        self.token_counter  = TokenCounter()
        
        # Initialize LlamaIndex components
        self._splitter      = None
        self._initialized   = False
        
        self._initialize_llamaindex()
        
        self.logger.info(f"Initialized LlamaIndexChunker: chunk_size={self.chunk_size}, overlap={self.overlap}, splitter_type={self.splitter_type}")
    

    def _initialize_llamaindex(self):
        """
        Initialize LlamaIndex splitter with proper error handling
        """
        try:
            # Initialize embedding model
            embed_model = HuggingFaceEmbedding(model_name = settings.EMBEDDING_MODEL)
            
            # Initialize appropriate splitter based on type
            if (self.splitter_type == "semantic"):
                self._splitter = SemanticSplitterNodeParser(buffer_size         = 1,
                                                            breakpoint_percentile_threshold = 95,
                                                            embed_model        = embed_model,
                                                           )
            
            elif (self.splitter_type == "sentence"):
                self._splitter = SentenceSplitter(chunk_size       = self.chunk_size,
                                                  chunk_overlap    = self.overlap,
                                                 )
            
            elif (self.splitter_type == "token"):
                self._splitter = TokenTextSplitter(chunk_size       = self.chunk_size,
                                                   chunk_overlap    = self.overlap,
                                                  )
            
            else:
                self.logger.warning(f"Unknown splitter type: {self.splitter_type}, using semantic")
                self._splitter = SemanticSplitterNodeParser(buffer_size                     = 1,
                                                            breakpoint_percentile_threshold = 95,
                                                            embed_model                     = embed_model,
                                                           )
            
            self._initialized = True
            self.logger.info(f"Successfully initialized LlamaIndex {self.splitter_type} splitter")
            
        except ImportError as e:
            self.logger.error(f"LlamaIndex not available: {repr(e)}")
            self._initialized = False
        
        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaIndex: {repr(e)}")
            self._initialized = False
    

    def chunk_text(self, text: str, metadata: Optional[DocumentMetadata] = None) -> List[DocumentChunk]:
        """
        Chunk text using LlamaIndex semantic splitting
        
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
        
        # Fallback if LlamaIndex not available
        if not self._initialized:
            self.logger.warning("LlamaIndex not available, falling back to simple semantic chunking")
            return self._fallback_chunking(text        = text, 
                                           metadata    = metadata,
                                          )
        
        document_id = metadata.document_id if metadata else "unknown"
        
        try:
            # Create LlamaIndex document
            llama_doc = LlamaDocument(text = text)
            
            # Get nodes from splitter
            nodes     = self._splitter.get_nodes_from_documents([llama_doc])
            
            # Convert nodes to our DocumentChunk format
            chunks    = list()
            start_pos = 0
            
            for i, node in enumerate(nodes):
                chunk_text = node.text
                
                # Create chunk
                chunk      = self._create_chunk(text          = self._clean_chunk_text(chunk_text),
                                                chunk_index   = i,
                                                document_id   = document_id,
                                                start_char    = start_pos,
                                                end_char      = start_pos + len(chunk_text),
                                                metadata      = {"llamaindex_splitter" : self.splitter_type,
                                                                 "node_id"             : node.node_id,
                                                                 "chunk_type"          : "llamaindex_semantic",
                                                                }
                                               )
                
                chunks.append(chunk)
                start_pos += len(chunk_text)
            
            # Filter out chunks that are too small
            chunks = [c for c in chunks if (c.token_count >= self.min_chunk_size)]
            
            self.logger.debug(f"Created {len(chunks)} chunks using LlamaIndex {self.splitter_type} splitter")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"LlamaIndex chunking failed: {repr(e)}")
            return self._fallback_chunking(text     = text, 
                                           metadata = metadata,
                                          )
    

    def _fallback_chunking(self, text: str, metadata: Optional[DocumentMetadata] = None) -> List[DocumentChunk]:
        """
        Fallback to basic semantic chunking when LlamaIndex fails
        
        Arguments:
        ----------
            text            { str }       : Input text

            metadata { DocumentMetaData } : Document metadata
        
        Returns:
        --------
                     { list }             : List of chunks
        """
        fallback_chunker = SemanticChunker(chunk_size           = self.chunk_size,
                                           overlap              = self.overlap,
                                           similarity_threshold = 0.95,
                                           min_chunk_size       = self.min_chunk_size,
                                          )
        
        return fallback_chunker.chunk_text(text, metadata)
    

    def get_splitter_info(self) -> dict:
        """
        Get information about the LlamaIndex splitter configuration
        
        Returns:
        --------
            { dict }    : Splitter information
        """
        return {"splitter_type"   : self.splitter_type,
                "chunk_size"      : self.chunk_size,
                "overlap"         : self.overlap,
                "initialized"     : self._initialized,
                "min_chunk_size"  : self.min_chunk_size,
               }
    

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'LlamaIndexChunker':
        """
        Create LlamaIndexChunker from configuration
        
        Arguments:
        ----------
            config { ChunkerConfig } : ChunkerConfig object
        
        Returns:
        --------
            { LlamaIndexChunker }    : LlamaIndexChunker instance
        """
        return cls(chunk_size    = config.chunk_size,
                   overlap       = config.overlap,
                   splitter_type = config.extra.get('llamaindex_splitter', 'semantic'),
                   min_chunk_size = config.min_chunk_size,
                  )