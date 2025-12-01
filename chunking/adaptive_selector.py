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
from chunking.fixed_chunker import FixedChunker
from chunking.semantic_chunker import SemanticChunker
from chunking.llamaindex_chunker import LlamaIndexChunker
from chunking.hierarchical_chunker import HierarchicalChunker



# Setup Settings and Logging
logger   = get_logger(__name__)
settings = get_settings()


class AdaptiveChunkingSelector:
    """
    Intelligent chunking strategy selector:
    - Analyzes document characteristics (size, structure, content type)
    - Automatically selects optimal chunking strategy
    - Provides strategy recommendations with reasoning
    
    Strategy Selection Logic:
    - Small docs (<50K tokens)      â†’ Fixed chunking (simplicity)
    - Medium docs (50K-500K tokens) â†’ Semantic/LlamaIndex chunking (coherence)  
    - Large docs (>500K tokens)     â†’ Hierarchical chunking (scalability)
    - LlamaIndex preferred when available for semantic chunking
    - Fallback to fixed chunking for reliability
    """
    def __init__(self, prefer_llamaindex: bool = True):
        """
        Initialize adaptive selector with all chunking strategies
        
        Arguments:
        ----------
            prefer_llamaindex { bool } : Prefer LlamaIndex over custom semantic chunking when available
        """
        self.logger               = logger
        self.token_counter        = TokenCounter()
        self.prefer_llamaindex    = prefer_llamaindex
        
        # Initialize all chunking strategies
        self.fixed_chunker        = FixedChunker()
        self.semantic_chunker     = SemanticChunker()
        self.hierarchical_chunker = HierarchicalChunker()
        self.llamaindex_chunker   = LlamaIndexChunker()
        
        # Strategy thresholds (from settings)
        self.small_doc_threshold  = settings.SMALL_DOC_THRESHOLD     
        self.large_doc_threshold  = settings.LARGE_DOC_THRESHOLD    
        
        # Check LlamaIndex availability
        self.llamaindex_available = self.llamaindex_chunker._initialized
        
        self.logger.info(f"Initialized AdaptiveChunkingSelector: LlamaIndex available={self.llamaindex_available}, prefer_llamaindex={self.prefer_llamaindex}")
    

    def select_chunking_strategy(self, text: str, metadata: Optional[DocumentMetadata] = None) -> tuple[ChunkingStrategy, dict]:
        """
        Analyze document and select optimal chunking strategy
        
        Arguments:
        ----------
            text            { str }       : Document text

            metadata { DocumentMetaData } : Document metadata
        
        Returns:
        --------
                  { tuple }               : Tuple of (selected_strategy, analysis_results)
        """
        analysis = self._analyze_document(text     = text, 
                                          metadata = metadata,
                                         )
        
        # Strategy selection logic
        if (analysis["total_tokens"] <= self.small_doc_threshold):
            strategy = ChunkingStrategy.FIXED
            reason   = f"Small document ({analysis['total_tokens']} tokens) - fixed chunking for simplicity"
        
        elif (analysis["total_tokens"] <= self.large_doc_threshold):
            # Medium documents: prefer LlamaIndex if available and preferred
            if (self.llamaindex_available and self.prefer_llamaindex):
                # LlamaIndex uses SEMANTIC strategy enum
                strategy = ChunkingStrategy.SEMANTIC  
                reason   = f"Medium document ({analysis['total_tokens']} tokens) - LlamaIndex semantic chunking for superior coherence"
            
            else:
                strategy = ChunkingStrategy.SEMANTIC
                reason   = f"Medium document ({analysis['total_tokens']} tokens) - semantic chunking for coherence"
        
        else:
            strategy = ChunkingStrategy.HIERARCHICAL
            reason   = f"Large document ({analysis['total_tokens']} tokens) - hierarchical chunking for scalability"
        
        # Override based on document structure if available
        if metadata and self._has_clear_structure(metadata):
            if (strategy == ChunkingStrategy.FIXED):
                # Upgrade to semantic/LlamaIndex for structured documents
                if (self.llamaindex_available and self.prefer_llamaindex):
                    strategy = ChunkingStrategy.SEMANTIC
                    reason   = "Document has clear structure - LlamaIndex semantic chunking preferred"
                
                else:
                    strategy = ChunkingStrategy.SEMANTIC
                    reason   = "Document has clear structure - semantic chunking preferred"
        
        analysis["selected_strategy"] = strategy
        analysis["selection_reason"]  = reason
        analysis["llamaindex_used"]   = (strategy == ChunkingStrategy.SEMANTIC and self.llamaindex_available and self.prefer_llamaindex)
        
        self.logger.info(f"Selected {strategy.value}: {reason}")
        
        return strategy, analysis
    

    def chunk_text(self, text: str, metadata: Optional[DocumentMetadata] = None, force_strategy: Optional[ChunkingStrategy] = None) -> List[DocumentChunk]:
        """
        Automatically select strategy and chunk text
        
        Arguments:
        ----------
            text                  { str }        : Document text

            metadata        { DocumentMetaData } : Document metadata
            
            force_strategy  { ChunkingStrategy } : Force specific strategy (optional)
        
        Returns:
        --------
                        { list }                 : List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Select strategy (or use forced strategy)
        if force_strategy:
            strategy        = force_strategy
            analysis        = self._analyze_document(text     = text, 
                                                     metadata = metadata,
                                                    )

            reason          = f"Forced strategy: {force_strategy.value}"
            llamaindex_used = False
        
        else:
            strategy, analysis = self.select_chunking_strategy(text, metadata)
            reason             = analysis["selection_reason"]
            llamaindex_used    = analysis["llamaindex_used"]
        
        # Get appropriate chunker (use LlamaIndex for semantic if preferred and available)
        if ((strategy == ChunkingStrategy.SEMANTIC) and llamaindex_used):
            chunker      = self.llamaindex_chunker
            chunker_name = "LlamaIndex Semantic"
        
        else:
            chunker      = self._get_chunker_for_strategy(strategy)
            chunker_name = strategy.value
        
        # Update metadata with strategy information
        if metadata:
            metadata.chunking_strategy          = strategy
            metadata.extra["chunking_analysis"] = {"strategy"         : strategy.value,
                                                   "chunker_used"     : chunker_name,
                                                   "reason"           : reason,
                                                   "total_tokens"     : analysis["total_tokens"],
                                                   "estimated_chunks" : analysis[f"estimated_{strategy.value.lower()}_chunks"],
                                                   "llamaindex_used"  : llamaindex_used,
                                                  }
        
        self.logger.info(f"Using {chunker_name} chunker for document")
        
        # Perform chunking
        try:
            chunks = chunker.chunk_text(text, metadata)
            
            # Add strategy metadata to chunks
            for chunk in chunks:
                chunk.metadata["chunking_strategy"] = strategy.value
                chunk.metadata["chunker_used"]      = chunker_name
                
                if llamaindex_used:
                    chunk.metadata["llamaindex_splitter"] = self.llamaindex_chunker.splitter_type
            
            self.logger.info(f"Successfully created {len(chunks)} chunks using {chunker_name}")
            
            return chunks
        
        except Exception as e:
            self.logger.error(f"{chunker_name} chunking failed: {repr(e)}, falling back to fixed chunking")
            
            # Fallback to fixed chunking
            return self.fixed_chunker.chunk_text(text     = text, 
                                                 metadata = metadata,
                                                )
    

    def _analyze_document(self, text: str, metadata: Optional[DocumentMetadata] = None) -> dict:
        """
        Analyze document characteristics for strategy selection
        
        Arguments:
        ----------
            text            { str }       : Document text

            metadata { DocumentMetaData } : Document metadata
        
        Returns:
        --------
                 { dict }                 : Analysis results
        """
        # Basic token analysis
        total_tokens                  = self.token_counter.count_tokens(text)
        total_chars                   = len(text)
        total_words                   = len(text.split())
        
        # Estimate chunks for each strategy
        estimated_fixed_chunks        = max(1, total_tokens // settings.FIXED_CHUNK_SIZE)
        estimated_semantic_chunks     = max(1, total_tokens // (settings.FIXED_CHUNK_SIZE * 2)) 
        estimated_hierarchical_chunks = max(1, total_tokens // settings.CHILD_CHUNK_SIZE)  
        estimated_llamaindex_chunks   = max(1, total_tokens // (settings.FIXED_CHUNK_SIZE * 1.5))
        
        # Structure analysis (simple heuristics)
        sentence_count                = len(self.token_counter._split_into_sentences(text))
        avg_sentence_length           = total_words / sentence_count if (sentence_count > 0) else 0
        
        # Paragraph detection (rough)
        paragraphs                    = [p for p in text.split('\n\n') if p.strip()]
        paragraph_count               = len(paragraphs)
        
        analysis                      = {"total_tokens"                  : total_tokens,
                                         "total_chars"                   : total_chars,
                                         "total_words"                   : total_words,
                                         "sentence_count"                : sentence_count,
                                         "paragraph_count"               : paragraph_count,
                                         "avg_sentence_length"           : avg_sentence_length,
                                         "estimated_fixed_chunks"        : estimated_fixed_chunks,
                                         "estimated_semantic_chunks"     : estimated_semantic_chunks,
                                         "estimated_llamaindex_chunks"   : estimated_llamaindex_chunks,
                                         "estimated_hierarchical_chunks" : estimated_hierarchical_chunks,
                                         "document_size_category"        : self._get_size_category(total_tokens),
                                         "llamaindex_available"          : self.llamaindex_available,
                                        }
        
        # Add metadata-based insights if available
        if metadata:
            analysis.update({"document_type"       : metadata.document_type.value,
                             "file_size_mb"        : metadata.file_size_mb,
                             "num_pages"           : metadata.num_pages,
                             "has_clear_structure" : self._has_clear_structure(metadata),
                           })
        
        return analysis
    

    def _get_chunker_for_strategy(self, strategy: ChunkingStrategy) -> BaseChunker:
        """
        Get chunker instance for specified strategy
        
        Arguments:
        ----------
            strategy { ChunkingStrategy } : Chunking strategy
        
        Returns:
        --------
            { BaseChunker }               : Chunker instance
        """
        chunkers = {ChunkingStrategy.FIXED        : self.fixed_chunker,
                    ChunkingStrategy.SEMANTIC     : self.semantic_chunker,
                    ChunkingStrategy.HIERARCHICAL : self.hierarchical_chunker,
                   }
        
        # Fallback to fixed
        return chunkers.get(strategy, self.fixed_chunker)  
    

    def _get_size_category(self, total_tokens: int) -> str:
        """
        Categorize document by size
        
        Arguments:
        ----------
            total_tokens { int } : Total tokens in document
        
        Returns:
        --------
                 { str }         : Size category
        """
        if (total_tokens <= self.small_doc_threshold):
            return "small"

        elif (total_tokens <= self.large_doc_threshold):
            return "medium"

        else:
            return "large"
    

    def _has_clear_structure(self, metadata: DocumentMetadata) -> bool:
        """
        Check if document has clear structural elements
        
        Arguments:
        ----------
            metadata { DocumentMetadata } : Document metadata
        
        Returns:
        --------
                 { bool }                 : True if clear structure detected
        """
        # Check for structural indicators in metadata
        if metadata.extra:
            # DOCX with multiple sections/headings
            if (metadata.document_type.value == "docx"):
                if (metadata.extra.get("num_sections", 0) > 1):
                    return True
                
                # Likely structured
                if (metadata.extra.get("num_paragraphs", 0) > 50): 
                    return True
            
            # PDF with multiple pages and likely structure
            if (metadata.document_type.value == "pdf"):
                if (metadata.num_pages and (metadata.num_pages > 10)):
                    return True
        
        return False
    

    def get_strategy_recommendations(self, text: str, metadata: Optional[DocumentMetadata] = None) -> dict:
        """
        Get detailed strategy recommendations with pros/cons
        
        Arguments:
        ----------
            text            { str }       : Document text

            metadata { DocumentMetaData } : Document metadata
        
        Returns:
        --------
                 { dict }                 : Strategy recommendations
        """
        analysis                             = self._analyze_document(text, metadata)
        
        # LlamaIndex recommendation
        llamaindex_recommendation            = {"recommended_for"  : ["Medium documents", "Structured content", "Superior semantic analysis"],
                                                "pros"             : ["Best semantic boundary detection", "LlamaIndex ecosystem integration", "Advanced embedding-based splitting"],
                                                "cons"             : ["Additional dependency", "Slower initialization", "More complex setup"],
                                                "estimated_chunks" : analysis["estimated_llamaindex_chunks"],
                                                "available"        : self.llamaindex_available,
                                               }
        
        recommendations                      = {"fixed"        : {"recommended_for"  : ["Small documents", "Homogeneous content", "Simple processing"],
                                                                  "pros"             : ["Fast", "Reliable", "Predictable chunk sizes"],
                                                                  "cons"             : ["May break semantic boundaries", "Less contextually coherent"],
                                                                  "estimated_chunks" : analysis["estimated_fixed_chunks"],
                                                                 },
                                                "semantic"     : {"recommended_for"  : ["Medium documents", "Structured content", "When coherence matters"],
                                                                  "pros"             : ["Preserves topic boundaries", "Better context coherence", "Intelligent breaks"],
                                                                  "cons"             : ["Slower (requires embeddings)", "Less predictable chunk sizes"],
                                                                  "estimated_chunks" : analysis["estimated_semantic_chunks"],
                                                                 },
                                                "llamaindex"   : llamaindex_recommendation,
                                                "hierarchical" : {"recommended_for"  : ["Large documents", "Complex structure", "Granular search needs"],
                                                                  "pros"             : ["Best for large docs", "Granular + context search", "Scalable"],
                                                                  "cons"             : ["Complex implementation", "More chunks to manage", "Higher storage"],
                                                                  "estimated_chunks" : analysis["estimated_hierarchical_chunks"],
                                                                 }
                                               }
        
        # Add selected strategy
        selected_strategy, analysis_result   = self.select_chunking_strategy(text     = text, 
                                                                             metadata = metadata,
                                                                            )
        
        recommendations["selected_strategy"] = selected_strategy.value
        recommendations["selection_reason"]  = analysis_result["selection_reason"]
        recommendations["llamaindex_used"]   = analysis_result["llamaindex_used"]
        
        return recommendations


# Global adaptive selector instance
_adaptive_selector = None


def get_adaptive_selector() -> AdaptiveChunkingSelector:
    """
    Get global adaptive selector instance (singleton)
    
    Returns:
    --------
        { AdaptiveChunkingSelector } : AdaptiveChunkingSelector instance
    """
    global _adaptive_selector

    if _adaptive_selector is None:
        _adaptive_selector = AdaptiveChunkingSelector()
    
    return _adaptive_selector


def adaptive_chunk_text(text: str, metadata: Optional[DocumentMetadata] = None, force_strategy: Optional[ChunkingStrategy] = None) -> List[DocumentChunk]:
    """
    Convenience function for adaptive chunking
    
    Arguments:
    ----------
        text            { str }       : Document text

        metadata { DocumentMetaData } : Document metadata
        
        force_strategy  { ChunkingStrategy } : Force specific strategy
    
    Returns:
    --------
             { list }                 : List of DocumentChunk objects
    """
    selector = get_adaptive_selector()

    return selector.chunk_text(text, metadata, force_strategy)


def analyze_document(text: str, metadata: Optional[DocumentMetadata] = None) -> dict:
    """
    Analyze document without chunking
    
    Arguments:
    ----------
        text            { str }       : Document text

        metadata { DocumentMetaData } : Document metadata
    
    Returns:
    --------
           { dict }                   : Analysis results
    """
    selector = get_adaptive_selector()

    return selector._analyze_document(text, metadata)