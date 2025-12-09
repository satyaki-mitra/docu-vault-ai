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
from chunking.fixed_chunker import FixedChunker
from chunking.semantic_chunker import SemanticChunker
from chunking.llamaindex_chunker import LlamaIndexChunker
from chunking.hierarchical_chunker import HierarchicalChunker



# Setup Settings and Logging
logger = get_logger(__name__)
settings = get_settings()


class AdaptiveChunkingSelector:
    """
    Intelligent chunking strategy selector with structure detection:
    - Analyzes document characteristics (size, structure, content type)
    - Detects structured documents (projects, sections, hierarchies)
    - Automatically selects optimal chunking strategy
    - Prioritizes section-aware chunking for structured content
    
    Strategy Selection Logic (UPDATED):
    - Small docs (< 1K tokens) → Fixed chunking
    - Medium structured docs → Semantic (section-aware)
    - Medium unstructured docs → LlamaIndex or basic semantic
    - Large docs (>500K tokens) → Hierarchical chunking
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
        self.semantic_chunker     = SemanticChunker(respect_section_boundaries = True)
        self.hierarchical_chunker = HierarchicalChunker()
        self.llamaindex_chunker   = LlamaIndexChunker()
        
        # Strategy thresholds (from settings)
        self.small_doc_threshold  = settings.SMALL_DOC_THRESHOLD     
        self.large_doc_threshold  = settings.LARGE_DOC_THRESHOLD    
        
        # Check LlamaIndex availability
        self.llamaindex_available = self.llamaindex_chunker._initialized
        
        self.logger.info(f"Initialized AdaptiveChunkingSelector: LlamaIndex available={self.llamaindex_available}, prefer_llamaindex={self.prefer_llamaindex}, section_aware_semantic=True")
    

    def select_chunking_strategy(self, text: str, metadata: Optional[DocumentMetadata] = None) -> tuple[ChunkingStrategy, dict]:
        """
        Analyze document and select optimal chunking strategy: Detects structured documents and prioritizes section-aware chunking
        
        Arguments:
        ----------
            text     { str }                : Document text

            metadata { DocumentMetadata }   : Document metadata
        
        Returns:
        --------
                     { tuple }              : Tuple of (selected_strategy, analysis_results)
        """
        analysis        = self._analyze_document(text     = text, 
                                                 metadata = metadata,
                                                )
        
        # Check if document has clear structure (projects, sections)
        has_structure   = analysis.get("has_structure", False)
        structure_score = analysis.get("structure_score", 0)
        
        # Strategy selection logic
        if (analysis["total_tokens"] <= self.small_doc_threshold):
            strategy = ChunkingStrategy.FIXED
            reason   = f"Small document ({analysis['total_tokens']} tokens) - fixed chunking for simplicity"
        
        elif (analysis["total_tokens"] <= self.large_doc_threshold):
            # Medium documents: check for structure
            if (has_structure and (structure_score > 0.3)):
                # Structured document detected - use section-aware semantic chunking
                strategy = ChunkingStrategy.SEMANTIC
                reason  = (f"Medium structured document ({analysis['total_tokens']} tokens, structure_score={structure_score:.2f}) - section-aware semantic chunking")
            
            elif self.llamaindex_available and self.prefer_llamaindex:
                strategy = ChunkingStrategy.SEMANTIC
                reason   = f"Medium document ({analysis['total_tokens']} tokens) - LlamaIndex semantic chunking"
            
            else:
                strategy = ChunkingStrategy.SEMANTIC
                reason   = f"Medium document ({analysis['total_tokens']} tokens) - semantic chunking"
        
        else:
            strategy = ChunkingStrategy.HIERARCHICAL
            reason   = f"Large document ({analysis['total_tokens']} tokens) - hierarchical chunking"
        
        # Override based on document structure if available
        if (metadata and self._has_clear_structure(metadata)):
            if (strategy == ChunkingStrategy.FIXED):
                # Upgrade to semantic for structured documents
                strategy = ChunkingStrategy.SEMANTIC
                reason   = "Document has clear structure - section-aware semantic chunking preferred"
        
        analysis["selected_strategy"] = strategy
        analysis["selection_reason"]  = reason
        analysis["llamaindex_used"]   = ((strategy == ChunkingStrategy.SEMANTIC) and self.llamaindex_available and self.prefer_llamaindex and not has_structure)
        
        self.logger.info(f"Selected {strategy.value}: {reason}")
        
        return strategy, analysis
    

    def chunk_text(self, text: str, metadata: Optional[DocumentMetadata] = None, force_strategy: Optional[ChunkingStrategy] = None) -> List[DocumentChunk]:
        """
        Automatically select strategy and chunk text
        
        Arguments:
        ----------
            text           { str }                : Document text
            
            metadata       { DocumentMetadata }   : Document metadata
            
            force_strategy { ChunkingStrategy }   : Force specific strategy (optional)
        
        Returns:
        --------
                           { list }               : List of DocumentChunk objects
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
            strategy, analysis = self.select_chunking_strategy(text     = text,
                                                               metadata = metadata,
                                                              )
            reason             = analysis["selection_reason"]
            llamaindex_used    = analysis["llamaindex_used"]
        
        # Get appropriate chunker
        if ((strategy == ChunkingStrategy.SEMANTIC) and llamaindex_used):
            chunker      = self.llamaindex_chunker
            chunker_name = "LlamaIndex Semantic"

        else:
            chunker      = self._get_chunker_for_strategy(strategy = strategy)
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
                                                   "has_structure"    : analysis.get("has_structure", False),
                                                   "structure_score"  : analysis.get("structure_score", 0),
                                                  }
        
        self.logger.info(f"Using {chunker_name} chunker for document")
        
        # Perform chunking
        try:
            chunks = chunker.chunk_text(text     = text, 
                                        metadata = metadata,
                                       )
            
            # Add strategy metadata to chunks
            for chunk in chunks:
                chunk.metadata["chunking_strategy"] = strategy.value
                chunk.metadata["chunker_used"]      = chunker_name
                
                if llamaindex_used:
                    chunk.metadata["llamaindex_splitter"] = self.llamaindex_chunker.splitter_type
            
            self.logger.info(f"Successfully created {len(chunks)} chunks using {chunker_name}")
            
            # Log section coverage statistics
            chunks_with_sections = sum(1 for c in chunks if c.section_title)
            if (chunks_with_sections > 0):
                self.logger.info(f"Section coverage: {chunks_with_sections}/{len(chunks)} chunks ({chunks_with_sections/len(chunks)*100:.1f}%) have section titles")
            
            return chunks
        
        except Exception as e:
            self.logger.error(f"{chunker_name} chunking failed: {repr(e)}, falling back to fixed chunking")
            
            # Fallback to fixed chunking
            return self.fixed_chunker.chunk_text(text     = text, 
                                                 metadata = metadata,
                                                )
    

    def _analyze_document(self, text: str, metadata: Optional[DocumentMetadata] = None) -> dict:
        """
        Analyze document characteristics for strategy selection: Includes structure detection
        
        Arguments:
        ----------
            text     { str }                : Document text

            metadata { DocumentMetadata }   : Document metadata
        
        Returns:
        --------
                     { dict }               : Analysis results
        """
        # Basic token analysis
        total_tokens                   = self.token_counter.count_tokens(text = text)
        total_chars                    = len(text)
        total_words                    = len(text.split())
        
        # Estimate chunks for each strategy
        estimated_fixed_chunks         = max(1, total_tokens // settings.FIXED_CHUNK_SIZE)
        estimated_semantic_chunks      = max(1, total_tokens // (settings.FIXED_CHUNK_SIZE * 2)) 
        estimated_hierarchical_chunks  = max(1, total_tokens // settings.CHILD_CHUNK_SIZE)  
        estimated_llamaindex_chunks    = max(1, total_tokens // (settings.FIXED_CHUNK_SIZE * 1.5))
        
        # Structure analysis (simple heuristics)
        sentence_count                 = len(self.token_counter._split_into_sentences(text = text))
        avg_sentence_length            = total_words / sentence_count if (sentence_count > 0) else 0
        
        # Paragraph detection (rough)
        paragraphs                     = [p for p in text.split('\n\n') if p.strip()]
        paragraph_count                = len(paragraphs)
        
        # NEW: Detect document structure
        has_structure, structure_score = self._detect_document_structure(text)
        
        analysis                       = {"total_tokens"                  : total_tokens,
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
                                          "has_structure"                 : has_structure,
                                          "structure_score"               : structure_score,
                                         }
        
        # Add metadata-based insights if available
        if metadata:
            analysis.update({"document_type"       : metadata.document_type.value,
                             "file_size_mb"        : metadata.file_size_mb,
                             "num_pages"           : metadata.num_pages,
                             "has_clear_structure" : self._has_clear_structure(metadata),
                           })
        
        return analysis
    

    def _detect_document_structure(self, text: str) -> tuple[bool, float]:
        """
        Analyzes text for structural patterns and detect if document has clear structural elements (projects, sections, etc.) 
        & returns: (has_structure, structure_score)
        """
        structure_indicators = 0
        max_indicators       = 5
        
        # Check for project-style headers: "a) Project Name", "b) Project Name"
        project_headers      = len(re.findall(r'^[a-z]\)\s+[A-Z]', text, re.MULTILINE))
        
        if (project_headers > 2):
            structure_indicators += 1
        
        # Check for bullet point lists: "●" or "❖"
        bullet_points = text.count('●') + text.count('❖')
        
        if (bullet_points > 5):
            structure_indicators += 1
        
        # Check for numbered sections: "1.", "2.", etc.
        numbered_sections = len(re.findall(r'^\d+\.\s+[A-Z]', text, re.MULTILINE))
        
        if (numbered_sections > 2):
            structure_indicators += 1
        
        # Check for subsection markers ending with ":"
        subsection_markers = len(re.findall(r'^●\s+\w+.*:', text, re.MULTILINE))
        
        if (subsection_markers > 3):
            structure_indicators += 1
        
        # Check for consistent indentation patterns
        lines          = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('   ') or line.startswith('\t'))
        
        # >20% indented
        if (indented_lines > len(lines) * 0.2): 
            structure_indicators += 1
        
        has_structure   = (structure_indicators >= 2)
        structure_score = structure_indicators / max_indicators
        
        if has_structure:
            self.logger.info(f"Document structure detected: score={structure_score:.2f} (project_headers={project_headers}, bullets={bullet_points}, "
                             f"numbered_sections={numbered_sections}, subsections={subsection_markers})")
        
        return has_structure, structure_score
    

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
        
        return chunkers.get(strategy, self.fixed_chunker)
    

    def _get_size_category(self, total_tokens: int) -> str:
        """
        Categorize document by size
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
        """
        if metadata.extra:
            # DOCX with multiple sections/headings
            if (metadata.document_type.value == "docx"):
                if (metadata.extra.get("num_sections", 0) > 1):
                    return True

                if (metadata.extra.get("num_paragraphs", 0) > 50):
                    return True
            
            # PDF with multiple pages and likely structure
            if (metadata.document_type.value == "pdf"):
                if metadata.num_pages and metadata.num_pages > 10:
                    return True
        
        return False
    

    def get_strategy_recommendations(self, text: str, metadata: Optional[DocumentMetadata] = None) -> dict:
        """
        Get detailed strategy recommendations with pros/cons
        """
        analysis                              = self._analyze_document(text, metadata)
        
        # LlamaIndex recommendation
        llamaindex_recommendation             = {"recommended_for"  : ["Medium documents", "Structured content", "Superior semantic analysis"],
                                                 "pros"             : ["Best semantic boundary detection", "LlamaIndex ecosystem integration", "Advanced embedding-based splitting"],
                                                 "cons"             : ["Additional dependency", "Slower initialization", "More complex setup"],
                                                 "estimated_chunks" : analysis["estimated_llamaindex_chunks"],
                                                 "available"        : self.llamaindex_available,
                                                }
        
        recommendations                       = {"fixed"        : {"recommended_for"  : ["Small documents", "Homogeneous content", "Simple processing"],
                                                                   "pros"             : ["Fast", "Reliable", "Predictable chunk sizes"],
                                                                   "cons"             : ["May break semantic boundaries", "Ignores document structure"],
                                                                   "estimated_chunks" : analysis["estimated_fixed_chunks"],
                                                                  },
                                                 "semantic"     : {"recommended_for"  : ["Medium documents", "Structured content", "When coherence matters"],
                                                                   "pros"             : ["Preserves topic boundaries", "Respects section structure", "Better context coherence"],
                                                                   "cons"             : ["Slower (requires embeddings)", "Less predictable chunk sizes"],
                                                                   "estimated_chunks" : analysis["estimated_semantic_chunks"],
                                                                   "section_aware"    : True,
                                                                  },
                                                 "llamaindex"   : llamaindex_recommendation,
                                                 "hierarchical" : {"recommended_for"  : ["Large documents", "Complex structure", "Granular search needs"],
                                                                   "pros"             : ["Best for large docs", "Granular + context search", "Scalable"],
                                                                   "cons"             : ["Complex implementation", "More chunks to manage", "Higher storage"],
                                                                   "estimated_chunks" : analysis["estimated_hierarchical_chunks"],
                                                                  }
                                                }
        
        # Add selected strategy
        selected_strategy, analysis_result    = self.select_chunking_strategy(text     = text, 
                                                                              metadata = metadata,
                                                                             )
        
        recommendations["selected_strategy"]  = selected_strategy.value
        recommendations["selection_reason"]   = analysis_result["selection_reason"]
        recommendations["llamaindex_used"]    = analysis_result["llamaindex_used"]
        recommendations["structure_detected"] = analysis_result.get("has_structure", False)
        
        return recommendations


# Global adaptive selector instance
_adaptive_selector = None


def get_adaptive_selector() -> AdaptiveChunkingSelector:
    """
    Get global adaptive selector instance (singleton)
    """
    global _adaptive_selector

    if _adaptive_selector is None:
        _adaptive_selector = AdaptiveChunkingSelector()

    return _adaptive_selector


def adaptive_chunk_text(text: str, metadata: Optional[DocumentMetadata] = None, force_strategy: Optional[ChunkingStrategy] = None) -> List[DocumentChunk]:
    """
    Convenience function for adaptive chunking
    """
    selector = get_adaptive_selector()

    return selector.chunk_text(text, metadata, force_strategy)


def analyze_document(text: str, metadata: Optional[DocumentMetadata] = None) -> dict:
    """
    Analyze document without chunking
    """
    selector = get_adaptive_selector()
    
    return selector._analyze_document(text, metadata)