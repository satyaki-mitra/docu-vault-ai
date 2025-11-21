"""
Adaptive Chunking Selector
Intelligently selects optimal chunking strategy based on document characteristics
"""

from typing import List, Optional, Tuple
from pathlib import Path

from config.logging_config import get_logger
from config.models import DocumentChunk, DocumentMetadata, ChunkingStrategy
from config.settings import get_settings
from chunking.base_chunker import BaseChunker, ChunkerConfig
from chunking.fixed_chunker import FixedChunker
from chunking.token_counter import TokenCounter

logger = get_logger(__name__)
settings = get_settings()


class AdaptiveSelector:
    """
    Adaptive chunking strategy selector.
    Analyzes document characteristics and selects the optimal chunking approach.
    
    Strategy Selection Rules:
    - < 50K tokens: Fixed chunking (simple, fast)
    - 50K-500K tokens: Semantic chunking (context-aware)
    - > 500K tokens: Hierarchical chunking (scalable)
    """
    
    def __init__(self):
        """Initialize adaptive selector"""
        self.logger = logger
        self.token_counter = TokenCounter()
        
        # Thresholds from settings
        self.small_doc_threshold = settings.SMALL_DOC_THRESHOLD
        self.large_doc_threshold = settings.LARGE_DOC_THRESHOLD
        
        self.logger.info(
            f"Initialized AdaptiveSelector: "
            f"small<{self.small_doc_threshold}, large>{self.large_doc_threshold}"
        )
    
    def select_strategy(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> ChunkingStrategy:
        """
        Select optimal chunking strategy for given text.
        
        Args:
            text: Document text
            metadata: Document metadata
        
        Returns:
            Recommended chunking strategy
        """
        # Count tokens
        token_count = self.token_counter.count_tokens(text)
        
        # Analyze document structure
        has_clear_structure = self._has_clear_structure(text)
        is_code = self._is_code_document(text, metadata)
        is_conversational = self._is_conversational(text)
        
        # Decision tree
        if token_count < self.small_doc_threshold:
            # Small document: use fixed chunking
            strategy = ChunkingStrategy.FIXED
            reason = "Document is small (<50K tokens)"
        
        elif token_count > self.large_doc_threshold:
            # Large document: use hierarchical chunking
            strategy = ChunkingStrategy.HIERARCHICAL
            reason = "Document is large (>500K tokens)"
        
        else:
            # Medium document: choose based on structure
            if has_clear_structure or is_code:
                # Well-structured: use semantic chunking
                strategy = ChunkingStrategy.SEMANTIC
                reason = "Document has clear structure or is code"
            else:
                # Unstructured: use fixed chunking
                strategy = ChunkingStrategy.FIXED
                reason = "Document lacks clear structure"
        
        self.logger.info(
            f"Selected {strategy} strategy: {reason} "
            f"(tokens={token_count}, structure={has_clear_structure}, "
            f"code={is_code})"
        )
        
        return strategy
    
    def _has_clear_structure(self, text: str) -> bool:
        """
        Check if document has clear structural markers.
        
        Args:
            text: Document text
        
        Returns:
            True if well-structured
        """
        import re
        
        # Count structural markers
        heading_count = len(re.findall(r'\n[A-Z][A-Za-z\s]{3,50}\n', text))
        bullet_count = len(re.findall(r'\n[\s]*[•\-\*]\s', text))
        numbered_list_count = len(re.findall(r'\n[\s]*\d+[\.\)]\s', text))
        section_markers = len(re.findall(r'\[HEADING|SECTION|CHAPTER\]', text))
        
        # Check for markdown-style headers
        markdown_headers = len(re.findall(r'\n#{1,6}\s', text))
        
        total_markers = (
            heading_count + 
            bullet_count + 
            numbered_list_count + 
            section_markers + 
            markdown_headers
        )
        
        # If more than 5 structural markers per 1000 tokens, consider structured
        tokens = self.token_counter.count_tokens(text)
        markers_per_1k = (total_markers / tokens * 1000) if tokens > 0 else 0
        
        is_structured = markers_per_1k > 5
        
        self.logger.debug(
            f"Structure analysis: {total_markers} markers, "
            f"{markers_per_1k:.1f} per 1K tokens, structured={is_structured}"
        )
        
        return is_structured
    
    def _is_code_document(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> bool:
        """
        Check if document is primarily code.
        
        Args:
            text: Document text
            metadata: Document metadata
        
        Returns:
            True if appears to be code
        """
        # Check file extension
        if metadata and metadata.filename:
            code_extensions = [
                '.py', '.js', '.java', '.cpp', '.c', '.go', '.rs',
                '.ts', '.jsx', '.tsx', '.rb', '.php', '.swift'
            ]
            if any(metadata.filename.endswith(ext) for ext in code_extensions):
                return True
        
        # Check content patterns
        import re
        
        # Look for code-like patterns
        function_defs = len(re.findall(r'def\s+\w+|function\s+\w+|class\s+\w+', text))
        imports = len(re.findall(r'import\s+\w+|from\s+\w+|#include|require\s*\(', text))
        braces = text.count('{') + text.count('}')
        brackets = text.count('[') + text.count(']')
        semicolons = text.count(';')
        
        # Calculate code score
        tokens = self.token_counter.count_tokens(text)
        code_indicators = function_defs + imports + (braces + brackets + semicolons) / 10
        code_score = (code_indicators / tokens * 1000) if tokens > 0 else 0
        
        is_code = code_score > 10
        
        self.logger.debug(f"Code analysis: score={code_score:.1f}, is_code={is_code}")
        
        return is_code
    
    def _is_conversational(self, text: str) -> bool:
        """
        Check if text is conversational (chat, dialogue).
        
        Args:
            text: Document text
        
        Returns:
            True if conversational
        """
        import re
        
        # Look for conversational patterns
        questions = len(re.findall(r'\?', text))
        short_sentences = len(re.findall(r'[.!?]\s+[A-Z]', text))
        personal_pronouns = len(re.findall(
            r'\b(I|you|we|they|he|she|it)\b',
            text,
            re.IGNORECASE
        ))
        
        tokens = self.token_counter.count_tokens(text)
        conversational_score = (
            (questions + personal_pronouns) / tokens * 1000
        ) if tokens > 0 else 0
        
        is_conversational = conversational_score > 20
        
        return is_conversational
    
    def get_chunker(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None,
        config: Optional[ChunkerConfig] = None
    ) -> BaseChunker:
        """
        Get appropriate chunker instance for text.
        
        Args:
            text: Document text
            metadata: Document metadata
            config: Optional chunker configuration
        
        Returns:
            Chunker instance
        """
        strategy = self.select_strategy(text, metadata)
        
        # Use provided config or create default
        if config is None:
            config = self._get_default_config(strategy)
        
        # Create chunker based on strategy
        if strategy == ChunkingStrategy.FIXED:
            return FixedChunker.from_config(config)
        
        elif strategy == ChunkingStrategy.SEMANTIC:
            from chunking.semantic_chunker import SemanticChunker
            try:
                return SemanticChunker.from_config(config)
            except Exception as e:
                self.logger.warning(
                    f"Failed to create semantic chunker: {e}, falling back to fixed"
                )
                return FixedChunker.from_config(config)
        
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            from chunking.hierarchical_chunker import HierarchicalChunker
            return HierarchicalChunker.from_config(config)
        
        else:
            # Default to fixed
            return FixedChunker.from_config(config)
    
    def _get_default_config(self, strategy: ChunkingStrategy) -> ChunkerConfig:
        """
        Get default configuration for strategy.
        
        Args:
            strategy: Chunking strategy
        
        Returns:
            ChunkerConfig
        """
        if strategy == ChunkingStrategy.FIXED:
            return ChunkerConfig(
                chunk_size=settings.FIXED_CHUNK_SIZE,
                overlap=settings.FIXED_CHUNK_OVERLAP,
                respect_boundaries=True
            )
        
        elif strategy == ChunkingStrategy.SEMANTIC:
            return ChunkerConfig(
                chunk_size=settings.FIXED_CHUNK_SIZE,
                overlap=settings.FIXED_CHUNK_OVERLAP,
                respect_boundaries=True,
                semantic_threshold=settings.SEMANTIC_BREAKPOINT_THRESHOLD
            )
        
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return ChunkerConfig(
                chunk_size=settings.PARENT_CHUNK_SIZE,
                overlap=settings.FIXED_CHUNK_OVERLAP,
                respect_boundaries=True,
                parent_size=settings.PARENT_CHUNK_SIZE,
                child_size=settings.CHILD_CHUNK_SIZE
            )
        
        else:
            return ChunkerConfig()
    
    def chunk_document(
        self,
        text: str,
        metadata: DocumentMetadata,
        config: Optional[ChunkerConfig] = None
    ) -> List[DocumentChunk]:
        """
        Adaptively chunk document.
        
        Args:
            text: Document text
            metadata: Document metadata
            config: Optional configuration
        
        Returns:
            List of chunks
        """
        # Get appropriate chunker
        chunker = self.get_chunker(text, metadata, config)
        
        # Perform chunking
        chunks = chunker.chunk_document(text, metadata)
        
        # Log results
        self.logger.info(
            f"Adaptive chunking complete: "
            f"strategy={chunker.strategy_name}, "
            f"chunks={len(chunks)}, "
            f"avg_tokens={sum(c.token_count for c in chunks) / len(chunks) if chunks else 0:.1f}"
        )
        
        return chunks
    
    def analyze_document(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> dict:
        """
        Analyze document and provide detailed recommendations.
        
        Args:
            text: Document text
            metadata: Document metadata
        
        Returns:
            Analysis dictionary
        """
        token_count = self.token_counter.count_tokens(text)
        has_structure = self._has_clear_structure(text)
        is_code = self._is_code_document(text, metadata)
        is_conversational = self._is_conversational(text)
        
        recommended_strategy = self.select_strategy(text, metadata)
        
        # Estimate chunk count for each strategy
        fixed_chunks = token_count // settings.FIXED_CHUNK_SIZE
        semantic_chunks = token_count // (settings.FIXED_CHUNK_SIZE * 1.2)  # Semantic typically 20% fewer
        hierarchical_chunks = token_count // settings.PARENT_CHUNK_SIZE
        
        analysis = {
            "token_count": token_count,
            "character_count": len(text),
            "word_count": len(text.split()),
            "has_clear_structure": has_structure,
            "is_code": is_code,
            "is_conversational": is_conversational,
            "recommended_strategy": recommended_strategy.value,
            "estimated_chunks": {
                "fixed": fixed_chunks,
                "semantic": semantic_chunks,
                "hierarchical": hierarchical_chunks,
            },
            "thresholds": {
                "small_doc": self.small_doc_threshold,
                "large_doc": self.large_doc_threshold,
            }
        }
        
        return analysis


# Global selector instance
_selector = None


def get_adaptive_selector() -> AdaptiveSelector:
    """
    Get global adaptive selector instance.
    
    Returns:
        AdaptiveSelector
    """
    global _selector
    if _selector is None:
        _selector = AdaptiveSelector()
    return _selector


def adaptive_chunk(
    text: str,
    metadata: DocumentMetadata,
    config: Optional[ChunkerConfig] = None
) -> List[DocumentChunk]:
    """
    Convenience function for adaptive chunking.
    
    Args:
        text: Document text
        metadata: Document metadata
        config: Optional configuration
    
    Returns:
        List of chunks
    """
    selector = get_adaptive_selector()
    return selector.chunk_document(text, metadata, config)


if __name__ == "__main__":
    # Test adaptive selector
    print("=== Adaptive Selector Tests ===\n")
    
    from config.models import DocumentMetadata, DocumentType
    
    selector = AdaptiveSelector()
    
    # Test 1: Small structured document
    print("Test 1: Small structured document")
    small_text = """
    # Introduction
    This is a test document with clear structure.
    
    # Section 1
    This section discusses various topics.
    - Point 1
    - Point 2
    - Point 3
    
    # Section 2
    Another section with more content.
    1. First item
    2. Second item
    3. Third item
    
    # Conclusion
    Final thoughts on the matter.
    """ * 10  # Repeat to get ~5K tokens
    
    analysis = selector.analyze_document(small_text)
    print(f"  Tokens: {analysis['token_count']}")
    print(f"  Has structure: {analysis['has_clear_structure']}")
    print(f"  Recommended: {analysis['recommended_strategy']}")
    print(f"  Estimated chunks: {analysis['estimated_chunks']}")
    print()
    
    # Test 2: Medium code document
    print("Test 2: Medium code document")
    code_text = """
    def process_data(data):
        result = []
        for item in data:
            if item.is_valid():
                result.append(item.transform())
        return result
    
    class DataProcessor:
        def __init__(self):
            self.cache = {}
        
        def process(self, data):
            return process_data(data)
    """ * 500  # Repeat to get ~100K tokens
    
    metadata = DocumentMetadata(
        document_id="code_doc",
        filename="processor.py",
        document_type=DocumentType.TXT,
        file_size_bytes=10000
    )
    
    analysis = selector.analyze_document(code_text, metadata)
    print(f"  Tokens: {analysis['token_count']}")
    print(f"  Is code: {analysis['is_code']}")
    print(f"  Recommended: {analysis['recommended_strategy']}")
    print()
    
    # Test 3: Actual chunking
    print("Test 3: Adaptive chunking")
    test_metadata = DocumentMetadata(
        document_id="test_doc",
        filename="test.txt",
        document_type=DocumentType.TXT,
        file_size_bytes=5000
    )
    
    chunks = selector.chunk_document(small_text, test_metadata)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Strategy used: {test_metadata.chunking_strategy}")
    print(f"  Avg tokens per chunk: {sum(c.token_count for c in chunks) / len(chunks):.1f}")
    print()
    
    # Test 4: Convenience function
    print("Test 4: Convenience function")
    chunks = adaptive_chunk(small_text, test_metadata)
    print(f"  Convenience function created {len(chunks)} chunks")
    print()
    
    print("✓ Adaptive selector module created successfully!")