"""
Hierarchical Chunker
Creates parent-child chunk relationships for large documents
"""

from typing import List, Optional, Tuple
from config.logging_config import get_logger
from config.models import DocumentChunk, DocumentMetadata, ChunkingStrategy
from config.settings import get_settings
from chunking.base_chunker import BaseChunker, ChunkerConfig
from chunking.token_counter import TokenCounter

logger = get_logger(__name__)
settings = get_settings()


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking strategy.
    Creates two-level structure: large parent chunks with smaller child chunks.
    
    Parent chunks: Provide broad context for semantic search
    Child chunks: Enable fine-grained retrieval
    
    Best for:
    - Large documents (>500K tokens)
    - When scalability is critical
    - Multi-level retrieval strategies
    """
    
    def __init__(
        self,
        parent_size: int = None,
        child_size: int = None,
        overlap: int = None,
        min_chunk_size: int = 100
    ):
        """
        Initialize hierarchical chunker.
        
        Args:
            parent_size: Tokens per parent chunk
            child_size: Tokens per child chunk
            overlap: Overlap tokens between children
            min_chunk_size: Minimum chunk size in tokens
        """
        super().__init__(ChunkingStrategy.HIERARCHICAL)
        
        self.parent_size = parent_size or settings.PARENT_CHUNK_SIZE
        self.child_size = child_size or settings.CHILD_CHUNK_SIZE
        self.overlap = overlap or settings.FIXED_CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size
        
        # Validate hierarchy
        if self.child_size >= self.parent_size:
            raise ValueError(
                f"Child size ({self.child_size}) must be less than parent size ({self.parent_size})"
            )
        
        # Initialize token counter
        self.token_counter = TokenCounter()
        
        self.logger.info(
            f"Initialized HierarchicalChunker: "
            f"parent_size={self.parent_size}, child_size={self.child_size}, "
            f"overlap={self.overlap}"
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> List[DocumentChunk]:
        """
        Chunk text into hierarchical structure.
        Returns flattened list of both parent and child chunks.
        
        Args:
            text: Input text
            metadata: Document metadata
        
        Returns:
            List of DocumentChunk objects (parents + children)
        """
        if not text or not text.strip():
            return []
        
        document_id = metadata.document_id if metadata else "unknown"
        
        # Split into sentences first
        sentences = self._split_sentences(text)
        
        # Create parent chunks
        parent_chunks = self._create_parent_chunks(sentences, document_id)
        
        # Create child chunks for each parent
        all_chunks = []
        
        for parent in parent_chunks:
            # Add parent chunk
            all_chunks.append(parent)
            
            # Create and add child chunks
            child_chunks = self._create_child_chunks(parent, document_id)
            all_chunks.extend(child_chunks)
            
            # Update parent with child references
            parent.child_chunk_ids = [c.chunk_id for c in child_chunks]
        
        self.logger.debug(
            f"Created {len(parent_chunks)} parent chunks with "
            f"{len(all_chunks) - len(parent_chunks)} child chunks"
        )
        
        return all_chunks
    
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
    
    def _create_parent_chunks(
        self,
        sentences: List[str],
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Create parent chunks from sentences.
        
        Args:
            sentences: List of sentences
            document_id: Document ID
        
        Returns:
            List of parent chunks
        """
        parent_chunks = []
        current_sentences = []
        current_tokens = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            # Check if adding this sentence exceeds parent size
            if current_tokens + sentence_tokens > self.parent_size and current_sentences:
                # Save current parent chunk
                chunk_text = " ".join(current_sentences)
                parent_chunk = self._create_chunk(
                    text=self._clean_chunk_text(chunk_text),
                    chunk_index=len(parent_chunks),
                    document_id=document_id,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        "is_parent": True,
                        "sentences": len(current_sentences)
                    }
                )
                parent_chunks.append(parent_chunk)
                
                # Start new parent chunk
                current_sentences = [sentence]
                current_tokens = sentence_tokens
                start_char += len(chunk_text)
            else:
                # Add sentence to current parent
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final parent chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            parent_chunk = self._create_chunk(
                text=self._clean_chunk_text(chunk_text),
                chunk_index=len(parent_chunks),
                document_id=document_id,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata={
                    "is_parent": True,
                    "sentences": len(current_sentences)
                }
            )
            parent_chunks.append(parent_chunk)
        
        return parent_chunks
    
    def _create_child_chunks(
        self,
        parent: DocumentChunk,
        document_id: str
    ) -> List[DocumentChunk]:
        """
        Create child chunks from a parent chunk.
        
        Args:
            parent: Parent chunk
            document_id: Document ID
        
        Returns:
            List of child chunks
        """
        # Split parent text into sentences
        parent_sentences = self._split_sentences(parent.text)
        
        if not parent_sentences:
            return []
        
        child_chunks = []
        current_sentences = []
        current_tokens = 0
        start_char = parent.start_char
        
        for sentence in parent_sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            # Check if adding this sentence exceeds child size
            if current_tokens + sentence_tokens > self.child_size and current_sentences:
                # Save current child chunk
                chunk_text = " ".join(current_sentences)
                child_chunk = self._create_chunk(
                    text=self._clean_chunk_text(chunk_text),
                    chunk_index=len(child_chunks),
                    document_id=document_id,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        "is_child": True,
                        "parent_id": parent.chunk_id
                    }
                )
                child_chunk.parent_chunk_id = parent.chunk_id
                child_chunks.append(child_chunk)
                
                # Handle overlap for next child
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
                
                start_char += len(chunk_text)
            else:
                # Add sentence to current child
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final child chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            child_chunk = self._create_chunk(
                text=self._clean_chunk_text(chunk_text),
                chunk_index=len(child_chunks),
                document_id=document_id,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                metadata={
                    "is_child": True,
                    "parent_id": parent.chunk_id
                }
            )
            child_chunk.parent_chunk_id = parent.chunk_id
            child_chunks.append(child_chunk)
        
        return child_chunks
    
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
    
    def get_parent_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Extract only parent chunks from mixed list.
        
        Args:
            chunks: Mixed list of parent and child chunks
        
        Returns:
            List of parent chunks only
        """
        return [
            c for c in chunks 
            if c.metadata.get("is_parent", False)
        ]
    
    def get_child_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Extract only child chunks from mixed list.
        
        Args:
            chunks: Mixed list of parent and child chunks
        
        Returns:
            List of child chunks only
        """
        return [
            c for c in chunks 
            if c.metadata.get("is_child", False)
        ]
    
    def get_children_for_parent(
        self,
        parent_id: str,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Get all child chunks for a specific parent.
        
        Args:
            parent_id: Parent chunk ID
            chunks: List of all chunks
        
        Returns:
            List of child chunks belonging to parent
        """
        return [
            c for c in chunks
            if c.parent_chunk_id == parent_id
        ]
    
    def expand_to_parent(
        self,
        child_chunk: DocumentChunk,
        all_chunks: List[DocumentChunk]
    ) -> Optional[DocumentChunk]:
        """
        Given a child chunk, return its parent chunk.
        Useful for retrieving broader context.
        
        Args:
            child_chunk: Child chunk
            all_chunks: List of all chunks
        
        Returns:
            Parent chunk or None
        """
        if not child_chunk.parent_chunk_id:
            return None
        
        for chunk in all_chunks:
            if chunk.chunk_id == child_chunk.parent_chunk_id:
                return chunk
        
        return None
    
    def get_hierarchy_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """
        Calculate statistics about the hierarchy.
        
        Args:
            chunks: List of all chunks
        
        Returns:
            Statistics dictionary
        """
        parents = self.get_parent_chunks(chunks)
        children = self.get_child_chunks(chunks)
        
        if not parents:
            return {
                "num_parents": 0,
                "num_children": 0,
                "avg_children_per_parent": 0,
            }
        
        # Calculate children per parent
        children_per_parent = []
        for parent in parents:
            num_children = len([
                c for c in children 
                if c.parent_chunk_id == parent.chunk_id
            ])
            children_per_parent.append(num_children)
        
        # Token statistics
        parent_tokens = [c.token_count for c in parents]
        child_tokens = [c.token_count for c in children]
        
        return {
            "num_parents": len(parents),
            "num_children": len(children),
            "avg_children_per_parent": sum(children_per_parent) / len(children_per_parent) if children_per_parent else 0,
            "min_children_per_parent": min(children_per_parent) if children_per_parent else 0,
            "max_children_per_parent": max(children_per_parent) if children_per_parent else 0,
            "avg_parent_tokens": sum(parent_tokens) / len(parent_tokens) if parent_tokens else 0,
            "avg_child_tokens": sum(child_tokens) / len(child_tokens) if child_tokens else 0,
        }
    
    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'HierarchicalChunker':
        """
        Create HierarchicalChunker from configuration.
        
        Args:
            config: ChunkerConfig object
        
        Returns:
            HierarchicalChunker instance
        """
        return cls(
            parent_size=config.extra.get('parent_size', settings.PARENT_CHUNK_SIZE),
            child_size=config.extra.get('child_size', settings.CHILD_CHUNK_SIZE),
            overlap=config.overlap,
            min_chunk_size=config.min_chunk_size
        )


if __name__ == "__main__":
    # Test hierarchical chunker
    print("=== Hierarchical Chunker Tests ===\n")
    
    from config.models import DocumentMetadata, DocumentType
    
    # Create test metadata
    metadata = DocumentMetadata(
        document_id="test_doc_hierarchical",
        filename="test.txt",
        document_type=DocumentType.TXT,
        file_size_bytes=50000
    )
    
    # Test text (large enough to create multiple parents)
    test_text = """
    Artificial intelligence is revolutionizing technology. Machine learning enables computers to 
    learn from data. Deep learning uses neural networks to process complex patterns. Natural 
    language processing allows machines to understand human language.
    
    Computer vision enables machines to interpret visual information. Robotics combines AI with 
    physical systems. Autonomous vehicles use AI for navigation and decision-making. Speech 
    recognition converts spoken words into text.
    
    The history of AI dates back to the 1950s. Early researchers dreamed of creating thinking 
    machines. Expert systems were among the first practical AI applications. The AI winter of 
    the 1970s slowed progress temporarily.
    
    Modern AI has achieved remarkable breakthroughs. AlphaGo defeated world champions at Go. 
    Large language models can generate human-like text. Image generation models create realistic 
    pictures from descriptions.
    
    Ethical concerns about AI are growing. Bias in algorithms can perpetuate discrimination. 
    Privacy issues arise from data collection. Job displacement worries workers in many industries.
    
    The future of AI holds great promise. Artificial general intelligence remains a long-term goal.
    AI could help solve major global challenges. Collaboration between humans and AI systems 
    will shape the future.
    """ * 10  # Repeat to get larger document
    
    # Test 1: Basic hierarchical chunking
    print("Test 1: Hierarchical chunking")
    chunker = HierarchicalChunker(
        parent_size=500,
        child_size=150,
        overlap=30
    )
    
    chunks = chunker.chunk_document(test_text, metadata)
    
    parents = chunker.get_parent_chunks(chunks)
    children = chunker.get_child_chunks(chunks)
    
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Parent chunks: {len(parents)}")
    print(f"  Child chunks: {len(children)}")
    print()
    
    # Test 2: Hierarchy structure
    print("Test 2: Hierarchy structure")
    for i, parent in enumerate(parents[:3]):  # Show first 3 parents
        parent_children = chunker.get_children_for_parent(parent.chunk_id, chunks)
        print(f"  Parent {i} ({parent.token_count} tokens):")
        print(f"    Has {len(parent_children)} children")
        print(f"    Preview: {parent.text[:100]}...")
        for j, child in enumerate(parent_children[:2]):  # Show first 2 children
            print(f"      Child {j} ({child.token_count} tokens): {child.text[:80]}...")
        print()
    
    # Test 3: Statistics
    print("Test 3: Hierarchy statistics")
    stats = chunker.get_hierarchy_statistics(chunks)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test 4: Expand child to parent
    print("Test 4: Expand child to parent")
    if children:
        test_child = children[0]
        parent = chunker.expand_to_parent(test_child, chunks)
        if parent:
            print(f"  Child belongs to parent: {parent.chunk_id}")
            print(f"  Parent has {len(parent.child_chunk_ids)} children")
        print()
    
    # Test 5: Different sizes
    print("Test 5: Different parent/child sizes")
    size_configs = [
        (1000, 200),
        (2000, 500),
        (500, 100)
    ]
    for parent_sz, child_sz in size_configs:
        test_chunker = HierarchicalChunker(parent_size=parent_sz, child_size=child_sz)
        test_chunks = test_chunker.chunk_text(test_text, metadata)
        test_parents = test_chunker.get_parent_chunks(test_chunks)
        test_children = test_chunker.get_child_chunks(test_chunks)
        print(f"  Parent={parent_sz}, Child={child_sz}: "
              f"{len(test_parents)} parents, {len(test_children)} children")
    print()
    
    print("✓ Hierarchical chunker module created successfully!")