"""
Overlap Manager
Manages chunk overlaps to maintain context continuity
"""

from typing import List, Optional, Tuple
from config.logging_config import get_logger
from config.models import DocumentChunk
from chunking.token_counter import TokenCounter

logger = get_logger(__name__)


class OverlapManager:
    """
    Manages overlapping regions between chunks.
    Ensures smooth context transitions and optimal retrieval.
    """
    
    def __init__(self, overlap_tokens: int = 50):
        """
        Initialize overlap manager.
        
        Args:
            overlap_tokens: Target overlap in tokens
        """
        self.overlap_tokens = overlap_tokens
        self.token_counter = TokenCounter()
        self.logger = logger
    
    def add_overlap(
        self,
        chunks: List[DocumentChunk],
        overlap_tokens: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Add overlap to existing chunks.
        
        Args:
            chunks: List of chunks without overlap
            overlap_tokens: Override default overlap
        
        Returns:
            List of chunks with overlap
        """
        if not chunks or len(chunks) < 2:
            return chunks
        
        overlap = overlap_tokens or self.overlap_tokens
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no prefix, add suffix from next
                new_text = chunk.text
                if i + 1 < len(chunks):
                    suffix = self._get_overlap_text(
                        chunks[i + 1].text,
                        overlap,
                        from_start=True
                    )
                    new_text = new_text + " " + suffix
            
            elif i == len(chunks) - 1:
                # Last chunk: add prefix from previous, no suffix
                prefix = self._get_overlap_text(
                    chunks[i - 1].text,
                    overlap,
                    from_start=False
                )
                new_text = prefix + " " + chunk.text
            
            else:
                # Middle chunk: add both prefix and suffix
                prefix = self._get_overlap_text(
                    chunks[i - 1].text,
                    overlap,
                    from_start=False
                )
                suffix = self._get_overlap_text(
                    chunks[i + 1].text,
                    overlap,
                    from_start=True
                )
                new_text = prefix + " " + chunk.text + " " + suffix
            
            # Create new chunk with overlapped text
            overlapped_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=new_text,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                token_count=self.token_counter.count_tokens(new_text),
                metadata=chunk.metadata
            )
            overlapped_chunks.append(overlapped_chunk)
        
        self.logger.debug(f"Added overlap to {len(chunks)} chunks")
        return overlapped_chunks
    
    def _get_overlap_text(
        self,
        text: str,
        overlap_tokens: int,
        from_start: bool
    ) -> str:
        """
        Extract overlap text from beginning or end.
        
        Args:
            text: Source text
            overlap_tokens: Number of tokens to extract
            from_start: True for start, False for end
        
        Returns:
            Overlap text
        """
        total_tokens = self.token_counter.count_tokens(text)
        
        if total_tokens <= overlap_tokens:
            return text
        
        if from_start:
            # Get first N tokens
            return self.token_counter.truncate_to_tokens(
                text,
                max_tokens=overlap_tokens,
                suffix=""
            )
        else:
            # Get last N tokens (approximate by taking from end)
            words = text.split()
            # Estimate: take proportional number of words
            ratio = overlap_tokens / total_tokens
            num_words = max(1, int(len(words) * ratio))
            overlap_text = " ".join(words[-num_words:])
            
            # Verify token count and adjust if needed
            while self.token_counter.count_tokens(overlap_text) > overlap_tokens and len(words) > 1:
                words = words[1:]
                overlap_text = " ".join(words[-num_words:])
            
            return overlap_text
    
    def remove_overlap(
        self,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        Remove overlap from chunks (get core content only).
        
        Args:
            chunks: List of chunks with overlap
        
        Returns:
            List of chunks without overlap
        """
        if not chunks or len(chunks) < 2:
            return chunks
        
        core_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: remove suffix
                core_text = self._remove_suffix_overlap(
                    chunk.text,
                    chunks[i + 1].text if i + 1 < len(chunks) else ""
                )
            elif i == len(chunks) - 1:
                # Last chunk: remove prefix
                core_text = self._remove_prefix_overlap(
                    chunk.text,
                    chunks[i - 1].text
                )
            else:
                # Middle chunk: remove both
                temp_text = self._remove_prefix_overlap(
                    chunk.text,
                    chunks[i - 1].text
                )
                core_text = self._remove_suffix_overlap(
                    temp_text,
                    chunks[i + 1].text
                )
            
            core_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=core_text,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                token_count=self.token_counter.count_tokens(core_text),
                metadata=chunk.metadata
            )
            core_chunks.append(core_chunk)
        
        return core_chunks
    
    def _remove_prefix_overlap(self, text: str, previous_text: str) -> str:
        """Remove overlap with previous chunk"""
        # Find common prefix
        words = text.split()
        prev_words = previous_text.split()
        
        # Simple approach: find where texts diverge
        common_length = 0
        for i, word in enumerate(words):
            if i < len(prev_words) and word == prev_words[-(len(words) - i)]:
                common_length += 1
            else:
                break
        
        if common_length > 0:
            return " ".join(words[common_length:])
        return text
    
    def _remove_suffix_overlap(self, text: str, next_text: str) -> str:
        """Remove overlap with next chunk"""
        # Find common suffix
        words = text.split()
        next_words = next_text.split()
        
        common_length = 0
        for i in range(1, min(len(words), len(next_words)) + 1):
            if words[-i] == next_words[i - 1]:
                common_length += 1
            else:
                break
        
        if common_length > 0:
            return " ".join(words[:-common_length])
        return text
    
    def calculate_overlap_percentage(
        self,
        chunks: List[DocumentChunk]
    ) -> float:
        """
        Calculate average overlap percentage.
        
        Args:
            chunks: List of chunks
        
        Returns:
            Average overlap percentage
        """
        if len(chunks) < 2:
            return 0.0
        
        overlaps = []
        for i in range(len(chunks) - 1):
            overlap = self._measure_overlap(chunks[i].text, chunks[i + 1].text)
            overlaps.append(overlap)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0
    
    def _measure_overlap(self, text1: str, text2: str) -> float:
        """
        Measure overlap between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Overlap percentage (0-100)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        common = words1 & words2
        overlap_pct = (len(common) / min(len(words1), len(words2))) * 100
        
        return overlap_pct
    
    def optimize_overlaps(
        self,
        chunks: List[DocumentChunk],
        target_overlap: int,
        tolerance: int = 10
    ) -> List[DocumentChunk]:
        """
        Optimize overlap sizes to target.
        
        Args:
            chunks: List of chunks
            target_overlap: Target overlap in tokens
            tolerance: Acceptable deviation in tokens
        
        Returns:
            Optimized chunks
        """
        if len(chunks) < 2:
            return chunks
        
        optimized = []
        
        for i in range(len(chunks)):
            chunk = chunks[i]
            
            # Check current overlap with next chunk
            if i < len(chunks) - 1:
                current_overlap = self._count_overlap_tokens(
                    chunk.text,
                    chunks[i + 1].text
                )
                
                # Adjust if outside tolerance
                if abs(current_overlap - target_overlap) > tolerance:
                    # Add or remove text to reach target
                    if current_overlap < target_overlap:
                        # Need more overlap
                        additional = self._get_overlap_text(
                            chunks[i + 1].text,
                            target_overlap - current_overlap,
                            from_start=True
                        )
                        new_text = chunk.text + " " + additional
                    else:
                        # Need less overlap
                        new_text = self.token_counter.truncate_to_tokens(
                            chunk.text,
                            self.token_counter.count_tokens(chunk.text) - (current_overlap - target_overlap)
                        )
                    
                    chunk = DocumentChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=new_text,
                        chunk_index=chunk.chunk_index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        page_number=chunk.page_number,
                        section_title=chunk.section_title,
                        token_count=self.token_counter.count_tokens(new_text),
                        metadata=chunk.metadata
                    )
            
            optimized.append(chunk)
        
        return optimized
    
    def _count_overlap_tokens(self, text1: str, text2: str) -> int:
        """Count overlapping tokens between two texts"""
        # Find longest common substring at the boundary
        words1 = text1.split()
        words2 = text2.split()
        
        max_overlap = 0
        for i in range(1, min(len(words1), len(words2)) + 1):
            if words1[-i:] == words2[:i]:
                overlap_text = " ".join(words1[-i:])
                max_overlap = self.token_counter.count_tokens(overlap_text)
        
        return max_overlap
    
    def get_overlap_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """
        Get statistics about overlaps.
        
        Args:
            chunks: List of chunks
        
        Returns:
            Statistics dictionary
        """
        if len(chunks) < 2:
            return {
                "num_chunks": len(chunks),
                "num_overlaps": 0,
                "avg_overlap_tokens": 0,
                "avg_overlap_percentage": 0,
            }
        
        overlap_tokens = []
        overlap_percentages = []
        
        for i in range(len(chunks) - 1):
            tokens = self._count_overlap_tokens(chunks[i].text, chunks[i + 1].text)
            pct = self._measure_overlap(chunks[i].text, chunks[i + 1].text)
            
            overlap_tokens.append(tokens)
            overlap_percentages.append(pct)
        
        return {
            "num_chunks": len(chunks),
            "num_overlaps": len(overlap_tokens),
            "avg_overlap_tokens": sum(overlap_tokens) / len(overlap_tokens) if overlap_tokens else 0,
            "min_overlap_tokens": min(overlap_tokens) if overlap_tokens else 0,
            "max_overlap_tokens": max(overlap_tokens) if overlap_tokens else 0,
            "avg_overlap_percentage": sum(overlap_percentages) / len(overlap_percentages) if overlap_percentages else 0,
        }


if __name__ == "__main__":
    # Test overlap manager
    print("=== Overlap Manager Tests ===\n")
    
    from config.models import DocumentMetadata, DocumentType
    
    # Create test chunks
    metadata = DocumentMetadata(
        document_id="test_doc",
        filename="test.txt",
        document_type=DocumentType.TXT,
        file_size_bytes=1000
    )
    
    test_chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="test_doc",
            text="This is the first chunk of text. It contains some information.",
            chunk_index=0,
            start_char=0,
            end_char=100,
            token_count=15
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_id="test_doc",
            text="This is the second chunk. It has different content from the first.",
            chunk_index=1,
            start_char=100,
            end_char=200,
            token_count=14
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            document_id="test_doc",
            text="And this is the third and final chunk with its own unique content.",
            chunk_index=2,
            start_char=200,
            end_char=300,
            token_count=13
        ),
    ]
    
    # Test 1: Add overlap
    print("Test 1: Add overlap")
    manager = OverlapManager(overlap_tokens=5)
    overlapped = manager.add_overlap(test_chunks)
    print(f"  Original chunks: {len(test_chunks)}")
    print(f"  Chunks with overlap: {len(overlapped)}")
    for i, chunk in enumerate(overlapped):
        print(f"  Chunk {i}: {chunk.token_count} tokens")
        print(f"    {chunk.text[:80]}...")
    print()
    
    # Test 2: Overlap statistics
    print("Test 2: Overlap statistics")
    stats = manager.get_overlap_statistics(overlapped)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test 3: Remove overlap
    print("Test 3: Remove overlap")
    core_chunks = manager.remove_overlap(overlapped)
    print(f"  Chunks after removing overlap: {len(core_chunks)}")
    for i, chunk in enumerate(core_chunks):
        print(f"  Chunk {i}: {chunk.token_count} tokens")
    print()
    
    # Test 4: Overlap percentage
    print("Test 4: Calculate overlap percentage")
    pct = manager.calculate_overlap_percentage(overlapped)
    print(f"  Average overlap: {pct:.2f}%")
    print()
    
    print("✓ Overlap manager module created successfully!")