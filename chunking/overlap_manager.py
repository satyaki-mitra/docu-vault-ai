# DEPENDENCIES
from typing import List
from typing import Tuple
from typing import Optional
from config.models import DocumentChunk
from config.logging_config import get_logger
from chunking.token_counter import TokenCounter


# Setup Logging
logger = get_logger(__name__)


class OverlapManager:
    """
    Manages overlapping regions between chunks : ensures smooth context transitions and optimal retrieval
    """
    def __init__(self, overlap_tokens: int = 50):
        """
        Initialize overlap manager
        
        Arguments:
        ----------
            overlap_tokens { int } : Target overlap in tokens
        """
        self.overlap_tokens = overlap_tokens
        self.token_counter  = TokenCounter()
        self.logger         = logger
    

    def add_overlap(self, chunks: List[DocumentChunk], overlap_tokens: Optional[int] = None) -> List[DocumentChunk]:
        """
        Add overlap to existing chunks
        
        Arguments:
        ----------
            chunks         { list } : List of chunks without overlap

            overlap_tokens { int }  : Override default overlap
        
        Returns:
        --------
                  { list }          : List of chunks with overlap
        """
        if (not chunks or (len(chunks) < 2)):
            return chunks
        
        overlap           = overlap_tokens or self.overlap_tokens
        overlapped_chunks = list()
        
        for i, chunk in enumerate(chunks):
            if (i == 0):
                # First chunk: no prefix, add suffix from next
                new_text = chunk.text
                if (i + 1 < len(chunks)):
                    suffix   = self._get_overlap_text(text           = chunks[i + 1].text,
                                                      overlap_tokens = overlap,
                                                      from_start     = True,
                                                     )

                    new_text = new_text + " " + suffix
            
            elif (i == len(chunks) - 1):
                # Last chunk: add prefix from previous, no suffix
                prefix   = self._get_overlap_text(text           = chunks[i - 1].text,
                                                  overlap_tokens = overlap,
                                                  from_start     = False,
                                                 )

                new_text = prefix + " " + chunk.text
            
            else:
                # Middle chunk: add both prefix and suffix
                prefix   = self._get_overlap_text(text           = chunks[i - 1].text,
                                                  overlap_tokens = overlap,
                                                  from_start     = False,
                                                 )

                suffix   = self._get_overlap_text(text           = chunks[i + 1].text,
                                                  overlap_tokens = overlap,
                                                  from_start     = True,
                                                 )

                new_text = prefix + " " + chunk.text + " " + suffix
            
            # Create new chunk with overlapped text
            overlapped_chunk = DocumentChunk(chunk_id      = chunk.chunk_id,
                                             document_id   = chunk.document_id,
                                             text          = new_text,
                                             chunk_index   = chunk.chunk_index,
                                             start_char    = chunk.start_char,
                                             end_char      = chunk.end_char,
                                             page_number   = chunk.page_number,
                                             section_title = chunk.section_title,
                                             token_count   = self.token_counter.count_tokens(new_text),
                                             metadata      = chunk.metadata,
                                            )

            overlapped_chunks.append(overlapped_chunk)
        
        self.logger.debug(f"Added overlap to {len(chunks)} chunks")
        return overlapped_chunks
    

    def _get_overlap_text(self, text: str, overlap_tokens: int, from_start: bool) -> str:
        """
        Extract overlap text from beginning or end
        
        Arguments:
        ----------
            text            { str } : Source text
            
            overlap_tokens  { int } : Number of tokens to extract
            
            from_start     { bool } : True for start, False for end
        
        Returns:
        --------
                  { str }           : Overlap text
        """
        total_tokens = self.token_counter.count_tokens(text)
    
        if (total_tokens <= overlap_tokens):
            return text
        
        if from_start:
            # Get first N tokens
            return self.token_counter.truncate_to_tokens(text       = text, 
                                                         max_tokens = overlap_tokens, 
                                                         suffix     = "",
                                                        )
        
        else:
            # Get last N tokens using token counter's boundary finding
            char_pos, overlap_text = self.token_counter.find_token_boundaries(text          = text, 
                                                                              target_tokens = overlap_tokens,
                                                                             )
            
            # Take from the end instead of beginning
            if (char_pos < len(text)):
                return text[-char_pos:] if (char_pos > 0) else text

            return overlap_text
    

    def remove_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Remove overlap from chunks (get core content only)
        
        Arguments:
        ----------
            chunks { list } : List of chunks with overlap
        
        Returns:
        --------
              { list }      : List of chunks without overlap
        """
        if (not chunks or (len(chunks) < 2)):
            return chunks
        
        core_chunks = list()
        
        for i, chunk in enumerate(chunks):
            if (i == 0):
                # First chunk: remove suffix
                core_text = self._remove_suffix_overlap(text      = chunk.text,
                                                        next_text = chunks[i + 1].text if i + 1 < len(chunks) else "",
                                                       )
            elif (i == len(chunks) - 1):
                # Last chunk: remove prefix
                core_text = self._remove_prefix_overlap(text = chunk.text,
                                                        previous_text = chunks[i - 1].text,
                                                       )
            else:
                # Middle chunk: remove both
                temp_text = self._remove_prefix_overlap(text = chunk.text,
                                                        previous_text = chunks[i - 1].text,
                                                       )

                core_text = self._remove_suffix_overlap(text      = temp_text,
                                                        next_text = chunks[i + 1].text,
                                                       )
            
            core_chunk = DocumentChunk(chunk_id      = chunk.chunk_id,
                                       document_id   = chunk.document_id,
                                       text          = core_text,
                                       chunk_index   = chunk.chunk_index,
                                       start_char    = chunk.start_char,
                                       end_char      = chunk.end_char,
                                       page_number   = chunk.page_number,
                                       section_title = chunk.section_title,
                                       token_count   = self.token_counter.count_tokens(core_text),
                                       metadata      = chunk.metadata,
                                      )

            core_chunks.append(core_chunk)
        
        return core_chunks

    
    def _remove_prefix_overlap(self, text: str, previous_text: str) -> str:
        """
        Remove overlap with previous chunk
        """
        if not text or not previous_text:
            return text
        
        words       = text.split()
        prev_words  = previous_text.split()
        
        # Find longest common suffix-prefix match
        max_overlap = 0

        for overlap_size in range(1, min(len(words), len(prev_words)) + 1):
            if (words[:overlap_size] == prev_words[-overlap_size:]):
                max_overlap = overlap_size
        
        if (max_overlap > 0):
            return " ".join(words[max_overlap:])
        
        return text
    

    def _remove_suffix_overlap(self, text: str, next_text: str) -> str:
        """
        Remove overlap with next chunk
        """
        # Find common suffix
        words         = text.split()
        next_words    = next_text.split()
        
        common_length = 0

        for i in range(1, min(len(words), len(next_words)) + 1):
            if (words[-i] == next_words[i - 1]):
                common_length += 1

            else:
                break
        
        if (common_length > 0):
            return " ".join(words[:-common_length])

        return text
    

    def calculate_overlap_percentage(self, chunks: List[DocumentChunk]) -> float:
        """
        Calculate average overlap percentage
        
        Arguments:
        ----------
            chunks { list } : List of chunks
        
        Returns:
        --------
              { float }     : Average overlap percentage
        """
        if (len(chunks) < 2):
            return 0.0
        
        overlaps = list()

        for i in range(len(chunks) - 1):
            overlap = self._measure_overlap(chunks[i].text, chunks[i + 1].text)
            
            overlaps.append(overlap)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0
    

    def _measure_overlap(self, text1: str, text2: str) -> float:
        """
        Measure overlap between two texts
        
        Arguments:
        ----------
            text1 { str } : First text

            text2 { str } : Second text
        
        Returns:
        --------
             { float }    : Overlap percentage (0-100)
        """
        words1      = set(text1.lower().split())
        words2      = set(text2.lower().split())
        
        if (not words1 or not words2):
            return 0.0
        
        common      = words1 & words2
        overlap_pct = (len(common) / min(len(words1), len(words2))) * 100
        
        return overlap_pct
    

    def optimize_overlaps(self, chunks: List[DocumentChunk], target_overlap: int, tolerance: int = 10) -> List[DocumentChunk]:
        """
        Optimize overlap sizes to target
        
        Arguments:
        ----------
            chunks        { list } : List of chunks
            
            target_overlap { int } : Target overlap in tokens
            
            tolerance      { int } : Acceptable deviation in tokens
        
        Returns:
        --------
                  { list }         : Optimized chunks
        """
        if (len(chunks) < 2):
            return chunks

        # Validate target_overlap is reasonable
        if (target_overlap <= 0):
            self.logger.warning("Target overlap must be positive, using default")
            target_overlap = self.overlap_tokens
        
        optimized = list()
        
        for i in range(len(chunks)):
            chunk = chunks[i]
            
            # Check current overlap with next chunk
            if (i < len(chunks) - 1):
                current_overlap = self._count_overlap_tokens(text1 = chunk.text,
                                                             text2 = chunks[i + 1].text,
                                                            )
                
                # Adjust if outside tolerance
                if (abs(current_overlap - target_overlap) > tolerance):
                    # Add or remove text to reach target
                    if (current_overlap < target_overlap):
                        # Need more overlap
                        additional = self._get_overlap_text(text           = chunks[i + 1].text,
                                                            overlap_tokens = target_overlap - current_overlap,
                                                            from_start     = True,
                                                           )

                        new_text   = chunk.text + " " + additional

                    else:
                        # Need less overlap
                        new_text = self.token_counter.truncate_to_tokens(text       = chunk.text,
                                                                         max_tokens = self.token_counter.count_tokens(chunk.text) - (current_overlap - target_overlap),
                                                                        )
                    
                    chunk = DocumentChunk(chunk_id      = chunk.chunk_id,
                                          document_id   = chunk.document_id,
                                          text          = new_text,
                                          chunk_index   = chunk.chunk_index,
                                          start_char    = chunk.start_char,
                                          end_char      = chunk.end_char,
                                          page_number   = chunk.page_number,
                                          section_title = chunk.section_title,
                                          token_count   = self.token_counter.count_tokens(new_text),
                                          metadata      = chunk.metadata,
                                         )
            
            optimized.append(chunk)
        
        return optimized

    
    def _count_overlap_tokens(self, text1: str, text2: str) -> int:
        """
        Count overlapping tokens between two texts
        """
        # Find longest common substring at the boundary
        words1      = text1.split()
        words2      = text2.split()
        
        max_overlap = 0

        for i in range(1, min(len(words1), len(words2)) + 1):
            if (words1[-i:] == words2[:i]):
                overlap_text = " ".join(words1[-i:])
                max_overlap  = self.token_counter.count_tokens(overlap_text)
        
        return max_overlap
    

    def get_overlap_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """
        Get statistics about overlaps
        
        Arguments:
        ----------
            chunks { list } : List of chunks
        
        Returns:
        --------
              { dict }      : Statistics dictionary
        """
        if (len(chunks) < 2):
            return {"num_chunks"             : len(chunks),
                    "num_overlaps"           : 0,
                    "avg_overlap_tokens"     : 0,
                    "avg_overlap_percentage" : 0,
                   }
        
        overlap_tokens      = list()
        overlap_percentages = list()
        
        for i in range(len(chunks) - 1):
            tokens = self._count_overlap_tokens(chunks[i].text, chunks[i + 1].text)
            pct    = self._measure_overlap(chunks[i].text, chunks[i + 1].text)
            
            overlap_tokens.append(tokens)
            overlap_percentages.append(pct)
        
        return {"num_chunks"             : len(chunks),
                "num_overlaps"           : len(overlap_tokens),
                "avg_overlap_tokens"     : sum(overlap_tokens) / len(overlap_tokens) if overlap_tokens else 0,
                "min_overlap_tokens"     : min(overlap_tokens) if overlap_tokens else 0,
                "max_overlap_tokens"     : max(overlap_tokens) if overlap_tokens else 0,
                "avg_overlap_percentage" : sum(overlap_percentages) / len(overlap_percentages) if overlap_percentages else 0,
               }