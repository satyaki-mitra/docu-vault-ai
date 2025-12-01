# DEPENDENCIES
import re
from typing import List
from typing import Dict
from typing import Tuple 
from typing import Optional
from collections import defaultdict
from config.models import DocumentChunk
from config.models import ChunkWithScore
from config.logging_config import get_logger
from utils.error_handler import CitationError
from utils.error_handler import handle_errors


# Setup Logging
logger = get_logger(__name__)


class CitationTracker:
    """
    Citation tracking and management: Tracks source citations in generated text and provides citation formatting and validation
    """
    def __init__(self):
        """
        Initialize citation tracker
        """
        self.logger           = logger
        self.citation_pattern = re.compile(r'\[(\d+)\]')
    

    def extract_citations(self, text: str) -> List[int]:
        """
        Extract citation numbers from text
        
        Arguments:
        ----------
            text { str } : Text containing citations
        
        Returns:
        --------
             { list }    : List of citation numbers found in text
        """
        if not text:
            return []
        
        try:
            matches          = self.citation_pattern.findall(text)
            citation_numbers = [int(match) for match in matches]
            
            # Remove duplicates and sort
            unique_citations = sorted(set(citation_numbers))
            
            self.logger.debug(f"Extracted {len(unique_citations)} citations from text")
            
            return unique_citations
        
        except Exception as e:
            self.logger.error(f"Citation extraction failed: {repr(e)}")
            return []
    

    def validate_citations(self, text: str, sources: List[ChunkWithScore]) -> Tuple[bool, List[int]]:
        """
        Validate that all citations in text reference existing sources
        
        Arguments:
        ----------
            text    { str }  : Text containing citations
            
            sources { list } : List of available sources
        
        Returns:
        --------
              { Tuple[bool, List[int]] } : (is_valid, invalid_citations)
        """
        citation_numbers = self.extract_citations(text = text)
        
        if not citation_numbers:
            return True, []
        
        # Check if all citation numbers are within valid range
        max_valid         = len(sources)
        invalid_citations = [num for num in citation_numbers if (num < 1) or (num > max_valid)]
        
        if invalid_citations:
            self.logger.warning(f"Invalid citations found: {invalid_citations}. Valid range: 1-{max_valid}")
            return False, invalid_citations
        
        return True, []
    

    def format_citations(self, sources: List[ChunkWithScore], style: str = "numeric") -> str:
        """
        Format citations as reference list
        
        Arguments:
        ----------
            sources { list } : List of sources to format
            
            style   { str }  : Citation style ('numeric', 'verbose')
        
        Returns:
        --------
               { str }       : Formatted citation text
        """
        if not sources:
            return ""
        
        try:
            citations = list()

            for i, source in enumerate(sources, 1):
                if (style == "verbose"):
                    citation = self._format_verbose_citation(source = source, 
                                                             number = i,
                                                            )
                
                else:
                    citation = self._format_numeric_citation(source = source, 
                                                             number = i,
                                                            )
                
                citations.append(citation)
            
            citation_text = "\n".join(citations)
            
            self.logger.debug(f"Formatted {len(citations)} citations in {style} style")
            
            return citation_text
        
        except Exception as e:
            self.logger.error(f"Citation formatting failed: {repr(e)}")
            return ""
    

    def _format_numeric_citation(self, source: ChunkWithScore, number: int) -> str:
        """
        Format citation in numeric style with sanitization
        
        Arguments:
        ----------
            source { ChunkWithScore } : Source to format

            number      { int }       : Citation number
        
        Returns:
        --------
                { str }              : Formatted citation
        """
        chunk = source.chunk
        
        parts = [f"[{number}]"]
        
        # Add source information with proper sanitization
        if (hasattr(chunk, 'metadata') and chunk.metadata):
            if ('filename' in chunk.metadata):
                # Sanitize filename more thoroughly
                filename = str(chunk.metadata['filename'])
                
                # Remove problematic characters that could break citation parsing: Keep only alphanumeric, spaces, dots, hyphens, underscores
                filename = re.sub(r'[^\w\s\.\-]', '_', filename)
                
                # Limit length to prevent overflow
                if (len(filename) > 50):
                    filename = filename[:47] + "..."
                
                parts.append(f"Source: {filename}")
        
        if chunk.page_number:
            parts.append(f"Page {chunk.page_number}")
        
        if chunk.section_title:
            # Sanitize section title similarly
            section = str(chunk.section_title)
            section = re.sub(r'[^\w\s\.\-]', '_', section)
            
            if (len(section) > 40):
                section = section[:37] + "..."
            
            parts.append(f"Section: {section}")
        
        # Add relevance score if available
        if (source.score > 0):
            parts.append(f"(Relevance: {source.score:.2f})")
        
        return " ".join(parts)
    

    def _format_verbose_citation(self, source: ChunkWithScore, number: int) -> str:
        """
        Format citation in verbose style - SAFER VERSION
        
        Arguments:
        ----------
            source { ChunkWithScore } : Source to format

            number     { int }        : Citation number
        
        Returns:
        --------
                { str }              : Formatted citation
        """
        chunk = source.chunk
        
        parts = [f"Citation {number}:"]
        
        # Document information with sanitization
        if (hasattr(chunk, 'metadata')):
            meta = chunk.metadata
            
            if ('filename' in meta):
                filename = str(meta['filename'])
                filename = re.sub(r'[^\w\s\.\-]', '_', filename)
                
                if (len(filename) > 50):
                    filename = filename[:47] + "..."

                parts.append(f"Document: {filename}")
            
            if ('title' in meta):
                title = str(meta['title'])
                title = re.sub(r'[^\w\s\.\-]', '_', title)
                
                if (len(title) > 60):
                    title = title[:57] + "..."
                
                parts.append(f"Title: {title}")
            
            if ('author' in meta):
                author = str(meta['author'])
                author = re.sub(r'[^\w\s\.\-]', '_', author)
                
                if (len(author) > 40):
                    author = author[:37] + "..."
                    
                parts.append(f"Author: {author}")
        
        # Location information
        location_parts = list()
        
        if chunk.page_number:
            location_parts.append(f"page {chunk.page_number}")
        
        if chunk.section_title:
            section = str(chunk.section_title)
            section = re.sub(r'[^\w\s\.\-]', '_', section)
            
            if (len(section) > 40):
                section = section[:37] + "..."
            
            location_parts.append(f"section '{section}'")
        
        if location_parts:
            parts.append("(" + ", ".join(location_parts) + ")")
        
        # Relevance information
        if (source.score > 0):
            parts.append(f"[Relevance score: {source.score:.3f}]")
        
        return " ".join(parts)
    

    def generate_citation_map(self, sources: List[ChunkWithScore]) -> Dict[int, Dict]:
        """
        Generate mapping from citation numbers to source details
        
        Arguments:
        ----------
            sources { list } : List of sources
        
        Returns:
        --------
               { dict }      : Dictionary mapping citation numbers to source details
        """
        citation_map = dict()

        for i, source in enumerate(sources, 1):
            chunk           = source.chunk
            
            citation_map[i] = {'chunk_id'      : chunk.chunk_id,
                               'document_id'   : chunk.document_id,
                               'score'         : source.score,
                               'text_preview'  : chunk.text[:200] + "..." if (len(chunk.text) > 200) else chunk.text,
                               'metadata'      : getattr(chunk, 'metadata', {}),
                               'page_number'   : chunk.page_number,
                               'section_title' : chunk.section_title,
                              }
        
        return citation_map
    

    def replace_citation_markers(self, text: str, citation_map: Dict[int, str]) -> str:
        """
        Replace citation markers with formatted citations - FIXED
        
        Arguments:
        ----------
            text         { str }  : Text containing citation markers
            
            citation_map { dict } : Mapping of citation numbers to formatted strings
        
        Returns:
        --------
                  { str }         : Text with replaced citations
        """
        def replacement(match):
            try:
                citation_num     = int(match.group(1))
                
                # Get replacement text and sanitize it
                replacement_text = citation_map.get(citation_num, match.group(0))
                
                return str(replacement_text)
            
            except (ValueError, IndexError):
                # Return original match if parsing fails
                return match.group(0)
        
        try:
            return self.citation_pattern.sub(replacement, text)
        
        except Exception as e:
            self.logger.error(f"Citation replacement failed: {repr(e)}")
            # Return original text on error
            return text  
    

    def get_citation_statistics(self, text: str, sources: List[ChunkWithScore]) -> Dict:
        """
        Get statistics about citations in text
        
        Arguments:
        ----------
            text    { str }  : Text containing citations
            
            sources { list } : List of sources
        
        Returns:
        --------
              { dict }       : Citation statistics
        """
        citation_numbers = self.extract_citations(text = text)
        
        if not citation_numbers:
            return {"total_citations": 0}
        
        # Calculate citation distribution
        citation_counts = defaultdict(int)

        for num in citation_numbers:
            if 1 <= num <= len(sources):
                source = sources[num - 1]
                doc_id = source.chunk.document_id
                citation_counts[doc_id] += 1
        
        return {"total_citations"      : len(citation_numbers),
                "unique_citations"     : len(set(citation_numbers)),
                "citation_distribution": dict(citation_counts),
                "citations_per_source" : {i: citation_numbers.count(i) for i in set(citation_numbers)},
               }
    

    def ensure_citation_consistency(self, text: str, sources: List[ChunkWithScore]) -> str:
        """
        Ensure citation numbers are consistent and sequential
        
        Arguments:
        ----------
            text    { str }  : Text containing citations
            
            sources { list } : List of sources
        
        Returns:
        --------
              { str }        : Text with consistent citations
        """
        is_valid, invalid_citations = self.validate_citations(text, sources)
        
        if not is_valid:
            self.logger.warning("Invalid citations found, attempting to fix consistency")
            
            # Extract current citations and create mapping
            current_citations = self.extract_citations(text = text)
            
            if not current_citations:
                return text
            
            # Create mapping from old to new citation numbers
            citation_mapping = dict()
            
            for i, old_num in enumerate(sorted(set(current_citations)), 1):
                if (old_num <= len(sources)):
                    citation_mapping[old_num] = i
            
            # Replace citations in text
            def consistent_replacement(match):
                old_num = int(match.group(1))
                new_num = citation_mapping.get(old_num, old_num)
                
                return f"[{new_num}]"
            
            fixed_text = self.citation_pattern.sub(consistent_replacement, text)
            
            self.logger.info(f"Fixed citation consistency: {current_citations} -> {list(citation_mapping.values())}")
            
            return fixed_text
        
        return text


# Global citation tracker instance
_citation_tracker = None


def get_citation_tracker() -> CitationTracker:
    """
    Get global citation tracker instance (singleton)
    
    Returns:
    --------
        { CitationTracker }    : CitationTracker instance
    """
    global _citation_tracker
    
    if _citation_tracker is None:
        _citation_tracker = CitationTracker()
    
    return _citation_tracker


@handle_errors(error_type = CitationError, log_error = True, reraise = False)
def extract_and_validate_citations(text: str, sources: List[ChunkWithScore]) -> Tuple[List[int], bool]:
    """
    Convenience function for citation extraction and validation
    
    Arguments:
    ----------
        text    { str }  : Text containing citations
        
        sources { list } : List of sources
    
    Returns:
    --------
             { Tuple[List[int], bool] } : (citation_numbers, is_valid)
    """
    tracker   = get_citation_tracker()
    citations = tracker.extract_citations(text = text)
    is_valid, _ = tracker.validate_citations(text    = text, 
                                             sources = sources,
                                            )
    
    return citations, is_valid