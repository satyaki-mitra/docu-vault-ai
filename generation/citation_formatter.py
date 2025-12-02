# DEPENDENCIES
import re
from enum import Enum
from typing import List
from typing import Dict
from typing import Optional
from collections import defaultdict
from config.models import CitationStyle
from config.models import ChunkWithScore
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import CitationFormattingError


# Setup Logging
logger = get_logger(__name__)


class CitationFormatter:
    """
    Citation formatting and management: Formats citations in generated text according to different styles and ensures citation consistency and validity
    """
    def __init__(self, style: CitationStyle = CitationStyle.NUMERIC):
        """
        Initialize citation formatter
        
        Arguments:
        ----------
            style { CitationStyle } : Citation style to use
        """
        self.logger           = logger
        self.style            = style
        self.citation_pattern = re.compile(r'\[(\d+)\]')
        
        # Style configurations
        self.style_configs    = {CitationStyle.NUMERIC  : {"inline_format" : "[{number}]", "reference_format" : "[{number}] {source_info}", "separator" : " ",},
                                 CitationStyle.VERBOSE  : {"inline_format" : "[{number}]", "reference_format" : "Citation {number}: {source_info}", "separator" : "\n",},
                                 CitationStyle.MINIMAL  : {"inline_format" : "[{number}]", "reference_format" : "[{number}]", "separator" : " ",},
                                 CitationStyle.ACADEMIC : {"inline_format" : "({number})", "reference_format" : "{number}. {source_info}", "separator" : "\n",},
                                 CitationStyle.LEGAL    : {"inline_format" : "[{number}]", "reference_format" : "[{number}] {source_info}", "separator" : "\n",}
                                }
    

    def format_citations_in_text(self, text: str, sources: List[ChunkWithScore]) -> str:
        """
        Format citations in generated text
        
        Arguments:
        ----------
            text    { str }  : Text containing citation markers
            
            sources { list } : List of sources for citation mapping
        
        Returns:
        --------
              { str }        : Text with formatted citations
        """
        if not text or not sources:
            return text
        
        try:
            # Extract citation numbers from text
            citation_numbers = self._extract_citation_numbers(text = text)
            
            if not citation_numbers:
                return text
            
            # Create citation mapping
            citation_map     = self._create_citation_map(sources = sources)
            
            # Replace citation markers with formatted citations
            formatted_text   = self._replace_citation_markers(text         = text, 
                                                              citation_map = citation_map,
                                                             )
            
            self.logger.debug(f"Formatted {len(citation_numbers)} citations in text")
            
            return formatted_text
        
        except Exception as e:
            self.logger.error(f"Citation formatting failed: {repr(e)}")
            raise CitationFormattingError(f"Citation formatting failed: {repr(e)}")
    

    def generate_reference_section(self, sources: List[ChunkWithScore], cited_numbers: List[int]) -> str:
        """
        Generate reference section for cited sources
        
        Arguments:
        ----------
            sources       { list } : All available sources
            
            cited_numbers { list } : Numbers of actually cited sources
        
        Returns:
        --------
                  { str }          : Formatted reference section
        """
        if not sources or not cited_numbers:
            return ""
        
        try:
            style_config  = self.style_configs[self.style]
            references    = list()
            
            # Get only cited sources
            cited_sources = [sources[num-1] for num in cited_numbers if (1 <= num <= len(sources))]
            
            for i, source in enumerate(cited_sources, 1):
                source_info = self._format_source_info(source, i)
                reference   = style_config["reference_format"].format(number = i, source_info = source_info)
                
                references.append(reference)
            
            separator         = style_config["separator"]
            reference_section = separator.join(references)
            
            # Add section header if appropriate
            if (self.style in [CitationStyle.VERBOSE, CitationStyle.ACADEMIC]):
                reference_section = "References:\n" + reference_section
            
            self.logger.debug(f"Generated reference section with {len(references)} entries")
            
            return reference_section
        
        except Exception as e:
            self.logger.error(f"Reference section generation failed: {repr(e)}")
            return ""
    

    def _extract_citation_numbers(self, text: str) -> List[int]:
        """
        Extract citation numbers from text
        """
        matches          = self.citation_pattern.findall(text)
        citation_numbers = [int(match) for match in matches]
        
        # Unique and sorted
        return sorted(set(citation_numbers))  
    

    def _create_citation_map(self, sources: List[ChunkWithScore]) -> Dict[int, str]:
        """
        Create mapping from citation numbers to formatted citations
        """
        citation_map = dict()
        style_config = self.style_configs[self.style]
        
        for i, source in enumerate(sources, 1):
            formatted_citation = style_config["inline_format"].format(number=i)
            citation_map[i]    = formatted_citation
        
        return citation_map
    

    def _replace_citation_markers(self, text: str, citation_map: Dict[int, str]) -> str:
        """
        Replace citation markers in text
        """
        def replacement(match):
            citation_num = int(match.group(1))

            return citation_map.get(citation_num, match.group(0))
        
        return self.citation_pattern.sub(replacement, text)
    

    def _format_source_info(self, source: ChunkWithScore, citation_number: int) -> str:
        """
        Format source information based on style
        """
        chunk = source.chunk
        
        if (self.style == CitationStyle.MINIMAL):
            return f"Source {citation_number}"
        
        # Build source components
        components = list()
        
        # Document information
        if hasattr(chunk, 'metadata'):
            meta = chunk.metadata
            
            if ('filename' in meta):
                components.append(f"Document: {meta['filename']}")
            
            if (('title' in meta) and meta['title']):
                components.append(f"\"{meta['title']}\"")
            
            if (('author' in meta) and meta['author']):
                components.append(f"by {meta['author']}")
        
        # Location information
        location_parts = list()

        if chunk.page_number:
            location_parts.append(f"p. {chunk.page_number}")
        
        if chunk.section_title:
            location_parts.append(f"Section: {chunk.section_title}")
        
        if location_parts:
            components.append("(" + ", ".join(location_parts) + ")")
        
        # Relevance score (for verbose styles)
        if ((self.style in [CitationStyle.VERBOSE, CitationStyle.ACADEMIC]) and (source.score > 0)):
            components.append(f"[relevance: {source.score:.3f}]")
        
        return " ".join(components)
    

    def validate_citations(self, text: str, sources: List[ChunkWithScore]) -> tuple[bool, List[int]]:
        """
        Validate citations in text
        
        Arguments:
        ----------
            text    { str }  : Text to validate
            
            sources { list } : Available sources
        
        Returns:
        --------
              { tuple }      : (is_valid, invalid_citations)
        """
        citation_numbers  = self._extract_citation_numbers(text = text)
        invalid_citations = list()
        
        for number in citation_numbers:
            if ((number < 1) or (number > len(sources))):
                invalid_citations.append(number)
        
        is_valid = (len(invalid_citations) == 0)
        
        if not is_valid:
            self.logger.warning(f"Invalid citations found: {invalid_citations}")
        
        return is_valid, invalid_citations
    

    def normalize_citations(self, text: str, sources: List[ChunkWithScore]) -> str:
        """
        Normalize citations to ensure sequential numbering
        
        Arguments:
        ----------
            text    { str }  : Text with citations
            
            sources { list } : Available sources
        
        Returns:
        --------
              { str }        : Text with normalized citations
        """
        citation_numbers = self._extract_citation_numbers(text = text)
        
        if not citation_numbers:
            return text
        
        # Create mapping from old to new numbers
        citation_mapping = dict()

        for i, old_num in enumerate(sorted(set(citation_numbers)), 1):
            if (1 <= old_num <= len(sources)):
                citation_mapping[old_num] = i
        
        # Replace citations
        def normalize_replacement(match):
            old_num      = int(match.group(1))
            new_num      = citation_mapping.get(old_num, old_num)
            style_config = self.style_configs[self.style]
            
            return style_config["inline_format"].format(number = new_num)
        
        normalized_text = self.citation_pattern.sub(normalize_replacement, text)
        
        if citation_mapping:
            self.logger.info(f"Normalized citations: {citation_numbers} -> {list(citation_mapping.values())}")
        
        return normalized_text
    

    def get_citation_statistics(self, text: str, sources: List[ChunkWithScore]) -> Dict:
        """
        Get citation statistics
        
        Arguments:
        ----------
            text    { str }  : Text with citations
            
            sources { list } : Available sources
        
        Returns:
        --------
              { dict }       : Citation statistics
        """
        citation_numbers = self._extract_citation_numbers(text = text)
        
        if not citation_numbers:
            return {"total_citations": 0}
        
        # Calculate distribution
        source_usage = defaultdict(int)

        for number in citation_numbers:
            if (1 <= number <= len(sources)):
                source                = sources[number-1]
                doc_id                = source.chunk.document_id
                source_usage[doc_id] += 1
        
        return {"total_citations"      : len(citation_numbers),
                "unique_citations"     : len(set(citation_numbers)),
                "sources_used"         : len(source_usage),
                "citations_per_source" : dict(source_usage),
                "citation_density"     : len(citation_numbers) / max(1, len(text.split())),  
               }
    

    def set_style(self, style: CitationStyle):
        """
        Set citation style
        
        Arguments:
        ----------
            style { CitationStyle } : New citation style
        """
        if (style not in self.style_configs):
            raise CitationFormattingError(f"Unsupported citation style: {style}")
        
        old_style  = self.style
        self.style = style
        
        self.logger.info(f"Citation style changed: {old_style} -> {style}")


# Global citation formatter instance
_citation_formatter = None


def get_citation_formatter() -> CitationFormatter:
    """
    Get global citation formatter instance (singleton)
    
    Returns:
    --------
        { CitationFormatter }    : CitationFormatter instance
    """
    global _citation_formatter
    
    if _citation_formatter is None:
        _citation_formatter = CitationFormatter()
    
    return _citation_formatter


@handle_errors(error_type = CitationFormattingError, log_error = True, reraise = False)
def format_citations(text: str, sources: List[ChunkWithScore], style: CitationStyle = None) -> str:
    """
    Convenience function for citation formatting
    
    Arguments:
    ----------
        text          { str }     : Text containing citations
        
        sources      { list }     : List of sources
        
        style   { CitationStyle } : Citation style to use
    
    Returns:
    --------
                 { str }          : Formatted text
    """
    formatter = get_citation_formatter()
    
    if style is not None:
        formatter.set_style(style)
    
    return formatter.format_citations_in_text(text, sources)