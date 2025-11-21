# DEPENDENCIES
import re
import html
import unicodedata
from typing import Optional, List
from config.logging_config import get_logger

# Setup Logger
logger = get_logger(__name__)


class TextCleaner:
    """
    Comprehensive text cleaning and normalization: Preserves semantic meaning while removing noise
    """
    # Common patterns
    URL_PATTERN         = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    EMAIL_PATTERN       = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN       = re.compile(r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')
    MULTIPLE_SPACES     = re.compile(r'\s+')
    MULTIPLE_NEWLINES   = re.compile(r'\n\s*\n\s*\n+')
    
    # HTML/XML patterns
    HTML_TAG_PATTERN    = re.compile(r'<[^>]+>')
    HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;')
    
    # Special characters
    BULLET_POINTS       = ['•', '◦', '▪', '▫', '⬩', '▹', '▸', '►', '▻', '→']
    QUOTATION_MARKS     = ['"', '"', ''', ''', '«', '»', '‹', '›']
    

    @classmethod
    def clean(cls, text: str, remove_urls: bool = False, remove_emails: bool = False, remove_phone_numbers: bool = False, remove_html: bool = True,
              normalize_whitespace: bool = True, normalize_quotes: bool = True, normalize_bullets: bool = True, lowercase: bool = False, 
              remove_extra_newlines: bool = True, preserve_structure: bool = True) -> str:
        """
        Clean text with configurable options
        
        Arguments:
        ----------
            text                  { str }  : Input text
            
            remove_urls           { bool } : Remove URLs
            
            remove_emails         { bool } : Remove email addresses
            
            remove_phone_numbers  { bool } : Remove phone numbers
            
            remove_html           { bool } : Remove HTML tags
            
            normalize_whitespace  { bool } : Normalize spaces/tabs
            
            normalize_quotes      { bool } : Convert fancy quotes to standard
            
            normalize_bullets     { bool } : Convert bullet points to standard
            
            lowercase             { bool } : Convert to lowercase
            
            remove_extra_newlines { bool } : Remove excessive blank lines
            
            preserve_structure    { bool } : Try to maintain document structure
        
        Returns:
        --------
                        { str }            : Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        # Original length for logging
        original_length = len(text)
        
        # Remove HTML if present
        if remove_html:
            text = cls.remove_html_tags(text)
            text = cls.decode_html_entities(text)
        
        # Remove specific patterns
        if remove_urls:
            text = cls.URL_PATTERN.sub(' ', text)
        
        if remove_emails:
            text = cls.EMAIL_PATTERN.sub(' ', text)
        
        if remove_phone_numbers:
            text = cls.PHONE_PATTERN.sub(' ', text)
        
        # Normalize unicode
        text = cls.normalize_unicode(text)
        
        # Normalize quotes
        if normalize_quotes:
            text = cls.normalize_quotation_marks(text)
        
        # Normalize bullets
        if normalize_bullets:
            text = cls.normalize_bullet_points(text)
        
        # Handle whitespace
        if normalize_whitespace:
            # Replace tabs with spaces
            text = text.replace('\t', '    ')
            
            # Normalize spaces (but not newlines if preserving structure)
            if preserve_structure:
                lines = text.split('\n')
                lines = [cls.MULTIPLE_SPACES.sub(' ', line) for line in lines]
                text  = '\n'.join(lines)
           
            else:
                text = cls.MULTIPLE_SPACES.sub(' ', text)
        
        # Remove extra newlines
        if remove_extra_newlines:
            text = cls.MULTIPLE_NEWLINES.sub('\n\n', text)
        
        # Lowercase if requested
        if lowercase:
            text = text.lower()
        
        # Final cleanup
        text = text.strip()
        
        # Log cleaning stats
        cleaned_length = len(text)
        reduction      = ((original_length - cleaned_length) / original_length * 100) if (original_length > 0) else 0

        logger.debug(f"Text cleaned: {original_length} -> {cleaned_length} chars ({reduction:.1f}% reduction)")
        
        return text
    

    @classmethod
    def remove_html_tags(cls, text: str) -> str:
        """
        Remove HTML tags
        """
        return cls.HTML_TAG_PATTERN.sub('', text)
    

    @classmethod
    def decode_html_entities(cls, text: str) -> str:
        """
        Decode HTML entities
        """
        return html.unescape(text)
    

    @classmethod
    def normalize_unicode(cls, text: str) -> str:
        """
        Normalize unicode characters : Converts to NFC form (canonical composition)
        """
        return unicodedata.normalize('NFC', text)
    

    @classmethod
    def normalize_quotation_marks(cls, text: str) -> str:
        """
        Convert fancy quotes to standard ASCII quotes
        """
        for fancy_quote in cls.QUOTATION_MARKS:
            if (fancy_quote in ['"', '"', '«', '»']):
                text = text.replace(fancy_quote, '"')
            
            elif (fancy_quote in [''', ''', '‹', '›']):
                text = text.replace(fancy_quote, "'")

        return text
    

    @classmethod
    def normalize_bullet_points(cls, text: str) -> str:
        """
        Convert various bullet points to standard bullet
        """
        for bullet in cls.BULLET_POINTS:
            text = text.replace(bullet, '•')
        
        return text
    

    @classmethod
    def remove_boilerplate(cls, text: str, remove_headers: bool = True, remove_footers: bool = True, remove_page_numbers: bool = True) -> str:
        """
        Remove common boilerplate text
        
        Arguments:
        ----------
            text                 { str } : Input text
            
            remove_headers      { bool } : Remove common header patterns
            
            remove_footers      { bool } : Remove common footer patterns
            
            remove_page_numbers { bool } : Remove standalone page numbers
        
        Returns:
        --------
                    { str }              : Text without boilerplate
        """
        lines         = text.split('\n')
        cleaned_lines = list()
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Remove page numbers (lines that are just numbers)
            if remove_page_numbers and line_stripped.isdigit():
                continue
            
            # Remove common header patterns
            if remove_headers:
                header_patterns = [r'^Page \d+ of \d+$', r'^\d+/\d+$', r'^Header:', r'^Draft', r'^Confidential']
                
                if (any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in header_patterns)):
                    continue
            
            # Remove common footer patterns
            if remove_footers:
                footer_patterns = [r'^Copyright ©', r'^All rights reserved', r'^Footer:', r'^\d{4} .+ Inc\.']
                
                if any(re.match(pattern, line_stripped, re.IGNORECASE) for pattern in footer_patterns):
                    continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    

    @classmethod
    def extract_sentences(cls, text: str) -> List[str]:
        """
        Split text into sentences : Handles common abbreviations and edge cases
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
            { list }     : List of sentences
        """
        # Common abbreviations that shouldn't trigger sentence breaks
        abbreviations  = {'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.', 'Prof.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'U.S.', 'U.K.'}
        
        # Protect abbreviations
        protected_text = text

        for abbr in abbreviations:
            protected_text = protected_text.replace(abbr, abbr.replace('.', '<DOT>'))
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences        = re.split(sentence_pattern, protected_text)
        
        # Restore abbreviations
        sentences        = [s.replace('<DOT>', '.') for s in sentences]
        
        # Clean and filter
        sentences        = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    

    @classmethod
    def truncate(cls, text: str, max_length: int, suffix: str = "...", word_boundary: bool = True) -> str:
        """
        Truncate text to maximum length
        
        Arguments:
        ----------
            text          { str }  : Input text

            max_length    { int }  : Maximum length
            
            suffix        { str }  : Suffix to append when truncated
            
            word_boundary { bool } : Truncate at word boundary
        
        Returns:
        --------
                  { str }          : Truncated text
        """
        if (len(text) <= max_length):
            return text
        
        # Account for suffix
        max_length -= len(suffix)
        
        if word_boundary:
            # Find last space before max_length
            truncated  = text[:max_length]
            last_space = truncated.rfind(' ')
            
            if (last_space > 0):
                truncated = truncated[:last_space]
        
        else:
            truncated = text[:max_length]
        
        return truncated + suffix
    

    @classmethod
    def remove_special_characters(cls, text: str, keep_punctuation: bool = True, keep_numbers: bool = True) -> str:
        """
        Remove special characters
        
        Arguments:
        ----------
            text             { str }  : Input text

            keep_punctuation { bool } : Keep basic punctuation
            
            keep_numbers     { bool } : Keep numbers
        
        Returns:
        --------
                    { str }           : Text with special characters removed
        """
        if keep_punctuation and keep_numbers:
            # Keep alphanumeric and basic punctuation
            pattern = r'[^a-zA-Z0-9\s.,!?;:\'-]'

        elif keep_punctuation:
            # Keep letters and punctuation
            pattern = r'[^a-zA-Z\s.,!?;:\'-]'

        elif keep_numbers:
            # Keep letters and numbers
            pattern = r'[^a-zA-Z0-9\s]'

        else:
            # Keep only letters
            pattern = r'[^a-zA-Z\s]'
        
        return re.sub(pattern, '', text)

    
    @classmethod
    def deduplicate_lines(cls, text: str, preserve_order: bool = True) -> str:
        """
        Remove duplicate lines
        
        Arguments:
        ----------
            text           { str }  : Input text

            preserve_order { bool } : Maintain original order
        
        Returns:
        --------
                  { str }           : Text with duplicate lines removed
        """
        lines = text.split('\n')
        
        if preserve_order:
            seen         = set()
            unique_lines = list()

            for line in lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)

        else:
            unique_lines = list(set(lines))
        
        return '\n'.join(unique_lines)

    
    @classmethod
    def count_tokens_estimate(cls, text: str) -> int:
        """
        Estimate token count: Rule of thumb is - ~4 characters per token for English.
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
            { int }      : Estimated token count
        """
        # More accurate estimation
        words         = text.split()
        chars         = len(text)
        
        # Average of word-based and char-based estimates
        word_estimate = len(words) * 1.3  # ~1.3 tokens per word

        # ~4 chars per token
        char_estimate = chars / 4  
        
        return int((word_estimate + char_estimate) / 2)
    

    @classmethod
    def preserve_structure_markers(cls, text: str) -> str:
        """
        Identify and mark structural elements: Useful for semantic chunking
        
        Arguments:
        ----------
            text { str } : Input text
        
        Returns:
        --------
             { str }     : Text with structure markers
        """
        lines        = text.split('\n')
        marked_lines = list()
        
        for line in lines:
            stripped = line.strip()
            
            # Mark headers (ALL CAPS, short lines)
            if (stripped.isupper() and (len(stripped) < 100)):
                marked_lines.append(f"[HEADER] {line}")
            
            # Mark list items
            elif re.match(r'^[\d•\-\*]\s', stripped):
                marked_lines.append(f"[LIST] {line}")

            # Regular text
            else:
                marked_lines.append(line)
        
        return '\n'.join(marked_lines)


def clean_for_rag(text: str) -> str:
    """
    Convenience function: clean text optimally for RAG
    
    Arguments:
    ----------
        text { str } : Input text
    
    Returns:
    --------
         { str }     : Cleaned text
    """
    return TextCleaner.clean(text,
                             remove_urls           = False,  # URLs might be useful context
                             remove_emails         = False,  # Emails might be useful
                             remove_phone_numbers  = False,  # Phone numbers might be useful
                             remove_html           = True,
                             normalize_whitespace  = True,
                             normalize_quotes      = True,
                             normalize_bullets     = True,
                             lowercase             = False,  # Keep original casing for proper nouns
                             remove_extra_newlines = True,
                             preserve_structure    = True,   # Important for chunking
                            )



if __name__ == "__main__":
    # Test text cleaner
    print("=== Text Cleaner Tests ===\n")
    
    # Test HTML removal
    html_text  = "<p>This is <strong>bold</strong> text with a <a href='url'>link</a>.</p>"
    clean_text = TextCleaner.clean(html_text, remove_html = True)
    
    print(f"HTML removal:\n{html_text}\n-> {clean_text}\n")
    
    # Test whitespace normalization
    messy_text = "This    has     multiple   spaces\n\n\n\nand  \n\n  newlines"
    clean_text = TextCleaner.clean(messy_text, normalize_whitespace = True)
    
    print(f"Whitespace normalization:\n{messy_text!r}\n-> {clean_text!r}\n")
    
    # Test quotation normalization
    fancy_text = "He said "hello" and she replied 'hi'"
    clean_text = TextCleaner.clean(fancy_text, normalize_quotes = True)
    
    print(f"Quote normalization:\n{fancy_text}\n-> {clean_text}\n")
    
    # Test sentence extraction
    text       = "Dr. Smith works at Inc. Corp. He is from the U.S. He has a Ph.D."
    sentences  = TextCleaner.extract_sentences(text)

    print("Sentence extraction:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    # Test token counting
    text       = "This is a sample text for token counting estimation."
    tokens     = TextCleaner.count_tokens_estimate(text)
    
    print(f"\nToken estimate: '{text}' -> ~{tokens} tokens")
    
    # Test truncation
    long_text  = "This is a very long piece of text that needs to be truncated at some point."
    truncated  = TextCleaner.truncate(long_text, max_length = 30, word_boundary = True)
    
    print(f"\nTruncation:\n{long_text}\n-> {truncated}")
    
    print("\n✓ All text cleaner tests passed!")
