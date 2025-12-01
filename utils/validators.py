# DEPENDENCIES
import re
import magic
from typing import List
from typing import Union
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import FileTooLargeError
from utils.error_handler import InvalidFileTypeError


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class FileValidator:
    """
    Comprehensive file validation utilities
    """
    @staticmethod
    def validate_file_size(file_path: Path, max_size_bytes: Optional[int] = None) -> bool:
        """
        Validate file size
        """
        if max_size_bytes is None:
            max_size_bytes = settings.max_file_size_bytes
        
        file_size = file_path.stat().st_size
        
        if file_size > max_size_bytes:
            raise FileTooLargeError(file_size = file_size,
                                    max_size  = max_size_bytes,
                                   )
        
        return True
    

    @staticmethod
    def validate_file_type(file_path: Path, allowed_extensions: Optional[List[str]] = None) -> bool:
        """
        Validate file type by both extension and content
        """
        if allowed_extensions is None:
            allowed_extensions = settings.ALLOWED_EXTENSIONS
        
        # Check extension
        extension = file_path.suffix.lstrip('.').lower()
        
        if extension not in allowed_extensions:
            raise InvalidFileTypeError(file_type     = extension,
                                       allowed_types = allowed_extensions,
                                      )
        
        # Verify actual file content
        try:
            mime               = magic.Magic(mime = True)
            mime_type          = mime.from_file(str(file_path))
            
            # Map MIME types to extensions
            mime_to_extension  = {'application/pdf'                                                         : 'pdf',
                                  'application/vnd.openxmlformats-officedocument.wordprocessingml.document' : 'docx',
                                  'text/plain'                                                              : 'txt',
                                  'application/zip'                                                         : 'zip',
                                 }
            
            detected_extension = mime_to_extension.get(mime_type)
            
            if (detected_extension and (detected_extension != extension)):
                # Still allowing it but logging warning message
                logger.warning(f"File extension mismatch: {extension} vs detected {detected_extension}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not verify file content type: {repr(e)}")
            # Fall back to extension validation only
            return True
    

    @staticmethod
    def validate_file_integrity(file_path: Path) -> bool:
        """
        Basic file integrity check
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if (file_path.stat().st_size == 0):
            raise ValueError(f"File is empty: {file_path}")
        
        return True
    

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """
        Validate filename safety
        """
        # Check for path traversal attempts
        if (('..' in filename) or ('/' in filename) or ('\\' in filename)):
            raise ValueError("Filename contains path traversal attempts")
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        
        for char in dangerous_chars:
            if char in filename:
                raise ValueError(f"Filename contains dangerous character: {char}")
        
        # Check length
        if (len(filename) > 255):
            raise ValueError("Filename too long")
        
        return True


class URLValidator:
    """
    URL validation utilities
    """
    
    @staticmethod
    def validate_url(url: str, allowed_domains: Optional[List[str]] = None) -> bool:
        """
        Validate URL format and domain
        """
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                raise ValueError("URL must use HTTP or HTTPS protocol")
            
            # Check netloc (domain)
            if not parsed.netloc:
                raise ValueError("Invalid URL: missing domain")
            
            # Check domain if restrictions exist
            if allowed_domains:
                domain_allowed = any(((parsed.netloc.endswith(domain)) or (parsed.netloc == domain)) for domain in allowed_domains)
                
                if not domain_allowed:
                    raise ValueError(f"Domain not allowed: {parsed.netloc}")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Invalid URL: {repr(e)}")
    

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Check if URL is valid without raising exceptions
        """
        try:
            return URLValidator.validate_url(url)

        except ValueError:
            return False
    

    @staticmethod
    def extract_domain(url: str) -> str:
        """
        Extract domain from URL
        """
        parsed = urlparse(url)
        
        return parsed.netloc


class TextValidator:
    """
    Text content validation
    """
    @staticmethod
    def validate_text_length(text: str, min_length: int = 1, max_length: Optional[int] = None) -> bool:
        """
        Validate text length
        """
        if (len(text.strip()) < min_length):
            raise ValueError(f"Text too short. Minimum {min_length} characters required.")
        
        if (max_length and len(text) > max_length):
            raise ValueError(f"Text too long. Maximum {max_length} characters allowed.")
        
        return True
    

    @staticmethod
    def is_meaningful_text(text: str, min_words: int = 3) -> bool:
        """
        Check if text contains meaningful content
        """
        words = text.strip().split()
        
        if (len(words) < min_words):
            return False
        
        # Check if it's not just special characters/numbers
        alpha_count = sum(1 for char in text if char.isalpha())
        
        if (alpha_count < min_words):
            return False
        
        return True
    

    @staticmethod
    def has_sufficient_content(text: str, min_chars: int = 50, min_sentences: int = 1) -> bool:
        """
        Check if text has sufficient content for processing
        """
        if len(text.strip()) < min_chars:
            return False
        
        # Count sentences (rough estimate)
        sentence_endings = re.findall(r'[.!?]+', text)
        
        if (len(sentence_endings) < min_sentences):
            return False
        
        return True


class DocumentValidator:
    """
    Document-specific validation
    """
    @staticmethod
    def validate_document_id(doc_id: str) -> bool:
        """
        Validate document ID format
        """
        pattern = r'^doc_\d+_[a-f0-9]{8}$'
        
        if (not re.match(pattern, doc_id)):
            raise ValueError(f"Invalid document ID format: {doc_id}")
        
        return True
    

    @staticmethod
    def validate_chunk_id(chunk_id: str) -> bool:
        """
        Validate chunk ID format
        """
        pattern = r'^chunk_doc_\d+_[a-f0-9]{8}_\d+$'
        
        if (not re.match(pattern, chunk_id)):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        return True


# Convenience functions
def validate_upload_file(file_path: Path) -> bool:
    """
    Comprehensive upload file validation
    """
    return (FileValidator.validate_file_integrity(file_path) and
            FileValidator.validate_file_size(file_path) and
            FileValidator.validate_file_type(file_path) and
            FileValidator.validate_filename(file_path.name)
           )


def validate_query_text(text: str) -> bool:
    """
    Validate query text for processing
    """
    return (TextValidator.validate_text_length(text, min_length = 1, max_length = 1000) and
            TextValidator.is_meaningful_text(text, min_words = 1)
           )