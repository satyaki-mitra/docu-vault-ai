# DEPENDENCIES
import chardet
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime
from config.models import DocumentType
from utils.text_cleaner import TextCleaner
from config.models import DocumentMetadata 
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import TextEncodingError


# Setup Logging
logger = get_logger(__name__)


class TXTParser:
    """
    Plain text file parser with automatic encoding detection : handles various text encodings and formats
    """
    # Common encodings to try
    COMMON_ENCODINGS = ['utf-8', 
                        'utf-16', 
                        'ascii', 
                        'latin-1', 
                        'cp1252', 
                        'iso-8859-1',
                       ]
    
    def __init__(self):
        self.logger = logger

    
    @handle_errors(error_type = TextEncodingError, log_error = True, reraise = True)
    def parse(self, file_path: Path, extract_metadata: bool = True, clean_text: bool = True, encoding: Optional[str] = None) -> tuple[str, Optional[DocumentMetadata]]:
        """
        Parse text file and extract content
        
        Arguments:
        -----------
            file_path        { Path } : Path to text file
            
            extract_metadata { bool } : Extract document metadata
            
            clean_text       { bool } : Clean extracted text
            
            encoding         { str }  : Force specific encoding (None = auto-detect)
        
        Returns:
        --------
                  { tuple }           : Tuple of (extracted_text, metadata)
        
        Raises:
        -------
            TextEncodingError         : If file cannot be decoded
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise TextEncodingError(str(file_path), encoding = "unknown", original_error = FileNotFoundError(f"Text file not found: {file_path}"))
        
        self.logger.info(f"Parsing TXT: {file_path}")
        
        # Detect encoding if not specified
        if encoding is None:
            encoding = self.detect_encoding(file_path)
            self.logger.info(f"Detected encoding: {encoding}")
        
        try:
            # Read file with detected/specified encoding
            with open(file_path, 'r', encoding = encoding, errors = 'replace') as f:
                text_content = f.read()
            
            # Extract metadata
            metadata = None

            if extract_metadata:
                metadata = self._extract_metadata(file_path   = file_path, 
                                                  encoding    = encoding, 
                                                  text_length = len(text_content),
                                                 )
            
            # Clean text
            if clean_text:
                text_content = TextCleaner.clean(text_content,
                                                 remove_html          = False,
                                                 normalize_whitespace = True,
                                                 preserve_structure   = True,
                                                )
            
            self.logger.info(f"Successfully parsed TXT: {len(text_content)} characters")
            
            return text_content, metadata
        
        except Exception as e:
            self.logger.error(f"Failed to parse TXT {file_path}: {repr(e)}")
            raise TextEncodingError(str(file_path), encoding = encoding, original_error = e)
    

    def detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding using chardet
        
        Arguments:
        ----------
            file_path { Path } : Path to text file
        
        Returns:
        --------
                { str }        : Detected encoding name
        """
        try:
            # Read raw bytes
            with open(file_path, 'rb') as f:
                # Read first 10KB for detection
                raw_data = f.read(10000)  
            
            # Detect encoding
            result     = chardet.detect(raw_data)
            encoding   = result['encoding']
            confidence = result['confidence']
            
            self.logger.debug(f"Encoding detection: {encoding} (confidence: {confidence:.2%})")
            
            # If confidence is low, try common encodings
            if (confidence < 0.7):
                self.logger.warning(f"Low confidence ({confidence:.2%}) for detected encoding {encoding}")
                encoding = self._try_common_encodings(file_path = file_path)
            
            # Fallback to UTF-8
            return encoding or 'utf-8'  
        
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {repr(e)}, using UTF-8")
            return 'utf-8'
    

    def _try_common_encodings(self, file_path: Path) -> Optional[str]:
        """
        Try reading file with common encodings
        
        Arguments:
        ----------
            file_path { Path } : Path to text file
        
        Returns:
        --------
                { str }        : Working encoding or None
        """
        for encoding in self.COMMON_ENCODINGS:
            try:
                with open(file_path, 'r', encoding = encoding) as f:
                    # Try reading first 1000 chars
                    f.read(1000)  

                self.logger.info(f"Successfully read with encoding: {encoding}")
                return encoding
            
            except (UnicodeDecodeError, LookupError):
                continue
        
        return None
    

    def _extract_metadata(self, file_path: Path, encoding: str, text_length: int) -> DocumentMetadata:
        """
        Extract metadata from text file
        
        Arguments:
        ----------
            file_path   { Path } : Path to text file

            encoding    { str }  : File encoding
            
            text_length { int }  : Length of text content
        
        Returns:
        --------
            { DocumentMetadata } : DocumentMetadata object
        """
        # Get file stats
        stat            = file_path.stat()
        file_size       = stat.st_size
        created_time    = datetime.fromtimestamp(stat.st_ctime)
        modified_time   = datetime.fromtimestamp(stat.st_mtime)
        
        # Generate document ID
        doc_hash        = hashlib.md5(str(file_path).encode()).hexdigest()
        doc_id          = f"doc_{int(datetime.now().timestamp())}_{doc_hash}"
        
        # Estimate pages (rough: 3000 characters per page)
        estimated_pages = max(1, text_length // 3000)
        
        # Count lines
        with open(file_path, 'r', encoding = encoding, errors = 'replace') as f:
            num_lines = sum(1 for _ in f)
        
        # Create metadata object
        metadata = DocumentMetadata(document_id     = doc_id,
                                    filename        = file_path.name,
                                    file_path       = file_path,
                                    document_type   = DocumentType.TXT,
                                    title           = file_path.stem,
                                    created_date    = created_time,
                                    modified_date   = modified_time,
                                    file_size_bytes = file_size,
                                    num_pages       = estimated_pages,
                                    extra           = {"encoding"    : encoding,
                                                       "num_lines"   : num_lines,
                                                       "text_length" : text_length,
                                                      }
                                   )
        
        return metadata

    
    def read_lines(self, file_path: Path, start_line: int = 0, end_line: Optional[int] = None, encoding: Optional[str] = None) -> list[str]:
        """
        Read specific lines from file
        
        Arguments:
        -----------
            file_path  { Path } : Path to text file

            start_line { int }  : Starting line (0-indexed)
            
            end_line   { int }  : Ending line (None = end of file)
            
            encoding   { str }  : File encoding (None = auto-detect)
        
        Returns:
        --------
                { list }        : List of lines
        """
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding = encoding, errors = 'replace') as f:
                lines = f.readlines()
            
            if end_line is None:
                return lines[start_line:]
            
            else:
                return lines[start_line:end_line]
        
        except Exception as e:
            self.logger.error(f"Failed to read lines: {repr(e)}")
            raise TextEncodingError(str(file_path), encoding = encoding, original_error = e)
    

    def count_lines(self, file_path: Path, encoding: Optional[str] = None) -> int:
        """
        Count number of lines in file
        
        Arguments:
        ----------
            file_path { Path } : Path to text file
            
            encoding   { str } : File encoding (None = auto-detect)
        
        Returns:
        --------
                { int }        : Number of lines
        """
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding = encoding, errors = 'replace') as f:
                return sum(1 for _ in f)
        
        except Exception as e:
            self.logger.error(f"Failed to count lines: {repr(e)}")
            raise TextEncodingError(str(file_path), encoding = encoding, original_error = e)
    

    def get_file_info(self, file_path: Path) -> dict:
        """
        Get comprehensive file information
        
        Arguments:
        ----------
            file_path { Path } : Path to text file
        
        Returns:
        --------
               { dict }        : Dictionary with file info
        """
        encoding = self.detect_encoding(file_path)
        
        with open(file_path, 'r', encoding = encoding, errors = 'replace') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        return {"encoding"        : encoding,
                "size_bytes"      : file_path.stat().st_size,
                "num_lines"       : len(lines),
                "num_characters"  : len(content),
                "num_words"       : len(content.split()),
                "avg_line_length" : sum(len(line) for line in lines) / len(lines) if lines else 0,
               }

    
    def is_empty(self, file_path: Path) -> bool:
        """
        Check if file is empty or contains only whitespace
        
        Arguments:
        ----------
            file_path { Path } : Path to text file
        
        Returns:
        --------
               { bool }        : True if empty
        """
        try:
            # Check file size first
            if file_path.stat().st_size == 0:
                return True
            
            # Read and check content
            encoding = self.detect_encoding(file_path)
            
            with open(file_path, 'r', encoding = encoding, errors = 'replace') as f:
                content = f.read().strip()
            
            return len(content) == 0

        except Exception as e:
            self.logger.warning(f"Error checking if file is empty: {repr(e)}")
            return True
