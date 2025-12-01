# DEPENDENCIES
import hashlib
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional
from datetime import datetime
from config.models import DocumentType
from utils.text_cleaner import TextCleaner
from config.models import DocumentMetadata
from config.logging_config import get_logger
from utils.error_handler import PDFParseError
from utils.error_handler import handle_errors
from document_parser.ocr_engine import OCREngine

try:
    import fitz 
    PYMUPDF_AVAILABLE = True
    
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    from PyPDF2 import PdfReader
    PYPdf2_AVAILABLE = True

except ImportError:
    PYPdf2_AVAILABLE = False


# Setup Logging
logger = get_logger(__name__)


class PDFParser:
    """
    Comprehensive PDF parsing with metadata extraction: Uses PyMuPDF (fitz) as primary parser with PyPDF2 fallback
    
    Handles various PDF formats including encrypted and scanned documents
    """
    def __init__(self, prefer_pymupdf: bool = True):
        """
        Initialize PDF parser.
        
        Arguments:
        ----------
            prefer_pymupdf { bool } : Use PyMuPDF as primary parser if available
        """
        self.logger         = logger
        self.prefer_pymupdf = prefer_pymupdf and PYMUPDF_AVAILABLE
        self.ocr_engine     = None  
        
        try:
            from document_parser.ocr_engine import OCREngine
            self.ocr_available = True

        except ImportError:
            self.ocr_available = False
            self.logger.warning("OCR engine not available - scanned PDFs may not be processed")

        if (not PYMUPDF_AVAILABLE and not PYPdf2_AVAILABLE):
            raise ImportError("Neither PyMuPDF nor PyPDF2 are available. Please install at least one.")
        
        self.logger.info(f"PDF Parser initialized - Primary: {'PyMuPDF' if self.prefer_pymupdf else 'PyPDF2'}, PyMuPDF available: {PYMUPDF_AVAILABLE}, PyPDF2 available: {PYPdf2_AVAILABLE}")
    

    @handle_errors(error_type=PDFParseError, log_error = True, reraise = True)
    def parse(self, file_path: Path, extract_metadata: bool = True, clean_text: bool = True, password: Optional[str] = None) -> tuple[str, Optional[DocumentMetadata]]:
        """
        Parse PDF and extract text and metadata : tries PyMuPDF first, falls back to PyPDF2 if needed
        
        Arguments:
        ----------
            file_path        { Path } : Path to PDF file
            
            extract_metadata { bool } : Extract document metadata
            
            clean_text       { bool } : Clean extracted text
            
            password         { str  } : Password for encrypted PDFs
        
        Returns:
        --------
                   { tuple }          : Tuple of (extracted_text, metadata)
        
        Raises:
        -------
            PDFParseError             : If parsing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PDFParseError(str(file_path), original_error = FileNotFoundError(f"PDF file not found: {file_path}"))
        
        self.logger.info(f"Parsing PDF: {file_path}")
        
        # Try PyMuPDF first if preferred and available
        if (self.prefer_pymupdf and PYMUPDF_AVAILABLE):
            try:
                parsed_text = self._parse_with_pymupdf(file_path        = file_path, 
                                                       extract_metadata = extract_metadata, 
                                                       clean_text       = clean_text, 
                                                       password         = password,
                                                      )

                return parsed_text
            
            except Exception as e:
                self.logger.warning(f"PyMuPDF parsing failed for {file_path}, falling back to PyPDF2: {repr(e)}")
        
        # Fall back to PyPDF2
        if PYPdf2_AVAILABLE:
            try:
                parsed_text = self._parse_with_pypdf2(file_path        = file_path, 
                                                      extract_metadata = extract_metadata, 
                                                      clean_text       = clean_text,
                                                      password         = password,
                                                     )

                return parsed_text
            
            except Exception as e:
                self.logger.error(f"PyPDF2 parsing also failed for {file_path}: {repr(e)}")
                raise PDFParseError(str(file_path), original_error = e)
        
        else:
            raise PDFParseError(str(file_path), original_error = RuntimeError("No PDF parsing libraries available"))

    
    def _parse_with_pymupdf(self, file_path: Path, extract_metadata: bool = True, clean_text: bool = True, password: Optional[str] = None) -> tuple[str, Optional[DocumentMetadata]]:
        """
        Parse PDF using PyMuPDF (fitz) with OCR fallback for scanned documents
        """
        self.logger.debug(f"Using PyMuPDF for parsing: {file_path}")
        
        doc = None

        try:
            # Open PDF with PyMuPDF
            self.logger.debug(f"Opening document: {file_path}")
            doc = fitz.open(str(file_path))
            
            self.logger.debug(f"Document opened successfully, {len(doc)} pages")
            
            # Handle encrypted PDFs
            if (doc.needs_pass and password):
                if not doc.authenticate(password):
                    raise PDFParseError(str(file_path), original_error = ValueError("Invalid password for encrypted PDF"))
            
            elif (doc.needs_pass and not password):
                raise PDFParseError(str(file_path), original_error = ValueError("PDF is encrypted but no password provided"))
            
            # Extract text with per-page OCR fallback
            text_content = self._extract_text_with_pymupdf(doc       = doc, 
                                                           file_path = file_path,
                                                          )

            # Extract metadata
            metadata = None
            if extract_metadata:
                metadata = self._extract_metadata_with_pymupdf(doc       = doc, 
                                                               file_path = file_path,
                                                              )
            
            # Clean text
            if clean_text:
                text_content = TextCleaner.clean(text_content,
                                                 remove_html          = True,
                                                 normalize_whitespace = True,
                                                 preserve_structure   = True,
                                                )
            
            self.logger.info(f"Successfully parsed PDF with PyMuPDF: {len(text_content)} characters, {len(doc)} pages")
            return text_content, metadata
            
        except Exception as e:
            self.logger.error(f"PyMuPDF parsing failed for {file_path}: {repr(e)}")
            raise
        
        finally:
            # Always close the document in finally block
            if doc:
                self.logger.debug("Closing PyMuPDF document")
                doc.close()

    def _parse_with_pypdf2(self, file_path: Path, extract_metadata: bool = True, clean_text: bool = True, password: Optional[str] = None) -> tuple[str, Optional[DocumentMetadata]]:
        """
        Parse PDF using PyPDF2
        
        Arguments:
        ----------
            file_path         { Path } : Path to PDF file
            
            extract_metadata  { bool } : Extract document metadata
            
            clean_text        { bool } : Clean extracted text
            
            password           { str } : Password for encrypted PDFs
        
        Returns:
        --------
                  { tuple }            : Tuple of (extracted_text, metadata)
        """
        self.logger.debug(f"Using PyPDF2 for parsing: {file_path}")
        
        try:
            # Open PDF with PyPDF2
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                
                # Handle encrypted PDFs
                if reader.is_encrypted:
                    if password:
                        reader.decrypt(password)
                        self.logger.info("Successfully decrypted PDF with PyPDF2")
                    
                    else:
                        raise PDFParseError(str(file_path), original_error = ValueError("PDF is encrypted but no password provided"))
                
                # Extract text from all pages
                text_content = self._extract_text_with_pypdf2(reader = reader)
                
                # Extract metadata
                metadata     = None

                if extract_metadata:
                    metadata = self._extract_metadata_with_pypdf2(reader    = reader, 
                                                                  file_path = file_path,
                                                                 )
                
                # Clean text
                if clean_text:
                    text_content = TextCleaner.clean(text_content,
                                                     remove_html          = True,
                                                     normalize_whitespace = True,
                                                     preserve_structure   = True,
                                                    )
                
                self.logger.info(f"Successfully parsed PDF with PyPDF2: {len(text_content)} characters, {len(reader.pages)} pages")
                
                return text_content, metadata
                
        except Exception as e:
            self.logger.error(f"PyPDF2 parsing failed for {file_path}: {repr(e)}")
            raise

    
    def _extract_text_with_pymupdf(self, doc: "fitz.Document", file_path: Path = None) -> str:
        """
        Extract text from all pages using PyMuPDF with per-page OCR fallback.
        
        Arguments:
        ----------
            doc        : PyMuPDF document object
            
            file_path  : Path to PDF file (for OCR fallback)
        
        Returns:
        --------
            { str }    : Combined text from all pages
        """
        text_parts = list()
        
        for page_num in range(len(doc)):
            try:
                page      = doc[page_num]
                page_text = page.get_text()
                
                if page_text and page_text.strip():
                    # Add page marker for citation purposes
                    text_parts.append(f"\n[PAGE {page_num + 1}]\n{page_text}")
                    self.logger.debug(f"Extracted {len(page_text)} chars from page {page_num + 1} with PyMuPDF")
                
                else:
                    # No text extracted - this page might be scanned
                    self.logger.warning(f"No text extracted from page {page_num + 1} with PyMuPDF (might be scanned)")
                    
                    # Try OCR for this specific page if available
                    if self.ocr_available and file_path:
                        try:
                            self.logger.info(f"Attempting OCR for page {page_num + 1}")
                            ocr_text = self._extract_page_text_with_ocr(file_path, page_num + 1)
                            
                            if ocr_text and ocr_text.strip():
                                text_parts.append(f"\n[PAGE {page_num + 1} - OCR]\n{ocr_text}")
                                self.logger.info(f"OCR extracted {len(ocr_text)} chars from page {page_num + 1}")
                            
                            else:
                                text_parts.append(f"\n[PAGE {page_num + 1} - NO TEXT]\n")
                                self.logger.warning(f"OCR also failed to extract text from page {page_num + 1}")
                        
                        except Exception as ocr_error:
                            self.logger.warning(f"OCR failed for page {page_num + 1}: {repr(ocr_error)}")
                            text_parts.append(f"\n[PAGE {page_num + 1} - OCR FAILED]\n")
                    
                    else:
                        # No OCR available or no file_path provided
                        text_parts.append(f"\n[PAGE {page_num + 1} - NO TEXT]\n")
            
            except Exception as e:
                self.logger.warning(f"Error extracting text from page {page_num + 1} with PyMuPDF: {repr(e)}")
                text_parts.append(f"\n[PAGE {page_num + 1} - ERROR: {str(e)}]\n")
                continue
        
        return "\n".join(text_parts)
    

    def _extract_text_with_pypdf2(self, reader: PdfReader) -> str:
        """
        Extract text from all pages using PyPDF2
        
        Arguments:
        ----------
            reader { PdfReader } : PdfReader object
        
        Returns:
        --------
                 { str }         : Combined text from all pages
        """
        text_parts = list()
        num_pages  = len(reader.pages)
        
        for page_num in range(num_pages):
            try:
                page      = reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text and page_text.strip():
                    # Add page marker for citation purposes
                    text_parts.append(f"\n[PAGE {page_num + 1}]\n{page_text}")
                    self.logger.debug(f"Extracted {len(page_text)} chars from page {page_num + 1} with PyPDF2")
                
                else:
                    self.logger.warning(f"No text extracted from page {page_num + 1} with PyPDF2 (might be scanned)")
            
            except Exception as e:
                self.logger.warning(f"Error extracting text from page {page_num + 1} with PyPDF2: {repr(e)}")
                continue
        
        return "\n".join(text_parts)

    
    def _extract_metadata_with_pymupdf(self, doc: "fitz.Document", file_path: Path) -> DocumentMetadata:
        """
        Extract metadata using PyMuPDF
        
        Arguments:
        -----------
            doc       { fitz.Document } : PyMuPDF document object

            file_path      { Path }     : Path to PDF file
        
        Returns:
        --------
              { DocumentMetadata }      : DocumentMetadata object
        """
        # Get PDF metadata
        pdf_metadata  = doc.metadata
        
        # Extract common fields
        title         = pdf_metadata.get('title', '').strip()
        author        = pdf_metadata.get('author', '').strip()
        
        # Parse dates
        created_date  = self._parse_pdf_date(pdf_metadata.get('creationDate'))
        modified_date = self._parse_pdf_date(pdf_metadata.get('modDate'))
        
        # Get file size
        file_size     = file_path.stat().st_size
        
        # Count pages
        num_pages     = len(doc)
        
        # Generate document ID
        doc_hash      = hashlib.md5(str(file_path).encode()).hexdigest()
        doc_id        = f"doc_{int(datetime.now().timestamp())}_{doc_hash}"
        
        # Create metadata object
        metadata      = DocumentMetadata(document_id     = doc_id,
                                         filename        = file_path.name,
                                         file_path       = file_path,
                                         document_type   = DocumentType.PDF,
                                         title           = title or file_path.stem,
                                         author          = author,
                                         created_date    = created_date,
                                         modified_date   = modified_date,
                                         file_size_bytes = file_size,
                                         num_pages       = num_pages,
                                         extra           = {"pdf_version"  : pdf_metadata.get('producer', ''),
                                                            "pdf_metadata" : {k: str(v) for k, v in pdf_metadata.items() if v},
                                                            "parser_used"  : "pymupdf"
                                                           }
                                        )
        
        return metadata

    
    def _extract_metadata_with_pypdf2(self, reader: PdfReader, file_path: Path) -> DocumentMetadata:
        """
        Extract metadata using PyPDF2
        
        Arguments:
        ----------
            reader    { PdfReader } : PdfReader object

            file_path    { Path }   : Path to PDF file
        
        Returns:
        --------
            { DocumentMetadata }    : DocumentMetadata object
        """
        # Get PDF metadata
        pdf_info      = reader.metadata if reader.metadata else {}
        
        # Extract common fields
        title         = self._get_metadata_field(pdf_info, ['/Title', 'title'])
        author        = self._get_metadata_field(pdf_info, ['/Author', 'author'])
        
        # Parse dates
        created_date  = self._parse_pdf_date(self._get_metadata_field(pdf_info, ['/CreationDate', 'creation_date']))
        modified_date = self._parse_pdf_date(self._get_metadata_field(pdf_info, ['/ModDate', 'mod_date']))
        
        # Get file size
        file_size     = file_path.stat().st_size
        
        # Count pages
        num_pages     = len(reader.pages)
        
        # Generate document ID
        doc_hash      = hashlib.md5(str(file_path).encode()).hexdigest()
        doc_id        = f"doc_{int(datetime.now().timestamp())}_{doc_hash}"
        
        # Create metadata object
        metadata      = DocumentMetadata(document_id     = doc_id,
                                         filename        = file_path.name,
                                         file_path       = file_path,
                                         document_type   = DocumentType.PDF,
                                         title           = title or file_path.stem,
                                         author          = author,
                                         created_date    = created_date,
                                         modified_date   = modified_date,
                                         file_size_bytes = file_size,
                                         num_pages       = num_pages,
                                         extra           = {"pdf_version"  : self._get_metadata_field(pdf_info, ['/Producer', 'producer']),
                                                            "pdf_metadata" : {k: str(v) for k, v in pdf_info.items() if v},
                                                            "parser_used"  : "pypdf2",
                                                           }
                                        )
        
        return metadata


    def _extract_text_with_ocr(self, file_path: Path) -> str:
        """
        Extract text from scanned PDF using OCR
        """
        if not self.ocr_available:
            raise PDFParseError(str(file_path), original_error = RuntimeError("OCR engine not available"))
        
        if self.ocr_engine is None:
            self.ocr_engine = OCREngine()
        
        return self.ocr_engine.extract_text_from_pdf(file_path)

    
    def _extract_page_text_with_ocr(self, file_path: Path, page_number: int) -> str:
        """
        Extract text from a specific page using OCR
        
        Arguments:
        ----------
            file_path   { Path }  : Path to PDF file

            page_number { int }   : Page number (1-indexed)
        
        Returns:
        --------
            { str }               : Extracted text from the page
        """
        if not self.ocr_available:
            raise PDFParseError(str(file_path), original_error = RuntimeError("OCR engine not available"))
        
        if self.ocr_engine is None:
            self.ocr_engine = OCREngine()
        
        try:
            # Use OCR engine to extract text from specific page
            return self.ocr_engine.extract_text_from_pdf(pdf_path = file_path, 
                                                         pages    = [page_number],
                                                        )
        
        except Exception as e:
            self.logger.error(f"OCR failed for page {page_number}: {repr(e)}")
            return ""
            

    @staticmethod
    def _get_metadata_field(metadata: Dict, field_names: List[str]) -> Optional[str]:
        """
        Get metadata field with fallback names
        
        Arguments:
        ----------
            metadata    { dict } : Metadata dictionary

            field_names { list } : List of possible field names
        
        Returns:
        --------
                 { str }         : Field value or None
        """
        for field_name in field_names:
            if field_name in metadata:
                value = metadata[field_name]

                if value:
                    return str(value).strip()
                    
        return None
    

    @staticmethod
    def _parse_pdf_date(date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse PDF date format : PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
        
        Arguments:
        ----------
            date_str { str } : PDF date string
        
        Returns:
        --------
             { datetime }    : Datetime object or None
        """
        if not date_str:
            return None
        
        try:
            # Remove 'D:' prefix if present
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Parse basic format: YYYYMMDDHHMMSS
            date_str = date_str[:14]
            
            return datetime.strptime(date_str, '%Y%m%d%H%M%S')

        except Exception:
            return None
    

    def extract_page_text(self, file_path: Path, page_number: int, clean_text: bool = True) -> str:
        """
        Extract text from a specific page
        
        Arguments:
        ----------
            file_path   { Path } : Path to PDF file
            
            page_number { int }  : Page number (1-indexed)
            
            clean_text  { bool } : Clean extracted text
        
        Returns:
        --------
                  { str }        : Page text
        """
        # Try PyMuPDF first if preferred and available
        if self.prefer_pymupdf and PYMUPDF_AVAILABLE:
            try:
                page_text = self._extract_page_text_pymupdf(file_path   = file_path, 
                                                            page_number = page_number, 
                                                            clean_text  = clean_text,
                                                           )
                return page_text
            
            except Exception as e:
                self.logger.warning(f"PyMuPDF page extraction failed, falling back to PyPDF2: {repr(e)}")
        
        # Fall back to PyPDF2
        if PYPdf2_AVAILABLE:
            page_text = self._extract_page_text_pypdf2(file_path   = file_path, 
                                                       pagse_number = page_number, 
                                                       clean_text  = clean_text,
                                                      )
            
            return page_text

        else:
            raise PDFParseError(str(file_path), original_error = RuntimeError("No PDF parsing libraries available"))

    
    def _extract_page_text_pymupdf(self, file_path: Path, page_number: int, clean_text: bool = True) -> str:
        """
        Extract page text using PyMuPDF
        """
        doc = None
        try:
            doc       = fitz.open(str(file_path))
            num_pages = len(doc)
            
            if ((page_number < 1) or (page_number > num_pages)):
                raise ValueError(f"Page number {page_number} out of range (1-{num_pages})")
            
            page      = doc[page_number - 1]
            page_text = page.get_text()
            
            if clean_text:
                page_text = TextCleaner.clean(page_text)
            
            return page_text
            
        except Exception as e:
            self.logger.error(f"Failed to extract page {page_number} with PyMuPDF: {repr(e)}")
            raise PDFParseError(str(file_path), original_error = e)
            
        finally:
            if doc:
                doc.close()
    
    
    def _extract_page_text_pypdf2(self, file_path: Path, page_number: int, clean_text: bool = True) -> str:
        """
        Extract page text using PyPDF2
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                reader    = PdfReader(pdf_file)
                num_pages = len(reader.pages)
                
                if ((page_number < 1) or (page_number > num_pages)):
                    raise ValueError(f"Page number {page_number} out of range (1-{num_pages})")
                
                page      = reader.pages[page_number - 1]
                page_text = page.extract_text()
                
                if clean_text:
                    page_text = TextCleaner.clean(page_text)
                
                return page_text
        
        except Exception as e:
            self.logger.error(f"Failed to extract page {page_number} with PyPDF2: {repr(e)}")
            raise PDFParseError(str(file_path), original_error = e)
    

    def get_page_count(self, file_path: Path) -> int:
        """
        Get number of pages in PDF
        
        Arguments:
        ----------
            file_path { Path } : Path to PDF file
        
        Returns:
        --------
                { int }        : Number of pages
        """
        # Try PyMuPDF first if available
        if PYMUPDF_AVAILABLE:
            doc = None
            try:
                doc        = fitz.open(str(file_path))
                page_count = len(doc)
                
                return page_count
            
            except Exception as e:
                self.logger.warning(f"PyMuPDF page count failed, trying PyPDF2: {repr(e)}")
            
            finally:
                if doc:
                    doc.close()
        
        # Fall back to PyPDF2
        if PYPdf2_AVAILABLE:
            try:
                with open(file_path, 'rb') as pdf_file:
                    reader = PdfReader(pdf_file)
                    
                    return len(reader.pages)

            except Exception as e:
                self.logger.error(f"Failed to get page count: {repr(e)}")
                raise PDFParseError(str(file_path), original_error = e)
        else:
            raise PDFParseError(str(file_path), original_error = RuntimeError("No PDF parsing libraries available"))

    
    def extract_page_range(self, file_path: Path, start_page: int, end_page: int, clean_text: bool = True) -> str:
        """
        Extract text from a range of pages
        
        Arguments:
        ----------
            file_path  { Path } : Path to PDF file
            
            start_page { int }  : Starting page (1-indexed, inclusive)
            
            end_page   { int }  : Ending page (1-indexed, inclusive)
            
            clean_text { bool } : Clean extracted text
        
        Returns:
        --------
                  { str }       : Combined text from pages
        """
        # Try PyMuPDF first if preferred and available
        if self.prefer_pymupdf and PYMUPDF_AVAILABLE:
            try:
                page_range = self._extract_page_range_pymupdf(file_path  = file_path, 
                                                              start_page = start_page, 
                                                              end_page   = end_page, 
                                                              clean_text = clean_text,
                                                             )
                return page_range
            
            except Exception as e:
                self.logger.warning(f"PyMuPDF page range extraction failed, falling back to PyPDF2: {repr(e)}")
        
        # Fall back to PyPDF2
        if PYPdf2_AVAILABLE:
            page_range = self._extract_page_range_pypdf2(file_path  = file_path, 
                                                         start_page = start_page, 
                                                         end_page   = end_page, 
                                                         clean_text = clean_text,
                                                        )
            
            return page_range

        else:
            raise PDFParseError(str(file_path), original_error = RuntimeError("No PDF parsing libraries available"))
    

    def _extract_page_range_pymupdf(self, file_path: Path, start_page: int, end_page: int, clean_text: bool = True) -> str:
        """
        Extract page range using PyMuPDF
        """
        doc = None
        try:
            doc       = fitz.open(str(file_path))
            num_pages = len(doc)
            
            if ((start_page < 1) or (end_page > num_pages) or (start_page > end_page)):
                raise ValueError(f"Invalid page range {start_page}-{end_page} for PDF with {num_pages} pages")
            
            text_parts = list()

            for page_num in range(start_page - 1, end_page):
                page      = doc[page_num]
                page_text = page.get_text()
                
                if page_text:
                    text_parts.append(f"\n[PAGE {page_num + 1}]\n{page_text}")
            
            combined_text = "\n".join(text_parts)
            
            if clean_text:
                combined_text = TextCleaner.clean(combined_text)
            
            return combined_text
            
        except Exception as e:
            self.logger.error(f"Failed to extract page range with PyMuPDF: {repr(e)}")
            raise PDFParseError(str(file_path), original_error = e)
            
        finally:
            if doc:
                doc.close()
    

    def _extract_page_range_pypdf2(self, file_path: Path, start_page: int, end_page: int, clean_text: bool = True) -> str:
        """
        Extract page range using PyPDF2
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                reader    = PdfReader(pdf_file)
                num_pages = len(reader.pages)
                
                if ((start_page < 1) or (end_page > num_pages) or (start_page > end_page)):
                    raise ValueError(f"Invalid page range {start_page}-{end_page} for PDF with {num_pages} pages")
                
                text_parts = list()

                for page_num in range(start_page - 1, end_page):
                    page      = reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_parts.append(f"\n[PAGE {page_num + 1}]\n{page_text}")
                
                combined_text = "\n".join(text_parts)
                
                if clean_text:
                    combined_text = TextCleaner.clean(combined_text)
                
                return combined_text
        
        except Exception as e:
            self.logger.error(f"Failed to extract page range with PyPDF2: {repr(e)}")
            raise PDFParseError(str(file_path), original_error = e)
    

    def is_scanned(self, file_path: Path) -> bool:
        """
        Check if PDF is scanned (image-based): Scanned PDFs have very little or no extractable text
        
        Arguments:
        ----------
            file_path { Path } : Path to PDF file
        
        Returns:
        --------
                { bool }       : True if appears to be scanned
        """
        # Try PyMuPDF first if available for better detection
        if PYMUPDF_AVAILABLE:
            try:
                return self._is_scanned_pymupdf(file_path = file_path)

            except Exception as e:
                self.logger.warning(f"PyMuPDF scanned detection failed, trying PyPDF2: {repr(e)}")
        
        # Fall back to PyPDF2
        if PYPdf2_AVAILABLE:
            return self._is_scanned_pypdf2(file_path = file_path)
        
        else:
            self.logger.warning("No PDF parsing libraries available for scanned detection")
            return False
    

    def _is_scanned_pymupdf(self, file_path: Path) -> bool:
        """
        Check if PDF is scanned using PyMuPDF
        """
        doc = None
        try:
            doc               = fitz.open(str(file_path))
            
            # Sample first 3 pages
            pages_to_check    = min(3, len(doc))
            total_text_length = 0
            
            for i in range(pages_to_check):
                page               = doc[i]
                text               = page.get_text()
                total_text_length += len(text.strip())
            
            # If average text per page is very low, likely scanned
            avg_text_per_page = total_text_length / pages_to_check

            # characters per page
            threshold         = 100 
            
            is_scanned        = (avg_text_per_page < threshold)
            
            if is_scanned:
                self.logger.info(f"PDF appears to be scanned (avg {avg_text_per_page:.0f} chars/page)")
            
            return is_scanned
            
        except Exception as e:
            self.logger.warning(f"Could not determine if PDF is scanned with PyMuPDF: {repr(e)}")
            return False
            
        finally:
            if doc:
                doc.close()
    

    def _is_scanned_pypdf2(self, file_path: Path) -> bool:
        """
        Check if PDF is scanned using PyPDF2
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                reader            = PdfReader(pdf_file)
                
                # Sample first 3 pages
                pages_to_check    = min(3, len(reader.pages))
                total_text_length = 0
                
                for i in range(pages_to_check):
                    page               = reader.pages[i]
                    text               = page.extract_text()
                    total_text_length += len(text.strip())
                
                # If average text per page is very low, likely scanned
                avg_text_per_page = total_text_length / pages_to_check

                # characters per page
                threshold         = 100  
                
                is_scanned        = (avg_text_per_page < threshold)
                
                if is_scanned:
                    self.logger.info(f"PDF appears to be scanned (avg {avg_text_per_page:.0f} chars/page)")
                
                return is_scanned
        
        except Exception as e:
            self.logger.warning(f"Could not determine if PDF is scanned with PyPDF2: {repr(e)}")
            return False