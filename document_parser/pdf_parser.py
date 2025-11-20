"""
PDF Parser
Extract text, metadata, and structure from PDF documents
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import PyPDF2
from PyPDF2 import PdfReader

from config.logging_config import get_logger
from config.models import DocumentMetadata, DocumentType
from utils.error_handler import PDFParseError, handle_errors
from utils.text_cleaner import TextCleaner

logger = get_logger(__name__)


class PDFParser:
    """
    Comprehensive PDF parsing with metadata extraction.
    Handles various PDF formats including encrypted and scanned documents.
    """
    
    def __init__(self):
        self.logger = logger
    
    @handle_errors(error_type=PDFParseError, log_error=True, reraise=True)
    def parse(
        self,
        file_path: Path,
        extract_metadata: bool = True,
        clean_text: bool = True,
        password: Optional[str] = None
    ) -> tuple[str, Optional[DocumentMetadata]]:
        """
        Parse PDF and extract text and metadata.
        
        Args:
            file_path: Path to PDF file
            extract_metadata: Extract document metadata
            clean_text: Clean extracted text
            password: Password for encrypted PDFs
        
        Returns:
            Tuple of (extracted_text, metadata)
        
        Raises:
            PDFParseError: If parsing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PDFParseError(
                str(file_path),
                original_error=FileNotFoundError(f"PDF file not found: {file_path}")
            )
        
        self.logger.info(f"Parsing PDF: {file_path}")
        
        try:
            # Open PDF
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                
                # Handle encrypted PDFs
                if reader.is_encrypted:
                    if password:
                        reader.decrypt(password)
                        self.logger.info("Successfully decrypted PDF")
                    else:
                        raise PDFParseError(
                            str(file_path),
                            original_error=ValueError("PDF is encrypted but no password provided")
                        )
                
                # Extract text from all pages
                text_content = self._extract_text_from_pages(reader)
                
                # Extract metadata
                metadata = None
                if extract_metadata:
                    metadata = self._extract_metadata(reader, file_path)
                
                # Clean text
                if clean_text:
                    text_content = TextCleaner.clean(
                        text_content,
                        remove_html=True,
                        normalize_whitespace=True,
                        preserve_structure=True
                    )
                
                self.logger.info(
                    f"Successfully parsed PDF: {len(text_content)} characters, "
                    f"{reader.numPages if hasattr(reader, 'numPages') else len(reader.pages)} pages"
                )
                
                return text_content, metadata
        
        except Exception as e:
            self.logger.error(f"Failed to parse PDF {file_path}: {str(e)}")
            raise PDFParseError(str(file_path), original_error=e)
    
    def _extract_text_from_pages(self, reader: PdfReader) -> str:
        """
        Extract text from all pages.
        
        Args:
            reader: PdfReader object
        
        Returns:
            Combined text from all pages
        """
        text_parts = []
        num_pages = len(reader.pages)
        
        for page_num in range(num_pages):
            try:
                page = reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text and page_text.strip():
                    # Add page marker for citation purposes
                    text_parts.append(f"\n[PAGE {page_num + 1}]\n{page_text}")
                    self.logger.debug(f"Extracted {len(page_text)} chars from page {page_num + 1}")
                else:
                    self.logger.warning(f"No text extracted from page {page_num + 1} (might be scanned)")
            
            except Exception as e:
                self.logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        return "\n".join(text_parts)
    
    def _extract_metadata(
        self,
        reader: PdfReader,
        file_path: Path
    ) -> DocumentMetadata:
        """
        Extract metadata from PDF.
        
        Args:
            reader: PdfReader object
            file_path: Path to PDF file
        
        Returns:
            DocumentMetadata object
        """
        # Get PDF metadata
        pdf_info = reader.metadata if reader.metadata else {}
        
        # Extract common fields
        title = self._get_metadata_field(pdf_info, ['/Title', 'title'])
        author = self._get_metadata_field(pdf_info, ['/Author', 'author'])
        
        # Parse dates
        created_date = self._parse_pdf_date(
            self._get_metadata_field(pdf_info, ['/CreationDate', 'creation_date'])
        )
        modified_date = self._parse_pdf_date(
            self._get_metadata_field(pdf_info, ['/ModDate', 'mod_date'])
        )
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Count pages
        num_pages = len(reader.pages)
        
        # Generate document ID
        import hashlib
        doc_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        doc_id = f"doc_{int(datetime.now().timestamp())}_{doc_hash}"
        
        # Create metadata object
        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=file_path.name,
            file_path=file_path,
            document_type=DocumentType.PDF,
            title=title or file_path.stem,
            author=author,
            created_date=created_date,
            modified_date=modified_date,
            file_size_bytes=file_size,
            num_pages=num_pages,
            extra={
                "pdf_version": self._get_metadata_field(pdf_info, ['/Producer', 'producer']),
                "pdf_metadata": {k: str(v) for k, v in pdf_info.items() if v}
            }
        )
        
        return metadata
    
    @staticmethod
    def _get_metadata_field(metadata: Dict, field_names: List[str]) -> Optional[str]:
        """
        Get metadata field with fallback names.
        
        Args:
            metadata: Metadata dictionary
            field_names: List of possible field names
        
        Returns:
            Field value or None
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
        Parse PDF date format.
        PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
        
        Args:
            date_str: PDF date string
        
        Returns:
            datetime object or None
        """
        if not date_str:
            return None
        
        try:
            # Remove 'D:' prefix if present
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Parse basic format: YYYYMMDDHHMMSS
            date_str = date_str[:14]  # Take first 14 characters
            
            return datetime.strptime(date_str, '%Y%m%d%H%M%S')
        except Exception:
            return None
    
    def extract_page_text(
        self,
        file_path: Path,
        page_number: int,
        clean_text: bool = True
    ) -> str:
        """
        Extract text from a specific page.
        
        Args:
            file_path: Path to PDF file
            page_number: Page number (1-indexed)
            clean_text: Clean extracted text
        
        Returns:
            Page text
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                
                # Validate page number
                num_pages = len(reader.pages)
                if page_number < 1 or page_number > num_pages:
                    raise ValueError(f"Page number {page_number} out of range (1-{num_pages})")
                
                # Extract page text (convert to 0-indexed)
                page = reader.pages[page_number - 1]
                page_text = page.extract_text()
                
                if clean_text:
                    page_text = TextCleaner.clean(page_text)
                
                return page_text
        
        except Exception as e:
            self.logger.error(f"Failed to extract page {page_number}: {str(e)}")
            raise PDFParseError(str(file_path), original_error=e)
    
    def get_page_count(self, file_path: Path) -> int:
        """
        Get number of pages in PDF.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Number of pages
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                return len(reader.pages)
        except Exception as e:
            self.logger.error(f"Failed to get page count: {str(e)}")
            raise PDFParseError(str(file_path), original_error=e)
    
    def extract_page_range(
        self,
        file_path: Path,
        start_page: int,
        end_page: int,
        clean_text: bool = True
    ) -> str:
        """
        Extract text from a range of pages.
        
        Args:
            file_path: Path to PDF file
            start_page: Starting page (1-indexed, inclusive)
            end_page: Ending page (1-indexed, inclusive)
            clean_text: Clean extracted text
        
        Returns:
            Combined text from pages
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                num_pages = len(reader.pages)
                
                # Validate range
                if start_page < 1 or end_page > num_pages or start_page > end_page:
                    raise ValueError(
                        f"Invalid page range {start_page}-{end_page} for PDF with {num_pages} pages"
                    )
                
                # Extract pages
                text_parts = []
                for page_num in range(start_page - 1, end_page):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"\n[PAGE {page_num + 1}]\n{page_text}")
                
                combined_text = "\n".join(text_parts)
                
                if clean_text:
                    combined_text = TextCleaner.clean(combined_text)
                
                return combined_text
        
        except Exception as e:
            self.logger.error(f"Failed to extract page range: {str(e)}")
            raise PDFParseError(str(file_path), original_error=e)
    
    def is_scanned(self, file_path: Path) -> bool:
        """
        Check if PDF is scanned (image-based).
        Scanned PDFs have very little or no extractable text.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            True if appears to be scanned
        """
        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                
                # Sample first 3 pages
                pages_to_check = min(3, len(reader.pages))
                total_text_length = 0
                
                for i in range(pages_to_check):
                    page = reader.pages[i]
                    text = page.extract_text()
                    total_text_length += len(text.strip())
                
                # If average text per page is very low, likely scanned
                avg_text_per_page = total_text_length / pages_to_check
                threshold = 100  # characters per page
                
                is_scanned = avg_text_per_page < threshold
                
                if is_scanned:
                    self.logger.info(
                        f"PDF appears to be scanned (avg {avg_text_per_page:.0f} chars/page)"
                    )
                
                return is_scanned
        
        except Exception as e:
            self.logger.warning(f"Could not determine if PDF is scanned: {str(e)}")
            return False


if __name__ == "__main__":
    # Test PDF parser
    print("=== PDF Parser Tests ===\n")
    
    parser = PDFParser()
    
    # Create a test PDF (you'll need an actual PDF file for this)
    test_pdf = Path("test_document.pdf")
    
    if test_pdf.exists():
        # Test basic parsing
        print("Test 1: Basic parsing")
        text, metadata = parser.parse(test_pdf)
        print(f"  Extracted: {len(text)} characters")
        print(f"  Pages: {metadata.num_pages}")
        print(f"  Title: {metadata.title}")
        print(f"  Author: {metadata.author}")
        print()
        
        # Test page count
        print("Test 2: Page count")
        page_count = parser.get_page_count(test_pdf)
        print(f"  Total pages: {page_count}")
        print()
        
        # Test single page extraction
        print("Test 3: Single page extraction")
        page_text = parser.extract_page_text(test_pdf, page_number=1)
        print(f"  Page 1 text: {page_text[:200]}...")
        print()
        
        # Test scanned detection
        print("Test 4: Scanned detection")
        is_scanned = parser.is_scanned(test_pdf)
        print(f"  Is scanned: {is_scanned}")
        print()
    else:
        print(f"Test PDF not found: {test_pdf}")
        print("Please create a test PDF file to run tests")
    
    print("✓ PDF parser module created successfully!")