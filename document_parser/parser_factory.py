"""
Parser Factory
Routes documents to appropriate parser based on file type
"""

from pathlib import Path
from typing import Optional, Union
from enum import Enum

from config.logging_config import get_logger
from config.models import DocumentMetadata, DocumentType
from utils.error_handler import InvalidFileTypeError, RAGException
from utils.file_handler import FileHandler

# Import parsers
from document_parser.pdf_parser import PDFParser
from document_parser.docx_parser import DOCXParser
from document_parser.txt_parser import TXTParser

logger = get_logger(__name__)


class ParserFactory:
    """
    Factory class for creating appropriate document parsers.
    Implements Factory pattern for extensible parser selection.
    """
    
    def __init__(self):
        self.logger = logger
        
        # Initialize parsers (reusable instances)
        self._parsers = {
            DocumentType.PDF: PDFParser(),
            DocumentType.DOCX: DOCXParser(),
            DocumentType.TXT: TXTParser(),
        }
        
        # File extension to DocumentType mapping
        self._extension_mapping = {
            'pdf': DocumentType.PDF,
            'docx': DocumentType.DOCX,
            'doc': DocumentType.DOCX,  # Treat .doc as .docx (may need conversion)
            'txt': DocumentType.TXT,
            'text': DocumentType.TXT,
            'md': DocumentType.TXT,  # Markdown as text
            'log': DocumentType.TXT,
            'csv': DocumentType.TXT,
            'json': DocumentType.TXT,
            'xml': DocumentType.TXT,
        }
    
    def get_parser(self, file_path: Path):
        """
        Get appropriate parser for file.
        
        Args:
            file_path: Path to document
        
        Returns:
            Parser instance
        
        Raises:
            InvalidFileTypeError: If file type not supported
        """
        doc_type = self.detect_document_type(file_path)
        
        if doc_type not in self._parsers:
            raise InvalidFileTypeError(
                file_type=str(doc_type),
                allowed_types=[t.value for t in self._parsers.keys()]
            )
        
        return self._parsers[doc_type]
    
    def detect_document_type(self, file_path: Path) -> DocumentType:
        """
        Detect document type from file extension and content.
        
        Args:
            file_path: Path to document
        
        Returns:
            DocumentType enum
        
        Raises:
            InvalidFileTypeError: If type cannot be determined
        """
        file_path = Path(file_path)
        
        # Get extension
        extension = file_path.suffix.lstrip('.').lower()
        
        # Check if extension is mapped
        if extension in self._extension_mapping:
            doc_type = self._extension_mapping[extension]
            self.logger.debug(f"Detected type {doc_type} from extension .{extension}")
            return doc_type
        
        # Try detecting from file content
        detected_type = FileHandler.detect_file_type(file_path)
        if detected_type and detected_type in self._extension_mapping:
            doc_type = self._extension_mapping[detected_type]
            self.logger.debug(f"Detected type {doc_type} from content")
            return doc_type
        
        raise InvalidFileTypeError(
            file_type=extension,
            allowed_types=list(self._extension_mapping.keys())
        )
    
    def parse(
        self,
        file_path: Union[str, Path],
        extract_metadata: bool = True,
        clean_text: bool = True,
        **kwargs
    ) -> tuple[str, Optional[DocumentMetadata]]:
        """
        Parse document using appropriate parser.
        
        Args:
            file_path: Path to document
            extract_metadata: Extract document metadata
            clean_text: Clean extracted text
            **kwargs: Additional parser-specific arguments
        
        Returns:
            Tuple of (extracted_text, metadata)
        
        Raises:
            InvalidFileTypeError: If file type not supported
            RAGException: If parsing fails
        """
        file_path = Path(file_path)
        
        self.logger.info(f"Parsing document: {file_path}")
        
        # Get appropriate parser
        parser = self.get_parser(file_path)
        
        # Parse document
        text, metadata = parser.parse(
            file_path,
            extract_metadata=extract_metadata,
            clean_text=clean_text,
            **kwargs
        )
        
        self.logger.info(
            f"Successfully parsed {file_path.name}: "
            f"{len(text)} chars, type={metadata.document_type if metadata else 'unknown'}"
        )
        
        return text, metadata
    
    def is_supported(self, file_path: Path) -> bool:
        """
        Check if file type is supported.
        
        Args:
            file_path: Path to document
        
        Returns:
            True if supported
        """
        try:
            self.detect_document_type(file_path)
            return True
        except InvalidFileTypeError:
            return False
    
    def get_supported_extensions(self) -> list[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of extensions (without dot)
        """
        return list(self._extension_mapping.keys())
    
    def register_parser(
        self,
        doc_type: DocumentType,
        parser_instance,
        extensions: Optional[list[str]] = None
    ):
        """
        Register a new parser type (for extensibility).
        
        Args:
            doc_type: Document type enum
            parser_instance: Parser instance
            extensions: File extensions to map to this parser
        """
        self._parsers[doc_type] = parser_instance
        
        if extensions:
            for ext in extensions:
                self._extension_mapping[ext.lstrip('.')] = doc_type
        
        self.logger.info(f"Registered parser for {doc_type}")
    
    def batch_parse(
        self,
        file_paths: list[Path],
        extract_metadata: bool = True,
        clean_text: bool = True,
        skip_errors: bool = True
    ) -> list[tuple[Path, str, Optional[DocumentMetadata]]]:
        """
        Parse multiple documents.
        
        Args:
            file_paths: List of file paths
            extract_metadata: Extract metadata
            clean_text: Clean text
            skip_errors: Skip files that fail to parse
        
        Returns:
            List of (file_path, text, metadata) tuples
        """
        results = []
        
        for file_path in file_paths:
            try:
                text, metadata = self.parse(
                    file_path,
                    extract_metadata=extract_metadata,
                    clean_text=clean_text
                )
                results.append((file_path, text, metadata))
            
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path}: {str(e)}")
                if not skip_errors:
                    raise
                # Add placeholder for failed file
                results.append((file_path, "", None))
        
        self.logger.info(
            f"Batch parsed {len(results)}/{len(file_paths)} files successfully"
        )
        
        return results
    
    def parse_directory(
        self,
        directory: Path,
        recursive: bool = False,
        pattern: str = "*",
        **kwargs
    ) -> list[tuple[Path, str, Optional[DocumentMetadata]]]:
        """
        Parse all supported documents in a directory.
        
        Args:
            directory: Directory path
            recursive: Search recursively
            pattern: File pattern (glob)
            **kwargs: Additional parse arguments
        
        Returns:
            List of (file_path, text, metadata) tuples
        """
        directory = Path(directory)
        
        # Get all files
        all_files = FileHandler.list_files(directory, pattern=pattern, recursive=recursive)
        
        # Filter to supported types
        supported_files = [f for f in all_files if self.is_supported(f)]
        
        self.logger.info(
            f"Found {len(supported_files)} supported files in {directory} "
            f"({len(all_files) - len(supported_files)} unsupported)"
        )
        
        # Parse all files
        return self.batch_parse(supported_files, **kwargs)
    
    def get_parser_info(self) -> dict:
        """
        Get information about registered parsers.
        
        Returns:
            Dictionary with parser information
        """
        info = {
            "supported_types": [t.value for t in self._parsers.keys()],
            "supported_extensions": self.get_supported_extensions(),
            "parser_classes": {
                t.value: type(p).__name__ 
                for t, p in self._parsers.items()
            }
        }
        return info


# Global factory instance
_factory = None


def get_parser_factory() -> ParserFactory:
    """
    Get global parser factory instance (singleton).
    
    Returns:
        ParserFactory instance
    """
    global _factory
    if _factory is None:
        _factory = ParserFactory()
    return _factory


# Convenience functions
def parse_document(
    file_path: Union[str, Path],
    **kwargs
) -> tuple[str, Optional[DocumentMetadata]]:
    """
    Convenience function to parse a document.
    
    Args:
        file_path: Path to document
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (text, metadata)
    """
    factory = get_parser_factory()
    return factory.parse(file_path, **kwargs)


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is supported.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if supported
    """
    factory = get_parser_factory()
    return factory.is_supported(Path(file_path))


def get_supported_extensions() -> list[str]:
    """
    Get list of supported extensions.
    
    Returns:
        List of extensions
    """
    factory = get_parser_factory()
    return factory.get_supported_extensions()


if __name__ == "__main__":
    # Test parser factory
    print("=== Parser Factory Tests ===\n")
    
    factory = ParserFactory()
    
    # Test 1: Get parser info
    print("Test 1: Parser info")
    info = factory.get_parser_info()
    print(f"  Supported types: {info['supported_types']}")
    print(f"  Supported extensions: {', '.join(info['supported_extensions'])}")
    print(f"  Parser classes: {info['parser_classes']}")
    print()
    
    # Test 2: Detect document type
    print("Test 2: Document type detection")
    test_files = [
        "document.pdf",
        "report.docx",
        "notes.txt",
        "data.csv",
        "config.json"
    ]
    for filename in test_files:
        try:
            doc_type = factory.detect_document_type(Path(filename))
            print(f"  {filename} -> {doc_type}")
        except Exception as e:
            print(f"  {filename} -> Error: {e}")
    print()
    
    # Test 3: Check if supported
    print("Test 3: Support check")
    test_files = ["test.pdf", "test.xlsx", "test.txt", "test.mp4"]
    for filename in test_files:
        is_supported = factory.is_supported(Path(filename))
        print(f"  {filename}: {'✓ Supported' if is_supported else '✗ Not supported'}")
    print()
    
    # Test 4: Parse actual files (if they exist)
    print("Test 4: Parsing test files")
    test_files = ["test_document.txt"]
    for test_file in test_files:
        if Path(test_file).exists():
            try:
                text, metadata = factory.parse(test_file)
                print(f"  ✓ Parsed {test_file}:")
                print(f"    - Characters: {len(text)}")
                print(f"    - Type: {metadata.document_type}")
                print(f"    - Title: {metadata.title}")
            except Exception as e:
                print(f"  ✗ Failed to parse {test_file}: {e}")
        else:
            print(f"  - Skipped {test_file} (not found)")
    print()
    
    # Test 5: Convenience functions
    print("Test 5: Convenience functions")
    print(f"  Supported extensions: {', '.join(get_supported_extensions())}")
    print(f"  Is 'test.pdf' supported: {is_supported_file('test.pdf')}")
    print()
    
    print("✓ Parser factory module created successfully!")