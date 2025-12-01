# DEPENDENCIES
from enum import Enum
from typing import List
from pathlib import Path
from typing import Union
from typing import Optional 
from utils.helpers import IDGenerator
from config.models import DocumentType
from config.models import DocumentMetadata
from utils.file_handler import FileHandler
from utils.error_handler import RAGException
from config.logging_config import get_logger
from document_parser.pdf_parser import PDFParser
from document_parser.txt_parser import TXTParser
from document_parser.ocr_engine import OCREngine
from document_parser.docx_parser import DOCXParser
from utils.error_handler import InvalidFileTypeError
from document_parser.zip_handler import ArchiveHandler


# Setup Logging
logger = get_logger(__name__)


class ParserFactory:
    """
    Factory class for creating appropriate document parsers: implements Factory pattern for extensible parser selection
    """
    def __init__(self):
        self.logger             = logger
        
        # Initialize parsers (reusable instances)
        self._parsers           = {DocumentType.PDF  : PDFParser(), 
                                   DocumentType.DOCX : DOCXParser(), 
                                   DocumentType.TXT  : TXTParser(),
                                  }
        
        # Initialize helper components
        self._ocr_engine        = None
        self._archive_handler   = None
        
        # File extension to DocumentType mapping
        self._extension_mapping = {'pdf'   : DocumentType.PDF,
                                   'docx'  : DocumentType.DOCX,
                                   'doc'   : DocumentType.DOCX,
                                   'txt'   : DocumentType.TXT,
                                   'text'  : DocumentType.TXT,
                                   'md'    : DocumentType.TXT, 
                                   'log'   : DocumentType.TXT,
                                   'csv'   : DocumentType.TXT,
                                   'json'  : DocumentType.TXT,
                                   'xml'   : DocumentType.TXT,
                                   'png'   : DocumentType.IMAGE, 
                                   'jpg'   : DocumentType.IMAGE, 
                                   'jpeg'  : DocumentType.IMAGE, 
                                   'gif'   : DocumentType.IMAGE, 
                                   'bmp'   : DocumentType.IMAGE, 
                                   'tiff'  : DocumentType.IMAGE, 
                                   'webp'  : DocumentType.IMAGE,
                                   'zip'   : DocumentType.ARCHIVE, 
                                   'tar'   : DocumentType.ARCHIVE, 
                                   'gz'    : DocumentType.ARCHIVE, 
                                   'tgz'   : DocumentType.ARCHIVE, 
                                   'rar'   : DocumentType.ARCHIVE, 
                                   '7z'    : DocumentType.ARCHIVE,
                                  }

    
    def get_parser(self, file_path: Path):
        """
        Get appropriate parser for file
        
        Arguments:
        ----------
            file_path { Path }   : Path to document
        
        Returns:
        --------
             { object }          : Parser instance or handler
        
        Raises:
        -------
            InvalidFileTypeError : If file type not supported
        """
        doc_type = self.detect_document_type(file_path = file_path)
        
        # Handle special types (image, archive)
        if (doc_type == DocumentType.IMAGE):
            return self._get_ocr_engine()
        
        elif (doc_type == DocumentType.ARCHIVE): 
            return self._get_archive_handler()
        
        # Handle standard document types
        elif doc_type in self._parsers:
            return self._parsers[doc_type]
        
        else:
            raise InvalidFileTypeError(file_type     = str(doc_type),
                                       allowed_types = [t.value for t in self._parsers.keys()] + [DocumentType.IMAGE.value, DocumentType.ARCHIVE.value],
                                      )
        
    
    def detect_document_type(self, file_path: Path) -> Union[DocumentType, str]:
        """
        Detect document type from file extension and content
        
        Arguments:
        ----------
            file_path { Path }   : Path to document
        
        Returns:
        --------
            { Union }            : DocumentType enum or string for special types
        
        Raises:
        -------
            InvalidFileTypeError : If type cannot be determined
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
        
        if (detected_type and (detected_type in self._extension_mapping)):
            doc_type = self._extension_mapping[detected_type]
            
            self.logger.debug(f"Detected type {doc_type} from content")
            
            return doc_type
        
        raise InvalidFileTypeError(file_type = extension, allowed_types = list(self._extension_mapping.keys()))
    

    def parse(self, file_path: Union[str, Path], extract_metadata: bool = True, clean_text: bool = True, **kwargs) -> tuple[str, Optional[DocumentMetadata]]:
        """
        Parse document using appropriate parser
        
        Arguments:
        ----------
            file_path        { Path } : Path to document

            extract_metadata { bool } : Extract document metadata
            
            clean_text       { bool } : Clean extracted text
            
            **kwargs                  : Additional parser-specific arguments
        
        Returns:
        --------
                { tuple }             : Tuple of (extracted_text, metadata)
        
        Raises:
        -------
            InvalidFileTypeError      : If file type not supported
           
            RAGException              : If parsing fails
        """
        file_path = Path(file_path)
        
        self.logger.info(f"Parsing document: {file_path}")
        
        # Get appropriate parser/handler
        parser    = self.get_parser(file_path)
        
        # Handle different parser types
        if isinstance(parser, (PDFParser, DOCXParser, TXTParser)):
            # Standard document parser
            text, metadata = parser.parse(file_path,
                                          extract_metadata = extract_metadata,
                                          clean_text       = clean_text,
                                          **kwargs
                                         )
        
        elif isinstance(parser, OCREngine):
            # Image file - use OCR
            text        = parser.extract_text_from_image(file_path)
            metadata    = self._create_image_metadata(file_path) if extract_metadata else None
        
        elif isinstance(parser, ArchiveHandler):
            # Archive file - extract and parse contents
            return self._parse_archive(file_path        = file_path,
                                       extract_metadata = extract_metadata,
                                       clean_text       = clean_text,
                                       **kwargs
                                      )
        
        else:
            raise InvalidFileTypeError(file_type = file_path.suffix, allowed_types = self.get_supported_extensions())
        
        self.logger.info(f"Successfully parsed {file_path.name}: {len(text)} chars, type={metadata.document_type if metadata else 'unknown'}")
        
        return text, metadata
    

    def _get_ocr_engine(self) -> OCREngine:
        """
        Get OCR engine instance (lazy initialization)
        
        Returns:
        --------
            { OCREngine } : OCR engine instance
        """
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine()
        
        return self._ocr_engine
    

    def _get_archive_handler(self) -> ArchiveHandler:
        """
        Get archive handler instance (lazy initialization)
        
        Returns:
        --------
            { ArchiveHandler } : Archive handler instance
        """
        if self._archive_handler is None:
            self._archive_handler = ArchiveHandler()
        
        return self._archive_handler
    

    def _create_image_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Create metadata for image file
        
        Arguments:
        ----------
            file_path { Path } : Path to image file
        
        Returns:
        --------
            { DocumentMetadata } : DocumentMetadata object
        """
        stat = file_path.stat()
        
        return DocumentMetadata(document_id     = IDGenerator.generate_document_id(),
                                filename        = file_path.name,
                                file_path       = file_path,
                                document_type   = DocumentType.IMAGE,
                                file_size_bytes = stat.st_size,
                                created_date    = stat.st_ctime,
                                modified_date   = stat.st_mtime,
                                extra           = {"file_type": "image"},
                               )
    

    def _parse_archive(self, file_path: Path, extract_metadata: bool = True, clean_text: bool = True, **kwargs) -> tuple[str, Optional[DocumentMetadata]]:
        """
        Parse archive file: extract contents and parse all supported files
        
        Arguments:
        ----------
            file_path        { Path } : Path to archive file

            extract_metadata { bool } : Extract document metadata
            
            clean_text       { bool } : Clean extracted text
            
            **kwargs                  : Additional arguments
        
        Returns:
        --------
                { tuple }             : Tuple of (combined_text, metadata)
        """
        archive_handler = self._get_archive_handler()
        
        # Extract archive contents
        extracted_files = archive_handler.extract_archive(file_path)
        
        # Parse all extracted files
        combined_text   = ""
        all_metadata    = list()
        
        for extracted_file in extracted_files:
            if self.is_supported(extracted_file):
                try:
                    file_text, file_metadata = self.parse(extracted_file,
                                                          extract_metadata = extract_metadata,
                                                          clean_text       = clean_text,
                                                          **kwargs
                                                         )
                    
                    combined_text           += f"\n\n[FILE: {extracted_file.name}]\n{file_text}"
                    
                    if file_metadata:
                        all_metadata.append(file_metadata)
                
                except Exception as e:
                    self.logger.warning(f"Failed to parse extracted file {extracted_file}: {repr(e)}")
                    continue
        
        # Create combined metadata
        combined_metadata = None
        
        if extract_metadata and all_metadata:
            combined_metadata = DocumentMetadata(document_id     = IDGenerator.generate_document_id(),
                                                 filename        = file_path.name,
                                                 file_path       = file_path,
                                                 document_type   = DocumentType.ARCHIVE,
                                                 file_size_bytes = file_path.stat().st_size,
                                                 extra           = {"archive_contents"    : len(extracted_files),
                                                                    "parsed_files"        : len(all_metadata),
                                                                    "contained_documents" : [meta.document_id for meta in all_metadata],
                                                                   }
                                                )
        
        return combined_text.strip(), combined_metadata
    

    def is_supported(self, file_path: Path) -> bool:
        """
        Check if file type is supported.
        
        Arguments:
        ----------
            file_path { Path } : Path to document
        
        Returns:
        --------
               { bool }        : True if supported
        """
        try:
            self.detect_document_type(file_path = file_path)
            return True

        except InvalidFileTypeError:
            return False
    

    def get_supported_extensions(self) -> list[str]:
        """
        Get list of supported file extensions.
        
        Returns:
        --------
            { list } : List of extensions (without dot)
        """
        return list(self._extension_mapping.keys())
    

    def register_parser(self, doc_type: DocumentType, parser_instance, extensions: Optional[list[str]] = None):
        """
        Register a new parser type (for extensibility)
        
        Arguments:
        ----------
            doc_type        { DocumentType } : Document type enum

            parser_instance                  : Parser instance
            
            extensions         { list }      : File extensions to map to this parser
        """
        self._parsers[doc_type] = parser_instance
        
        if extensions:
            for ext in extensions:
                self._extension_mapping[ext.lstrip('.')] = doc_type
        
        self.logger.info(f"Registered parser for {doc_type}")
    

    def batch_parse(self, file_paths: list[Path], extract_metadata: bool = True, clean_text: bool = True, skip_errors: bool = True) -> list[tuple[Path, str, Optional[DocumentMetadata]]]:
        """
        Parse multiple documents.
        
        Arguments:
        ----------
            file_paths       { list } : List of file paths

            extract_metadata { bool } : Extract metadata
            
            clean_text       { str }  : Clean text
            
            skip_errors      { bool } : Skip files that fail to parse
        
        Returns:
        --------
                    { list }          : List of (file_path, text, metadata) tuples
        """
        results = list()
        
        for file_path in file_paths:
            try:
                text, metadata = self.parse(file_path,
                                            extract_metadata = extract_metadata,
                                            clean_text       = clean_text,
                                           )

                results.append((file_path, text, metadata))
            
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path}: {repr(e)}")
                
                if not skip_errors:
                    raise
                # Add placeholder for failed file
                results.append((file_path, "", None))
        
        self.logger.info(f"Batch parsed {len(results)}/{len(file_paths)} files successfully")
        
        return results
    

    def parse_directory(self, directory: Path, recursive: bool = False, pattern: str = "*", **kwargs) -> list[tuple[Path, str, Optional[DocumentMetadata]]]:
        """
        Parse all supported documents in a directory
        
        Arguments:
        ----------
            directory { Path } : Directory path

            recursive { bool } : Search recursively
            
            pattern   { str }  : File pattern (glob)
            
            **kwargs           : Additional parse arguments
        
        Returns:
        --------
                { list }       : List of (file_path, text, metadata) tuples
        """
        directory       = Path(directory)
        
        # Get all files
        all_files       = FileHandler.list_files(directory, pattern=pattern, recursive=recursive)
        
        # Filter to supported types
        supported_files = [f for f in all_files if self.is_supported(f)]
        
        self.logger.info(f"Found {len(supported_files)} supported files in {directory} ({len(all_files) - len(supported_files)} unsupported)")
        
        # Parse all files
        return self.batch_parse(supported_files, **kwargs)
    

    def get_parser_info(self) -> dict:
        """
        Get information about registered parsers
        
        Returns:
        --------
            { dict }    : Dictionary with parser information
        """
        info = {"supported_types"      : [t.value for t in self._parsers.keys()] + ['image', 'archive'],
                "supported_extensions" : self.get_supported_extensions(),
                "parser_classes"       : {t.value: type(p).__name__  for t, p in self._parsers.items()},
                "special_handlers"     : {"image"   : "OCREngine", 
                                          "archive" : "ArchiveHandler"},
               }

        return info


# Global factory instance
_factory = None


def get_parser_factory() -> ParserFactory:
    """
    Get global parser factory instance (singleton)
    
    Returns:
    --------
        { ParserFactory }    : ParserFactory instance
    """
    global _factory

    if _factory is None:
        _factory = ParserFactory()
    
    return _factory


# Convenience functions
def parse_document(file_path: Union[str, Path], **kwargs) -> tuple[str, Optional[DocumentMetadata]]:
    """
    Convenience function to parse a document
    
    Arguments:
    ----------
        file_path { Path } : Path to document

        **kwargs           : Additional arguments
    
    Returns:
    --------
           { tuple }       : Tuple of (text, metadata)
    """
    factory = get_parser_factory()

    return factory.parse(file_path, **kwargs)


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is supported
    
    Arguments:
    ----------
        file_path { Path } : Path to file
    
    Returns:
    --------
           { bool }        : True if supported
    """
    factory = get_parser_factory()

    return factory.is_supported(Path(file_path))


def get_supported_extensions() -> list[str]:
    """
    Get list of supported extensions.
    
    Returns:
    --------
        { list }    : List of extensions
    """
    factory = get_parser_factory()

    return factory.get_supported_extensions()
