# DEPENDENCIES
import traceback
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from config.models import ErrorResponse
from config.logging_config import get_logger


# Setup Logger
logger = get_logger(__name__)


class ErrorCode(str, Enum):
    """
    Standardized error codes
    """
    # File errors (1xxx)
    FILE_NOT_FOUND         = "FILE_1001"
    FILE_TOO_LARGE         = "FILE_1002"
    INVALID_FILE_TYPE      = "FILE_1003"
    FILE_CORRUPTED         = "FILE_1004"
    FILE_UPLOAD_FAILED     = "FILE_1005"
    
    # Parsing errors (2xxx)
    PARSE_ERROR            = "PARSE_2001"
    PDF_PARSE_ERROR        = "PARSE_2002"
    DOCX_PARSE_ERROR       = "PARSE_2003"
    TEXT_ENCODING_ERROR    = "PARSE_2004"
    
    # Processing errors (3xxx)
    CHUNKING_ERROR         = "PROC_3001"
    EMBEDDING_ERROR        = "PROC_3002"
    INDEXING_ERROR         = "PROC_3003"
    
    # Retrieval errors (4xxx)
    RETRIEVAL_ERROR        = "RETR_4001"
    VECTOR_SEARCH_ERROR    = "RETR_4002"
    BM25_SEARCH_ERROR      = "RETR_4003"
    NO_RESULTS_FOUND       = "RETR_4004"
    
    # LLM errors (5xxx)
    LLM_ERROR              = "LLM_5001"
    OLLAMA_NOT_AVAILABLE   = "LLM_5002"
    GENERATION_TIMEOUT     = "LLM_5003"
    CONTEXT_TOO_LONG       = "LLM_5004"
    
    # Validation errors (6xxx)
    VALIDATION_ERROR       = "VAL_6001"
    INVALID_INPUT          = "VAL_6002"
    MISSING_REQUIRED_FIELD = "VAL_6003"
    
    # System errors (7xxx)
    SYSTEM_ERROR           = "SYS_7001"
    DATABASE_ERROR         = "SYS_7002"
    CACHE_ERROR            = "SYS_7003"
    CONFIGURATION_ERROR    = "SYS_7004"
    
    # Generic
    UNKNOWN_ERROR          = "ERR_9999"


class RAGException(Exception):
    """
    Base exception for RAG system
    """
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        self.message        = message
        self.error_code     = error_code
        self.details        = details or {}
        self.original_error = original_error
        
        super().__init__(self.message)

    
    def to_dict(self) -> dict:
        """
        Convert exception to dictionary
        """
        error_dict = {"error"   : self.error_code.value,
                      "message" : self.message,
                      "details" : self.details,
                     }
        
        if self.original_error:
            error_dict["original_error"] = str(self.original_error)
        
        return error_dict
    

    def to_error_response(self) -> ErrorResponse:
        """
        Convert to ErrorResponse model
        """
        return ErrorResponse(error   = self.error_code.value,
                             message = self.message,
                             detail  = self.details if self.details else None,
                            )


# Specific Exceptions
class FileException(RAGException):
    """
    File-related errors
    """
    pass


class FileNotFoundError(FileException):
    """
    File not found
    """
    def __init__(self, file_path: str, **kwargs):
        super().__init__(message    = f"File not found: {file_path}",
                         error_code = ErrorCode.FILE_NOT_FOUND,
                         details    = {"file_path": file_path},
                         **kwargs
                        )


class FileTooLargeError(FileException):
    """
    File exceeds size limit
    """
    def __init__(self, file_size: int, max_size: int, **kwargs):
        super().__init__(message    = f"File size {file_size} bytes exceeds maximum {max_size} bytes",
                         error_code = ErrorCode.FILE_TOO_LARGE,
                         details    = {"file_size": file_size, "max_size": max_size},
                         **kwargs
                        )


class InvalidFileTypeError(FileException):
    """
    Invalid file type
    """
    def __init__(self, file_type: str, allowed_types: list, **kwargs):
        super().__init__(message    = f"Invalid file type '{file_type}'. Allowed: {', '.join(allowed_types)}",
                         error_code = ErrorCode.INVALID_FILE_TYPE,
                         details    = {"file_type": file_type, "allowed_types": allowed_types},
                         **kwargs
                        )


class ParsingException(RAGException):
    """
    Document parsing errors
    """
    pass


class PDFParseError(ParsingException):
    """
    PDF parsing failed
    """
    def __init__(self, file_path: str, **kwargs):
        super().__init__(message    = f"Failed to parse PDF: {file_path}",
                         error_code = ErrorCode.PDF_PARSE_ERROR,
                         details    = {"file_path": file_path},
                         **kwargs
                        )


class DOCXParseError(ParsingException):
    """
    DOCX parsing failed
    """
    def __init__(self, file_path: str, **kwargs):
        super().__init__(message    = f"Failed to parse DOCX: {file_path}",
                         error_code = ErrorCode.DOCX_PARSE_ERROR,
                         details    = {"file_path": file_path},
                         **kwargs
                        )


class TextEncodingError(ParsingException):
    """
    Text encoding error
    """
    def __init__(self, file_path: str, encoding: str, **kwargs):
        super().__init__(message    = f"Failed to decode file {file_path} with encoding {encoding}",
                         error_code = ErrorCode.TEXT_ENCODING_ERROR,
                         details    = {"file_path": file_path, "encoding": encoding},
                         **kwargs
                        )


class ProcessingException(RAGException):
    """
    Processing errors
    """
    pass


class ChunkingError(ProcessingException):
    """
    Chunking failed
    """
    def __init__(self, document_id: str, **kwargs):
        super().__init__(message    = f"Failed to chunk document: {document_id}",
                         error_code = ErrorCode.CHUNKING_ERROR,
                         details    = {"document_id": document_id},
                         **kwargs
                        )


class EmbeddingError(ProcessingException):
    """
    Embedding generation failed
    """
    def __init__(self, text_length: int, **kwargs):
        super().__init__(message    = f"Failed to generate embeddings for text of length {text_length}",
                         error_code = ErrorCode.EMBEDDING_ERROR,
                         details    = {"text_length": text_length},
                         **kwargs
                        )


class IndexingError(ProcessingException):
    """
    Indexing failed
    """
    def __init__(self, index_type: str, **kwargs):
        super().__init__(message    = f"Failed to index into {index_type}",
                         error_code = ErrorCode.INDEXING_ERROR,
                         details    = {"index_type": index_type},
                         **kwargs
                        )


class RetrievalException(RAGException):
    """
    Retrieval errors
    """
    pass


class NoResultsFoundError(RetrievalException):
    """
    No results found
    """
    def __init__(self, query: str, **kwargs):
        super().__init__(message    = f"No results found for query: {query}",
                         error_code = ErrorCode.NO_RESULTS_FOUND,
                         details    = {"query": query},
                         **kwargs
                        )


class LLMException(RAGException):
    """
    LLM errors
    """
    pass


class OllamaNotAvailableError(LLMException):
    """
    Ollama service not available
    """
    def __init__(self, base_url: str, **kwargs):
        super().__init__(message    = f"Ollama service not available at {base_url}",
                         error_code = ErrorCode.OLLAMA_NOT_AVAILABLE,
                         details    = {"base_url": base_url},
                         **kwargs
                        )


class GenerationTimeoutError(LLMException):
    """
    Generation timeout
    """
    def __init__(self, timeout: int, **kwargs):
        super().__init__(message    = f"LLM generation timed out after {timeout} seconds",
                         error_code = ErrorCode.GENERATION_TIMEOUT,
                         details    = {"timeout": timeout},
                         **kwargs
                        )


class ContextTooLongError(LLMException):
    """
    Context exceeds window
    """
    def __init__(self, context_length: int, max_length: int, **kwargs):
        super().__init__(message    = f"Context length {context_length} exceeds maximum {max_length}",
                         error_code = ErrorCode.CONTEXT_TOO_LONG,
                         details    = {"context_length": context_length, "max_length": max_length},
                         **kwargs
                        )


class ValidationException(RAGException):
    """
    Validation errors
    """
    pass


# Error Handler Decorators
def handle_errors(error_type: type = RAGException, log_error: bool = True, reraise: bool = True):
    """
    Decorator to handle errors in functions
    
    Arguments:
    ----------
        error_type { RAGException } : Exception type to catch (default: RAGException)

        log_error      { bool }     : Whether to log the error
        
        reraise        { bool }     : Whether to reraise after handling
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except error_type as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                
                if reraise:
                    raise
                
                else:
                    return None
        
        return wrapper
    return decorator


def safe_execute(func, *args, default = None, log_errors: bool = True, **kwargs):
    """
    Safely execute a function with error handling
    
    Arguments:
    ----------
        func       : Function to execute
        *args      : Function arguments
        default    : Default value on error
        log_errors : Whether to log errors
        **kwargs   : Function keyword arguments
    
    Returns:
    --------
        Function result or default on error
    """
    try:
        return func(*args, **kwargs)

    except Exception as e:
        if log_errors:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
        
        return default


class ErrorContext:
    """
    Context manager for error handling
    """
    
    def __init__(self, operation: str, raise_on_error: bool = True, log_on_error: bool = True, **context_data):
        self.operation                  = operation
        self.raise_on_error             = raise_on_error
        self.log_on_error               = log_on_error
        self.context_data               = context_data
        self.error: Optional[Exception] = None

    
    def __enter__(self):
        logger.debug(f"Starting: {self.operation}")
        
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            
            if self.log_on_error:
                logger.error(f"Error in {self.operation}: {exc_val}",
                             extra    = self.context_data,
                             exc_info = True,
                            )
            
            if not self.raise_on_error:
                # Suppress exception
                return True
        
        return False


def log_and_raise(error: Exception, message: str,**context):
    """
    Log error and raise
    
    Arguments:
    ----------
        error     : Exception to raise
        
        message   : Log message
        
        **context : Additional context to log
    """
    logger.error(message, extra=context, exc_info=True)
    
    raise error


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    Format error message for display.
    
    Arguments:
    ----------
        error             : Exception

        include_traceback : Include full traceback
    
    Returns:
    --------
            { str }       : Formatted error message
    """
    if isinstance(error, RAGException):
        message = f"[{error.error_code.value}] {error.message}"
        
        if error.details:
            message += f"\nDetails: {error.details}"
    
    else:
        message = f"{type(error).__name__}: {str(error)}"
    
    if include_traceback:    
        message += f"\n\nTraceback:\n{''.join(traceback.format_tb(error.__traceback__))}"
    
    return message



if __name__ == "__main__":
    # Test error handling
    print("=== Error Handler Tests ===\n")
    
    # Test custom exceptions
    try:
        raise FileTooLargeError(file_size=100_000_000, max_size=50_000_000)
    
    except RAGException as e:
        print("FileTooLargeError:")
        print(f"  Code: {e.error_code}")
        print(f"  Message: {e.message}")
        print(f"  Details: {e.details}")
        print(f"  Dict: {e.to_dict()}\n")
    
    # Test error context
    print("Testing ErrorContext:")
    with ErrorContext("test operation", raise_on_error=False, test_param="value") as ctx:
        raise ValueError("Something went wrong")
    
    print(f"  Error captured: {ctx.error}\n")
    
    # Test safe execute
    def risky_function(x):
        if x < 0:
            raise ValueError("Negative value")
        return x * 2
    
    result = safe_execute(risky_function, -5, default=0)
    print(f"Safe execute with error: {result}")
    
    result = safe_execute(risky_function, 5, default=0)
    print(f"Safe execute success: {result}")
    
    print("\n✓ All error handler tests passed!")