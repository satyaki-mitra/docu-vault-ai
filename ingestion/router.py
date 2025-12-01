# DEPENDENCIES
import os
import tempfile
from enum import Enum
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path        
from typing import Optional
from config.settings import get_settings
from utils.file_handler import FileHandler
from config.models import IngestionInputType
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.validators import validate_upload_file
from utils.error_handler import ProcessingException
from ingestion.progress_tracker import get_progress_tracker
from document_parser.zip_handler import get_archive_handler
from ingestion.async_coordinator import get_async_coordinator


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class IngestionRouter:
    """
    Intelligent ingestion router: Determines optimal processing strategy based on input type and characteristics
    """
    def __init__(self):
        """
        Initialize ingestion router
        """
        self.logger                = logger
        self.async_coordinator     = get_async_coordinator()
        self.progress_tracker      = get_progress_tracker()
        self.file_handler          = FileHandler()
        
        # Processing strategies
        self.processing_strategies = {IngestionInputType.FILE    : self._process_single_file,
                                      IngestionInputType.ARCHIVE : self._process_archive,
                                      IngestionInputType.URL     : self._process_url,
                                      IngestionInputType.TEXT    : self._process_text,
                                     }
        
        self.logger.info("Initialized IngestionRouter")
    

    @handle_errors(error_type = ProcessingException, log_error = True, reraise = True)
    def route_and_process(self, input_data: Any, input_type: IngestionInputType, metadata: Optional[Dict[str, Any]] = None, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Route input to appropriate processing strategy
        
        Arguments:
        ----------
            input_data         { Any }          : Input data (file path, URL, text, etc.)

            input_type   { IngestionInputType } : Type of input
            
            metadata            { dict }        : Additional metadata
            
            progress_callback { callable }      : Progress callback
        
        Returns:
        --------
                        { dict }                : Processing results
        """
        self.logger.info(f"Routing {input_type.value} input for processing")
        
        # Validate input
        self._validate_input(input_data, input_type)
        
        # Get processing strategy
        processor = self.processing_strategies.get(input_type)
        
        if not processor:
            raise ProcessingException(f"No processor available for input type: {input_type}")
        
        # Process with progress tracking
        task_id   = self.progress_tracker.start_task(total_items = 1,
                                                     description = f"Processing {input_type.value}",
                                                     metadata    = metadata or {},
                                                    )
        
        try:
            result = processor(input_data, metadata, task_id, progress_callback)
            
            self.progress_tracker.complete_task(task_id, {"processed_items": 1})
            
            self.logger.info(f"Successfully processed {input_type.value} input")
            
            return result
            
        except Exception as e:
            self.progress_tracker.fail_task(task_id, str(e))
            raise ProcessingException(f"Failed to process {input_type.value} input: {repr(e)}")
    

    def _process_single_file(self, file_path: Path, metadata: Optional[Dict[str, Any]], task_id: str, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """
        Process single file
        
        Arguments:
        ----------
            file_path         { Path }      : File path to process

            metadata          { dict }      : File metadata
            
            task_id           { str }       : Progress task ID
            
            progress_callback { callable }  : Progress callback
        
        Returns:
        --------
                       { dict }             : Processing results
        """
        self.logger.info(f"Processing single file: {file_path}")
        
        # Update progress
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = file_path.name,
                                          current_status = "validating_file",
                                         )
        
        if progress_callback:
            progress_callback(self.progress_tracker.get_task_progress(task_id))
        
        # Validate file
        validate_upload_file(file_path)
        
        # Process file
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = file_path.name,
                                          current_status = "processing_file",
                                         )
        
        result = self.async_coordinator.process_documents_threaded([file_path], progress_callback)
        
        # Extract single result (since we processed one file)
        if result["results"]:
            file_result = result["results"][0]

        else:
            raise ProcessingException(f"File processing failed: {result.get('failures', [])}")
        
        return {"success"      : True,
                "file_path"    : str(file_path),
                "file_name"    : file_path.name,
                "text_length"  : file_result.get("text_length", 0),
                "content_type" : self._detect_content_type(file_path),
                "metadata"     : file_result.get("metadata", {}),
               }
    

    def _process_archive(self, archive_path: Path, metadata: Optional[Dict[str, Any]], task_id: str, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """
        Process archive file (ZIP, RAR, etc.)
        
        Arguments:
        ----------
            archive_path        { Path }    : Archive file path

            metadata            { dict }    : Archive metadata
            
            task_id             { str }     : Progress task ID
            
            progress_callback { callable }  : Progress callback
        
        Returns:
        --------
                       { dict }             : Processing results
        """
        self.logger.info(f"Processing archive: {archive_path}")
        
        # Update progress
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = archive_path.name,
                                          current_status = "extracting_archive",
                                         )
        
        if progress_callback:
            progress_callback(self.progress_tracker.get_task_progress(task_id))
        
        # Extract archive
        archive_handler = get_archive_handler()
        extracted_files = archive_handler.extract_archive(archive_path)
        
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = archive_path.name,
                                          current_status = f"processing_{len(extracted_files)}_files",
                                          processed_items = 0,
                                          total_items    = len(extracted_files),
                                         )
        
        # Process extracted files
        result = self.async_coordinator.process_documents_threaded(extracted_files, progress_callback)
        
        return {"success"          : True,
                "archive_path"     : str(archive_path),
                "archive_name"     : archive_path.name,
                "extracted_files"  : len(extracted_files),
                "processed_files"  : result["processed"],
                "failed_files"     : result["failed"],
                "success_rate"     : result["success_rate"],
                "results"          : result["results"],
                "failures"         : result["failures"],
               }
    

    def _process_url(self, url: str, metadata: Optional[Dict[str, Any]], task_id: str, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """
        Process URL (web page scraping)
        
        WARNING: URL processing not implemented yet. Requires Playwright/BeautifulSoup integration
        TODO: Implement web scraping functionality

        Arguments:
        ----------
            url                 { str }     : URL to process

            metadata           { dict }     : URL metadata
            
            task_id             { str }     : Progress task ID
            
            progress_callback { callable }  : Progress callback
        
        Returns:
        --------
                       { dict }             : Processing results
        """
        self.logger.info(f"Processing URL: {url}")
        
        # Update progress
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = url,
                                          current_status = "scraping_url",
                                         )
        
        if progress_callback:
            progress_callback(self.progress_tracker.get_task_progress(task_id))
        
        # Note: Web scraping would be implemented here: For now, return placeholder
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = url,
                                          current_status = "processing_content",
                                         )
        
        # Placeholder implementation: In production, this would use playwright/beautifulsoup scrapers
        return {"success"      : True,
                "url"          : url,
                "content_type" : "web_page",
                "text_length"  : 0,  # Would be actual content length
                "message"      : "URL processing placeholder - implement web scraping",
                "metadata"     : metadata or {},
               }
    

    def _process_text(self, text: str, metadata: Optional[Dict[str, Any]], task_id: str, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """
        Process raw text input
        
        Arguments:
        ----------
            text                { str }     : Text content to process

            metadata            { dict }    : Text metadata
            
            task_id             { str }     : Progress task ID
            
            progress_callback { callable }  : Progress callback
        
        Returns:
        --------
                       { dict }             : Processing results
        """
        self.logger.info(f"Processing text input ({len(text)} characters)")
        
        # Update progress
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = "text_input",
                                          current_status = "processing_text",
                                         )
        
        if progress_callback:
            progress_callback(self.progress_tracker.get_task_progress(task_id))
        
        # For text input, create a temporary file and process it
        with tempfile.NamedTemporaryFile(mode = 'w', suffix = '.txt', delete = False, encoding = 'utf-8') as temp_file:
            temp_file.write(text)
            temp_path = Path(temp_file.name)
        
        try:
            # Process as a file
            file_result                         = self._process_single_file(file_path         = temp_path, 
                                                                            metadata          = metadata, 
                                                                            task_id           = task_id, 
                                                                            progress_callback = progress_callback,
                                                                           )
            
            # Add text-specific metadata
            file_result["input_type"]           = "direct_text"
            file_result["original_text_length"] = len(text)
            
            return file_result
            
        finally:
            # Cleanup temporary file
            try:
                os.unlink(temp_path)

            except Exception as e:
                self.logger.warning(f"Failed to delete temporary file: {repr(e)}")
    

    def _validate_input(self, input_data: Any, input_type: IngestionInputType):
        """
        Validate input based on type
        
        Arguments:
        ----------
            input_data         { Any }        : Input data to validate

            input_type { IngestionInputType } : Type of input
        
        Raises:
        -------
            ProcessingException               : If validation fails
        """
        if (input_type == IngestionInputType.FILE):
            if not isinstance(input_data, (str, Path)):
                raise ProcessingException("File input must be a path string or Path object")
            
            file_path = Path(input_data)
            if not file_path.exists():
                raise ProcessingException(f"File not found: {file_path}")
        
        elif (input_type == IngestionInputType.URL):
            if not isinstance(input_data, str):
                raise ProcessingException("URL input must be a string")
            
            if not input_data.startswith(('http://', 'https://')):
                raise ProcessingException("URL must start with http:// or https://")
        
        elif (input_type == IngestionInputType.TEXT):
            if not isinstance(input_data, str):
                raise ProcessingException("Text input must be a string")
            
            if len(input_data.strip()) == 0:
                raise ProcessingException("Text input cannot be empty")
        
        elif (input_type == IngestionInputType.ARCHIVE):
            if not isinstance(input_data, (str, Path)):
                raise ProcessingException("Archive input must be a path string or Path object")
            
            file_path = Path(input_data)
            if not file_path.exists():
                raise ProcessingException(f"Archive file not found: {file_path}")
    

    def _detect_content_type(self, file_path: Path) -> str:
        """
        Detect content type from file extension
        
        Arguments:
        ----------
            file_path { Path } : File path
        
        Returns:
        --------
               { str }         : Content type
        """
        extension     = file_path.suffix.lower()
        
        content_types = {'.pdf'  : 'application/pdf',
                         '.docx' : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                         '.doc'  : 'application/msword',
                         '.txt'  : 'text/plain',
                         '.zip'  : 'application/zip',
                         '.rar'  : 'application/vnd.rar',
                         '.7z'   : 'application/x-7z-compressed',
                        }
        
        return content_types.get(extension, 'application/octet-stream')
    

    def batch_process(self, inputs: List[Dict[str, Any]], progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process multiple inputs in batch
        
        Arguments:
        ----------
            inputs            { list }     : List of input dictionaries

            progress_callback { callable } : Progress callback
        
        Returns:
        --------
                       { dict }            : Batch processing results
        """
        self.logger.info(f"Starting batch processing of {len(inputs)} inputs")
        
        task_id     = self.progress_tracker.start_task(total_items = len(inputs),
                                                       description = "Batch processing",
                                                      )
        
        results     = list()
        failed      = list()
        
        for i, input_config in enumerate(inputs):
            try:
                self.progress_tracker.update_task(task_id         = task_id,
                                                  processed_items = i,
                                                  current_item    = f"Input {i + 1}",
                                                  current_status  = "processing",
                                                 )
                
                input_type = IngestionInputType(input_config.get("type"))
                input_data = input_config.get("data")
                metadata   = input_config.get("metadata", {})
                
                result     = self.route_and_process(input_data        = input_data,
                                                    input_type        = input_type,
                                                    metadata          = metadata,
                                                    progress_callback = None,
                                                   )
                
                results.append(result)
                
                if progress_callback:
                    progress_callback(self.progress_tracker.get_task_progress(task_id))
            
            except Exception as e:
                self.logger.error(f"Failed to process input {i + 1}: {repr(e)}")
                failed.append({"input_index": i, "input_config": input_config, "error": str(e)})
        
        self.progress_tracker.complete_task(task_id, {"processed_items": len(results)})
        
        return {"processed"    : len(results),
                "failed"       : len(failed),
                "success_rate" : (len(results) / len(inputs)) * 100,
                "results"      : results,
                "failures"     : failed,
               }
    

    def cleanup(self):
        """
        Cleanup router resources
        """
        if hasattr(self, 'async_coordinator'):
            self.async_coordinator.cleanup()
        
        self.logger.debug("IngestionRouter cleanup completed")


    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Get supported processing capabilities
        
        Returns:
        --------
            { dict }    : Capabilities information
        """
        # URL type is not fully implemented yet
        supported_types = [t.value for t in IngestionInputType if (t != IngestionInputType.URL)]
        
        return {"supported_input_types" : supported_types,
                "max_file_size_mb"      : settings.MAX_FILE_SIZE_MB,
                "max_batch_files"       : settings.MAX_BATCH_FILES,
                "allowed_extensions"    : settings.ALLOWED_EXTENSIONS,
                "max_workers"           : settings.MAX_WORKERS,
                "async_supported"       : True,
                "batch_supported"       : True,
               }


# Global ingestion router instance
_ingestion_router = None


def get_ingestion_router() -> IngestionRouter:
    """
    Get global ingestion router instance
    
    Returns:
    --------
        { IngestionRouter } : IngestionRouter instance
    """
    global _ingestion_router

    if _ingestion_router is None:
        _ingestion_router = IngestionRouter()
    
    return _ingestion_router


def process_input(input_data: Any, input_type: IngestionInputType, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for input processing
    
    Arguments:
    ----------
        input_data         { Any }        : Input data
        
        input_type { IngestionInputType } : Input type
        
        **kwargs                          : Additional arguments
    
    Returns:
    --------
                  { dict }                : Processing results
    """
    router = get_ingestion_router()

    return router.route_and_process(input_data, input_type, **kwargs)