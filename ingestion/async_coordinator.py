# DEPENDENCIES
import asyncio
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
import concurrent.futures
from typing import Optional
from datetime import datetime
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import ProcessingException
from ingestion.progress_tracker import get_progress_tracker
from document_parser.parser_factory import get_parser_factory


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class AsyncCoordinator:
    """
    Asynchronous document processing coordinator: Manages parallel processing of multiple documents with resource optimization
    """
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize async coordinator
        
        Arguments:
        ----------
            max_workers { int } : Maximum parallel workers (default from settings)
        """
        self.logger               = logger
        self.max_workers          = max_workers or settings.MAX_WORKERS
        self.parser_factory       = get_parser_factory()
        self.progress_tracker     = get_progress_tracker()
        
        # Processing statistics
        self.total_processed      = 0
        self.total_failed         = 0
        self.avg_processing_time  = 0.0
        
        self.logger.info(f"Initialized AsyncCoordinator: max_workers={self.max_workers}")
    

    @handle_errors(error_type = ProcessingException, log_error = True, reraise = True)
    async def process_documents_async(self, file_paths: List[Path], progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process multiple documents asynchronously with progress tracking
        
        Arguments:
        ----------
            file_paths        { list }     : List of file paths to process

            progress_callback { callable } : Callback for progress updates
        
        Returns:
        --------
                       { dict }            : Processing results
        """
        if not file_paths:
            return {"processed" : 0, 
                    "failed"    : 0, 
                    "results"   : [],
                   }
        
        self.logger.info(f"Starting async processing of {len(file_paths)} documents")
        
        # Initialize progress tracking
        task_id = self.progress_tracker.start_task(total_items = len(file_paths),
                                                   description = "Document processing",
                                                  )
        
        try:
            # Process files in parallel with semaphore for resource control
            semaphore     = asyncio.Semaphore(self.max_workers)
            tasks         = [self._process_single_file_async(file_path        = file_path, 
                                                             semaphore        = semaphore, 
                                                             task_id          = task_id,
                                                             progress_callback = progress_callback,
                                                            ) for file_path in file_paths]
            
            results       = await asyncio.gather(*tasks, return_exceptions = True)
            
            # Process results
            processed     = list()
            failed        = list()
            
            for file_path, result in zip(file_paths, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to process {file_path}: {repr(result)}")
                    failed.append({"file_path": file_path, "error": str(result)})
                    self.total_failed += 1
                
                else:
                    processed.append(result)
                    self.total_processed += 1
            
            # Update statistics
            self._update_statistics(processed_count = len(processed))
            
            final_result = {"processed"      : len(processed),
                            "failed"         : len(failed),
                            "success_rate"   : (len(processed) / len(file_paths)) * 100,
                            "results"        : processed,
                            "failures"       : failed,
                            "task_id"        : task_id,
                           }
            
            self.progress_tracker.complete_task(task_id)
            
            self.logger.info(f"Async processing completed: {len(processed)} successful, {len(failed)} failed")
            
            return final_result
            
        except Exception as e:
            self.progress_tracker.fail_task(task_id, str(e))
            raise ProcessingException(f"Async processing failed: {repr(e)}")
    

    async def _process_single_file_async(self, file_path: Path, semaphore: asyncio.Semaphore, task_id: str, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process single file asynchronously
        
        Arguments:
        ----------
            file_path            { Path }   : File to process

            semaphore         { Semaphore } : Resource semaphore
            
            task_id              { str }    : Progress task ID
            
            progress_callback { callable }  : Progress callback
        
        Returns:
        --------
                       { dict }             : Processing result
        """
        async with semaphore:
            try:
                self.logger.debug(f"Processing file: {file_path}")
                
                # Update progress
                self.progress_tracker.update_task(task_id        = task_id, 
                                                  current_item   = file_path.name,
                                                  current_status = "parsing",
                                                 )

                processed_so_far = len([t for t in self.progress_tracker.active_tasks.get(task_id, {}) if t])
                self.progress_tracker.update_task(task_id = task_id, processed_items = processed_so_far)

                if progress_callback:
                    progress_callback(self.progress_tracker.get_task_progress(task_id))
                
                # Parse document
                start_time      = datetime.now()
                text, metadata  = await asyncio.to_thread(self.parser_factory.parse, 
                                                          file_path        = file_path,
                                                          extract_metadata = True,
                                                          clean_text       = True,
                                                         )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Update progress
                self.progress_tracker.update_task(task_id        = task_id,
                                                  current_item   = file_path.name, 
                                                  current_status = "completed",
                                                 )
                
                # Handle metadata (could be Pydantic model or dict)
                metadata_dict    = metadata.dict() if hasattr(metadata, 'dict') else (metadata if isinstance(metadata, dict) else {})
                
                result           = {"file_path"       : str(file_path),
                                    "file_name"       : file_path.name,
                                    "text_length"     : len(text),
                                    "processing_time" : processing_time,
                                    "metadata"        : metadata_dict,
                                    "success"         : True,
                                   }
                
                # Increment processed items count
                current_progress = self.progress_tracker.get_task_progress(task_id)

                if current_progress:
                    self.progress_tracker.update_task(task_id         = task_id, 
                                                      processed_items = current_progress.processed_items + 1,
                                                     )
                
                if progress_callback:
                    progress_callback(self.progress_tracker.get_task_progress(task_id))
                
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {repr(e)}")
                
                self.progress_tracker.update_task(task_id        = task_id,
                                                  current_item   = file_path.name,
                                                  current_status = f"failed: {str(e)}",
                                                 )
                
                raise ProcessingException(f"File processing failed: {repr(e)}")
    

    def process_documents_threaded(self, file_paths: List[Path], progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process documents using thread pool (alternative to async)
        
        Arguments:
        ----------
            file_paths          { list }   : List of file paths

            progress_callback { callable } : Progress callback
        
        Returns:
        --------
                       { dict }            : Processing results
        """
        self.logger.info(f"Starting threaded processing of {len(file_paths)} documents")
        
        task_id = self.progress_tracker.start_task(total_items = len(file_paths),
                                                   description = "Threaded document processing",
                                                  )
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers = self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(self._process_single_file_sync, file_path, task_id, progress_callback): file_path for file_path in file_paths}
                
                results        = list()
                failed         = list()
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        self.total_processed += 1
                    
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path}: {repr(e)}")
                        failed.append({"file_path": file_path, "error": str(e)})
                        self.total_failed += 1
                
                final_result = {"processed"    : len(results),
                                "failed"       : len(failed),
                                "success_rate" : (len(results) / len(file_paths)) * 100,
                                "results"      : results,
                                "failures"     : failed,
                                "task_id"      : task_id,
                               }
                
                self.progress_tracker.complete_task(task_id)
                
                self.logger.info(f"Threaded processing completed: {len(results)} successful, {len(failed)} failed")
                
                return final_result
                
        except Exception as e:
            self.progress_tracker.fail_task(task_id, str(e))
            raise ProcessingException(f"Threaded processing failed: {repr(e)}")
    

    def _process_single_file_sync(self, file_path: Path, task_id: str, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process single file synchronously (for thread pool)
        
        Arguments:
        ----------
            file_path            { Path }   : File to process

            task_id              { str }    : Progress task ID
            
            progress_callback { callable }  : Progress callback
        
        Returns:
        --------
                       { dict }             : Processing result
        """
        self.logger.debug(f"Processing file (sync): {file_path}")
        
        # Update progress
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = file_path.name,
                                          current_status = "parsing",
                                         )
        
        if progress_callback:
            progress_callback(self.progress_tracker.get_task_progress(task_id))
        
        # Parse document
        start_time      = datetime.now()
        text, metadata  = self.parser_factory.parse(file_path        = file_path,
                                                    extract_metadata = True,
                                                    clean_text       = True,
                                                   )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update progress
        self.progress_tracker.update_task(task_id        = task_id,
                                          current_item   = file_path.name,
                                          current_status = "completed",
                                         )
        
        # Handle metadata (could be Pydantic model or dict)
        metadata_dict = metadata.dict() if hasattr(metadata, 'dict') else (metadata if isinstance(metadata, dict) else {})
        
        result        = {"file_path"       : str(file_path),
                         "file_name"       : file_path.name,
                         "text_length"     : len(text),
                         "processing_time" : processing_time,
                         "metadata"        : metadata_dict,
                         "success"         : True,
                        }
        
        # Increment processed items count
        current_progress = self.progress_tracker.get_task_progress(task_id)
        
        if current_progress:
            self.progress_tracker.update_task(task_id         = task_id,
                                              processed_items = current_progress.processed_items + 1,
                                             )
        
        if progress_callback:
            progress_callback(self.progress_tracker.get_task_progress(task_id))
        
        return result
    

    def _update_statistics(self, processed_count: int):
        """
        Update processing statistics
        
        Arguments:
        ----------
            processed_count { int } : Number of documents processed in current batch
        """
        # Update average processing time (simplified)
        if (processed_count > 0):
            self.avg_processing_time = (self.avg_processing_time + (processed_count * 1.0)) / 2
    

    def get_coordinator_stats(self) -> Dict[str, Any]:
        """
        Get coordinator statistics
        
        Returns:
        --------
            { dict }    : Statistics dictionary
        """
        return {"total_processed"     : self.total_processed,
                "total_failed"        : self.total_failed,
                "success_rate"        : (self.total_processed / (self.total_processed + self.total_failed)) * 100 if (self.total_processed + self.total_failed) > 0 else 0,
                "avg_processing_time" : self.avg_processing_time,
                "max_workers"         : self.max_workers,
                "active_tasks"        : self.progress_tracker.get_active_task_count(),
               }
    

    def cleanup(self):
        """
        Cleanup resources
        """
        self.progress_tracker.cleanup_completed_tasks()
        
        self.logger.debug("AsyncCoordinator cleanup completed")


# Global async coordinator instance
_async_coordinator = None


def get_async_coordinator(max_workers: Optional[int] = None) -> AsyncCoordinator:
    """
    Get global async coordinator instance
    
    Arguments:
    ----------
        max_workers { int }  : Maximum workers
    
    Returns:
    --------
        { AsyncCoordinator } : AsyncCoordinator instance
    """
    global _async_coordinator

    if _async_coordinator is None:
        _async_coordinator = AsyncCoordinator(max_workers)
    
    return _async_coordinator


async def process_documents_async(file_paths: List[Path], **kwargs) -> Dict[str, Any]:
    """
    Convenience function for async document processing
    
    Arguments:
    ----------
        file_paths { list } : List of file paths
        
        **kwargs            : Additional arguments
    
    Returns:
    --------
             { dict }       : Processing results
    """
    coordinator = get_async_coordinator()

    return await coordinator.process_documents_async(file_paths, **kwargs)