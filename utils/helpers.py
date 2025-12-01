# DEPENDENCIES
import os
import uuid
import psutil
import asyncio
import hashlib
import requests
from typing import Any
from typing import List
from typing import Dict
from typing import Union
from datetime import datetime
from config.settings import get_settings
from config.logging_config import get_logger
from config.logging_config import TimedLogger
from concurrent.futures import ThreadPoolExecutor


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class IDGenerator:
    """
    Unique ID generation utilities
    """
    @staticmethod
    def generate_document_id() -> str:
        """
        Generate unique document ID
        """
        timestamp   = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_hash = hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]
        
        return f"doc_{timestamp}_{unique_hash}"
    

    @staticmethod
    def generate_chunk_id(document_id: str, chunk_index: int) -> str:
        """
        Generate unique chunk ID
        """
        return f"chunk_{document_id}_{chunk_index}"
    

    @staticmethod
    def generate_request_id() -> str:
        """
        Generate unique request ID
        """
        return f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    

    @staticmethod
    def generate_session_id() -> str:
        """
        Generate session ID
        """
        return f"sess_{uuid.uuid4().hex}"


class PerformanceTimer:
    """
    Performance timing utilities
    """
    @staticmethod
    def time_sync(func):
        """
        Decorator to time synchronous functions
        """
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result     = func(*args, **kwargs)
            end_time   = datetime.now()
            
            duration   = (end_time - start_time).total_seconds()

            logger.debug(f"Function {func.__name__} took {duration:.3f} seconds")
            
            return result
        
        return wrapper
    

    @staticmethod
    def time_async(func):
        """
        Decorator to time asynchronous functions
        """
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result     = await func(*args, **kwargs)
            end_time   = datetime.now()
            
            duration   = (end_time - start_time).total_seconds()

            logger.debug(f"Async function {func.__name__} took {duration:.3f} seconds")
            
            return result
        
        return wrapper
    

    @staticmethod
    def get_timestamp_ms() -> int:
        """
        Get current timestamp in milliseconds
        """
        return int(datetime.now().timestamp() * 1000)


class BatchProcessor:
    """
    Batch processing utilities
    """
    @staticmethod
    def process_batch_sync(items: List[Any], process_func: callable, batch_size: int = None, **kwargs) -> List[Any]:
        """
        Process items in batches (synchronous)
        """
        if batch_size is None:
            batch_size = settings.EMBEDDING_BATCH_SIZE
        
        results = list()
        
        for i in range(0, len(items), batch_size):
            batch         = items[i:i + batch_size]
            batch_results = [process_func(item, **kwargs) for item in batch]
            
            results.extend(batch_results)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
        
        return results
    

    @staticmethod
    async def process_batch_async(items: List[Any], process_func: callable, batch_size: int = None, **kwargs) -> List[Any]:
        """
        Process items in batches (asynchronous)
        """
        if batch_size is None:
            batch_size = settings.ASYNC_BATCH_SIZE
        
        results = list()
        
        for i in range(0, len(items), batch_size):
            batch         = items[i:i + batch_size]
            
            # Create tasks for all items in batch
            tasks         = [process_func(item, **kwargs) for item in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions = True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if (isinstance(result, Exception)):
                    logger.error(f"Error processing item {i + j}: {result}")
                    results.append(None)
                
                else:
                    results.append(result)
            
            logger.debug(f"Processed async batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1}")
        
        return results
    

    @staticmethod
    def process_batch_parallel(items: List[Any], process_func: callable, max_workers: int = None, **kwargs) -> List[Any]:
        """
        Process items in parallel using thread pool
        """
        if max_workers is None:
            max_workers = settings.MAX_WORKERS
        
        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            results = list(executor.map(process_func, items))
        
        return results


class TextUtilities:
    """
    Text processing utilities
    """
    @staticmethod
    def truncate_text(text: str, max_tokens: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum token count (rough estimate)
        """
        # Rough token estimation: ~4 chars per token
        max_chars = max_tokens * 4
        
        if (len(text) <= max_chars):
            return text
        
        # Truncate at word boundary
        truncated  = text[:max_chars - len(suffix)]
        last_space = truncated.rfind(' ')
        
        if (last_space > 0):
            truncated = truncated[:last_space]
        
        return truncated + suffix
    

    @staticmethod
    def count_tokens_accurate(text: str) -> int:
        """
        More accurate token counting (approximation)
        """
        # For English: words * 1.3 + special characters
        words            = text.split()
        word_count       = len(words)
        
        # Count punctuation and special chars
        special_chars    = len(re.findall(r'[^\w\s]', text))
        
        # Estimate tokens (words + punctuation + special formatting)
        estimated_tokens = int(word_count * 1.3 + special_chars * 0.5)
        
        return max(estimated_tokens, 1)
    

    @staticmethod
    def split_into_sections(text: str, max_section_length: int = 1000) -> List[str]:
        """
        Split text into sections at natural boundaries
        """
        if (len(text) <= max_section_length):
            return [text]
        
        sections        = list()
        current_section = ""
        sentences       = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed limit, start new section
            if ((len(current_section) + len(sentence) > max_section_length) and current_section):
                sections.append(current_section.strip())
                current_section = sentence
            
            else:
                if current_section:
                    current_section += ". " + sentence
                
                else:
                    current_section = sentence
        
        if current_section:
            sections.append(current_section.strip())
        
        return sections


class SystemUtilities:
    """
    System-level utilities
    """
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """
        Get memory usage information
        """
        try:
            process     = psutil.Process()
            memory_info = process.memory_info()
            
            return {"rss_mb"  : memory_info.rss / 1024 / 1024,  # Resident Set Size
                    "vms_mb"  : memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                    "percent" : process.memory_percent(),
                   }

        except ImportError:
            return {"error": "psutil not available"}
    

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get system information
        """
        try:
            return {"cpu_percent"         : psutil.cpu_percent(interval=1),
                    "memory_total_gb"     : psutil.virtual_memory().total / 1024 / 1024 / 1024,
                    "memory_available_gb" : psutil.virtual_memory().available / 1024 / 1024 / 1024,
                    "memory_used_percent" : psutil.virtual_memory().percent,
                    "disk_usage_percent"  : psutil.disk_usage('/').percent,
                   }

        except ImportError:
            return {"error": "psutil not available"}
    

    @staticmethod
    def is_ollama_available() -> bool:
        """
        Check if Ollama service is available
        """
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            return (response.status_code == 200)

        except:
            return False


# Convenience functions
def generate_document_id() -> str:
    """
    Generate a document ID
    """
    return IDGenerator.generate_document_id()


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate a chunk ID
    """
    return IDGenerator.generate_chunk_id(document_id, chunk_index)


async def process_batch_async(items: List[Any], process_func: callable, **kwargs) -> List[Any]:
    """
    Process batch asynchronously
    """
    return await BatchProcessor.process_batch_async(items, process_func, **kwargs)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    """
    if (seconds < 1):
        return f"{seconds * 1000:.0f}ms"

    elif (seconds < 60):
        return f"{seconds:.1f}s"

    elif (seconds < 3600):
        return f"{seconds / 60:.1f}m"

    else:
        return f"{seconds / 3600:.1f}h"