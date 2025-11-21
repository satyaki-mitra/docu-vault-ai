# DEPENDENCIES
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with color support for console output
    """
    # ANSI color codes
    COLORS = {'DEBUG'    : '\033[36m',   # Cyan
              'INFO'     : '\033[32m',   # Green
              'WARNING'  : '\033[33m',   # Yellow
              'ERROR'    : '\033[31m',   # Red
              'CRITICAL' : '\033[35m',   # Magenta
              'RESET'    : '\033[0m',    # Reset
             }
    

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with color
        """
        # Add color to level name
        levelname = record.levelname
        
        if levelname in self.COLORS:
            record.levelname = (f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}")
        
        # Add color to logger name
        record.name = f"\033[34m{record.name}\033[0m"  # Blue
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON-like formatter for file logging
    """
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured data
        """
        log_data = {"timestamp" : datetime.fromtimestamp(record.created).isoformat(),
                    "level"     : record.levelname,
                    "logger"    : record.name,
                    "message"   : record.getMessage(),
                    "module"    : record.module,
                    "function"  : record.funcName,
                    "line"      : record.lineno,
                   }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Format as key=value pairs (easier to read than JSON)
        parts = [f"{k}={v}" for k, v in log_data.items()]
        
        return " | ".join(parts)


def setup_logging(log_level: str = "INFO", log_dir: Optional[Path] = None, log_format: Optional[str] = None, enable_console: bool = True,  
                  enable_file: bool = True, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Arguments:
    ----------
        log_level      { str }  : Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
        log_dir        { Path } : Directory for log files
        
        log_format     { str }  : Custom log format string
        
        enable_console { bool } : Enable console output
        
        enable_file    { bool } : Enable file output
        
        max_bytes      { int }  : Max file size before rotation
        
        backup_count   { int }  : Number of backup files to keep
    
    Returns:
    --------
        { logging.Logger }      : Configured root logger
    """
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Console handler with colors
    if enable_console:
        console_handler   = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(log_format, datefmt = "%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents = True, exist_ok = True)
        
        # Main log file
        log_file       = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler   = RotatingFileHandler(log_file, maxBytes = max_bytes, backupCount = backup_count, encoding = "utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        # Use structured formatter for files
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Separate error log
        error_file     = log_dir / f"error_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler  = RotatingFileHandler(error_file, maxBytes = max_bytes, backupCount = backup_count, encoding = "utf-8")

        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured: level={log_level}, console={enable_console}, file={enable_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Arguments:
    ----------
        name       { str }    : Logger name (typically __name__)
    
    Returns:
    --------
        { logging.Logger }    : Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds contextual information: Useful for tracking request IDs, user IDs, etc
    """
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """
        Add extra context to log messages
        """
        extra = self.extra.copy()
        
        # Add to structured logging
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(extra)
        
        # Add to message
        context_parts = [f"{k}={v}" for k, v in extra.items()]
        
        if context_parts:
            msg = f"[{', '.join(context_parts)}] {msg}"
        
        return msg, kwargs


def get_context_logger(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with contextual information
    
    Arguments:
    ----------
        name      { str } : Logger name

        **context         : Context key-value pairs
    
    Returns:
    --------
        { LoggerAdapter } : Logger adapter with context
    """
    base_logger = get_logger(name)
    
    return LoggerAdapter(base_logger, context)


# Performance logging utilities
class TimedLogger:
    """
    Context manager for timing operations and logging
    """
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO, log_start: bool = False):
        self.logger     = logger
        self.operation  = operation
        self.level      = level
        self.log_start  = log_start
        self.start_time = None
    

    def __enter__(self):
        if self.log_start:
            self.logger.log(self.level, f"{self.operation} started")
        
        self.start_time = datetime.now()
        
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log(self.level, f"{self.operation} completed in {duration:.2f}s")

        else:
            self.logger.error(f"{self.operation} failed after {duration:.2f}s: {exc_val}")


# Logging decorators
def log_execution(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    Decorator to log function execution time
    """
    def decorator(func):
        nonlocal logger
        
        if logger is None:
            logger = get_logger(func.__module__)
        
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            with TimedLogger(logger, func_name, level=level):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_exceptions(logger: Optional[logging.Logger] = None, reraise: bool = True):
    """
    Decorator to log exceptions with full traceback
    """
    def decorator(func):
        nonlocal logger
        
        if logger is None:
            logger = get_logger(func.__module__)
        
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                func_name = f"{func.__module__}.{func.__name__}"
                
                logger.exception(f"Exception in {func_name}: {str(e)}")
                
                if reraise:
                    raise
                
                return None
        
        return wrapper
    return decorator


# Initialize logging on module import (with defaults) : This will be overridden by app.py with actual settings
_default_logger = setup_logging(log_level      = "INFO",
                                enable_console = True,
                                enable_file    = False, 
                               )


if __name__ == "__main__":
    # Test logging configuration
    from config.settings import settings
    
    # Setup with actual settings
    logger      = setup_logging(log_level      = settings.LOG_LEVEL,
                                log_dir        = settings.LOG_DIR,
                                log_format     = settings.LOG_FORMAT,
                                enable_console = True,
                                enable_file    = True,
                               )
    
    # Test different log levels
    test_logger = get_logger(__name__)

    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test context logger
    ctx_logger  = get_context_logger(__name__, request_id = "test123", user = "admin")

    ctx_logger.info("Processing with context")
    
    # Test timed logger
    with TimedLogger(test_logger, "test_operation"):
        import time
        time.sleep(0.1)
    
    # Test decorator
    @log_execution(test_logger)
    def test_function():
        return "success"
    
    result = test_function()
    
    print(f"\nFunction result: {result}")