# DEPENDENCIES
import os
import shutil
import asyncio
import hashlib
import aiofiles
import filetype
from typing import List
from typing import Union
from pathlib import Path
from typing import BinaryIO
from typing import Optional
from datetime import datetime
from config.settings import get_settings
from config.logging_config import get_logger


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class FileHandler:
    """
    Comprehensive file handling operations
    """
    @staticmethod
    def generate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
        """
        Generate cryptographic hash of file content.
        
        Arguments:
        ----------
            file_path { Path } : Path to file

            algorithm { str }  : Hash algorithm (md5, sha1, sha256)
        
        Returns:
        --------
                { str }        : Hexadecimal hash string
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    

    @staticmethod
    def detect_file_type(file_path: Path) -> Optional[str]:
        """
        Detect actual file type (not just extension)
        
        Arguments:
        ----------
            file_path { Path } : Path to file
        
        Returns:
        --------
                  { str }      : File extension (e.g., 'pdf', 'docx') or None
        """
        try:
            kind = filetype.guess(str(file_path))
            if kind:
                return kind.extension
            
            # Fallback to extension
            return file_path.suffix.lstrip('.').lower()

        except Exception as e:
            logger.warning(f"Could not detect file type for {file_path}: {repr(e)}")
            return file_path.suffix.lstrip('.').lower()
    

    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: Optional[List[str]] = None) -> bool:
        """
        Validate file has allowed extension
        
        Arguments:
        ----------
            filename            { str } : Name of file

            allowed_extensions { list } : List of allowed extensions (without dot)
        
        Returns:
        --------
                    {bool }             : True if valid, False otherwise
        """
        if allowed_extensions is None:
            allowed_extensions = settings.ALLOWED_EXTENSIONS
        
        extension = Path(filename).suffix.lstrip('.').lower()
        
        return extension in allowed_extensions
    

    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """
        Get file size in bytes.
        
        Arguments:
        ----------
            file_path { Path } : Path to file
        
        Returns:
        --------
                { int }        : Size in bytes
        """
        return file_path.stat().st_size
    

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format
        
        Arguments:
        ----------
            size_bytes { int } : Size in bytes
        
        Returns:
        --------
                { str }        : Formatted string (e.g., '1.5 MB')
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if (size_bytes < 1024.0):
                return f"{size_bytes:.2f} {unit}"

            size_bytes /= 1024.0

        return f"{size_bytes:.2f} PB"
    

    @staticmethod
    def safe_filename(filename: str) -> str:
        """
        Sanitize filename for safe filesystem use
        
        Arguments:
        ----------
            filename { str } : Original filename
        
        Returns:
        --------
               { str }       : Sanitized filename
        """
        # Remove path components
        filename     = os.path.basename(filename)
        
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'

        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        filename     = filename.strip(' .')
        
        # Ensure not empty
        if not filename:
            filename = "unnamed_file"
        
        # Limit length (keeping extension)
        max_length = 255
        
        if (len(filename) > max_length):
            name, extension = os.path.splitext(filename)
            name            = name[:max_length - len(extension) - 1]
            filename        = name + extension
        
        return filename
    

    @staticmethod
    def generate_unique_filename(original_filename: str, base_dir: Optional[Path] = None) -> str:
        """
        Generate unique filename to avoid collisions
        
        Arguments:
        ----------
            original_filename { str } : Original filename

            base_dir         { Path } : Directory to check for existing files
        
        Returns:
        --------
                   { str }            : Unique filename with timestamp and hash
        """
        # Sanitize original filename
        safe_name   = FileHandler.safe_filename(original_filename)
        name, ext   = os.path.splitext(safe_name)
        
        # Add timestamp
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add short hash for uniqueness
        hash_short  = hashlib.md5(f"{original_filename}{timestamp}".encode()).hexdigest()[:8]
        
        # Construct unique filename
        unique_name = f"{name}_{timestamp}_{hash_short}{ext}"
        
        # If base_dir provided, ensure uniqueness
        if base_dir:
            base_dir  = Path(base_dir)
            counter   = 1
            test_path = base_dir / unique_name
            
            while test_path.exists():
                unique_name = f"{name}_{timestamp}_{hash_short}_{counter}{ext}"
                test_path   = base_dir / unique_name
                counter    += 1
        
        return unique_name
    

    @staticmethod
    def ensure_directory(directory: Path) -> Path:
        """
        Ensure directory exists, create if not.
        
        Arguments:
        ----------
            directory { Path } : Path to directory
        
        Returns:
        --------
                { Path }       : Path object
        """
        directory = Path(directory)

        directory.mkdir(parents = True, exist_ok = True)
        return directory
    

    @staticmethod
    def copy_file(source: Path, destination: Path, overwrite: bool = False) -> Path:
        """
        Copy file with safety checks
        
        Arguments:
        ----------
            source      { Path } : Source file path

            destination { Path } : Destination file path
            
            overwrite   { bool } : Whether to overwrite existing file
        
        Returns:
        --------
                { Path }         : Destination path
        
        Raises:
        --------
            FileExistsError      : If destination exists and overwrite=False

            FileNotFoundError    : If source doesn't exist
        """
        source      = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        
        # Ensure destination directory exists
        destination.parent.mkdir(parents = True, exist_ok = True)
        
        shutil.copy2(source, destination)
        logger.info(f"Copied file: {source} -> {destination}")
        
        return destination
    

    @staticmethod
    def move_file(source: Path, destination: Path, overwrite: bool = False) -> Path:
        """
        Move file with safety checks
        
        Arguments:
        ----------
            source      { Path } : Source file path

            destination { Path } : Destination file path
            
            overwrite   { bool } : Whether to overwrite existing file
        
        Returns:
        --------
                { Path }         : Destination path
        """
        source      = Path(source)
        destination = Path(destination)
        
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        
        # Ensure destination directory exists
        destination.parent.mkdir(parents = True, exist_ok = True)
        
        shutil.move(str(source), str(destination))
        logger.info(f"Moved file: {source} -> {destination}")
        
        return destination
    

    @staticmethod
    def delete_file(file_path: Path, missing_ok: bool = True) -> bool:
        """
        Delete file safely
        
        Arguments:
        ----------
            file_path  { Path } : Path to file

            missing_ok { bool } : Don't raise error if file doesn't exist
        
        Returns:
        --------
                { bool }        : True if deleted, False if not found
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            if missing_ok:
                return False

            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_path.unlink()

        logger.info(f"Deleted file: {file_path}")
        return True
    

    @staticmethod
    def delete_directory(directory: Path, missing_ok: bool = True, recursive: bool = True) -> bool:
        """
        Delete directory safely
        
        Arguments:
        ----------
            directory  { Path } : Path to directory

            missing_ok { bool } : Don't raise error if directory doesn't exist
            
            recursive  { bool } : Delete contents recursively
        
        Returns:
        --------
                { bool }        : True if deleted, False if not found
        """
        directory = Path(directory)
        
        if not directory.exists():
            if missing_ok:
                return False
        
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if recursive:
            shutil.rmtree(directory)
        
        else:
            directory.rmdir()
        
        logger.info(f"Deleted directory: {directory}")
        return True
    

    @staticmethod
    async def read_file_async(file_path: Path, encoding: str = "utf-8") -> str:
        """
        Read file asynchronously
        
        Arguments:
        ----------
            file_path { Path } : Path to file
            
            encoding  { str }  : Text encoding
        
        Returns:
        --------
                { str }        : File content as string
        """
        async with aiofiles.open(file_path, mode = 'r', encoding = encoding) as f:
            content = await f.read()
        
        return content
    

    @staticmethod
    async def read_file_binary_async(file_path: Path) -> bytes:
        """
        Read file as binary asynchronously
        
        Arguments:
        ----------
            file_path { Path } : Path to file
        
        Returns:
        --------
               { bytes }       : File content as bytes
        """
        async with aiofiles.open(file_path, mode = 'rb') as f:
            content = await f.read()
        
        return content
    

    @staticmethod
    async def write_file_async(file_path: Path, content: str, encoding: str = "utf-8") -> None:
        """
        Write file asynchronously
        
        Arguments:
        ----------
            file_path { Path } : Path to file

            content   { str }  : Content to write
            
            encoding  { str }  : Text encoding
        """
        # Ensure directory exists
        file_path.parent.mkdir(parents = True, exist_ok = True)
        
        async with aiofiles.open(file_path, mode = 'w', encoding = encoding) as f:
            await f.write(content)
    

    @staticmethod
    def list_files(directory: Path, pattern: str = "*", recursive: bool = False) -> List[Path]:
        """
        List files in directory
        
        Arguments:
        ----------
            directory { Path } : Directory to search
           
            pattern   { str }  : Glob pattern
           
            recursive { bool } : Search recursively
        
        Returns:
        --------
                { list }       : List of file paths
        """
        directory = Path(directory)
        
        if recursive:
            files = list(directory.rglob(pattern))
        
        else:
            files = list(directory.glob(pattern))
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        return sorted(files)
    

    @staticmethod
    def get_file_metadata(file_path: Path) -> dict:
        """
        Get comprehensive file metadata
        
        Arguments:
        ----------
            file_path { Path } : Path to file
        
        Returns:
        --------
               { dict }        : Dictionary with metadata
        """
        file_path = Path(file_path)
        stat      = file_path.stat()
        
        return {"name"           : file_path.name,
                "path"           : str(file_path.absolute()),
                "size_bytes"     : stat.st_size,
                "size_formatted" : FileHandler.format_file_size(stat.st_size),
                "extension"      : file_path.suffix.lstrip('.'),
                "created"        : datetime.fromtimestamp(stat.st_ctime),
                "modified"       : datetime.fromtimestamp(stat.st_mtime),
                "accessed"       : datetime.fromtimestamp(stat.st_atime),
                "is_file"        : file_path.is_file(),
                "is_dir"         : file_path.is_dir(),
               }


class TempFileManager:
    """
    Context manager for temporary files : Ensures cleanup even if exceptions occur
    """
    def __init__(self, suffix: str = "", prefix: str = "tmp_", dir: Optional[Path] = None):
        self.suffix                     = suffix
        self.prefix                     = prefix
        self.dir                        = Path(dir) if dir else Path(settings.UPLOAD_DIR) / "temp"
        self.temp_file : Optional[Path] = None
    

    def __enter__(self) -> Path:
        """
        Create temporary file
        """
        self.dir.mkdir(parents = True, exist_ok = True)
        
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_str       = hashlib.md5(f"{timestamp}".encode()).hexdigest()[:8]
        
        filename       = f"{self.prefix}{timestamp}_{hash_str}{self.suffix}"
        self.temp_file = self.dir / filename
        
        return self.temp_file
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Delete temporary file
        """
        if self.temp_file and self.temp_file.exists():
            try:
                self.temp_file.unlink()
                logger.debug(f"Cleaned up temporary file: {self.temp_file}")
            
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {self.temp_file}: {e}")
