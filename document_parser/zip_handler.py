# DEPENDENCIES
import py7zr
import zipfile
import tarfile
import rarfile
from typing import List
from typing import Dict
from pathlib import Path
from typing import Optional
from tempfile import TemporaryDirectory
from config.settings import get_settings
from utils.file_handler import FileHandler
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import ArchiveException


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class ArchiveHandler:
    """
    Comprehensive archive file handler supporting multiple formats
    ZIP, TAR, RAR, 7Z with recursive extraction and validation
    """
    # Supported archive formats and their handlers
    SUPPORTED_FORMATS = {'.zip' : 'zip',
                         '.tar' : 'tar', 
                         '.gz'  : 'tar',
                         '.tgz' : 'tar',
                         '.rar' : 'rar',
                         '.7z'  : '7z',
                        }

    
    def __init__(self, max_size_mb: int = 2048, max_files: int = 10000, allow_recursive: bool = True):
        """
        Initialize archive handler
        
        Arguments:
        ----------
            max_size_mb     { int }  : Maximum uncompressed size in MB

            max_files       { int }  : Maximum number of files to extract
            
            allow_recursive { bool } : Allow nested archives
        """
        self.logger                = logger
        self.max_size_bytes        = max_size_mb * 1024 * 1024
        self.max_files             = max_files
        self.allow_recursive       = allow_recursive
        self.extracted_files_count = 0
        self.total_extracted_size  = 0
        self._temp_dirs            = list()
    

    @handle_errors(error_type = ArchiveException, log_error = True, reraise = True)
    def extract_archive(self, archive_path: Path, extract_dir: Optional[Path] = None, flatten_structure: bool = False, filter_extensions: Optional[List[str]] = None) -> List[Path]:
        """
        Extract archive and return list of extracted file paths
        
        Arguments:
        ----------
            archive_path      { Path } : Path to archive file
            
            extract_dir       { Path } : Directory to extract to (None = temp directory)
            
            flatten_structure { bool } : Ignore directory structure, extract all files to root
            
            filter_extensions { list } : Only extract files with these extensions
            
        Returns:
        --------
                   { list }            : List of paths to extracted files
        """
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            raise ArchiveException(f"Archive file not found: {archive_path}")
        
        # Validate archive size
        self._validate_archive_size(archive_path = archive_path)
        
        # Determine extraction directory
        if extract_dir is None:
            temp_dir    = TemporaryDirectory()
            extract_dir = Path(temp_dir.name)
            # Keep reference to prevent cleanup
            self._temp_dirs.append(temp_dir) 

        else:
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(parents = True, exist_ok = True)
        
        self.logger.info(f"Extracting archive: {archive_path} to {extract_dir}")
        
        # Reset counters
        self.extracted_files_count = 0
        self.total_extracted_size  = 0
        
        # Extract based on format
        archive_format             = self._detect_archive_format(archive_path = archive_path)

        extracted_files            = self._extract_by_format(archive_path      = archive_path, 
                                                             extract_dir       = extract_dir, 
                                                             format            = archive_format,  
                                                             flatten_structure = flatten_structure, 
                                                             filter_extensions = filter_extensions,
                                                            )
        
        self.logger.info(f"Extracted {len(extracted_files)} files from {archive_path} ({self.total_extracted_size} bytes)")
        
        return extracted_files
    

    def _extract_by_format(self, archive_path: Path, extract_dir: Path, format: str, flatten_structure: bool, filter_extensions: Optional[List[str]]) -> List[Path]:
        """
        Extract archive based on format
        """
        try:
            if (format == 'zip'):
                return self._extract_zip(archive_path      = archive_path, 
                                         extract_dir       = extract_dir, 
                                         flatten_structure = flatten_structure, 
                                         filter_extensions = filter_extensions,
                                        )

            elif (format == 'tar'):
                return self._extract_tar(archive_path      = archive_path, 
                                         extract_dir       = extract_dir, 
                                         flatten_structure = flatten_structure, 
                                         filter_extensions = filter_extensions,
                                        )

            elif (format == 'rar'):
                return self._extract_rar(archive_path      = archive_path, 
                                         extract_dir       = extract_dir, 
                                         flatten_structure = flatten_structure, 
                                         filter_extensions = filter_extensions,
                                        )

            elif (format == '7z'):
                return self._extract_7z(archive_path      = archive_path, 
                                        extract_dir       = extract_dir, 
                                        flatten_structure = flatten_structure, 
                                        filter_extensions = filter_extensions,
                                       )

            else:
                raise ArchiveException(f"Unsupported archive format: {format}")

        except Exception as e:
            raise ArchiveException(f"Failed to extract {format} archive: {repr(e)}")
    

    def _extract_zip(self, archive_path: Path, extract_dir: Path, flatten_structure: bool, filter_extensions: Optional[List[str]]) -> List[Path]:
        """
        Extract ZIP archive
        """
        extracted_files = list()
        
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Validate files before extraction
            file_list = zip_ref.namelist()

            self._validate_file_count(file_count = len(file_list))
            
            for file_info in zip_ref.infolist():
                # Use enhanced filtering
                if self._should_extract_file(file_info.filename, filter_extensions):
                    try:
                        extracted_path = self._extract_zip_file(zip_ref           = zip_ref, 
                                                                file_info         = file_info, 
                                                                extract_dir       = extract_dir, 
                                                                flatten_structure = flatten_structure,
                                                               )
                        
                        if extracted_path:
                            extracted_files.append(extracted_path)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to extract {file_info.filename}: {repr(e)}")
                        continue
        
        return extracted_files
    

    def _extract_zip_file(self, zip_ref, file_info, extract_dir: Path, flatten_structure: bool) -> Optional[Path]:
        """
        Extract single file from ZIP
        """
        # Skip directories
        if file_info.filename.endswith('/'):
            return None
        
        # Determine extraction path
        if flatten_structure:
            target_filename = Path(file_info.filename).name
            extract_path    = extract_dir / self._safe_filename(filename = target_filename)
        
        else:
            extract_path = extract_dir / self._safe_archive_path(archive_path = file_info.filename)
        
        # Ensure parent directory exists
        extract_path.parent.mkdir(parents = True, exist_ok = True)
        
        # Check limits
        self._check_extraction_limits(file_size = file_info.file_size)
        
        # Extract file
        zip_ref.extract(file_info, extract_dir)
        
        # Rename if flattening structure
        if flatten_structure:
            original_path = extract_dir / file_info.filename
            if (original_path != extract_path):
                original_path.rename(extract_path)
        
        self.extracted_files_count += 1
        self.total_extracted_size  += file_info.file_size
        
        return extract_path
    

    def _extract_tar(self, archive_path: Path, extract_dir: Path, flatten_structure: bool, filter_extensions: Optional[List[str]]) -> List[Path]:
        """
        Extract TAR archive
        """
        extracted_files = list()
        
        # Determine compression
        mode = 'r'

        if (archive_path.suffix.lower() in ['.gz', '.tgz']):
            mode = 'r:gz'

        elif (archive_path.suffix.lower() == '.bz2'):
            mode = 'r:bz2'

        elif (archive_path.suffix.lower() == '.xz'):
            mode = 'r:xz'
        
        with tarfile.open(archive_path, mode) as tar_ref:
            # Validate files before extraction
            file_list = tar_ref.getnames()
            self._validate_file_count(file_count = len(file_list))
            
            for member in tar_ref.getmembers():
                if self._should_extract_file(member.name, filter_extensions) and member.isfile():
                    try:
                        extracted_path = self._extract_tar_file(tar_ref           = tar_ref, 
                                                                member            = member, 
                                                                extract_dir       = extract_dir, 
                                                                flatten_structure = flatten_structure,
                                                               )

                        if extracted_path:
                            extracted_files.append(extracted_path)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to extract {member.name}: {repr(e)}")
                        continue
        
        return extracted_files
    

    def _extract_tar_file(self, tar_ref, member, extract_dir: Path, flatten_structure: bool) -> Optional[Path]:
        """
        Extract single file from TAR
        """
        # Determine extraction path
        if flatten_structure:
            target_filename = Path(member.name).name
            extract_path    = extract_dir / self._safe_filename(filename = target_filename)
        
        else:
            extract_path = extract_dir / self._safe_archive_path(archive_path = member.name)
        
        # Ensure parent directory exists
        extract_path.parent.mkdir(parents = True, exist_ok = True)
        
        # Check limits
        self._check_extraction_limits(file_size = member.size)
        
        # Extract file
        tar_ref.extract(member, extract_dir)
        
        # Rename if flattening structure
        if flatten_structure:
            original_path = extract_dir / member.name
            
            if (original_path != extract_path):
                original_path.rename(extract_path)
        
        self.extracted_files_count += 1
        self.total_extracted_size  += member.size
        
        return extract_path
    

    def _extract_rar(self, archive_path: Path, extract_dir: Path, flatten_structure: bool, filter_extensions: Optional[List[str]]) -> List[Path]:
        """
        Extract RAR archive
        """
        extracted_files = list()
        
        try:
            with rarfile.RarFile(archive_path) as rar_ref:
                # Validate files before extraction
                file_list = rar_ref.namelist()

                self._validate_file_count(file_count = len(file_list))
                
                for file_info in rar_ref.infolist():
                    if (self._should_extract_file(filename = file_info.filename, filter_extensions = filter_extensions) and not file_info.isdir()):
                        try:
                            extracted_path = self._extract_rar_file(rar_ref           = rar_ref, 
                                                                    file_info         = file_info, 
                                                                    extract_dir       = extract_dir, 
                                                                    flatten_structure = flatten_structure,
                                                                   )
                            if extracted_path:
                                extracted_files.append(extracted_path)
                        
                        except Exception as e:
                            self.logger.warning(f"Failed to extract {file_info.filename}: {repr(e)}")
                            continue
        
        except rarfile.NotRarFile:
            raise ArchiveException(f"Not a valid RAR file: {archive_path}")

        except rarfile.BadRarFile:
            raise ArchiveException(f"Corrupted RAR file: {archive_path}")
        
        return extracted_files
    

    def _extract_rar_file(self, rar_ref, file_info, extract_dir: Path, flatten_structure: bool) -> Optional[Path]:
        """
        Extract single file from RAR
        """
        # Determine extraction path
        if flatten_structure:
            target_filename = Path(file_info.filename).name
            extract_path    = extract_dir / self._safe_filename(filename = target_filename)
        
        else:
            extract_path = extract_dir / self._safe_archive_path(archive_path = file_info.filename)
        
        # Ensure parent directory exists
        extract_path.parent.mkdir(parents = True, exist_ok = True)
        
        # Check limits
        self._check_extraction_limits(file_size = file_info.file_size)
        
        # Extract file
        rar_ref.extract(file_info.filename, extract_dir)
        
        # Rename if flattening structure
        if flatten_structure:
            original_path = extract_dir / file_info.filename
            
            if (original_path != extract_path):
                original_path.rename(extract_path)
        
        self.extracted_files_count += 1
        self.total_extracted_size  += file_info.file_size
        
        return extract_path
    

    def _extract_7z(self, archive_path: Path, extract_dir: Path, flatten_structure: bool, filter_extensions: Optional[List[str]]) -> List[Path]:
        """
        Extract 7Z archive
        """
        extracted_files = list()
        
        with py7zr.SevenZipFile(archive_path, 'r') as sevenz_ref:
            # Get file list
            file_list = sevenz_ref.getnames()
            self._validate_file_count(file_count = len(file_list))
            
            # Extract all files
            sevenz_ref.extractall(extract_dir)
            
            # Process extracted files
            for filename in file_list:
                if self._should_extract_file(filename = filename, filter_extensions = filter_extensions):
                    original_path = extract_dir / filename

                    if original_path.is_file():
                        if flatten_structure:
                            target_path = extract_dir / self._safe_filename(filename = Path(filename).name)
                            
                            if (original_path != target_path):
                                original_path.rename(target_path)
                                extracted_files.append(target_path)
                            
                            else:
                                extracted_files.append(original_path)
                        
                        else:
                            extracted_files.append(original_path)
                        
                        # Update counters
                        self.extracted_files_count += 1
                        self.total_extracted_size  += original_path.stat().st_size
        
        return extracted_files
    
    
    def _is_system_file(self, filename: str) -> bool:
        """
        Check if file is a system/metadata file that should be skipped
        """
        system_patterns = ['__MACOSX',
                           '.DS_Store', 
                           'Thumbs.db',
                           'desktop.ini',
                           '~$',           # Temporary office files
                           '._',           # macOS resource fork
                           '#recycle',     # Recycle bin
                           '@eaDir',       # Synology index
                          ]
                        
        filename_str    = str(filename).lower()
        path_parts      = Path(filename).parts
        
        # Check for system patterns in filename or path
        for pattern in system_patterns:
            if pattern.lower() in filename_str:
                return True
        
        # Skip hidden files and directories (except current/parent dir references)
        for part in path_parts:
            if part.startswith(('.', '_')) and part not in ['.', '..']:
                return True
        
        # Skip common backup and temporary files
        temp_extensions = ['.tmp', '.temp', '.bak', '.backup']

        if any(Path(filename).suffix.lower() == ext for ext in temp_extensions):
            return True
        
        return False


    def _should_extract_file(self, filename: str, filter_extensions: Optional[List[str]]) -> bool:
        """
        Check if file should be extracted based on filters and system files
        """
        # Skip system files and metadata
        if self._is_system_file(filename):
            self.logger.debug(f"Skipping system file: {filename}")
            return False
        
        # Apply extension filters if provided
        if filter_extensions is not None:
            file_ext = Path(filename).suffix.lower()
            return file_ext in [ext.lower() for ext in filter_extensions]
        
        return True


    def _safe_archive_path(self, archive_path: str) -> Path:
        """
        Convert archive path to safe filesystem path
        """
        safe_parts = list()

        for part in Path(archive_path).parts:
            safe_parts.append(self._safe_filename(filename = part))
        
        return Path(*safe_parts)
    

    def _safe_filename(self, filename: str) -> str:
        """
        Ensure filename is safe for filesystem
        """
        return FileHandler.safe_filename(filename)
    

    def _detect_archive_format(self, archive_path: Path) -> str:
        """
        Detect archive format from file extension
        """
        suffix = archive_path.suffix.lower()
        
        for ext, format_type in self.SUPPORTED_FORMATS.items():
            if ((suffix == ext) or (ext == '.tar' and suffix in ['.gz', '.tgz', '.bz2', '.xz'])):
                return format_type
        
        raise ArchiveException(f"Unsupported archive format: {suffix}")
    

    def _validate_archive_size(self, archive_path: Path):
        """
        Validate archive size against limits
        """
        file_size = archive_path.stat().st_size
        
        if (file_size > self.max_size_bytes):
            raise ArchiveException(f"Archive size {file_size} exceeds maximum {self.max_size_bytes}")
    

    def _validate_file_count(self, file_count: int):
        """
        Validate number of files in archive
        """
        if (file_count > self.max_files):
            raise ArchiveException(f"Archive contains {file_count} files, exceeds maximum {self.max_files}")
    

    def _check_extraction_limits(self, file_size: int):
        """
        Check if extraction limits are exceeded
        """
        if (self.extracted_files_count >= self.max_files):
            raise ArchiveException(f"Maximum file count ({self.max_files}) exceeded")
        
        if (self.total_extracted_size + file_size > self.max_size_bytes):
            raise ArchiveException(f"Maximum extraction size ({self.max_size_bytes}) exceeded")
    

    def list_contents(self, archive_path: Path) -> List[Dict]:
        """
        List contents of archive without extraction
        
        Arguments:
        ----------
            archive_path { Path } : Path to archive file
            
        Returns:
        --------
                { list }          : List of file information dictionaries
        """
        archive_path = Path(archive_path)
        format_type  = self._detect_archive_format(archive_path)
        
        try:
            if (format_type == 'zip'):
                return self._list_zip_contents(archive_path)

            elif (format_type == 'tar'):
                return self._list_tar_contents(archive_path)
            
            elif (format_type == 'rar'):
                return self._list_rar_contents(archive_path)

            elif (format_type == '7z'):
                return self._list_7z_contents(archive_path)

            else:
                return []

        except Exception as e:
            self.logger.error(f"Failed to list archive contents: {repr(e)}")
            return []
    

    def _list_zip_contents(self, archive_path: Path) -> List[Dict]:
        """
        List ZIP archive contents
        """
        contents = list()

        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                contents.append({'filename'      : file_info.filename,
                                 'file_size'     : file_info.file_size,
                                 'compress_size' : file_info.compress_size,
                                 'is_dir'        : file_info.filename.endswith('/'),
                               })
        return contents
    

    def _list_tar_contents(self, archive_path: Path) -> List[Dict]:
        """
        List TAR archive contents
        """
        contents = list()
        mode     = 'r'

        if archive_path.suffix.lower() in ['.gz', '.tgz']:
            mode = 'r:gz'
        
        with tarfile.open(archive_path, mode) as tar_ref:
            for member in tar_ref.getmembers():
                contents.append({'filename'  : member.name,
                                 'file_size' : member.size,
                                 'is_dir'    : member.isdir(),
                                 'is_file'   : member.isfile(),
                               })
        return contents
    

    def _list_rar_contents(self, archive_path: Path) -> List[Dict]:
        """
        List RAR archive contents
        """
        contents = list()

        with rarfile.RarFile(archive_path) as rar_ref:
            for file_info in rar_ref.infolist():
                contents.append({'filename'      : file_info.filename,
                                 'file_size'     : file_info.file_size,
                                 'compress_size' : file_info.compress_size,
                                 'is_dir'        : file_info.isdir(),
                               })
        return contents
    

    def _list_7z_contents(self, archive_path: Path) -> List[Dict]:
        """
        List 7Z archive contents
        """
        contents = list()

        with py7zr.SevenZipFile(archive_path, 'r') as sevenz_ref:
            for filename in sevenz_ref.getnames():
                # 7z doesn't provide detailed file info in listing
                contents.append({'filename'  : filename,
                                 'file_size' : 0,
                                 'is_dir'    : filename.endswith('/'),
                               })
        return contents
    

    def is_supported_archive(self, file_path: Path) -> bool:
        """
        Check if file is a supported archive format
        
        Arguments:
        ----------
            file_path { Path } : Path to file
            
        Returns:
        --------
               { bool }        : True if supported archive format
        """
        try:
            self._detect_archive_format(file_path)
            return True

        except ArchiveException:
            return False
    

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported archive formats
        
        Returns:
        --------
            { list }    : List of supported file extensions
        """
        return list(self.SUPPORTED_FORMATS.keys())


# Global archive handler instance
_global_archive_handler = None


def get_archive_handler() -> ArchiveHandler:
    """
    Get global archive handler instance (singleton)
    
    Returns:
    --------
        { ArchiveHandler }    : ArchiveHandler instance
    """
    global _global_archive_handler

    if _global_archive_handler is None:
        _global_archive_handler = ArchiveHandler()

    return _global_archive_handler


def extract_archive(archive_path: Path, **kwargs) -> List[Path]:
    """
    Convenience function for archive extraction
    
    Arguments:
    ----------
        archive_path { Path } : Path to archive file

        **kwargs              : Additional arguments for ArchiveHandler
        
    Returns:
    --------
              { list }        : List of extracted file paths
    """
    handler = get_archive_handler()

    return handler.extract_archive(archive_path, **kwargs)


def is_archive_file(file_path: Path) -> bool:
    """
    Check if file is a supported archive
    
    Arguments:
    ----------
        file_path { Path } : Path to file
        
    Returns:
    --------
            { bool }       : True if supported archive
    """
    handler = get_archive_handler()

    return handler.is_supported_archive(file_path)