# DEPENDENCIES
import json
import shutil
import zipfile
from typing import List
from pathlib import Path
from typing import Optional
from datetime import datetime
from config.settings import get_settings
from utils.file_handler import FileHandler
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import IndexingError
from vector_store.index_persister import get_index_persister


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class BackupManager:
    """
    Automated backup management for vector indexes and metadata: Creates compressed backups with versioning
    """
    def __init__(self, backup_dir: Optional[Path] = None, vector_store_dir: Optional[Path] = None):
        """
        Initialize backup manager
        
        Arguments:
        ----------
            backup_dir       { Path } : Directory for backups
            
            vector_store_dir { Path } : Directory containing indexes to backup
        """
        self.logger           = logger
        self.backup_dir       = Path(backup_dir or settings.BACKUP_DIR)
        self.vector_store_dir = Path(vector_store_dir or settings.VECTOR_STORE_DIR)
        
        # Ensure directories exist
        FileHandler.ensure_directory(self.backup_dir)
        FileHandler.ensure_directory(self.vector_store_dir)
        
        # Backup configuration
        self.auto_backup      = settings.AUTO_BACKUP

        # Documents between auto-backups
        self.backup_interval  = settings.BACKUP_INTERVAL  
        
        self.backup_count     = 0
        
        self.logger.info(f"Initialized BackupManager: backup_dir={self.backup_dir}, auto_backup={self.auto_backup}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def create_backup(self, backup_name: Optional[str] = None, description: Optional[str] = None) -> str:
        """
        Create a backup of vector store
        
        Arguments:
        ----------
            backup_name { str } : Name for backup (default: timestamp-based)
            
            description { str } : Description of backup
        
        Returns:
        --------
               { str }          : Path to backup file
        """
        if not backup_name:
            timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name  = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        self.logger.info(f"Creating backup: {backup_path}")
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all index files
                index_files = list(self.vector_store_dir.glob("*"))
                
                for file_path in index_files:
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.vector_store_dir)

                        zipf.write(file_path, arcname)
                        self.logger.debug(f"Added to backup: {file_path.name}")
                
                # Add backup metadata
                backup_metadata = {"backup_name"      : backup_name,
                                   "description"      : description,
                                   "created_at"       : datetime.now().isoformat(),
                                   "file_count"       : len(index_files),
                                   "vector_store_dir" : str(self.vector_store_dir),
                                  }
                
                zipf.writestr("backup_metadata.json", json.dumps(backup_metadata, indent = 4))
            
            self.backup_count += 1
            
            backup_size        = backup_path.stat().st_size / (1024 * 1024)  # MB
            
            self.logger.info(f"Backup created: {backup_path} ({backup_size:.2f} MB)")
            
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {repr(e)}")
            raise IndexingError(f"Backup creation failed: {repr(e)}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def restore_backup(self, backup_path: Path, restore_dir: Optional[Path] = None) -> dict:
        """
        Restore from a backup file
        
        Arguments:
        ----------
            backup_path { Path } : Path to backup file
            
            restore_dir { Path } : Directory to restore to (default: vector_store_dir)
        
        Returns:
        --------
                 { dict }        : Restore statistics
        """
        backup_path = Path(backup_path)
        restore_dir = Path(restore_dir or self.vector_store_dir)
        
        if not backup_path.exists():
            raise IndexingError(f"Backup file not found: {backup_path}")
        
        self.logger.info(f"Restoring backup: {backup_path} to {restore_dir}")
        
        # Ensure restore directory exists
        FileHandler.ensure_directory(restore_dir)
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Extract all files
                zipf.extractall(restore_dir)
                
                # Get backup metadata
                backup_metadata = dict()

                if "backup_metadata.json" in zipf.namelist():
                    metadata_str    = zipf.read("backup_metadata.json").decode('utf-8')
                    backup_metadata = json.loads(metadata_str)
            
            # Exclude metadata file
            restored_files = len(zipf.namelist()) - 1  
            
            self.logger.info(f"Restored {restored_files} files from backup")
            
            return {"restored_files" : restored_files,
                    "backup_name"    : backup_metadata.get("backup_name", "unknown"),
                    "backup_date"    : backup_metadata.get("created_at", "unknown"),
                    "restore_dir"    : str(restore_dir),
                   }
            
        except Exception as e:
            self.logger.error(f"Backup restoration failed: {repr(e)}")
            raise IndexingError(f"Backup restoration failed: {repr(e)}")
    

    def list_backups(self) -> List[dict]:
        """
        List all available backups
        
        Returns:
        --------
            { list }    : List of backup information dictionaries
        """
        backups = list()
        
        for backup_file in self.backup_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    if "backup_metadata.json" in zipf.namelist():
                        metadata_str = zipf.read("backup_metadata.json").decode('utf-8')
                        metadata     = json.loads(metadata_str)
                    
                    else:
                        metadata = {"backup_name": backup_file.stem}
                
                file_stat   = backup_file.stat()
                
                backup_info = {"name"      : backup_file.name,
                               "path"      : str(backup_file),
                               "size_mb"   : file_stat.st_size / (1024 * 1024),
                               "created"   : datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                               "metadata"  : metadata,
                              }
                
                backups.append(backup_info)
                
            except Exception as e:
                self.logger.warning(f"Could not read backup info for {backup_file}: {repr(e)}")
        
        # Sort by creation time (newest first)
        backups.sort(key     = lambda x: x["created"], 
                     reverse = True,
                    )
        
        return backups
    

    def auto_backup_check(self, documents_processed: int) -> Optional[str]:
        """
        Check if auto-backup should be triggered
        
        Arguments:
        ----------
            documents_processed { int } : Number of documents processed since last backup
        
        Returns:
        --------
                         { str }       : Backup path if backup was created, None otherwise
        """
        if not self.auto_backup:
            return None
        
        if (documents_processed >= self.backup_interval):
            self.logger.info(f"Auto-backup triggered after {documents_processed} documents")
            
            backup_name = f"auto_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            description = f"Auto-backup after {documents_processed} documents"
            
            return self.create_backup(backup_name, description)
        
        return None
    

    def cleanup_old_backups(self, keep_count: int = 5, keep_days: int = 30) -> dict:
        """
        Clean up old backups based on count and age
        
        Arguments:
        ----------
            keep_count { int } : Number of most recent backups to keep
            
            keep_days  { int } : Maximum age of backups in days
        
        Returns:
        --------
                 { dict }      : Cleanup results
        """
        backups = self.list_backups()
        
        if (len(backups) <= keep_count):
            return {"deleted" : 0, 
                    "kept"    : len(backups), 
                    "message" : "No cleanup needed",
                   }
        
        # Sort by date (oldest first for deletion)
        backups.sort(key = lambda x: x["created"])
        
        deleted_count = 0
        cutoff_date   = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        
        # Keep most recent N
        for backup in backups[:-keep_count]:  
            backup_date = datetime.fromisoformat(backup["created"]).timestamp()
            
            # Delete if too old or beyond keep count
            if ((backup_date < cutoff_date) or (len(backups) - deleted_count > keep_count)):
                try:
                    Path(backup["path"]).unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted old backup: {backup['name']}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to delete backup {backup['name']}: {repr(e)}")
        
        remaining = len(backups) - deleted_count
        
        self.logger.info(f"Backup cleanup: deleted {deleted_count}, kept {remaining}")
        
        return {"deleted"      : deleted_count,
                "kept"         : remaining,
                "total_before" : len(backups),
                "message"      : f"Cleanup completed: {deleted_count} backups deleted",
               }
    

    def get_backup_stats(self) -> dict:
        """
        Get backup statistics
        
        Returns:
        --------
            { dict }    : Backup statistics
        """
        backups       = self.list_backups()
        
        total_size    = sum(backup["size_mb"] for backup in backups)
        oldest_backup = min(backups, key = lambda x: x["created"])["created"] if backups else None
        newest_backup = max(backups, key = lambda x: x["created"])["created"] if backups else None
        
        return {"total_backups"   : len(backups),
                "total_size_mb"   : total_size,
                "oldest_backup"   : oldest_backup,
                "newest_backup"   : newest_backup,
                "auto_backup"     : self.auto_backup,
                "backup_interval" : self.backup_interval,
                "backup_dir"      : str(self.backup_dir),
               }
    

    def verify_backup(self, backup_path: Path) -> bool:
        """
        Verify backup integrity
        
        Arguments:
        ----------
            backup_path { Path } : Path to backup file
        
        Returns:
        --------
               { bool }          : True if backup is valid
        """
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Test zip integrity
                bad_file = zipf.testzip()
                
                if bad_file is not None:
                    self.logger.error(f"Backup corrupted: {bad_file}")
                    return False
                
                # Check for essential files
                persister       = get_index_persister()
                
                essential_files = [persister.faiss_index_path.name,
                                   persister.faiss_metadata_path.name,
                                   persister.bm25_index_path.name,
                                  ]
                                  
                existing_files  = zipf.namelist()
                
                missing_files   = [f for f in essential_files if f not in existing_files]
                
                if missing_files:
                    self.logger.warning(f"Backup missing files: {missing_files}")
                    # Not necessarily invalid, but incomplete
                
                return True
                
        except Exception as e:
            self.logger.error(f"Backup verification failed: {repr(e)}")
            return False


# Global backup manager instance
_backup_manager = None


def get_backup_manager(backup_dir: Optional[Path] = None, vector_store_dir: Optional[Path] = None) -> BackupManager:
    """
    Get global backup manager instance
    
    Arguments:
    ----------
        backup_dir       { Path } : Backup directory
        
        vector_store_dir { Path } : Vector store directory
    
    Returns:
    --------
        { BackupManager }         : BackupManager instance
    """
    global _backup_manager

    if _backup_manager is None:
        _backup_manager = BackupManager(backup_dir, vector_store_dir)
    
    return _backup_manager


def create_backup(backup_name: Optional[str] = None, **kwargs) -> str:
    """
    Convenience function to create backup
    
    Arguments:
    ----------
        backup_name { str } : Backup name
        
        **kwargs            : Additional arguments
    
    Returns:
    --------
               { str }      : Backup path
    """
    manager = get_backup_manager()

    return manager.create_backup(backup_name, **kwargs)