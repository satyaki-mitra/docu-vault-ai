# DEPENDENCIES
import json
import faiss
import pickle
from pathlib import Path
from typing import Optional
from datetime import datetime
from config.settings import get_settings
from utils.file_handler import FileHandler
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import IndexingError


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class IndexPersister:
    """
    Handles persistence of indexes to disk: Saves and loads FAISS indexes, BM25 indexes, and metadata
    """
    def __init__(self, vector_store_dir: Optional[Path] = None):
        """
        Initialize index persister
        
        Arguments:
        ----------
            vector_store_dir { Path } : Directory for index storage
        """
        self.logger              = logger
        self.vector_store_dir    = Path(vector_store_dir or settings.VECTOR_STORE_DIR)
        
        # Ensure directory exists
        FileHandler.ensure_directory(self.vector_store_dir)
        
        # File paths
        self.faiss_index_path    = self.vector_store_dir / "faiss.index"
        self.faiss_metadata_path = self.vector_store_dir / "faiss_metadata.pkl"
        self.bm25_index_path     = self.vector_store_dir / "bm25_index.pkl"
        self.metadata_db_path    = self.vector_store_dir / "metadata.db"
        
        self.logger.info(f"Initialized IndexPersister: store_dir={self.vector_store_dir}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def save_faiss_index(self, index: faiss.Index, chunk_ids: list, metadata: Optional[dict] = None) -> bool:
        """
        Save FAISS index and metadata to disk
        
        Arguments:
        ----------
            index     { faiss.Index } : FAISS index object

            chunk_ids { list }        : List of chunk IDs in order
            
            metadata  { dict }        : Additional metadata
        
        Returns:
        --------
               { bool }               : True if successful
        """
        try:
            self.logger.info(f"Saving FAISS index with {len(chunk_ids)} chunks")
            
            # Save FAISS index
            faiss.write_index(index, str(self.faiss_index_path))
            
            # Save metadata
            faiss_metadata = {"chunk_ids"    : chunk_ids,
                              "total_chunks" : len(chunk_ids),
                              "timestamp"    : self._get_timestamp(),
                              "index_type"   : type(index).__name__,
                             }
            
            if metadata:
                faiss_metadata.update(metadata)
            
            with open(self.faiss_metadata_path, 'wb') as f:
                pickle.dump(faiss_metadata, f)
            
            self.logger.info(f"FAISS index saved: {self.faiss_index_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {repr(e)}")
            raise IndexingError(f"FAISS index save failed: {repr(e)}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def load_faiss_index(self) -> tuple[Optional[faiss.Index], list, dict]:
        """
        Load FAISS index and metadata from disk
        
        Returns:
        --------
            { tuple }                 : Tuple of (index, chunk_ids, metadata)
        """
        if not self.faiss_index_path.exists():
            self.logger.warning("FAISS index file not found")
            return None, [], {}
        
        try:
            self.logger.info("Loading FAISS index from disk")
            
            # Load FAISS index
            index     = faiss.read_index(str(self.faiss_index_path))
            
            # Load metadata
            chunk_ids = list()
            metadata  = dict()
            
            if self.faiss_metadata_path.exists():
                with open(self.faiss_metadata_path, 'rb') as f:
                    loaded_metadata = pickle.load(f)
                
                chunk_ids = loaded_metadata.get("chunk_ids", [])
                metadata  = loaded_metadata
            
            self.logger.info(f"Loaded FAISS index with {len(chunk_ids)} chunks")
            
            return index, chunk_ids, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index: {repr(e)}")
            raise IndexingError(f"FAISS index load failed: {repr(e)}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def save_bm25_index(self, bm25_index, chunk_ids: list, metadata: Optional[dict] = None) -> bool:
        """
        Save BM25 index to disk
        
        Arguments:
        ----------
            bm25_index              : BM25 index object

            chunk_ids    { list }   : List of chunk IDs
            
            metadata     { dict }   : Additional metadata
        
        Returns:
        --------
               { bool }             : True if successful
        """
        try:
            self.logger.info(f"Saving BM25 index with {len(chunk_ids)} chunks")
            
            bm25_data = {"index"      : bm25_index,
                         "chunk_ids"  : chunk_ids,
                         "timestamp"  : self._get_timestamp(),
                         "total_chunks": len(chunk_ids),
                        }
            
            if metadata:
                bm25_data.update(metadata)
            
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump(bm25_data, f)
            
            self.logger.info(f"BM25 index saved: {self.bm25_index_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save BM25 index: {repr(e)}")
            raise IndexingError(f"BM25 index save failed: {repr(e)}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def load_bm25_index(self) -> tuple[Optional[object], list, dict]:
        """
        Load BM25 index from disk
        
        Returns:
        --------
            { tuple }    : Tuple of (index, chunk_ids, metadata)
        """
        if not self.bm25_index_path.exists():
            self.logger.warning("BM25 index file not found")
            return None, [], {}
        
        try:
            self.logger.info("Loading BM25 index from disk")
            
            with open(self.bm25_index_path, 'rb') as f:
                bm25_data = pickle.load(f)
            
            index     = bm25_data.get("index")
            chunk_ids = bm25_data.get("chunk_ids", [])
            metadata  = {k: v for k, v in bm25_data.items() if k not in ["index", "chunk_ids"]}
            
            self.logger.info(f"Loaded BM25 index with {len(chunk_ids)} chunks")
            
            return index, chunk_ids, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {repr(e)}")
            raise IndexingError(f"BM25 index load failed: {repr(e)}")
    

    def save_index_metadata(self, metadata: dict, filename: str = "index_metadata.json") -> bool:
        """
        Save general index metadata
        
        Arguments:
        ----------
            metadata  { dict } : Metadata dictionary

            filename  { str }  : Metadata filename
        
        Returns:
        --------
               { bool }        : True if successful
        """
        try:
            metadata_path          = self.vector_store_dir / filename
            
            # Add timestamp
            metadata["last_saved"] = self._get_timestamp()
            
            with open(metadata_path, 'w') as f:
                json.dump(obj    = metadata, 
                          fp     = f, 
                          indent = 4,
                         )
            
            self.logger.debug(f"Index metadata saved: {metadata_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index metadata: {repr(e)}")
            return False
    

    def load_index_metadata(self, filename: str = "index_metadata.json") -> dict:
        """
        Load general index metadata
        
        Arguments:
        ----------
            filename { str } : Metadata filename
        
        Returns:
        --------
               { dict }      : Metadata dictionary
        """
        metadata_path = self.vector_store_dir / filename
        
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load index metadata: {repr(e)}")
            return {}
    

    def index_files_exist(self) -> bool:
        """
        Check if index files exist on disk
        
        Returns:
        --------
            { bool }    : True if index files exist
        """
        faiss_exists    = self.faiss_index_path.exists()
        bm25_exists     = self.bm25_index_path.exists()
        metadata_exists = self.faiss_metadata_path.exists()
        
        return faiss_exists and bm25_exists and metadata_exists
    

    def get_index_files_info(self) -> dict:
        """
        Get information about index files
        
        Returns:
        --------
            { dict }    : File information
        """
        files_info = dict()
        
        for file_path in [self.faiss_index_path, self.faiss_metadata_path, self.bm25_index_path]:
            if file_path.exists():
                stat                       = file_path.stat()
                files_info[file_path.name] = {"size_bytes"    : stat.st_size,
                                              "size_mb"       : stat.st_size / (1024 * 1024),
                                              "modified_time" : stat.st_mtime,
                                              "exists"        : True,
                                             }
            else:
                files_info[file_path.name] = {"exists": False}
        
        return files_info
    

    def cleanup_old_indexes(self, keep_latest: bool = True) -> dict:
        """
        Clean up old index files
        
        Arguments:
        ----------
            keep_latest { bool } : Whether to keep the latest indexes
        
        Returns:
        --------
                 { dict }        : Cleanup results
        """
        # This would be implemented for versioned indexes
        files_info = self.get_index_files_info()
        
        return {"cleaned_files": 0,
                "kept_files"   : len([f for f in files_info.values() if f.get("exists")]),
                "files_info"   : files_info,
                "message"      : "Index cleanup completed (basic implementation)",
               }
    

    @staticmethod
    def _get_timestamp() -> str:
        """
        Get current timestamp string
        
        Returns:
        --------
            { str }    : Timestamp string
        """
        return datetime.now().isoformat()
    

    def get_persistence_stats(self) -> dict:
        """
        Get persistence statistics
        
        Returns:
        --------
            { dict }    : Persistence statistics
        """
        files_info = self.get_index_files_info()
        
        total_size = sum(info.get("size_mb", 0) for info in files_info.values())
        file_count = sum(1 for info in files_info.values() if info.get("exists"))
        
        return {"total_size_mb"    : total_size,
                "file_count"       : file_count,
                "store_directory"  : str(self.vector_store_dir),
                "files"            : files_info,
               }


# Global index persister instance
_index_persister = None


def get_index_persister(vector_store_dir: Optional[Path] = None) -> IndexPersister:
    """
    Get global index persister instance
    
    Arguments:
    ----------
        vector_store_dir { Path } : Vector store directory
    
    Returns:
    --------
        { IndexPersister }        : IndexPersister instance
    """
    global _index_persister

    if _index_persister is None:
        _index_persister = IndexPersister(vector_store_dir)
    
    return _index_persister