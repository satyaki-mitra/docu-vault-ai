# DEPENDENCIES
import json
import sqlite3
import numpy as np
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
from typing import Optional
from datetime import datetime
from config.models import DocumentType
from config.models import DocumentChunk
from config.settings import get_settings
from config.models import DocumentMetadata
from config.models import ProcessingStatus
from config.models import ChunkingStrategy
from utils.file_handler import FileHandler
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import IndexingError


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class MetadataStore:
    """
    SQLite-based metadata storage for documents and chunks: Provides fast metadata retrieval and relationship management
    """
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize metadata store
        
        Arguments:
        ----------
            db_path { Path } : Path to SQLite database file
        """
        self.logger  = logger
        self.db_path = Path(db_path or settings.METADATA_DB_PATH)
        
        # Ensure directory exists
        FileHandler.ensure_directory(self.db_path.parent)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"Initialized MetadataStore: db_path={self.db_path}")
    

    def _init_database(self):
        """
        Initialize database schema
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Documents table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT,
                        document_type TEXT NOT NULL,
                        title TEXT,
                        author TEXT,
                        created_date TEXT,
                        modified_date TEXT,
                        upload_date TEXT NOT NULL,
                        processed_date TEXT,
                        status TEXT NOT NULL,
                        file_size_bytes INTEGER NOT NULL,
                        num_pages INTEGER,
                        num_tokens INTEGER,
                        num_chunks INTEGER,
                        chunking_strategy TEXT,
                        processing_time_seconds REAL,
                        error_message TEXT,
                        extra_data TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # Chunks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding BLOB,
                        chunk_index INTEGER NOT NULL,
                        start_char INTEGER NOT NULL,
                        end_char INTEGER NOT NULL,
                        page_number INTEGER,
                        section_title TEXT,
                        token_count INTEGER NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE,
                        UNIQUE(document_id, chunk_index)
                    )
                ''')
                
                # Indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date)')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {repr(e)}")
            raise IndexingError(f"Database initialization failed: {repr(e)}")
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = True)
    def store_chunks(self, chunks: List[DocumentChunk], rebuild: bool = False) -> dict:
        """
        Store chunks and their document metadata
        
        Arguments:
        ----------
            chunks  { list } : List of DocumentChunk objects

            rebuild { bool } : Whether to rebuild the storage
        
        Returns:
        --------
               { dict }      : Storage statistics
        """
        if not chunks:
            return {"stored": 0, "message": "No chunks to store"}
        
        if rebuild:
            self.clear()
        
        # Group chunks by document
        chunks_by_doc = dict()

        for chunk in chunks:
            if chunk.document_id not in chunks_by_doc:
                chunks_by_doc[chunk.document_id] = []

            chunks_by_doc[chunk.document_id].append(chunk)
        
        total_stored = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for document_id, doc_chunks in chunks_by_doc.items():
                # Extract document metadata from first chunk
                first_chunk       = doc_chunks[0]
                document_metadata = self._extract_document_metadata(first_chunk, len(doc_chunks))
                
                # Store document
                self._store_document(cursor, document_metadata)
                
                # Store chunks
                for chunk in doc_chunks:
                    self._store_chunk(cursor, chunk)
                    total_stored += 1
            
            conn.commit()
        
        self.logger.info(f"Stored {total_stored} chunks for {len(chunks_by_doc)} documents")
        
        return {"stored_chunks"    : total_stored,
                "stored_documents" : len(chunks_by_doc),
                "message"          : "Metadata storage completed",
               }
    

    def _extract_document_metadata(self, chunk: DocumentChunk, num_chunks: int) -> DocumentMetadata:
        """
        Extract document metadata from chunk
        
        Arguments:
        ----------
            chunk      { DocumentChunk } : Chunk with document metadata

            num_chunks { int }           : Number of chunks in document
        
        Returns:
        --------
            { DocumentMetadata }         : Document metadata
        """
        # Extract metadata from chunk with proper validation
        chunk_metadata    = chunk.metadata or {}
        
        # Determine document type with proper validation
        document_type_str = chunk_metadata.get('document_type', 'unknown')
        
        try:
            document_type = DocumentType(document_type_str)
        
        except ValueError:
            # Try to infer from filename or other metadata
            filename = chunk_metadata.get('file_name', '') or chunk_metadata.get('filename', '')
            
            if filename:
                extension = filename.split('.')[-1].lower()

                if (extension == 'pdf'):
                    document_type = DocumentType.PDF

                elif (extension in ['docx', 'doc']):
                    document_type = DocumentType.DOCX

                elif (extension == 'txt'):
                    document_type = DocumentType.TXT

                elif (extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']):
                    document_type = DocumentType.IMAGE
                
                elif (extension in ['zip', 'tar', 'gz', 'rar', '7z']):
                    document_type = DocumentType.ARCHIVE
                
                elif (extension in ['html', 'htm'] or filename.startswith('http')):
                    document_type = DocumentType.URL
                
                else:
                    # default fallback
                    document_type = DocumentType.TXT  
            
            else:
                document_type = DocumentType.TXT  # default fallback
        
        # Ensure file_size_bytes is valid
        file_size_bytes = chunk_metadata.get('file_size_bytes', 0)
        if (file_size_bytes <= 0):
            # Estimate file size based on text content as fallback
            file_size_bytes = len(chunk.text.encode('utf-8')) if chunk.text else 1
        
        # Get filename with fallback
        filename  = chunk_metadata.get('file_name') or chunk_metadata.get('filename') or f"document_{chunk.document_id}"
        
        # Get other metadata with fallbacks
        file_path = chunk_metadata.get('file_path')
        title     = chunk_metadata.get('title') or filename
        author    = chunk_metadata.get('author')
        
        # Handle dates
        upload_date = chunk_metadata.get('upload_date')
        
        if upload_date and isinstance(upload_date, datetime):
            upload_date = upload_date
        
        else:
            upload_date = datetime.now()
        
        created_date = chunk_metadata.get('created_date')
        if created_date and isinstance(created_date, datetime):
            created_date = created_date
        
        modified_date = chunk_metadata.get('modified_date')
        
        if modified_date and isinstance(modified_date, datetime):
            modified_date = modified_date
        
        # Calculate token count estimate if not provided
        num_tokens = chunk_metadata.get('num_tokens', 0)
        
        if (num_tokens <= 0 and chunk.text):
            # Rough estimate: ~4 characters per token
            num_tokens = len(chunk.text) // 4
        
        # Get chunking strategy
        chunking_strategy_str = chunk_metadata.get('chunking_strategy')
        chunking_strategy     = None

        if chunking_strategy_str:
            try:
                chunking_strategy = ChunkingStrategy(chunking_strategy_str)
            
            except ValueError:
                pass
        
        return DocumentMetadata(document_id             = chunk.document_id,
                                filename                = filename,
                                file_path               = Path(file_path) if file_path else None,
                                document_type           = document_type,
                                title                   = title,
                                author                  = author,
                                created_date            = created_date,
                                modified_date           = modified_date,
                                upload_date             = upload_date,
                                processed_date          = datetime.now(),
                                status                  = ProcessingStatus.COMPLETED,
                                file_size_bytes         = file_size_bytes,
                                num_pages               = chunk_metadata.get('num_pages', 1),
                                num_tokens              = num_tokens,
                                num_chunks              = num_chunks,
                                chunking_strategy       = chunking_strategy,
                                processing_time_seconds = chunk_metadata.get('processing_time_seconds', 0.0),
                                error_message           = chunk_metadata.get('error_message'),
                                extra                   = chunk_metadata.get('extra_data') or {},
                               )
    

    def _store_document(self, cursor: sqlite3.Cursor, metadata: DocumentMetadata):
        """
        Store document metadata
        
        Arguments:
        ----------
            cursor   { sqlite3.Cursor }   : Database cursor

            metadata { DocumentMetadata } : Document metadata
        """
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (document_id, filename, file_path, document_type, title, author, 
             created_date, modified_date, upload_date, processed_date, status,
             file_size_bytes, num_pages, num_tokens, num_chunks, chunking_strategy,
             processing_time_seconds, error_message, extra_data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.document_id,
            metadata.filename,
            str(metadata.file_path) if metadata.file_path else None,
            metadata.document_type.value,
            metadata.title,
            metadata.author,
            metadata.created_date.isoformat() if metadata.created_date else None,
            metadata.modified_date.isoformat() if metadata.modified_date else None,
            metadata.upload_date.isoformat(),
            metadata.processed_date.isoformat() if metadata.processed_date else None,
            metadata.status.value,
            metadata.file_size_bytes,
            metadata.num_pages,
            metadata.num_tokens,
            metadata.num_chunks,
            metadata.chunking_strategy.value if metadata.chunking_strategy else None,
            metadata.processing_time_seconds,
            metadata.error_message,
            json.dumps(metadata.extra) if metadata.extra else None,
            now,
            now
        ))
    

    def _store_chunk(self, cursor: sqlite3.Cursor, chunk: DocumentChunk):
        """
        Store chunk metadata
        
        Arguments:
        ----------
            cursor { sqlite3.Cursor } : Database cursor

            chunk  { DocumentChunk }  : Chunk to store
        """
        now = datetime.now().isoformat()
        
        # Convert embedding to bytes if present
        embedding_blob = None
        if chunk.embedding:
            embedding_array = np.array(chunk.embedding, dtype='float32')
            embedding_blob  = embedding_array.tobytes()
        
        cursor.execute('''
            INSERT OR REPLACE INTO chunks 
            (chunk_id, document_id, text, embedding, chunk_index, start_char, end_char,
             page_number, section_title, token_count, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk.chunk_id,
            chunk.document_id,
            chunk.text,
            embedding_blob,
            chunk.chunk_index,
            chunk.start_char,
            chunk.end_char,
            chunk.page_number,
            chunk.section_title,
            chunk.token_count,
            json.dumps(chunk.metadata) if chunk.metadata else None,
            now
        ))
    

    @handle_errors(error_type = IndexingError, log_error = True, reraise = False)
    def add_chunks(self, chunks: List[DocumentChunk]) -> dict:
        """
        Add new chunks to storage
        
        Arguments:
        ----------
            chunks { list } : New chunks to add
        
        Returns:
        --------
               { dict }     : Add operation statistics
        """
        return self.store_chunks(chunks, rebuild = False)
    

    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific chunk
        
        Arguments:
        ----------
            chunk_id { str } : Chunk ID
        
        Returns:
        --------
               { dict }      : Chunk metadata or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.*, d.filename, d.document_type, d.title
                FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.document_id
                WHERE c.chunk_id = ?
            ''', (chunk_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_chunk_dict(row)
    

    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document
        
        Arguments:
        ----------
            document_id { str } : Document ID
        
        Returns:
        --------
                     { list }   : List of chunk metadata dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.*, d.filename, d.document_type, d.title
                FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.document_id
                WHERE c.document_id = ?
                ORDER BY c.chunk_index
            ''', (document_id,))
            
            rows = cursor.fetchall()
            
            return [self._row_to_chunk_dict(row) for row in rows]
    

    def get_all_chunks(self) -> List[DocumentChunk]:
        """
        Get all chunks from database
        
        Returns:
        --------
            { List[DocumentChunk] }    : List of all chunks
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                                SELECT chunk_id, document_id, text, embedding, chunk_index, 
                                    start_char, end_char, page_number, section_title, 
                                    token_count, metadata
                                FROM chunks
                                ORDER BY document_id, chunk_index
                           '''
                          )
            
            rows   = cursor.fetchall()
            
            chunks = list()
            
            for row in rows:
                # Parse embedding from bytes
                embedding = None

                if row[3]:  # embedding column
                    embedding_array = np.frombuffer(row[3], dtype='float32')
                    embedding       = embedding_array.tolist()
                
                # Parse metadata JSON
                metadata = None
                if row[10]:  # metadata column
                    try:
                        metadata = json.loads(row[10])
                    
                    except:
                        metadata  = dict()
                
                # Create DocumentChunk object
                chunk = DocumentChunk(chunk_id      = row[0],
                                      document_id   = row[1],
                                      text          = row[2],
                                      embedding     = embedding,
                                      chunk_index   = row[4],
                                      start_char    = row[5],
                                      end_char      = row[6],
                                      page_number   = row[7],
                                      section_title = row[8],
                                      token_count   = row[9],
                                      metadata      = metadata or {},
                                     )
                
                chunks.append(chunk)
            
            self.logger.info(f"Retrieved {len(chunks)} chunks from database")
            
            return chunks


    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document
        
        Arguments:
        ----------
            document_id { str } : Document ID
        
        Returns:
        --------
               { dict }         : Document metadata or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM documents WHERE document_id = ?', (document_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_document_dict(row)
    

    def _row_to_chunk_dict(self, row) -> Dict[str, Any]:
        """
        Convert database row to chunk dictionary
        
        Arguments:
        ----------
            row : Database row
        
        Returns:
        --------
            { dict }    : Chunk dictionary
        """
        columns    = ['chunk_id', 'document_id', 'text', 'embedding', 'chunk_index', 
                      'start_char', 'end_char', 'page_number', 'section_title', 
                      'token_count', 'metadata', 'created_at', 'filename', 'document_type', 'title',
                     ]
        
        chunk_dict = dict(zip(columns, row))
        
        # Parse JSON fields
        if chunk_dict['metadata']:
            chunk_dict['metadata'] = json.loads(chunk_dict['metadata'])
        
        # Convert embedding bytes back to list
        if chunk_dict['embedding']:
            embedding_array         = np.frombuffer(chunk_dict['embedding'], dtype='float32')
            chunk_dict['embedding'] = embedding_array.tolist()
        
        return chunk_dict
    

    def _row_to_document_dict(self, row) -> Dict[str, Any]:
        """
        Convert database row to document dictionary
        
        Arguments:
        ----------
            row         : Database row
        
        Returns:
        --------
            { dict }    : Document dictionary
        """
        columns  = ['document_id', 'filename', 'file_path', 'document_type', 'title', 'author',
                    'created_date', 'modified_date', 'upload_date', 'processed_date', 'status',
                    'file_size_bytes', 'num_pages', 'num_tokens', 'num_chunks', 'chunking_strategy',
                    'processing_time_seconds', 'error_message', 'extra_data', 'created_at', 'updated_at',
                   ]
        
        doc_dict = dict(zip(columns, row))
        
        # Parse JSON fields
        if doc_dict['extra_data']:
            doc_dict['extra_data'] = json.loads(doc_dict['extra_data'])
        
        return doc_dict
    

    def get_stats(self) -> dict:
        """
        Get metadata store statistics
        
        Returns:
        --------
            { dict }    : Statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor      = conn.cursor()
            
            # Document count
            cursor.execute('SELECT COUNT(*) FROM documents')
            doc_count   = cursor.fetchone()[0]
            
            # Chunk count
            cursor.execute('SELECT COUNT(*) FROM chunks')
            chunk_count = cursor.fetchone()[0]
            
            # Database size
            db_size     = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {"documents"        : doc_count,
                    "chunks"           : chunk_count,
                    "database_size_mb" : db_size / (1024 * 1024),
                    "db_path"          : str(self.db_path),
                   }
    

    def is_ready(self) -> bool:
        """
        Check if metadata store is ready
        
        Returns:
        --------
            { bool }    : True if ready
        """
        return self.db_path.exists()
    

    def clear(self):
        """
        Clear all metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM chunks')
            cursor.execute('DELETE FROM documents')
            
            conn.commit()
        
        self.logger.info("Cleared all metadata")
    

    def get_size(self) -> dict:
        """
        Get storage size information
        
        Returns:
        --------
            { dict }    : Size information
        """
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return {"disk_mb"    : db_size / (1024 * 1024),
                "memory_mb"  : 0,  # SQLite is file-based
                "documents"  : self.get_stats()["documents"],
                "chunks"     : self.get_stats()["chunks"],
               }
    

# Global metadata store instance
_metadata_store = None


def get_metadata_store(db_path: Optional[Path] = None) -> MetadataStore:
    """
    Get global metadata store instance
    
    Arguments:
    ----------
        db_path { Path } : Database path
    
    Returns:
    --------
        { MetadataStore } : MetadataStore instance
    """
    global _metadata_store

    if _metadata_store is None:
        _metadata_store = MetadataStore(db_path)
    
    return _metadata_store