"""
Metadata Store
SQLite-based storage for document and chunk metadata
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

from config.logging_config import get_logger
from config.settings import get_settings
from config.models import DocumentMetadata, DocumentChunk, ProcessingStatus

logger = get_logger(__name__)
settings = get_settings()


class MetadataStore:
    """
    SQLite-based metadata storage.
    Stores document metadata, chunk information, and query history.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize metadata store.
        
        Args:
            db_path: Path to SQLite database (default from settings)
        """
        self.db_path = Path(db_path or settings.METADATA_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger
        self._initialize_database()
        
        self.logger.info(f"MetadataStore initialized: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
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
                    extra TEXT
                )
            """)
            
            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    page_number INTEGER,
                    section_title TEXT,
                    parent_chunk_id TEXT,
                    token_count INTEGER NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (document_id)
                )
            """)
            
            # Query history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    retrieval_time_ms REAL,
                    generation_time_ms REAL,
                    total_time_ms REAL,
                    num_results INTEGER,
                    model_used TEXT
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_status 
                ON documents(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document 
                ON chunks(document_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_query_timestamp 
                ON query_history(timestamp)
            """)
            
            self.logger.debug("Database tables initialized")
    
    # ========== Document Operations ==========
    
    def add_document(self, metadata: DocumentMetadata):
        """
        Add document metadata.
        
        Args:
            metadata: DocumentMetadata object
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO documents (
                    document_id, filename, file_path, document_type,
                    title, author, created_date, modified_date,
                    upload_date, processed_date, status,
                    file_size_bytes, num_pages, num_tokens, num_chunks,
                    chunking_strategy, processing_time_seconds,
                    error_message, extra
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
                json.dumps(metadata.extra) if metadata.extra else None
            ))
            
            self.logger.info(f"Added document: {metadata.document_id}")
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        Get document metadata by ID.
        
        Args:
            document_id: Document ID
        
        Returns:
            DocumentMetadata or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_document_metadata(row)
    
    def list_documents(
        self,
        status: Optional[ProcessingStatus] = None,
        limit: Optional[int] = None
    ) -> List[DocumentMetadata]:
        """
        List all documents.
        
        Args:
            status: Filter by status
            limit: Maximum number of results
        
        Returns:
            List of DocumentMetadata
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM documents"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status.value)
            
            query += " ORDER BY upload_date DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_document_metadata(row) for row in rows]
    
    def update_document_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ):
        """
        Update document processing status.
        
        Args:
            document_id: Document ID
            status: New status
            error_message: Error message if failed
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            updates = {
                "status": status.value,
                "error_message": error_message
            }
            
            if status == ProcessingStatus.COMPLETED:
                updates["processed_date"] = datetime.now().isoformat()
            
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [document_id]
            
            cursor.execute(
                f"UPDATE documents SET {set_clause} WHERE document_id = ?",
                values
            )
            
            self.logger.info(f"Updated document status: {document_id} -> {status.value}")
    
    def delete_document(self, document_id: str):
        """
        Delete document and its chunks.
        
        Args:
            document_id: Document ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete chunks first
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            
            # Delete document
            cursor.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            
            self.logger.info(f"Deleted document: {document_id}")
    
    # ========== Chunk Operations ==========
    
    def add_chunk(self, chunk: DocumentChunk):
        """
        Add chunk metadata.
        
        Args:
            chunk: DocumentChunk object
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, chunk_index, text,
                    start_char, end_char, page_number, section_title,
                    parent_chunk_id, token_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.text,
                chunk.start_char,
                chunk.end_char,
                chunk.page_number,
                chunk.section_title,
                chunk.parent_chunk_id,
                chunk.token_count,
                json.dumps(chunk.metadata) if chunk.metadata else None
            ))
    
    def add_chunks_batch(self, chunks: List[DocumentChunk]):
        """
        Add multiple chunks efficiently.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            data = [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.page_number,
                    chunk.section_title,
                    chunk.parent_chunk_id,
                    chunk.token_count,
                    json.dumps(chunk.metadata) if chunk.metadata else None
                )
                for chunk in chunks
            ]
            
            cursor.executemany("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, chunk_index, text,
                    start_char, end_char, page_number, section_title,
                    parent_chunk_id, token_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            
            self.logger.info(f"Added {len(chunks)} chunks")
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get chunk by ID.
        
        Args:
            chunk_id: Chunk ID
        
        Returns:
            DocumentChunk or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?",
                (chunk_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_chunk(row)
    
    def get_chunks_by_document(self, document_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
        
        Returns:
            List of DocumentChunk objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_chunk(row) for row in rows]
    
    def get_chunks_batch(self, chunk_ids: List[str]) -> List[DocumentChunk]:
        """
        Get multiple chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs
        
        Returns:
            List of DocumentChunk objects
        """
        if not chunk_ids:
            return []
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            placeholders = ",".join(["?" for _ in chunk_ids])
            cursor.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids
            )
            rows = cursor.fetchall()
            
            # Maintain order from input
            chunk_dict = {self._row_to_chunk(row).chunk_id: self._row_to_chunk(row) for row in rows}
            return [chunk_dict.get(cid) for cid in chunk_ids if cid in chunk_dict]
    
    # ========== Query History Operations ==========
    
    def add_query_history(
        self,
        query: str,
        retrieval_time_ms: float,
        generation_time_ms: float,
        total_time_ms: float,
        num_results: int,
        model_used: str
    ):
        """
        Add query to history.
        
        Args:
            query: Search query
            retrieval_time_ms: Retrieval time
            generation_time_ms: Generation time
            total_time_ms: Total time
            num_results: Number of results
            model_used: Model identifier
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO query_history (
                    query, timestamp, retrieval_time_ms, generation_time_ms,
                    total_time_ms, num_results, model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query,
                datetime.now().isoformat(),
                retrieval_time_ms,
                generation_time_ms,
                total_time_ms,
                num_results,
                model_used
            ))
    
    def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent query history.
        
        Args:
            limit: Maximum number of queries
        
        Returns:
            List of query dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM query_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    # ========== Statistics ==========
    
    def get_statistics(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Document stats
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = ?", ("completed",))
            completed_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(file_size_bytes) FROM documents")
            total_size = cursor.fetchone()[0] or 0
            
            # Chunk stats
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(token_count) FROM chunks")
            avg_chunk_tokens = cursor.fetchone()[0] or 0
            
            # Query stats
            cursor.execute("SELECT COUNT(*) FROM query_history")
            total_queries = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(total_time_ms) FROM query_history")
            avg_query_time = cursor.fetchone()[0] or 0
            
            return {
                "total_documents": total_docs,
                "completed_documents": completed_docs,
                "total_file_size_mb": total_size / (1024 * 1024),
                "total_chunks": total_chunks,
                "avg_chunk_tokens": avg_chunk_tokens,
                "total_queries": total_queries,
                "avg_query_time_ms": avg_query_time,
            }
    
    # ========== Helper Methods ==========
    
    def _row_to_document_metadata(self, row: sqlite3.Row) -> DocumentMetadata:
        """Convert database row to DocumentMetadata"""
        from config.models import DocumentType, ProcessingStatus, ChunkingStrategy
        
        return DocumentMetadata(
            document_id=row["document_id"],
            filename=row["filename"],
            file_path=Path(row["file_path"]) if row["file_path"] else None,
            document_type=DocumentType(row["document_type"]),
            title=row["title"],
            author=row["author"],
            created_date=datetime.fromisoformat(row["created_date"]) if row["created_date"] else None,
            modified_date=datetime.fromisoformat(row["modified_date"]) if row["modified_date"] else None,
            upload_date=datetime.fromisoformat(row["upload_date"]),
            processed_date=datetime.fromisoformat(row["processed_date"]) if row["processed_date"] else None,
            status=ProcessingStatus(row["status"]),
            file_size_bytes=row["file_size_bytes"],
            num_pages=row["num_pages"],
            num_tokens=row["num_tokens"],
            num_chunks=row["num_chunks"],
            chunking_strategy=ChunkingStrategy(row["chunking_strategy"]) if row["chunking_strategy"] else None,
            processing_time_seconds=row["processing_time_seconds"],
            error_message=row["error_message"],
            extra=json.loads(row["extra"]) if row["extra"] else {}
        )
    
    def _row_to_chunk(self, row: sqlite3.Row) -> DocumentChunk:
        """Convert database row to DocumentChunk"""
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            text=row["text"],
            chunk_index=row["chunk_index"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            page_number=row["page_number"],
            section_title=row["section_title"],
            parent_chunk_id=row["parent_chunk_id"],
            token_count=row["token_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )
    
    def close(self):
        """Close database connection (cleanup)"""
        # SQLite connections are closed automatically in context manager
        self.logger.info("Metadata store closed")


if __name__ == "__main__":
    # Test metadata store
    print("=== Metadata Store Tests ===\n")
    
    from config.models import DocumentType, ProcessingStatus
    
    # Create test database
    test_db = Path("test_metadata.db")
    store = MetadataStore(test_db)
    
    # Test 1: Add document
    print("Test 1: Add document")
    doc_metadata = DocumentMetadata(
        document_id="doc_test_123",
        filename="test.pdf",
        document_type=DocumentType.PDF,
        file_size_bytes=50000,
        status=ProcessingStatus.PENDING
    )
    store.add_document(doc_metadata)
    print(f"  Added document: {doc_metadata.document_id}")
    print()
    
    # Test 2: Get document
    print("Test 2: Get document")
    retrieved = store.get_document("doc_test_123")
    print(f"  Retrieved: {retrieved.document_id}")
    print(f"  Filename: {retrieved.filename}")
    print(f"  Status: {retrieved.status}")
    print()
    
    # Test 3: Add chunks
    print("Test 3: Add chunks")
    chunks = [
        DocumentChunk(
            chunk_id=f"chunk_test_{i}",
            document_id="doc_test_123",
            text=f"This is test chunk {i}",
            chunk_index=i,
            start_char=i*100,
            end_char=(i+1)*100,
            token_count=20
        )
        for i in range(5)
    ]
    store.add_chunks_batch(chunks)
    print(f"  Added {len(chunks)} chunks")
    print()
    
    # Test 4: Get chunks
    print("Test 4: Get chunks by document")
    doc_chunks = store.get_chunks_by_document("doc_test_123")
    print(f"  Retrieved {len(doc_chunks)} chunks")
    print()
    
    # Test 5: Update status
    print("Test 5: Update document status")
    store.update_document_status("doc_test_123", ProcessingStatus.COMPLETED)
    updated = store.get_document("doc_test_123")
    print(f"  Status updated to: {updated.status}")
    print()
    
    # Test 6: Query history
    print("Test 6: Add query history")
    store.add_query_history(
        query="test query",
        retrieval_time_ms=100.5,
        generation_time_ms=250.3,
        total_time_ms=350.8,
        num_results=5,
        model_used="mistral-7b"
    )
    history = store.get_query_history(limit=10)
    print(f"  Added query, history size: {len(history)}")
    print()
    
    # Test 7: Statistics
    print("Test 7: Get statistics")
    stats = store.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Cleanup
    test_db.unlink()
    print("✓ Metadata store module created successfully!")