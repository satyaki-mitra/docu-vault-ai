# DEPENDENCIES
import re
from enum import Enum
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
from typing import Literal
from pydantic import Field
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator


# Enums
class DocumentType(str, Enum):
    """
    Supported document types
    """
    PDF     = "pdf"
    DOCX    = "docx"
    TXT     = "txt"
    URL     = "url"
    IMAGE   = "image"    
    ARCHIVE = "archive"


class IngestionInputType(str, Enum):
    """
    Supported input types for ingestion
    """
    FILE    = "file"
    URL     = "url"
    ARCHIVE = "archive"
    TEXT    = "text"


class ProcessingStatus(str, Enum):
    """
    Document processing status
    """
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class TokenizerType(str, Enum):
    """
    Supported tokenizer types
    """
    CL100K      = "cl100k_base"  # GPT-4, GPT-3.5-turbo
    P50K        = "p50k_base"    # Codex, text-davinci-002/003
    R50K        = "r50k_base"    # GPT-3, text-davinci-001
    GPT2        = "gpt2"         # GPT-2
    APPROXIMATE = "approximate"  # Fast approximation


class ChunkingStrategy(str, Enum):
    """
    Available chunking strategies
    """
    FIXED        = "fixed"
    SEMANTIC     = "semantic"
    HIERARCHICAL = "hierarchical"


class LLMProvider(str, Enum):
    """
    Supported LLM providers
    """
    OLLAMA    = "ollama"
    OPENAI    = "openai"


class TemperatureStrategy(str, Enum):
    """
    Temperature control strategies
    """
    FIXED       = "fixed"
    ADAPTIVE    = "adaptive"
    CONFIDENCE  = "confidence"
    PROGRESSIVE = "progressive"


class CitationStyle(str, Enum):
    """
    Supported citation styles
    """
    NUMERIC  = "numeric"
    VERBOSE  = "verbose"
    MINIMAL  = "minimal"
    ACADEMIC = "academic"
    LEGAL    = "legal"


class PromptType(str, Enum):
    """
    Supported prompt types
    """
    QA             = "qa"
    SUMMARY        = "summary"
    ANALYTICAL     = "analytical"
    COMPARISON     = "comparison"
    EXTRACTION     = "extraction"
    CREATIVE       = "creative"
    CONVERSATIONAL = "conversational"


# Document Models
class DocumentMetadata(BaseModel):
    """
    Metadata extracted from documents
    """
    model_config                                         = ConfigDict(arbitrary_types_allowed = True)
    
    document_id             : str                        = Field(..., description = "Unique document identifier")
    filename                : str                        = Field(..., description = "Original filename")
    file_path               : Optional[Path]             = Field(None, description = "Path to uploaded file")
    document_type           : DocumentType               = Field(..., description = "Type of document")
    
    # Content metadata
    title                   : Optional[str]              = Field(None, description = "Document title")
    author                  : Optional[str]              = Field(None, description = "Document author")
    created_date            : Optional[datetime]         = Field(None, description = "Document creation date")
    modified_date           : Optional[datetime]         = Field(None, description = "Last modification date")
    
    # Processing metadata
    upload_date             : datetime                   = Field(default_factory = datetime.now)
    processed_date          : Optional[datetime]         = Field(None)
    status                  : ProcessingStatus           = Field(default = ProcessingStatus.PENDING)
    
    # Size metrics
    file_size_bytes         : int                        = Field(..., gt = 0, description = "File size in bytes")
    num_pages               : Optional[int]              = Field(None, ge = 1, description = "Number of pages (PDFs)")
    num_tokens              : Optional[int]              = Field(None, ge = 0, description = "Total tokens")
    num_chunks              : Optional[int]              = Field(None, ge = 0, description = "Number of chunks")
    
    # Processing info
    chunking_strategy       : Optional[ChunkingStrategy] = Field(None)
    processing_time_seconds : Optional[float]            = Field(None, ge = 0.0)
    error_message           : Optional[str]              = Field(None)
    
    # Additional metadata
    extra                   : Dict[str, Any]             = Field(default_factory = dict)
    

    @field_validator("file_size_bytes")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """
        Ensure file size is reasonable
        """
        max_size = 2 * 1024 * 1024 * 1024  # 2GB
        
        if (v > max_size):
            raise ValueError(f"File size {v} exceeds maximum {max_size}")

        return v
    
    @property
    def file_size_mb(self) -> float:
        """
        File size in megabytes
        """
        return self.file_size_bytes / (1024 * 1024)



class DocumentChunk(BaseModel):
    """
    A single chunk of text from a document
    """
    chunk_id        : str                   = Field(..., description = "Unique chunk identifier")
    document_id     : str                   = Field(..., description = "Parent document ID")
    
    # Content
    text            : str                   = Field(..., min_length = 1, description = "Chunk text content")
    embedding       : Optional[List[float]] = Field(None, description = "Vector embedding")
    
    # Position metadata
    chunk_index     : int                   = Field(..., ge = 0, description = "Chunk position in document")
    start_char      : int                   = Field(..., ge = 0, description = "Start character position")
    end_char        : int                   = Field(..., ge = 0, description = "End character position")
    
    # Page/section info
    page_number     : Optional[int]         = Field(None, ge = 1, description = "Page number (if applicable)")
    section_title   : Optional[str]         = Field(None, description = "Section heading")
    
    # Hierarchical info (for hierarchical chunking)
    parent_chunk_id : Optional[str]         = Field(None)
    child_chunk_ids : List[str]             = Field(default_factory = list)
    
    # Token info
    token_count     : int                   = Field(..., gt = 0, description = "Number of tokens")
    
    # Metadata
    metadata        : Dict[str, Any]        = Field(default_factory = dict)
    
    
    @property
    def char_count(self) -> int:
        """
        Number of characters in chunk
        """
        return self.end_char - self.start_char


class ChunkWithScore(BaseModel):
    """
    Chunk with retrieval score
    """
    chunk            : DocumentChunk
    score            : float          = Field(..., description = "Relevance score (can be any real number)")
    rank             : int            = Field(..., ge = 1, description = "Rank in results")
    retrieval_method : str            = Field('vector', description = "Retrieval method used")
    

    @property
    def citation(self) -> str:
        parts = [self.chunk.document_id]
        
        # Add source filename if available
        if ((hasattr(self.chunk, 'metadata')) and ('filename' in self.chunk.metadata)):
            parts.append(f"file: {self.chunk.metadata['filename']}")
        
        if self.chunk.page_number:
            parts.append(f"page {self.chunk.page_number}")
        
        if self.chunk.section_title:
            parts.append(f"section: {self.chunk.section_title}")
        
        return ", ".join(parts)


# Embedding Request 
class EmbeddingRequest(BaseModel):
    texts      : List[str]
    normalize  : bool          = True
    device     : Optional[str] = None
    batch_size : Optional[int] = None


# Query Models 
class QueryRequest(BaseModel):
    """
    User query request
    """
    model_config                           = ConfigDict(protected_namespaces = ())

    query            : str                 = Field(..., min_length = 1, max_length = 1000, description = "User question")
    
    # Retrieval parameters
    top_k            : Optional[int]       = Field(5, ge = 1, le = 20, description = "Number of chunks to retrieve")
    enable_reranking : Optional[bool]      = Field(False)
    
    # Generation parameters
    temperature      : Optional[float]     = Field(0.1, ge = 0.0, le = 1.0)
    top_p            : Optional[float]     = Field(0.9, ge = 0.0, le = 1.0)
    max_tokens       : Optional[int]       = Field(1000, ge = 50, le = 4000)
    
    # Filters
    document_ids     : Optional[List[str]] = Field(None, description = "Filter by specific documents")
    date_from        : Optional[datetime]  = Field(None)
    date_to          : Optional[datetime]  = Field(None)
    
    # Response preferences
    include_sources  : bool                = Field(True, description = "Include source citations")
    include_metrics  : bool                = Field(False, description = "Include quality metrics")
    stream           : bool                = Field(False, description = "Stream response tokens")


class QueryResponse(BaseModel):
    """
    Response to user query
    """
    query              : str                        = Field(..., description = "Original query")
    answer             : str                        = Field(..., description = "Generated answer")
    
    # Retrieved context
    sources            : List[ChunkWithScore]       = Field(default_factory = list)
    
    # Metrics
    retrieval_time_ms  : float                      = Field(..., ge = 0.0)
    generation_time_ms : float                      = Field(..., ge = 0.0)
    total_time_ms      : float                      = Field(..., ge = 0.0)
    
    tokens_used        : Optional[Dict[str, int]]   = Field(None)  # {input: X, output: Y}
    
    # Quality metrics (if enabled)
    metrics            : Optional[Dict[str, float]] = Field(None)
    
    # Metadata
    timestamp          : datetime                   = Field(default_factory = datetime.now)
    model_used         : str                        = Field(...)
    
    model_config                                    = ConfigDict(protected_namespaces = ())

    @property
    def citation_text(self) -> str:
        """
        Format citations as text
        """
        if not self.sources:
            return ""
        
        citations = list()

        for i, source in enumerate(self.sources, 1):
            citations.append(f"[{i}] {source.citation}")
        
        return "\n".join(citations)


# Upload Models
class UploadRequest(BaseModel):
    """
    File upload request metadata
    """
    filename        : str           = Field(..., min_length = 1)
    file_size_bytes : int           = Field(..., gt = 0)
    content_type    : Optional[str] = Field(None)
    

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """
        Ensure filename is safe
        """
        # Remove path traversal attempts
        v = Path(v).name
        
        if not v or v.startswith("."):
            raise ValueError("Invalid filename")
        
        return v


class UploadResponse(BaseModel):
    """
    File upload response
    """
    document_id : str              = Field(..., description = "Generated document ID")
    filename    : str              = Field(...)
    status      : ProcessingStatus = Field(...)
    message     : str              = Field(...)
    upload_date : datetime         = Field(default_factory = datetime.now)


class ProcessingProgress(BaseModel):
    """
    Real-time processing progress
    """
    document_id                 : str              = Field(...)
    status                      : ProcessingStatus = Field(...)
    
    # Progress tracking
    progress_percentage         : float            = Field(0.0, ge = 0.0, le = 100.0)
    current_step                : str              = Field(..., description = "Current processing step")
    
    # Stats
    chunks_processed            : int              = Field(0, ge = 0)
    total_chunks                : Optional[int]    = Field(None)
    
    # Timing
    start_time                  : datetime         = Field(...)
    elapsed_seconds             : float            = Field(0.0, ge = 0.0)
    estimated_remaining_seconds : Optional[float]  = Field(None)
    
    # Messages
    log_messages                : List[str]        = Field(default_factory = list)
    error_message               : Optional[str]    = Field(None)


# Embedding Models
class EmbeddingRequest(BaseModel):
    """
    Request to generate embeddings
    """
    texts      : List[str]     = Field(..., min_length = 1, max_length = 1000)
    batch_size : Optional[int] = Field(32, ge = 1, le = 128)
    normalize  : bool          = Field(True, description = "Normalize embeddings to unit length")


class EmbeddingResponse(BaseModel):
    """
    Embedding generation response
    """
    embeddings         : List[List[float]] = Field(...)
    dimension          : int               = Field(..., gt = 0)
    num_embeddings     : int               = Field(..., gt = 0)
    processing_time_ms : float             = Field(..., ge = 0.0)


# Retrieval Models
class RetrievalRequest(BaseModel):
    """
    Request for document retrieval
    """
    query         : str                 = Field(..., min_length = 1)
    top_k         : int                 = Field(10, ge = 1, le = 100)
    
    # Retrieval method
    use_vector    : bool                = Field(True)
    use_bm25      : bool                = Field(True)
    vector_weight : Optional[float]     = Field(0.6, ge = 0.0, le = 1.0)
    
    # Filters
    document_ids  : Optional[List[str]] = Field(None)
    min_score     : Optional[float]     = Field(None, ge = 0.0, le = 1.0)


class RetrievalResponse(BaseModel):
    """
    Document retrieval response
    """
    chunks            : List[ChunkWithScore] = Field(...)
    retrieval_time_ms : float                = Field(..., ge = 0.0)
    num_candidates    : int                  = Field(..., ge = 0)


# Evaluation Models
class EvaluationRequest(BaseModel):
    """
    Request for RAG evaluation
    """
    query            : str                  = Field(..., description = "Original user query")
    reference_answer : Optional[str]        = Field(None, description = "Ground truth answer for evaluation")
    context_chunks   : List[ChunkWithScore] = Field(..., description = "Retrieved context chunks")
    generated_answer : str                  = Field(..., description = "LLM-generated answer to evaluate")


class EvaluationResult(BaseModel):
    """
    RAG evaluation results using Ragas metrics
    """
    answer_relevancy   : float           = Field(..., ge = 0.0, le = 1.0, description = "How well answer addresses question")
    faithfulness       : float           = Field(..., ge = 0.0, le = 1.0, description = "Is answer grounded in context")
    context_precision  : float           = Field(..., ge = 0.0, le = 1.0, description = "Are relevant chunks ranked high")
    context_recall     : Optional[float] = Field(None, ge = 0.0, le = 1.0, description = "Was all necessary info retrieved")
    overall_score      : float           = Field(..., ge = 0.0, le = 1.0, description = "Composite evaluation score")
    evaluation_time_ms : float           = Field(..., ge=0.0)
    model_used         : str             = Field(..., description = "Ragas evaluation model")
    timestamp          : datetime        = Field(default_factory = datetime.now)


# System Models
class HealthCheck(BaseModel):
    """
    System health check response
    """
    status                    : Literal["healthy", "degraded", "unhealthy"] = Field(...)
    timestamp                 : datetime                                    = Field(default_factory = datetime.now)
    
    # Component status
    ollama_available          : bool                                        = Field(...)
    vector_store_available    : bool                                        = Field(...)
    embedding_model_available : bool                                        = Field(...)
    
    # Stats
    total_documents           : int                                         = Field(0, ge = 0)
    total_chunks              : int                                         = Field(0, ge = 0)
    
    # Version info
    version                   : str                                         = Field(...)
    
    # Issues
    warnings                  : List[str]                                   = Field(default_factory = list)
    errors                    : List[str]                                   = Field(default_factory = list)


class SystemStats(BaseModel):
    """
    System statistics
    """
    # Document stats
    total_documents     : int            = Field(0, ge = 0)
    documents_by_type   : Dict[str, int] = Field(default_factory = dict)
    total_file_size_mb  : float          = Field(0.0, ge = 0.0)
    
    # Chunk stats
    total_chunks        : int            = Field(0, ge = 0)
    avg_chunk_size      : float          = Field(0.0, ge = 0.0)
    
    # Query stats
    total_queries       : int            = Field(0, ge = 0)
    avg_query_time_ms   : float          = Field(0.0, ge = 0.0)
    avg_retrieval_score : float          = Field(0.0, ge = 0.0)
    
    # Timestamp
    generated_at        : datetime       = Field(default_factory = datetime.now)


class ErrorResponse(BaseModel):
    """
    Standard error response
    """
    error      : str                      = Field(..., description = "Error type")
    message    : str                      = Field(..., description = "Human-readable error message")
    detail     : Optional[Dict[str, Any]] = Field(None, description = "Additional error details")
    timestamp  : datetime                 = Field(default_factory = datetime.now)
    request_id : Optional[str]            = Field(None)


# Configuration Models
class ChunkingConfig(BaseModel):
    """
    Chunking configuration
    """
    strategy           : ChunkingStrategy = Field(...)
    chunk_size         : int              = Field(..., gt = 0)
    overlap            : int              = Field(..., ge = 0)
    
    # Strategy-specific params
    semantic_threshold : Optional[float]  = Field(None, ge = 0.0, le = 1.0)
    parent_size        : Optional[int]    = Field(None, gt = 0)
    child_size         : Optional[int]    = Field(None, gt = 0)


class RetrievalConfig(BaseModel):
    """
    Retrieval configuration
    """
    top_k            : int   = Field(10, ge = 1, le = 100)
    vector_weight    : float = Field(0.6, ge = 0.0, le = 1.0)
    bm25_weight      : float = Field(0.4, ge = 0.0, le = 1.0)
    enable_reranking : bool  = Field(False)
    faiss_nprobe     : int   = Field(10, ge = 1, le = 100)
    

    @field_validator("bm25_weight")
    @classmethod
    def validate_weights(cls, v: float, info) -> float:
        """
        Ensure weights sum to 1.0
        """
        if ("vector_weight" in info.data):
            vector_weight = info.data["vector_weight"]

            if (abs(vector_weight + v - 1.0) > 0.01):
                raise ValueError("vector_weight + bm25_weight must equal 1.0")

        return v


# Chat Response
class ChatRequest(BaseModel):
    message    : str
    session_id : Optional[str] = None


# Validation Utilities
def validate_document_id(document_id: str) -> bool:
    """
    Validate document ID format
    """
    # Format: doc_<timestamp>_<hash>
    pattern = r'^doc_\d{10,}_[a-f0-9]{8}$'

    return bool(re.match(pattern, document_id))


def validate_chunk_id(chunk_id: str) -> bool:
    """
    Validate chunk ID format
    """
    # Format: chunk_<doc_id>_<index>
    pattern = r'^chunk_doc_\d+_[a-f0-9]{8}_\d+$'

    return bool(re.match(pattern, chunk_id))

    