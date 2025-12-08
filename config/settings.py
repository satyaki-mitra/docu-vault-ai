# DEPENDENCIES
import os
import time
import torch
from pathlib import Path
from pydantic import Field
from typing import Literal
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration with environment variable support
    
    Environment variables take precedence over defaults
    """
    # Application Settings
    APP_NAME                      : str                                                      = "AI Universal Knowledge Ingestion System"
    APP_VERSION                   : str                                                      = "1.0.0"
    DEBUG                         : bool                                                     = Field(default = False, description = "Enable debug mode")
    HOST                          : str                                                      = Field(default = "0.0.0.0", description = "API host")
    PORT                          : int                                                      = Field(default = 8000, description = "API port")
    
    # File Upload Settings
    MAX_FILE_SIZE_MB              : int                                                      = Field(default = 100, description = "Max file size in MB")
    MAX_BATCH_FILES               : int                                                      = Field(default = 10, description = "Max files per upload")
    ALLOWED_EXTENSIONS            : list[str]                                                = Field(default = ["pdf", "docx", "txt"], description = "Allowed file extensions")
    UPLOAD_DIR                    : Path                                                     = Field(default = Path("data/uploads"), description = "Directory for uploaded files")
    
    # Ollama LLM Settings
    OLLAMA_BASE_URL               : str                                                      = Field(default = "http://localhost:11434", description = "Ollama API endpoint")
    OLLAMA_MODEL                  : str                                                      = Field(default = "mistral:7b", description = "Ollama model name")
    OLLAMA_TIMEOUT                : int                                                      = Field(default = 120, description = "Ollama request timeout (seconds)")
    
    # Generation parameters
    DEFAULT_TEMPERATURE           : float                                                    = Field(default = 0.1, ge = 0.0, le = 1.0, description = "LLM temperature (0=deterministic, 1=creative)")
    TOP_P                         : float                                                    = Field(default = 0.9, ge = 0.0, le = 1.0, description = "Nucleus sampling threshold")
    MAX_TOKENS                    : int                                                      = Field(default = 1000, description = "Max output tokens")
    CONTEXT_WINDOW                : int                                                      = Field(default = 8192, description = "Model context window size")
    
    # OpenAI Settings
    OPENAI_API_KEY                : Optional[str]                                            = Field(default = None, description = "Open AI API secret key")
    OPENAI_MODEL                  : str                                                      = Field(default = "gpt-3.5-turbo", description = "Ollama model name")
   
    # Embedding Settings
    EMBEDDING_MODEL               : str                                                      = Field(default = "BAAI/bge-small-en-v1.5", description = "HuggingFace embedding model")
    EMBEDDING_DIMENSION           : int                                                      = Field(default = 384, description = "Embedding vector dimension")
    EMBEDDING_DEVICE              : Literal["cpu", "cuda", "mps"]                            = Field(default = "cpu", description = "Device for embedding generation")
    EMBEDDING_BATCH_SIZE          : int                                                      = Field(default = 32, description = "Batch size for embedding generation")
    
    # Chunking Settings
    # Fixed chunking
    FIXED_CHUNK_SIZE              : int                                                      = Field(default = 512, description = "Fixed chunk size in tokens")
    FIXED_CHUNK_OVERLAP           : int                                                      = Field(default = 25, description = "Overlap between chunks")
    
    # Semantic chunking
    SEMANTIC_BREAKPOINT_THRESHOLD : float                                                    = Field(default = 0.80, description = "Percentile for semantic breakpoints")

    # Hierarchical chunking
    PARENT_CHUNK_SIZE             : int                                                      = Field(default = 2048, description = "Parent chunk size")
    CHILD_CHUNK_SIZE              : int                                                      = Field(default = 512, description = "Child chunk size")
    
    # Adaptive thresholds
    SMALL_DOC_THRESHOLD           : int                                                      = Field(default = 1000, description = "Token threshold for fixed chunking")
    LARGE_DOC_THRESHOLD           : int                                                      = Field(default = 500000, description = "Token threshold for hierarchical chunking")
    
    # Retrieval Settings
    # Vector search
    TOP_K_RETRIEVE                : int                                                      = Field(default = 10, description = "Top chunks to retrieve")
    TOP_K_FINAL                   : int                                                      = Field(default = 5, description = "Final chunks after reranking")
    FAISS_NPROBE                  : int                                                      = Field(default = 10, description = "FAISS search probes")
    
    # Hybrid search weights
    VECTOR_WEIGHT                 : float                                                    = Field(default = 0.6, description = "Vector search weight")
    BM25_WEIGHT                   : float                                                    = Field(default = 0.4, description = "BM25 search weight")
    
    # BM25 parameters
    BM25_K1                       : float                                                    = Field(default = 1.5, description = "BM25 term saturation")
    BM25_B                        : float                                                    = Field(default = 0.75, description = "BM25 length normalization")
    
    # Reranking
    ENABLE_RERANKING              : bool                                                     = Field(default = True, description = "Enable cross-encoder reranking")
    RERANKER_MODEL                : str                                                      = Field(default = "cross-encoder/ms-marco-MiniLM-L-6-v2", description = "Reranker model")
    
    # Storage Settings
    VECTOR_STORE_DIR              : Path                                                     = Field(default = Path("data/vector_store"), description = "FAISS index storage")
    METADATA_DB_PATH              : Path                                                     = Field(default = Path("data/metadata.db"), description = "SQLite metadata database")
    
    # Backup
    AUTO_BACKUP                   : bool                                                     = Field(default = True, description = "Enable auto-backup")
    BACKUP_INTERVAL               : int                                                      = Field(default = 1000, description = "Backup every N documents")
    BACKUP_DIR                    : Path                                                     = Field(default = Path("data/backups"), description = "Backup directory")
    
    # Cache Settings
    ENABLE_CACHE                  : bool                                                     = Field(default = True, description = "Enable embedding cache")
    CACHE_TYPE                    : Literal["memory", "redis"]                               = Field(default = "memory", description = "Cache backend")
    CACHE_TTL                     : int                                                      = Field(default = 3600, description = "Cache TTL in seconds")
    CACHE_MAX_SIZE                : int                                                      = Field(default = 1000, description = "Max cached items")
    
    # Redis (if used)
    REDIS_HOST                    : str                                                      = Field(default = "localhost", description = "Redis host")
    REDIS_PORT                    : int                                                      = Field(default = 6379, description = "Redis port")
    REDIS_DB                      : int                                                      = Field(default = 0, description = "Redis database number")
    
    # Logging Settings
    LOG_LEVEL                     : Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default = "INFO", description = "Logging level")
    LOG_DIR                       : Path                                                     = Field(default = Path("logs"), description = "Log file directory")
    LOG_FORMAT                    : str                                                      = Field(default = "%(asctime)s - %(name)s - %(levelname)s - %(message)s", description = "Log format string")
    LOG_ROTATION                  : str                                                      = Field(default = "500 MB", description = "Log rotation size")
    LOG_RETENTION                 : str                                                      = Field(default = "30 days", description = "Log retention period")
    
    # Evaluation Settings
    ENABLE_RAGAS                  : bool                                                     = Field(default = True, description = "Enable Ragas evaluation")
    RAGAS_ENABLE_GROUND_TRUTH     : bool                                                     = Field(default = False, description = "Enable RAGAS metrics requiring ground truth")
    RAGAS_METRICS                 : list[str]                                                = Field(default = ["answer_relevancy", "faithfulness", "context_utilization", "context_relevancy"], description = "Ragas metrics to compute (base metrics without ground truth)")
    RAGAS_GROUND_TRUTH_METRICS    : list[str]                                                = Field(default = ["context_precision", "context_recall", "answer_similarity", "answer_correctness"], description = "Ragas metrics requiring ground truth")
    RAGAS_EVALUATION_TIMEOUT      : int                                                      = Field(default = 60, description = "RAGAS evaluation timeout in seconds")
    RAGAS_BATCH_SIZE              : int                                                      = Field(default = 10, description = "Batch size for RAGAS evaluations")

    # Web Scraping Settings (for future)
    SCRAPING_ENABLED              : bool                                                     = Field(default = False, description = "Enable web scraping")
    USER_AGENT                    : str                                                      = Field(default = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", description = "User agent for scraping")
    REQUEST_DELAY                 : float                                                    = Field(default = 2.0, description = "Delay between requests (seconds)")
    MAX_RETRIES                   : int                                                      = Field(default = 3, description = "Max scraping retries")
    
    # Performance Settings
    MAX_WORKERS                   : int                                                      = Field(default = 4, description = "Max parallel workers")
    ASYNC_BATCH_SIZE              : int                                                      = Field(default = 10, description = "Async batch size")
    
    # Security Settings
    ENABLE_AUTH                   : bool                                                     = Field(default = False, description = "Enable authentication")
    SECRET_KEY                    : str                                                      = Field(default = os.getenv("SECRET_KEY", "dev-key-change-in-production"))
    
    FIXED_CHUNK_STRATEGY          : str                                                      = Field(default = "fixed", description = "Default chunking strategy")
    

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"
        case_sensitive    = True
    

    @field_validator("UPLOAD_DIR", "VECTOR_STORE_DIR", "LOG_DIR", "BACKUP_DIR", "METADATA_DB_PATH")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """
        Ensure directories exist
        """
        if v.suffix:  # It's a file path (like metadata.db)
            v.parent.mkdir(parents = True, exist_ok = True)

        else:  # It's a directory
            v.mkdir(parents = True, exist_ok = True)

        return v

    
    @field_validator("VECTOR_WEIGHT", "BM25_WEIGHT")
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """
        Ensure vector and BM25 weights are valid
        """
        if ((info.field_name == "BM25_WEIGHT") and ("VECTOR_WEIGHT" in info.data)):
            vector_weight = info.data["VECTOR_WEIGHT"]
            
            if (abs(vector_weight + v - 1.0) > 0.01):
                raise ValueError(f"VECTOR_WEIGHT ({vector_weight}) + BM25_WEIGHT ({v}) must sum to 1.0")
        
        return v
    

    @property
    def max_file_size_bytes(self) -> int:
        """
        Convert MB to bytes
        """
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    

    @property
    def is_cuda_available(self) -> bool:
        """
        Check if CUDA device is requested and available
        """
        if self.EMBEDDING_DEVICE == "cuda":
            try:
                return torch.cuda.is_available()

            except ImportError:
                return False

        return False
    

    def get_ollama_url(self, endpoint: str) -> str:
        """
        Construct full Ollama API URL
        """
        return f"{self.OLLAMA_BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    

    @classmethod
    def get_timestamp_ms(cls) -> int:
        """
        Get current timestamp in milliseconds
        """
        return int(time.time() * 1000)


    def summary(self) -> dict:
        """
        Get configuration summary (excluding sensitive data)
        """
        return {"app_name"           : self.APP_NAME,
                "version"            : self.APP_VERSION,
                "ollama_model"       : self.OLLAMA_MODEL,
                "embedding_model"    : self.EMBEDDING_MODEL,
                "embedding_device"   : self.EMBEDDING_DEVICE,
                "max_file_size_mb"   : self.MAX_FILE_SIZE_MB,
                "allowed_extensions" : self.ALLOWED_EXTENSIONS,
                "chunking_strategy"  : {"small_threshold" : self.SMALL_DOC_THRESHOLD, "large_threshold" : self.LARGE_DOC_THRESHOLD},
                "retrieval"          : {"top_k" : self.TOP_K_RETRIEVE, "hybrid_weights" : {"vector" : self.VECTOR_WEIGHT, "bm25" : self.BM25_WEIGHT}},
                "evaluation"         : {"ragas_enabled" : self.ENABLE_RAGAS, "ragas_ground_truth" : self.RAGAS_ENABLE_GROUND_TRUTH, "ragas_metrics" : self.RAGAS_METRICS},
               }


# Global settings instance
settings = Settings()


# Convenience function for getting settings
def get_settings() -> Settings:
    """
    Get global settings instance
    """
    return settings