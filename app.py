# DEPENDENCIES
import os
import gc
import io
import csv
import json
import time
import signal
import atexit
import shutil
import asyncio
import logging
import uvicorn
import tempfile
import traceback
import threading
from typing import Set
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
from typing import Tuple
from fastapi import File
from fastapi import Form
from signal import SIGINT
from signal import SIGTERM
from pydantic import Field
from fastapi import FastAPI
from typing import Optional
from datetime import datetime
from datetime import timedelta
from fastapi import UploadFile
from pydantic import BaseModel
from fastapi import HTTPException
from config.models import PromptType
from config.models import ChatRequest
from config.models import LLMProvider
from utils.helpers import IDGenerator
from config.models import QueryRequest 
from config.settings import get_settings
from config.models import RAGASStatistics
from config.models import RAGASExportData
from config.models import DocumentMetadata
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from utils.file_handler import FileHandler
from utils.validators import FileValidator
from fastapi.staticfiles import StaticFiles
from utils.error_handler import RAGException
from utils.error_handler import FileException
from config.models import RAGASEvaluationResult
from config.logging_config import setup_logging
from generation.llm_client import get_llm_client
from embeddings.bge_embedder import get_embedder
from concurrent.futures import ThreadPoolExecutor
from ingestion.router import get_ingestion_router
from utils.validators import validate_upload_file
from fastapi.middleware.cors import CORSMiddleware
from vector_store.index_builder import get_index_builder
from document_parser.parser_factory import ParserFactory
from evaluation.ragas_evaluator import get_ragas_evaluator
from vector_store.metadata_store import get_metadata_store
from embeddings.embedding_cache import get_embedding_cache
from ingestion.progress_tracker import get_progress_tracker
from retrieval.hybrid_retriever import get_hybrid_retriever
from chunking.adaptive_selector import get_adaptive_selector
from retrieval.context_assembler import get_context_assembler
from document_parser.parser_factory import get_parser_factory
from chunking.adaptive_selector import AdaptiveChunkingSelector 
from generation.response_generator import get_response_generator
from config.models import ProcessingStatus as ProcessingStatusEnum


# Setup logging and settings
settings = get_settings()
logger   = setup_logging(log_level      = settings.LOG_LEVEL,
                         log_dir        = settings.LOG_DIR,
                         enable_console = True,
                         enable_file    = True,
                        )


# Global Cleanup Variables 
_cleanup_registry : Set[str] = set()
_cleanup_lock                = threading.RLock()
_is_cleaning                 = False
_cleanup_executor            = ThreadPoolExecutor(max_workers        = 2, 
                                                  thread_name_prefix = "cleanup_",
                                                 )


# Analytics Cache Structure
class AnalyticsCache:
    """
    Cache for analytics data to avoid recalculating on every request
    """
    def __init__(self, ttl_seconds: int = 30):
        self.data            = None
        self.last_calculated = None
        self.ttl_seconds     = ttl_seconds
        self.is_calculating  = False
    

    def is_valid(self) -> bool:
        """
        Check if cache is still valid
        """
        if self.data is None or self.last_calculated is None:
            return False
        
        elapsed = (datetime.now() - self.last_calculated).total_seconds()
        
        return (elapsed < self.ttl_seconds)
    

    def update(self, data: Dict):
        """
        Update cache with new data
        """
        self.data            = data
        self.last_calculated = datetime.now()
    
    
    def get(self) -> Optional[Dict]:
        """
        Get cached data if valid
        """
        return self.data if self.is_valid() else None


class CleanupManager:
    """
    Centralized cleanup manager with multiple redundancy layers
    """
    @staticmethod
    def register_resource(resource_id: str, cleanup_func, *args, **kwargs):
        """
        Register a resource for cleanup
        """
        with _cleanup_lock:
            _cleanup_registry.add(resource_id)
        
        # Register with atexit for process termination
        atexit.register(cleanup_func, *args, **kwargs)
        
        return resource_id
    

    @staticmethod
    def unregister_resource(resource_id: str):
        """
        Unregister a resource (if already cleaned up elsewhere)
        """
        with _cleanup_lock:
            if resource_id in _cleanup_registry:
                _cleanup_registry.remove(resource_id)
    

    @staticmethod
    async def full_cleanup(state: Optional['AppState'] = None) -> bool:
        """
        Perform full system cleanup with redundancy
        """
        global _is_cleaning
        
        with _cleanup_lock:
            if _is_cleaning:
                logger.warning("Cleanup already in progress")
                return False
            
            _is_cleaning = True
        
        try:
            logger.info("Starting comprehensive system cleanup...")
            
            # Layer 1: Memory cleanup
            success1 = await CleanupManager._cleanup_memory(state)
            
            # Layer 2: Disk cleanup (async to not block)
            success2 = await CleanupManager._cleanup_disk_async()
            
            # Layer 3: Component cleanup
            success3 = await CleanupManager._cleanup_components(state)
            
            # Layer 4: External resources
            success4 = CleanupManager._cleanup_external_resources()
            
            # Clear registry
            with _cleanup_lock:
                _cleanup_registry.clear()
            
            overall_success = all([success1, success2, success3, success4])
            
            if overall_success:
                logger.info("Comprehensive cleanup completed successfully")
            
            else:
                logger.warning("Cleanup completed with some failures")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Cleanup failed catastrophically: {e}", exc_info=True)
            return False
        
        finally:
            with _cleanup_lock:
                _is_cleaning = False
    
    @staticmethod
    async def _cleanup_memory(state: Optional['AppState']) -> bool:
        """
        Memory cleanup
        """
        try:
            if not state:
                logger.warning("No AppState provided for memory cleanup")
                return True
            
            # Session cleanup
            session_count = len(state.active_sessions)
            state.active_sessions.clear()
            state.config_overrides.clear()
            logger.info(f"Cleared {session_count} sessions from memory")
            
            # Document data cleanup
            doc_count   = len(state.processed_documents)
            chunk_count = sum(len(chunks) for chunks in state.document_chunks.values())

            state.processed_documents.clear()
            state.document_chunks.clear()
            state.uploaded_files.clear()
            logger.info(f"Cleared {doc_count} documents ({chunk_count} chunks) from memory")
            
            # Performance data cleanup
            state.query_timings.clear()
            state.chunking_statistics.clear()
            
            # State reset
            state.is_ready          = False
            state.processing_status = "idle"
            
            # Cache cleanup
            if hasattr(state, 'analytics_cache'):
                state.analytics_cache.data = None
            
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"🧹 Garbage collection freed {collected} objects")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    @staticmethod
    async def _cleanup_disk_async() -> bool:
        """
        Asynchronous disk cleanup
        """
        try:
            # Run in thread pool to avoid blocking
            loop    = asyncio.get_event_loop()
            success = await loop.run_in_executor(_cleanup_executor, CleanupManager._cleanup_disk_sync)

            return success

        except Exception as e:
            logger.error(f"Async disk cleanup failed: {e}")
            return False
    

    @staticmethod
    def _cleanup_disk_sync() -> bool:
        """
        Synchronous disk cleanup
        """
        try:
            logger.info("Starting disk cleanup...")
            
            # Track what we clean
            cleaned_paths = list()
            
            # Vector store directory
            if settings.VECTOR_STORE_DIR.exists():
                vector_files = list(settings.VECTOR_STORE_DIR.glob("*"))
                for file in vector_files:
                    try:
                        if file.is_file():
                            file.unlink()
                            cleaned_paths.append(str(file))
                        
                        elif file.is_dir():
                            shutil.rmtree(file)
                            cleaned_paths.append(str(file))
                    
                    except Exception as e:
                        logger.warning(f"Failed to delete {file}: {e}")
                
                logger.info(f"Cleaned {len(cleaned_paths)} vector store files")
            
            # Upload directory (preserve directory structure)
            if settings.UPLOAD_DIR.exists():
                upload_files = list(settings.UPLOAD_DIR.glob("*"))
                for file in upload_files:
                    try:
                        if file.is_file():
                            file.unlink()
                            cleaned_paths.append(str(file))
                        
                        elif file.is_dir():
                            shutil.rmtree(file)
                            cleaned_paths.append(str(file))
                    
                    except Exception as e:
                        logger.warning(f"Failed to delete {file}: {e}")
                
                # Recreate empty directory
                settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleaned {len(upload_files)} uploaded files")
            
            # Metadata database
            metadata_path = Path(settings.METADATA_DB_PATH)
            if metadata_path.exists():
                try:
                    metadata_path.unlink(missing_ok=True)
                    cleaned_paths.append(str(metadata_path))
                    logger.info("Cleaned metadata database")
                
                except Exception as e:
                    logger.warning(f"Failed to delete metadata DB: {e}")
            
            # Backup directory
            if settings.BACKUP_DIR.exists():
                backup_files = list(settings.BACKUP_DIR.glob("*"))
                for file in backup_files:
                    try:
                        if file.is_file():
                            file.unlink()
                        
                        elif file.is_dir():
                            shutil.rmtree(file)
                    
                    except:
                        pass  # Silently fail for backups
                logger.info(f"Cleaned {len(backup_files)} backup files")
            
            # Temp files cleanup
            CleanupManager._cleanup_temp_files()
            
            logger.info(f"Disk cleanup completed: {len(cleaned_paths)} items cleaned")
            
            return True
            
        except Exception as e:
            logger.error(f"Disk cleanup failed: {e}")
            return False
    

    @staticmethod
    def _cleanup_temp_files():
        """
        Clean up temporary files
        """
        temp_dir = tempfile.gettempdir()
        
        # Clean our specific temp files (if any)
        for pattern in ["rag_*", "faiss_*", "embedding_*"]:
            for file in Path(temp_dir).glob(pattern):
                try:
                    file.unlink(missing_ok=True)
                except:
                    pass
    

    @staticmethod
    async def _cleanup_components(state: Optional['AppState']) -> bool:
        """
        Component-specific cleanup
        """
        try:
            if not state:
                return True
            
            components_cleaned = 0
            
            # Vector store components
            if state.index_builder:
                try:
                    state.index_builder.clear_indexes()
                    components_cleaned += 1
                
                except Exception as e:
                    logger.warning(f"Index builder cleanup failed: {e}")
            
            if state.metadata_store and hasattr(state.metadata_store, 'clear_all'):
                try:
                    state.metadata_store.clear_all()
                    components_cleaned += 1
                
                except Exception as e:
                    logger.warning(f"Metadata store cleanup failed: {e}")
            
            # RAGAS evaluator
            if state.ragas_evaluator and hasattr(state.ragas_evaluator, 'clear_history'):
                try:
                    state.ragas_evaluator.clear_history()
                    components_cleaned += 1
                
                except Exception as e:
                    logger.warning(f"RAGAS evaluator cleanup failed: {e}")
            
            logger.info(f"Cleaned {components_cleaned} components")
            return True
            
        except Exception as e:
            logger.error(f"Component cleanup failed: {e}")
            return False
    

    @staticmethod
    def _cleanup_external_resources() -> bool:
        """
        External resource cleanup
        """
        try:
            # Close database connections
            CleanupManager._close_db_connections()
            
            # Clean up thread pool
            _cleanup_executor.shutdown(wait = False)
            
            logger.info("External resources cleaned")
            return True
            
        except Exception as e:
            logger.error(f"External resource cleanup failed: {e}")
            return False
    

    @staticmethod
    def _close_db_connections():
        """
        Close any open database connections
        """
        try:
            # SQLite handles this automatically in most cases
            pass
        except:
            pass

    
    @staticmethod
    def handle_signal(signum, frame):
        """
        Signal handler for graceful shutdown
        """
        global _is_cleaning
        
        # If already cleaning up, don't raise KeyboardInterrupt
        with _cleanup_lock:
            if _is_cleaning:
                logger.info(f"Signal {signum} received during cleanup - ignoring")
                return
            
        if (signum == SIGINT):
            logger.info("Ctrl+C received - shutdown initiated")
            raise KeyboardInterrupt
        
        elif (signum == SIGTERM):
            logger.info("SIGTERM received - shutdown initiated")
            # Just log, not scheduling anything
        
        else:
            logger.info(f"Signal {signum} received")


# Global state manager
class AppState:
    """
    Manages application state and components
    """
    def __init__(self):
        self.is_ready            = False
        self.processing_status   = "idle"
        self.uploaded_files      = list()
        self.active_sessions     = dict()
        self.processed_documents = dict()
        self.document_chunks     = dict()
        
        # Performance tracking
        self.query_timings       = list()  
        self.chunking_statistics = dict()
        
        # Core components
        self.file_handler        = None
        self.parser_factory      = None
        self.chunking_selector   = None
        
        # Embeddings components
        self.embedder            = None
        self.embedding_cache     = None
        
        # Ingestion components
        self.ingestion_router    = None
        self.progress_tracker    = None
        
        # Vector store components
        self.index_builder       = None
        self.metadata_store      = None
        
        # Retrieval components
        self.hybrid_retriever    = None
        self.context_assembler   = None
        
        # Generation components
        self.response_generator  = None
        self.llm_client          = None
        
        # RAGAS component
        self.ragas_evaluator     = None

        # Processing tracking
        self.current_processing  = None
        self.processing_progress = {"status"       : "idle",
                                    "current_step" : "Waiting",
                                    "progress"     : 0,
                                    "processed"    : 0,
                                    "total"        : 0,
                                    "details"      : {},
                                   }

        # Session-based configuration overrides
        self.config_overrides    = dict()
        
        # Analytics cache
        self.analytics_cache     = AnalyticsCache(ttl_seconds = 30)
        
        # System start time
        self.start_time          = datetime.now()

        # Add cleanup tracking 
        self._cleanup_registered = False
        self._cleanup_resources  = list()
        
        # Register with cleanup manager
        self._register_for_cleanup()


    def _register_for_cleanup(self):
        """
        Register this AppState instance for cleanup
        """
        if not self._cleanup_registered:
            resource_id              = f"appstate_{id(self)}"

            CleanupManager.register_resource(resource_id, self._emergency_cleanup)
            self._cleanup_resources.append(resource_id)
            
            self._cleanup_registered = True
    

    def _emergency_cleanup(self):
        """
        Emergency cleanup if regular cleanup fails
        """
        try:
            logger.warning("Performing emergency cleanup...")
            
            # Brutal but effective memory clearing
            for attr in ['active_sessions', 'processed_documents', 'document_chunks', 'uploaded_files', 'query_timings', 'chunking_statistics']:
                if hasattr(self, attr):
                    getattr(self, attr).clear()
            
            # Nullify heavy objects
            self.index_builder  = None
            self.metadata_store = None
            self.embedder       = None
            
            logger.warning("Emergency cleanup completed")
        
        except:
            # Last resort - don't crash during emergency cleanup
            pass  
    

    async def graceful_shutdown(self):
        """
        Graceful shutdown procedure
        """
        logger.info("Starting graceful shutdown...")
        
        # Notify clients (if any WebSocket connections)
        await self._notify_clients()
        
        # Perform cleanup
        await CleanupManager.full_cleanup(self)
        
        # Unregister from cleanup manager
        for resource_id in self._cleanup_resources:
            CleanupManager.unregister_resource(resource_id)
        
        logger.info("Graceful shutdown completed")
    

    async def _notify_clients(self):
        """
        Notify connected clients of shutdown
        """
        # Placeholder for WebSocket notifications
        pass
    

    def add_query_timing(self, duration_ms: float):
        """
        Record query timing for analytics
        """
        self.query_timings.append((datetime.now(), duration_ms))
        # Keep only last 1000 timings to prevent memory issues
        if (len(self.query_timings) > 1000):
            self.query_timings = self.query_timings[-1000:]
    

    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics from recorded timings
        """
        if not self.query_timings:
            return {"avg_response_time" : 0,
                    "min_response_time" : 0,
                    "max_response_time" : 0,
                    "total_queries"     : 0,
                    "queries_last_hour" : 0,
                   }
        

        # Get recent timings (last hour)
        one_hour_ago   = datetime.now() - timedelta(hours = 1)
        recent_timings = [t for t, _ in self.query_timings if (t > one_hour_ago)]
        
        # Calculate statistics
        durations      = [duration for _, duration in self.query_timings]
        
        return {"avg_response_time" : int(sum(durations) / len(durations)),
                "min_response_time" : int(min(durations)) if durations else 0,
                "max_response_time" : int(max(durations)) if durations else 0,
                "total_queries"     : len(self.query_timings),
                "queries_last_hour" : len(recent_timings),
                "p95_response_time" : int(sorted(durations)[int(len(durations) * 0.95)]) if (len(durations) > 10) else 0,
               }

    
    def get_chunking_statistics(self) -> Dict:
        """
        Get statistics about chunking strategies used
        """
        if not self.chunking_statistics:
            return {"primary_strategy" : "adaptive",
                    "total_chunks"     : 0,
                    "avg_chunk_size"   : 0,
                    "strategies_used"  : {},
                   }
        
        total_chunks = sum(self.chunking_statistics.values())
        strategies   = {k: v for k, v in self.chunking_statistics.items() if (v > 0)}
        
        return {"primary_strategy" : max(strategies.items(), key=lambda x: x[1])[0] if strategies else "adaptive",
                "total_chunks"     : total_chunks,
                "strategies_used"  : strategies,
               }

    
    def get_system_health(self) -> Dict:
        """
        Get comprehensive system health status
        """
        llm_healthy        = self.llm_client.check_health() if self.llm_client else False
        vector_store_ready = self.is_ready
        
        # Check embedding model
        embedding_ready    = self.embedder is not None
        
        # Check retrieval components
        retrieval_ready    = (self.hybrid_retriever is not None and self.context_assembler is not None)
        
        # Determine overall status
        if all([llm_healthy, vector_store_ready, embedding_ready, retrieval_ready]):
            overall_status = "healthy"
        
        elif vector_store_ready and embedding_ready and retrieval_ready:
            # LLM issues but RAG works
            overall_status = "degraded"  
        
        else:
            overall_status = "unhealthy"
        
        return {"overall"      : overall_status,
                "llm"          : llm_healthy,
                "vector_store" : vector_store_ready,
                "embeddings"   : embedding_ready,
                "retrieval"    : retrieval_ready,
                "generation"   : self.response_generator is not None,
               }

    
    def get_system_information(self) -> Dict:
        """
        Get current system information
        """
        # Get chunking strategy
        chunking_strategy = "adaptive"
        
        if self.chunking_selector:
            try:
                # Try to get strategy from selector
                if (hasattr(self.chunking_selector, 'get_current_strategy')):
                    chunking_strategy = self.chunking_selector.get_current_strategy()
                
                elif (hasattr(self.chunking_selector, 'prefer_llamaindex')):
                    chunking_strategy = "llama_index" if self.chunking_selector.prefer_llamaindex else "adaptive"
            
            except:
                pass
        
        # Get vector store status
        vector_store_status = "Not Ready"

        if self.is_ready:
            try:
                index_stats  = self.index_builder.get_index_stats() if self.index_builder else {}
                total_chunks = index_stats.get('total_chunks_indexed', 0)
                
                if (total_chunks > 0):
                    vector_store_status = f"Ready ({total_chunks} chunks)"
                
                else:
                    vector_store_status = "Empty"
            
            except:
                vector_store_status = "Ready"
        
        # Get model info
        current_model   = settings.OLLAMA_MODEL
        embedding_model = settings.EMBEDDING_MODEL
        
        # Uptime
        uptime_seconds  = (datetime.now() - self.start_time).total_seconds()
        
        return {"vector_store_status"   : vector_store_status,
                "current_model"         : current_model,
                "embedding_model"       : embedding_model,
                "chunking_strategy"     : chunking_strategy,
                "system_uptime_seconds" : int(uptime_seconds),
                "last_updated"          : datetime.now().isoformat(),
               }
    

    def calculate_quality_metrics(self) -> Dict:
        """
        Calculate quality metrics for the system
        """
        total_queries = 0
        total_sources = 0
        source_counts = list()
        
        # Analyze session data
        for session_id, messages in self.active_sessions.items():
            total_queries += len(messages)
            
            for msg in messages:
                sources        = len(msg.get('sources', []))
                total_sources += sources

                source_counts.append(sources)
        
        # Calculate averages
        avg_sources_per_query = total_sources / total_queries if total_queries > 0 else 0
        
        # Calculate metrics based on heuristics
        # These are simplified - for production, use RAGAS or similar framework
        
        if (total_queries == 0):
            return {"answer_relevancy"  : 0.0,
                    "faithfulness"      : 0.0,
                    "context_precision" : 0.0,
                    "context_recall"    : None,
                    "overall_score"     : 0.0,
                    "confidence"        : "low",
                    "metrics_available" : False
                   }
        
        # Heuristic calculations
        answer_relevancy  = min(0.9, 0.7 + (avg_sources_per_query * 0.1))
        faithfulness      = min(0.95, 0.8 + (avg_sources_per_query * 0.05))
        context_precision = min(0.85, 0.6 + (avg_sources_per_query * 0.1))
        
        # Overall score weighted average
        overall_score     = (answer_relevancy * 0.4 + faithfulness * 0.3 + context_precision * 0.3)
        
        confidence        = "high" if total_queries > 10 else ("medium" if (total_queries > 3) else "low")
        
        return {"answer_relevancy"      : round(answer_relevancy, 3),
                "faithfulness"          : round(faithfulness, 3),
                "context_precision"     : round(context_precision, 3),
                "context_recall"        : None,  # Requires ground truth
                "overall_score"         : round(overall_score, 3),
                "avg_sources_per_query" : round(avg_sources_per_query, 2),
                "queries_with_sources"  : sum(1 for count in source_counts if count > 0),
                "confidence"            : confidence,
                "metrics_available"     : True,
                "evaluation_note"       : "Metrics are heuristic estimates. For accurate evaluation, use RAGAS framework.",
               }

    
    def calculate_comprehensive_analytics(self) -> Dict:
        """
        Calculate comprehensive analytics data
        """
        # Performance metrics
        performance     = self.get_performance_metrics()
        
        # System information
        system_info     = self.get_system_information()
        
        # Quality metrics
        quality_metrics = self.calculate_quality_metrics()
        
        # Health status
        health_status   = self.get_system_health()
        
        # Chunking statistics
        chunking_stats  = self.get_chunking_statistics()
        
        # Document statistics
        total_docs      = len(self.processed_documents)
        total_chunks    = sum(len(chunks) for chunks in self.document_chunks.values())
        
        # Session statistics
        total_sessions  = len(self.active_sessions)
        total_messages  = sum(len(msgs) for msgs in self.active_sessions.values())
        
        # File statistics
        uploaded_files  = len(self.uploaded_files)
        total_file_size = sum(f.get('size', 0) for f in self.uploaded_files)
        
        # Index statistics
        index_stats     = dict()

        if self.index_builder:
            try:
                index_stats = self.index_builder.get_index_stats()
            
            except:
                index_stats = {"error": "Could not retrieve index stats"}
        
        return {"performance_metrics" : performance,
                "quality_metrics"     : quality_metrics,
                "system_information"  : system_info,
                "health_status"       : health_status,
                "chunking_statistics" : chunking_stats,
                "document_statistics" : {"total_documents"         : total_docs,
                                         "total_chunks"            : total_chunks,
                                         "uploaded_files"          : uploaded_files,
                                         "total_file_size_bytes"   : total_file_size,
                                         "total_file_size_mb"      : round(total_file_size / (1024 * 1024), 2) if (total_file_size > 0) else 0,
                                         "avg_chunks_per_document" : round(total_chunks / total_docs, 2) if (total_docs > 0) else 0,
                                        },
                "session_statistics"  : {"total_sessions"           : total_sessions,
                                         "total_messages"           : total_messages,
                                         "avg_messages_per_session" : round(total_messages / total_sessions, 2) if (total_sessions > 0) else 0
                                        },
                "index_statistics"    : index_stats,
                "calculated_at"       : datetime.now().isoformat(),
                "cache_info"          : {"from_cache"      : False,
                                         "next_refresh_in" : self.analytics_cache.ttl_seconds,
                                        }
               }


def _setup_signal_handlers():
    """
    Setup signal handlers for graceful shutdown
    """
    try:
        signal.signal(signal.SIGINT, CleanupManager.handle_signal)
        signal.signal(signal.SIGTERM, CleanupManager.handle_signal)
        logger.debug("Signal handlers registered")
    
    except Exception as e:
        logger.warning(f"Failed to setup signal handlers: {e}")


def _atexit_cleanup():
    """
    Atexit handler as last resort
    """
    logger.info("Atexit cleanup triggered")
    
    # Check if it's already in a cleanup process
    with _cleanup_lock:
        if _is_cleaning:
            logger.info("Cleanup already in progress, skipping atexit cleanup")
            return

    try:
        # Check if app exists
        if (('app' in globals()) and (hasattr(app.state, 'app'))):
            # Run cleanup in background thread
            cleanup_thread = threading.Thread(target  = lambda: asyncio.run(CleanupManager.full_cleanup(app.state.app)),
                                              name    = "atexit_cleanup",
                                              daemon  = True,
                                             )
            cleanup_thread.start()
            cleanup_thread.join(timeout = 5.0)
    
    except Exception as e:
        logger.error(f"Atexit cleanup error: {e}")
        # Don't crash during atexit


async def _brute_force_cleanup_app_state(state: AppState):
    """
    Brute force AppState cleanup
    """
    try:
        # Clear all collections
        for attr_name in dir(state):
            if not attr_name.startswith('_'):
                attr = getattr(state, attr_name)
                
                if isinstance(attr, (list, dict, set)):
                    attr.clear()
        
        # Nullify heavy components
        for attr_name in ['index_builder', 'metadata_store', 'embedder', 'llm_client', 'ragas_evaluator']:
            if hasattr(state, attr_name):
                setattr(state, attr_name, None)
        
    except:
        pass


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown with multiple cleanup guarantees
    """
    # Setup signal handlers FIRST
    _setup_signal_handlers()
    
    # Register atexit cleanup
    atexit.register(_atexit_cleanup)

    logger.info("Starting DocuVault AI ...")
    
    try:
        # Initialize application state
        app.state.app = AppState()
        
        # Initialize core components
        await initialize_components(app.state.app)
        
        logger.info("Application startup complete. System ready.")
        
        # Yield control to FastAPI
        yield
    
    except Exception as e:
        logger.error(f"Application runtime error: {e}", exc_info = True)
        raise
    
    finally:
        # GUARANTEED cleanup (even on crash)
        logger.info("Beginning guaranteed cleanup sequence...")
        
        # Set the cleaning flag
        with _cleanup_lock:
            _is_cleaning = True
        
        try:
            # Simple cleanup
            if (hasattr(app.state, 'app')):
                # Just clear the state, don't run full cleanup again
                await _brute_force_cleanup_app_state(app.state.app)
                
                # Clean up disk resources
                await CleanupManager._cleanup_disk_async()
                
                # Shutdown the executor
                _cleanup_executor.shutdown(wait = True)
        
        except Exception as e:
            logger.error(f"Cleanup error in lifespan finally: {e}")
    


# Create FastAPI application
app = FastAPI(title       = "DocuVault AI",
              description = "RAG Platform with Multi-Source & Multi-Format Document Ingestion Support",
              version     = "1.0.0",
              lifespan    = lifespan,
             )


# Add CORS middleware
app.add_middleware(CORSMiddleware,
                   allow_origins     = ["*"],
                   allow_credentials = True,
                   allow_methods     = ["*"],
                   allow_headers     = ["*"],
                  )


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================
async def initialize_components(state: AppState):
    """
    Initialize all application components
    """
    try:
        logger.info("Initializing components...")
        
        # Create necessary directories
        create_directories()
        
        # Initialize utilities
        state.file_handler      = FileHandler()
        logger.info("FileHandler initialized")
        
        # Initialize document parsing
        state.parser_factory    = get_parser_factory()
        logger.info(f"ParserFactory initialized with support for: {', '.join(state.parser_factory.get_supported_extensions())}")
        
        # Initialize chunking
        state.chunking_selector = get_adaptive_selector()
        logger.info("AdaptiveChunkingSelector initialized")
        
        # Initialize embeddings
        state.embedder          = get_embedder()
        state.embedding_cache   = get_embedding_cache()
        logger.info(f"Embedder initialized: {state.embedder.get_model_info()}")
        
        # Initialize ingestion
        state.ingestion_router  = get_ingestion_router()
        state.progress_tracker  = get_progress_tracker()
        logger.info("Ingestion components initialized")
        
        # Initialize vector store
        state.index_builder     = get_index_builder()
        state.metadata_store    = get_metadata_store()
        logger.info("Vector store components initialized")
        
        # Check if indexes exist and load them
        if state.index_builder.is_index_built():
            logger.info("Existing indexes found - loading...")
            index_stats    = state.index_builder.get_index_stats()

            logger.info(f"Indexes loaded: {index_stats}")
            state.is_ready = True
        
        # Initialize retrieval
        state.hybrid_retriever  = get_hybrid_retriever()
        state.context_assembler = get_context_assembler()
        logger.info("Retrieval components initialized")
        
        # Initialize generation components
        state.response_generator = get_response_generator(provider   = LLMProvider.OLLAMA,
                                                          model_name = settings.OLLAMA_MODEL,
                                                         )

        state.llm_client         = get_llm_client(provider = LLMProvider.OLLAMA)

        logger.info(f"Generation components initialized: model={settings.OLLAMA_MODEL}")

        # Check LLM health
        if state.llm_client.check_health():
            logger.info("LLM provider health check: PASSED")
        
        else:
            logger.warning("LLM provider health check: FAILED - Ensure Ollama is running")
            logger.warning("- Run: ollama serve (in a separate terminal)")
            logger.warning("- Run: ollama pull mistral (if model not downloaded)")

        # Initialize RAGAS evaluator
        if settings.ENABLE_RAGAS:
            state.ragas_evaluator = get_ragas_evaluator(enable_ground_truth_metrics = settings.RAGAS_ENABLE_GROUND_TRUTH)

            logger.info("RAGAS evaluator initialized")

        else:
            logger.info("RAGAS evaluation disabled in settings")

        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}", exc_info = True)
        raise


async def cleanup_components(state: AppState):
    """
    Cleanup components on shutdown
    """
    try:
        logger.info("Starting component cleanup...")
    
        # Use the cleanup manager
        await CleanupManager.full_cleanup(state)
        
        logger.info("Component cleanup complete")
        
    except Exception as e:
        logger.error(f"Component cleanup error: {e}", exc_info = True)
        
        # Last-ditch effort
        await _brute_force_cleanup_app_state(state)


def create_directories():
    """
    Create necessary directories
    """
    directories = [settings.UPLOAD_DIR,
                   settings.VECTOR_STORE_DIR,
                   settings.BACKUP_DIR,
                   Path(settings.METADATA_DB_PATH).parent,
                   settings.LOG_DIR,
                  ]
    
    for directory in directories:
        Path(directory).mkdir(parents = True, exist_ok = True)
    
    logger.info("Directories created/verified")


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/", response_class = HTMLResponse)
async def serve_frontend():
    """
    Serve the main frontend HTML
    """
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    
    raise HTTPException(status_code = 404, 
                        detail      = "Frontend not found",
                       )


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    state         = app.state.app
    
    health_status = state.get_system_health()
    
    return {"status"     : health_status["overall"],
            "timestamp"  : datetime.now().isoformat(),
            "version"    : "1.0.0",
            "components" : {"vector_store"     : health_status["vector_store"],
                            "llm"              : health_status["llm"],
                            "embeddings"       : health_status["embeddings"],
                            "retrieval"        : health_status["retrieval"],
                            "generation"       : health_status["generation"],
                            "hybrid_retriever" : health_status["retrieval"],
                           },
            "details"    : health_status
           }


@app.get("/api/system-info")
async def get_system_info():
    """
    Get system information and status
    """
    state       = app.state.app
    
    # Get system information
    system_info = state.get_system_information()
    
    # Get LLM provider info
    llm_info    = dict()

    if state.llm_client:
        llm_info = state.llm_client.get_provider_info()
    
    # Get current configuration
    current_config = {"inference_model"  : settings.OLLAMA_MODEL,
                      "embedding_model"  : settings.EMBEDDING_MODEL,
                      "vector_weight"    : settings.VECTOR_WEIGHT,
                      "bm25_weight"      : settings.BM25_WEIGHT,
                      "temperature"      : settings.DEFAULT_TEMPERATURE,
                      "max_tokens"       : settings.MAX_TOKENS,
                      "chunk_size"       : settings.FIXED_CHUNK_SIZE,
                      "chunk_overlap"    : settings.FIXED_CHUNK_OVERLAP,
                      "top_k_retrieve"   : settings.TOP_K_RETRIEVE,
                      "enable_reranking" : settings.ENABLE_RERANKING,
                     }
                    
    return {"system_state"       : {"is_ready"          : state.is_ready,
                                    "processing_status" : state.processing_status,
                                    "total_documents"   : len(state.uploaded_files),
                                    "active_sessions"   : len(state.active_sessions),
                                   },
            "configuration"      : current_config,
            "llm_provider"       : llm_info,
            "system_information" : system_info,
            "timestamp"          : datetime.now().isoformat()
           }


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple files
    """
    state = app.state.app
    
    try:
        logger.info(f"Received {len(files)} files for upload")
        uploaded_info = list()
        
        for file in files:
            try:
                # Validate file type
                if not state.parser_factory.is_supported(Path(file.filename)):
                    logger.warning(f"Unsupported file type: {file.filename}")
                    continue
                
                # Save file to upload directory
                file_path     = settings.UPLOAD_DIR / FileHandler.generate_unique_filename(file.filename, settings.UPLOAD_DIR)
                
                # Write file content
                content       = await file.read()

                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # Get file metadata
                file_metadata = FileHandler.get_file_metadata(file_path)
                
                file_info     = {"filename"      : file_path.name,
                                 "original_name" : file.filename,
                                 "size"          : file_metadata["size_bytes"],
                                 "upload_time"   : datetime.now().isoformat(),
                                 "file_path"     : str(file_path),
                                 "status"        : "uploaded",
                                }
                
                uploaded_info.append(file_info)
                state.uploaded_files.append(file_info)
                
                logger.info(f"Uploaded: {file.filename} -> {file_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to upload {file.filename}: {e}")
                continue
        
        # Clear analytics cache since we have new data
        state.analytics_cache.data = None
        
        return {"success" : True,
                "message" : f"Successfully uploaded {len(uploaded_info)} files",
                "files"   : uploaded_info,
               }
        
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info = True)

        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.post("/api/start-processing")
async def start_processing():
    """
    Start processing uploaded documents
    """
    state = app.state.app
    
    if not state.uploaded_files:
        raise HTTPException(status_code = 400, 
                            detail      = "No files uploaded",
                           )

     
    if (state.processing_status == "processing"):
        raise HTTPException(status_code = 400, 
                            detail      = "Processing already in progress",
                           )
    
    try:
        state.processing_status   = "processing"
        state.processing_progress = {"status"       : "processing",
                                     "current_step" : "Starting document processing...",
                                     "progress"     : 0,
                                     "processed"    : 0,
                                     "total"        : len(state.uploaded_files),
                                     "details"      : {},
                                    }
        
        logger.info("Starting document processing pipeline...")
        
        all_chunks     = list()
        chunking_stats = dict()
        
        # Process each file
        for idx, file_info in enumerate(state.uploaded_files):
            try:
                file_path                                 = Path(file_info["file_path"])
                
                # Update progress - Parsing
                state.processing_progress["current_step"] = f"Parsing {file_info['original_name']}..."
                state.processing_progress["progress"]     = int((idx / len(state.uploaded_files)) * 20)
                
                # Parse document
                logger.info(f"Parsing document: {file_path}")
                text, metadata                            = state.parser_factory.parse(file_path,
                                                                                       extract_metadata = True,
                                                                                       clean_text       = True,
                                                                                      )
                
                if not text:
                    logger.warning(f"No text extracted from {file_path}")
                    continue
                
                logger.info(f"Extracted {len(text)} characters from {file_path}")
                
                # Update progress - Chunking
                state.processing_progress["current_step"] = f"Chunking {file_info['original_name']}..."
                state.processing_progress["progress"]     = int((idx / len(state.uploaded_files)) * 40) + 20
                
                # Chunk document
                logger.info(f"Chunking document: {metadata.document_id}")
                chunks                                    = state.chunking_selector.chunk_text(text     = text,
                                                                                               metadata = metadata,
                                                                                              )
                
                # Get strategy used from metadata or selector
                strategy_used = "adaptive"  # Default

                if (metadata and hasattr(metadata, 'chunking_strategy')):
                    strategy_used = metadata.chunking_strategy.value if metadata.chunking_strategy else "adaptive"
                
                elif (hasattr(state.chunking_selector, 'last_strategy_used')):
                    strategy_used = state.chunking_selector.last_strategy_used

                # Track chunking strategy usage
                if strategy_used not in chunking_stats:
                    chunking_stats[strategy_used] = 0

                chunking_stats[strategy_used] += len(chunks)
                
                logger.info(f"Created {len(chunks)} chunks for {metadata.document_id} using {strategy_used}")
                
                # Update progress - Embedding
                state.processing_progress["current_step"] = f"Generating embeddings for {file_info['original_name']}..."
                state.processing_progress["progress"]     = int((idx / len(state.uploaded_files)) * 60) + 40
                
                # Generate embeddings for chunks
                logger.info(f"Generating embeddings for {len(chunks)} chunks...")
                chunks_with_embeddings                    = state.embedder.embed_chunks(chunks     = chunks,
                                                                                        batch_size = settings.EMBEDDING_BATCH_SIZE,
                                                                                        normalize  = True,
                                                                                       )
                
                logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
                
                # Store chunks
                all_chunks.extend(chunks_with_embeddings)
                
                # Store processed document and chunks
                state.processed_documents[metadata.document_id] = {"metadata"          : metadata,
                                                                   "text"              : text,
                                                                   "file_info"         : file_info,
                                                                   "chunks_count"      : len(chunks_with_embeddings),
                                                                   "processed_time"    : datetime.now().isoformat(),
                                                                   "chunking_strategy" : strategy_used,
                                                                  }
                
                state.document_chunks[metadata.document_id]     = chunks_with_embeddings
                
                # Update progress
                state.processing_progress["processed"]          = idx + 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_info['original_name']}: {e}", exc_info=True)
                continue
        
        # Update chunking statistics
        state.chunking_statistics = chunking_stats
        
        if not all_chunks:
            raise Exception("No chunks were successfully processed")
        
        # Update progress - Building indexes
        state.processing_progress["current_step"] = "Building vector and keyword indexes..."
        state.processing_progress["progress"]     = 80
        
        # Build indexes (FAISS + BM25 + Metadata)
        logger.info(f"Building indexes for {len(all_chunks)} chunks...")
        index_stats                               = state.index_builder.build_indexes(chunks  = all_chunks,
                                                                                      rebuild = True,
                                                                                     )
        
        logger.info(f"Indexes built: {index_stats}")
        
        # Update progress - Indexing for hybrid retrieval
        state.processing_progress["current_step"] = "Indexing for hybrid retrieval..."
        state.processing_progress["progress"]     = 95
        
        # Mark as ready
        state.processing_status                   = "ready"
        state.is_ready                            = True
        state.processing_progress["status"]       = "ready"
        state.processing_progress["current_step"] = "Processing complete"
        state.processing_progress["progress"]     = 100
        
        # Clear analytics cache
        state.analytics_cache.data                = None
        
        logger.info(f"Processing complete. Processed {len(state.processed_documents)} documents with {len(all_chunks)} total chunks.")
        
        return {"success"             : True,
                "message"             : "Processing completed successfully",
                "status"              : "ready",
                "documents_processed" : len(state.processed_documents),
                "total_chunks"        : len(all_chunks),
                "chunking_statistics" : chunking_stats,
                "index_stats"         : index_stats,
               }
        
    except Exception as e:
        state.processing_status             = "error"
        state.processing_progress["status"] = "error"

        logger.error(f"Processing error: {e}", exc_info = True)

        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.get("/api/processing-status")
async def get_processing_status():
    """
    Get current processing status
    """
    state = app.state.app
    
    return {"status"              : state.processing_progress["status"],
            "progress"            : state.processing_progress["progress"],
            "current_step"        : state.processing_progress["current_step"],
            "processed_documents" : state.processing_progress["processed"],
            "total_documents"     : state.processing_progress["total"],
            "details"             : state.processing_progress["details"],
           }


@app.get("/api/ragas/history")
async def get_ragas_history():
    """
    Get RAGAS evaluation history for current session
    """
    state = app.state.app
    
    if not settings.ENABLE_RAGAS or not state.ragas_evaluator:

        raise HTTPException(status_code = 400,
                            detail      = "RAGAS evaluation is not enabled. Set ENABLE_RAGAS=True in settings.",
                           )
    
    try:
        history = state.ragas_evaluator.get_evaluation_history()
        stats   = state.ragas_evaluator.get_session_statistics()
        
        return {"success"     : True,
                "total_count" : len(history),
                "statistics"  : stats.model_dump(), 
                "history"     : history
               }
        
    except Exception as e:
        logger.error(f"RAGAS history retrieval error: {e}", exc_info = True)

        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.post("/api/ragas/clear")
async def clear_ragas_history():
    """
    Clear RAGAS evaluation history
    """
    state = app.state.app
    
    if not settings.ENABLE_RAGAS or not state.ragas_evaluator:

        raise HTTPException(status_code = 400,
                            detail      = "RAGAS evaluation is not enabled.",
                           )
    
    try:
        state.ragas_evaluator.clear_history()
        
        return {"success" : True,
                "message" : "RAGAS evaluation history cleared, new session started",
               }
        
    except Exception as e:
        logger.error(f"RAGAS history clear error: {e}", exc_info = True)

        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.get("/api/ragas/statistics")
async def get_ragas_statistics():
    """
    Get aggregate RAGAS statistics for current session
    """
    state = app.state.app
    
    if not settings.ENABLE_RAGAS or not state.ragas_evaluator:

        raise HTTPException(status_code = 400,
                            detail      = "RAGAS evaluation is not enabled.",
                           )
    
    try:
        stats = state.ragas_evaluator.get_session_statistics()
        
        return {"success"    : True,
                "statistics" : stats.model_dump(),
               }
        
    except Exception as e:
        logger.error(f"RAGAS statistics error: {e}", exc_info = True)

        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.get("/api/ragas/export")
async def export_ragas_data():
    """
    Export all RAGAS evaluation data
    """
    state = app.state.app
    
    if not settings.ENABLE_RAGAS or not state.ragas_evaluator:

        raise HTTPException(status_code = 400,
                            detail      = "RAGAS evaluation is not enabled.",
                           )
    
    try:
        export_data = state.ragas_evaluator.export_to_dict()
        
        return JSONResponse(content = json.loads(export_data.model_dump_json()))
        
    except Exception as e:
        logger.error(f"RAGAS export error: {e}", exc_info = True)

        raise HTTPException(status_code = 500,
                            detail      = str(e),
                           )


@app.get("/api/ragas/config")
async def get_ragas_config():
    """
    Get current RAGAS configuration
    """
    return {"enabled"              : settings.ENABLE_RAGAS,
            "ground_truth_enabled" : settings.RAGAS_ENABLE_GROUND_TRUTH,
            "base_metrics"         : settings.RAGAS_METRICS,
            "ground_truth_metrics" : settings.RAGAS_GROUND_TRUTH_METRICS,
            "evaluation_timeout"   : settings.RAGAS_EVALUATION_TIMEOUT,
            "batch_size"           : settings.RAGAS_BATCH_SIZE,
           }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Handle chat queries with LLM-based intelligent routing (generic vs RAG)
    Supports both conversational queries and document-based queries
    """
    state      = app.state.app
    
    message    = request.message
    session_id = request.session_id
    
    # Check LLM health (required for both general and RAG queries)
    if not state.llm_client.check_health():
        raise HTTPException(status_code = 503,
                            detail      = "LLM service unavailable. Please ensure Ollama is running.",
                           )
    
    try:
        logger.info(f"Chat query received: {message}")
        
        # Check if documents are available
        has_documents = state.is_ready and (len(state.processed_documents) > 0)
        
        logger.debug(f"System state - Documents available: {has_documents}, Processed docs: {len(state.processed_documents)}, System ready: {state.is_ready}")
        
        # Get conversation history for this session (for general queries)
        conversation_history = None
        
        if (session_id and (session_id in state.active_sessions)):
            # Convert to format expected by general_responder
            conversation_history = list()
            
            # Last 10 messages for context
            for msg in state.active_sessions[session_id][-10:]:
                conversation_history.append({"role"    : "user",
                                             "content" : msg.get("query", ""),
                                           })

                conversation_history.append({"role"    : "assistant",
                                             "content" : msg.get("response", ""),
                                           })
        
        # Create QueryRequest object
        query_request     = QueryRequest(query            = message,
                                         top_k            = settings.TOP_K_RETRIEVE,
                                         enable_reranking = settings.ENABLE_RERANKING,
                                         temperature      = settings.DEFAULT_TEMPERATURE,
                                         top_p            = settings.TOP_P,
                                         max_tokens       = settings.MAX_TOKENS,
                                         include_sources  = True,
                                         include_metrics  = False,
                                         stream           = False,
                                        )
        
        # Generate response using response generator (with LLM-based routing)
        start_time        = time.time()

        query_response    = await state.response_generator.generate_response(request              = query_request,
                                                                             conversation_history = conversation_history,
                                                                             has_documents        = has_documents,  # Pass document availability
                                                                            )
        
        # Convert to ms
        total_time        = (time.time() - start_time) * 1000
        
        # Record timing for analytics
        state.add_query_timing(total_time)

        # Determine query type using response metadata
        is_general_query  = False

        # Default to rag
        actual_query_type = "rag"  

        # Check if response has metadata
        if (hasattr(query_response, 'query_type')):
            actual_query_type = query_response.query_type
            is_general_query  = (actual_query_type == "general")

        elif (hasattr(query_response, 'is_general_query')):
            is_general_query  = query_response.is_general_query
            actual_query_type = "general" if is_general_query else "rag"

        else:
            # Method 2: Check sources (fallback)
            has_sources       = query_response.sources and len(query_response.sources) > 0
            is_general_query  = not has_sources
            actual_query_type = "general" if is_general_query else "rag"
        
        logger.debug(f"Query classification: actual_query_type={actual_query_type}, has_sources={query_response.sources and len(query_response.sources) > 0}")
    
        # Extract contexts for RAGAS evaluation (only if RAG was used)
        contexts          = list()

        if query_response.sources:
            contexts = [chunk.chunk.text for chunk in query_response.sources]
        
        # Run RAGAS evaluation (only if RAGAS enabled)
        ragas_result   = None

        if (settings.ENABLE_RAGAS and state.ragas_evaluator):
            try:
                logger.info("Running RAGAS evaluation...")
                
                ragas_result = state.ragas_evaluator.evaluate_single(query              = message,
                                                                     answer             = query_response.answer,
                                                                     contexts           = contexts,
                                                                     ground_truth       = None,
                                                                     retrieval_time_ms  = int(query_response.retrieval_time_ms),
                                                                     generation_time_ms = int(query_response.generation_time_ms),
                                                                     total_time_ms      = int(query_response.total_time_ms),
                                                                     chunks_retrieved   = len(query_response.sources),
                                                                     query_type         = actual_query_type,
                                                                    )
                
                logger.info(f"RAGAS evaluation complete: type={actual_query_type.upper()}, relevancy={ragas_result.answer_relevancy:.3f}, faithfulness={ragas_result.faithfulness:.3f}, overall={ragas_result.overall_score:.3f}")
            
            except Exception as e:
                logger.error(f"RAGAS evaluation failed: {e}", exc_info = True)
                # Continue without RAGAS metrics - don't fail the request
        
        # Format sources for response
        sources = list()

        for i, chunk_with_score in enumerate(query_response.sources[:5], 1):
            chunk  = chunk_with_score.chunk

            source = {"rank"             : i,
                      "score"            : chunk_with_score.score,
                      "document_id"      : chunk.document_id,
                      "chunk_id"         : chunk.chunk_id,
                      "text_preview"     : chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                      "page_number"      : chunk.page_number,
                      "section_title"    : chunk.section_title,
                      "retrieval_method" : chunk_with_score.retrieval_method,
                     }

            sources.append(source)
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.now().timestamp()}"
        
        # Determine query type for response metadata
        is_general_query = (actual_query_type == "general")
        
        # Prepare response
        response         = {"session_id"       : session_id,
                            "response"         : query_response.answer,
                            "sources"          : sources,
                            "is_general_query" : is_general_query,
                            "metrics"          : {"retrieval_time"    : int(query_response.retrieval_time_ms),
                                                  "generation_time"   : int(query_response.generation_time_ms),
                                                  "total_time"        : int(query_response.total_time_ms),
                                                  "chunks_retrieved"  : len(query_response.sources),
                                                  "chunks_used"       : len(sources),
                                                  "tokens_used"       : query_response.tokens_used.get("total", 0) if query_response.tokens_used else 0,
                                                  "actual_total_time" : int(total_time),
                                                  "query_type"        : actual_query_type,
                                                  "llm_classified"    : True,  # Now using LLM for classification
                                                 },
                           }
        
        # Add RAGAS metrics if evaluation succeeded
        if ragas_result:
            response["ragas_metrics"] = {"answer_relevancy"   : round(ragas_result.answer_relevancy, 3),
                                         "faithfulness"       : round(ragas_result.faithfulness, 3),
                                         "context_precision"  : round(ragas_result.context_precision, 3) if ragas_result.context_precision else None,
                                         "context_relevancy"  : round(ragas_result.context_relevancy, 3),
                                         "overall_score"      : round(ragas_result.overall_score, 3),
                                         "context_recall"     : round(ragas_result.context_recall, 3) if ragas_result.context_recall else None,
                                         "answer_similarity"  : round(ragas_result.answer_similarity, 3) if ragas_result.answer_similarity else None,
                                         "answer_correctness" : round(ragas_result.answer_correctness, 3) if ragas_result.answer_correctness else None,
                                         "query_type"         : ragas_result.query_type, 
                                        }
        else:
            response["ragas_metrics"] = None
        
        # Store in session
        if session_id not in state.active_sessions:
            state.active_sessions[session_id] = list()
        
        state.active_sessions[session_id].append({"query"            : message,
                                                  "response"         : query_response.answer,
                                                  "sources"          : sources,
                                                  "timestamp"        : datetime.now().isoformat(),
                                                  "metrics"          : response["metrics"],
                                                  "ragas_metrics"    : response.get("ragas_metrics", {}),
                                                  "is_general_query" : is_general_query,
                                                })
        
        # Clear analytics cache when new data is available
        state.analytics_cache.data = None
        
        logger.info(f"Chat response generated successfully in {int(total_time)}ms | (type: {actual_query_type.upper()})")
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info = True)
        
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.get("/api/configuration")
async def get_configuration():
    """
    Get current configuration
    """
    state        = app.state.app
    
    # Get system health
    health_status = state.get_system_health()
    
    return {"configuration" : {"inference_model"   : settings.OLLAMA_MODEL,
                               "embedding_model"   : settings.EMBEDDING_MODEL,
                               "chunking_strategy" : "adaptive",
                               "chunk_size"        : settings.FIXED_CHUNK_SIZE,
                               "chunk_overlap"     : settings.FIXED_CHUNK_OVERLAP,
                               "retrieval_top_k"   : settings.TOP_K_RETRIEVE,
                               "vector_weight"     : settings.VECTOR_WEIGHT,
                               "bm25_weight"       : settings.BM25_WEIGHT,
                               "temperature"       : settings.DEFAULT_TEMPERATURE,
                               "max_tokens"        : settings.MAX_TOKENS,
                               "enable_reranking"  : settings.ENABLE_RERANKING,
                               "is_ready"          : state.is_ready,
                               "llm_healthy"       : health_status["llm"],
                              },
            "health"        : health_status,
           }


@app.post("/api/configuration")
async def update_configuration(temperature: float = Form(None), max_tokens: int = Form(None), retrieval_top_k: int = Form(None),
                               vector_weight: float = Form(None), bm25_weight: float = Form(None), enable_reranking: bool = Form(None),
                               session_id: str = Form(None)):
    """
    Update system configuration (runtime parameters only)
    """
    state = app.state.app
    
    try:
        updates = dict()
        
        # Runtime parameters (no rebuild required)
        if (temperature is not None):
            updates["temperature"] = temperature
        
        if (max_tokens and (max_tokens != settings.MAX_TOKENS)):
            updates["max_tokens"] = max_tokens
        
        if (retrieval_top_k and (retrieval_top_k != settings.TOP_K_RETRIEVE)):
            updates["retrieval_top_k"] = retrieval_top_k
        
        if ((vector_weight is not None) and (vector_weight != settings.VECTOR_WEIGHT)):
            updates["vector_weight"] = vector_weight
            
            # Update hybrid retriever weights
            if bm25_weight is not None:
                state.hybrid_retriever.update_weights(vector_weight, bm25_weight)
        
        if ((bm25_weight is not None) and (bm25_weight != settings.BM25_WEIGHT)):
            updates["bm25_weight"] = bm25_weight
        
        if (enable_reranking is not None):
            updates["enable_reranking"] = enable_reranking
        
        # Store session-based config overrides
        if session_id:
            if session_id not in state.config_overrides:
                state.config_overrides[session_id] = {}
            
            state.config_overrides[session_id].update(updates)
        
        logger.info(f"Configuration updated: {updates}")
        
        # Clear analytics cache since configuration changed
        state.analytics_cache.data = None
        
        return {"success" : True,
                "message" : "Configuration updated successfully",
                "updates" : updates,
               }
        
    except Exception as e:
        logger.error(f"Configuration update error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.get("/api/analytics")
async def get_analytics():
    """
    Get comprehensive system analytics and metrics with caching
    """
    state = app.state.app
    
    try:
        # Check cache first
        cached_data = state.analytics_cache.get()
        
        if cached_data:
            cached_data["cache_info"]["from_cache"] = True
            
            return cached_data
        
        # Calculate fresh analytics
        analytics_data = state.calculate_comprehensive_analytics()
        
        # Update cache
        state.analytics_cache.update(analytics_data)
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Analytics calculation error: {e}", exc_info = True)
        
        # Return basic analytics even if calculation fails
        return {"performance_metrics" : {"avg_response_time" : 0,
                                         "total_queries"     : 0,
                                         "queries_last_hour" : 0,
                                         "error"             : "Could not calculate performance metrics"
                                        },
                "quality_metrics"     : {"answer_relevancy"  : 0.0,
                                         "faithfulness"      : 0.0,
                                         "context_precision" : 0.0,
                                         "overall_score"     : 0.0,
                                         "confidence"        : "low",
                                         "metrics_available" : False,
                                         "error"             : "Could not calculate quality metrics"
                                        },
                "system_information"  : state.get_system_information() if hasattr(state, 'get_system_information') else {},
                "health_status"       : {"overall" : "unknown"},
                "document_statistics" : {"total_documents" : len(state.processed_documents),
                                         "total_chunks"    : sum(len(chunks) for chunks in state.document_chunks.values()),
                                         "uploaded_files"  : len(state.uploaded_files)
                                        },
                "session_statistics"  : {"total_sessions" : len(state.active_sessions),
                                         "total_messages" : sum(len(msgs) for msgs in state.active_sessions.values())
                                        },
                "calculated_at"       : datetime.now().isoformat(),
                "error"               : str(e)
               }


@app.get("/api/analytics/refresh")
async def refresh_analytics():
    """
    Force refresh analytics cache
    """
    state = app.state.app
    
    try:
        # Clear cache
        state.analytics_cache.data = None
        
        # Calculate fresh analytics
        analytics_data             = state.calculate_comprehensive_analytics()
        
        return {"success" : True,
                "message" : "Analytics cache refreshed successfully",
                "data"    : analytics_data,
               }
        
    except Exception as e:
        logger.error(f"Analytics refresh error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.get("/api/analytics/detailed")
async def get_detailed_analytics():
    """
    Get detailed analytics including query history and component performance
    """
    state = app.state.app
    
    try:
        # Get basic analytics
        analytics         = await get_analytics()
        
        # Add detailed session information
        detailed_sessions = list()

        for session_id, messages in state.active_sessions.items():
            session_info = {"session_id"            : session_id,
                            "message_count"         : len(messages),
                            "first_message"         : messages[0]["timestamp"] if messages else None,
                            "last_message"          : messages[-1]["timestamp"] if messages else None,
                            "total_response_time"   : sum(msg.get("metrics", {}).get("total_time", 0) for msg in messages),
                            "avg_sources_per_query" : sum(len(msg.get("sources", [])) for msg in messages) / len(messages) if messages else 0,
                           }

            detailed_sessions.append(session_info)
        
        # Add component performance if available
        component_performance = dict()

        if state.hybrid_retriever:
            try:
                retrieval_stats                    = state.hybrid_retriever.get_retrieval_stats()
                component_performance["retrieval"] = retrieval_stats

            except:
                component_performance["retrieval"] = {"error": "Could not retrieve stats"}
        
        if state.embedder:
            try:
                embedder_info                       = state.embedder.get_model_info()
                component_performance["embeddings"] = {"model"     : embedder_info.get("model_name", "unknown"),
                                                       "dimension" : embedder_info.get("embedding_dim", 0),
                                                       "device"    : embedder_info.get("device", "cpu"),
                                                      }
            except:
                component_performance["embeddings"] = {"error": "Could not retrieve stats"}
        
        analytics["detailed_sessions"]     = detailed_sessions
        analytics["component_performance"] = component_performance
        
        return analytics
        
    except Exception as e:
        logger.error(f"Detailed analytics error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.get("/api/export-chat/{session_id}")
async def export_chat(session_id: str, format: str = "json"):
    """
    Export chat history
    """
    state = app.state.app
    if session_id not in state.active_sessions:
        raise HTTPException(status_code = 404, 
                            detail      = "Session not found",
                           )

    try:
        chat_history = state.active_sessions[session_id]
        
        if (format == "json"):
            return JSONResponse(content = {"session_id"     : session_id,
                                           "export_time"    : datetime.now().isoformat(),
                                           "total_messages" : len(chat_history),
                                           "history"        : chat_history,
                                          }
                               )

        elif (format == "csv"):
            output = io.StringIO()
            
            if chat_history:
                fieldnames = ["timestamp", "query", "response", "sources_count", "response_time_ms"]
                writer     = csv.DictWriter(output, fieldnames = fieldnames)
                writer.writeheader()
                
                for entry in chat_history:
                    writer.writerow({"timestamp"        : entry.get("timestamp", ""),
                                     "query"            : entry.get("query", ""),
                                     "response"         : entry.get("response", ""),
                                     "sources_count"    : len(entry.get("sources", [])),
                                     "response_time_ms" : entry.get("metrics", {}).get("total_time", 0),
                                   })
            
            return JSONResponse(content = {"csv"        : output.getvalue(),
                                           "session_id" : session_id,
                                           "format"     : "csv",
                                          }
                               )

        else:
            raise HTTPException(status_code = 400, 
                                detail      = "Unsupported format. Use 'json' or 'csv'",
                               )
            
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )
                           

@app.post("/api/cleanup/session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up specific session
    """
    state = app.state.app
    
    if session_id in state.active_sessions:
        del state.active_sessions[session_id]
        
        if session_id in state.config_overrides:
            del state.config_overrides[session_id]
        
        # Check if no sessions left
        if not state.active_sessions:
            logger.info("No active sessions, suggesting vector store cleanup")
            
            return {"success"    : True, 
                    "message"    : f"Session {session_id} cleaned up",
                    "suggestion" : "No active sessions remaining. Consider cleaning vector store.",
                   }
        
        return {"success" : True, 
                "message" : f"Session {session_id} cleaned up",
               }
    
    return {"success" : False, 
            "message" : "Session not found",
           }


@app.post("/api/cleanup/vector-store")
async def cleanup_vector_store():
    """
    Manual vector store cleanup
    """
    state = app.state.app
    
    try:
        # Use cleanup manager
        success = await CleanupManager.full_cleanup(state)
        
        if success:
            return {"success" : True, 
                    "message" : "Vector store and all data cleaned up",
                   }
        
        else:
            return {"success" : False, 
                    "message" : "Cleanup completed with errors",
                   }
            
    except Exception as e:
        logger.error(f"Manual cleanup error: {e}")
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.post("/api/cleanup/full")
async def full_cleanup_endpoint():
    """
    Full system cleanup endpoint
    """
    state = app.state.app
    
    try:
        # Also clean up frontend sessions
        state.active_sessions.clear()
        state.config_overrides.clear()
        
        # Full cleanup
        success = await CleanupManager.full_cleanup(state)
        
        return {"success" : success,
                "message" : "Full system cleanup completed",
                "details" : {"sessions_cleaned" : 0,  # Already cleared above
                             "memory_freed"     : "All application state",
                             "disk_space_freed" : "All vector store and uploaded files",
                            }
               }
        
    except Exception as e:
        logger.error(f"Full cleanup endpoint error: {e}")
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.get("/api/cleanup/status")
async def get_cleanup_status():
    """
    Get cleanup status and statistics
    """
    state = app.state.app
    
    return {"sessions_active"       : len(state.active_sessions),
            "documents_processed"   : len(state.processed_documents),
            "total_chunks"          : sum(len(chunks) for chunks in state.document_chunks.values()),
            "vector_store_ready"    : state.is_ready,
            "cleanup_registry_size" : len(_cleanup_registry),
            "suggested_action"      : "cleanup_vector_store" if state.is_ready else "upload_documents",
           }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        # Run the app
        uvicorn.run("app:app",
                    host                      = settings.HOST,
                    port                      = settings.PORT,
                    reload                    = settings.DEBUG,
                    log_level                 = "info",
                    timeout_graceful_shutdown = 10.0,
                    access_log                = False,
                   )
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received - normal shutdown")
    
    except Exception as e:
        logger.error(f"Application crashed: {e}", exc_info = True)
    
    finally:
        # Simple final cleanup
        logger.info("Application stopping, final cleanup...")
        try:
            # Shutdown executor if it exists
            if '_cleanup_executor' in globals():
                _cleanup_executor.shutdown(wait = True)

        except:
            pass
        
        logger.info("Application stopped")