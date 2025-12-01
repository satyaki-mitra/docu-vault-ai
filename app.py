"""
AI Universal Knowledge Ingestion System - FastAPI Application
A production-grade RAG system with multi-source ingestion and zero API costs
"""

import os
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from config.models import ChatRequest
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import configuration and logging
from config.settings import get_settings
from config.logging_config import setup_logging
from config.models import DocumentMetadata, ProcessingStatus as ProcessingStatusEnum

# Import core modules
from utils.error_handler import RAGException, FileException
from utils.file_handler import FileHandler
from utils.validators import FileValidator, validate_upload_file
from utils.helpers import IDGenerator

# Import pipeline components
from document_parser.parser_factory import ParserFactory, get_parser_factory
from chunking.adaptive_selector import AdaptiveChunkingSelector, get_adaptive_selector

# Import embeddings module
from embeddings.bge_embedder import get_embedder
from embeddings.embedding_cache import get_embedding_cache

# Import ingestion module
from ingestion.router import get_ingestion_router
from ingestion.progress_tracker import get_progress_tracker

# Import vector store module
from vector_store.index_builder import get_index_builder
from vector_store.metadata_store import get_metadata_store

# Import retrieval module
from retrieval.hybrid_retriever import get_hybrid_retriever
from retrieval.context_assembler import get_context_assembler

# Import generation module
from generation.response_generator import get_response_generator
from generation.llm_client import get_llm_client
from config.models import PromptType, LLMProvider


# Setup logging and settings
settings = get_settings()
logger = setup_logging(
    log_level=settings.LOG_LEVEL,
    log_dir=settings.LOG_DIR,
    enable_console=True,
    enable_file=True
)

# Global state manager
class AppState:
    """Manages application state and components"""
    
    def __init__(self):
        self.is_ready = False
        self.processing_status = "idle"
        self.uploaded_files = []
        self.active_sessions = {}
        self.processed_documents = {}
        self.document_chunks = {}
        
        # Core components
        self.file_handler = None
        self.parser_factory = None
        self.chunking_selector = None
        
        # Embeddings components
        self.embedder = None
        self.embedding_cache = None
        
        # Ingestion components
        self.ingestion_router = None
        self.progress_tracker = None
        
        # Vector store components
        self.index_builder = None
        self.metadata_store = None
        
        # Retrieval components
        self.hybrid_retriever = None
        self.context_assembler = None
        
        # Generation components
        self.response_generator = None
        self.llm_client = None
        
        # Processing tracking
        self.current_processing = None
        self.processing_progress = {
            "status": "idle",
            "current_step": "Waiting",
            "progress": 0,
            "processed": 0,
            "total": 0,
            "details": {}
        }
        
        # Session-based configuration overrides
        self.config_overrides = {}


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    logger.info("Starting AI Universal Knowledge Ingestion System...")
    
    # Initialize application state
    app.state.app = AppState()
    
    # Initialize core components
    await initialize_components(app.state.app)
    
    logger.info("Application startup complete. System ready.")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application...")
    await cleanup_components(app.state.app)
    logger.info("Application shutdown complete.")


# Create FastAPI application
app = FastAPI(
    title="AI Universal Knowledge Ingestion System",
    description="Enterprise RAG Platform with Multi-Source Ingestion",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

async def initialize_components(state: AppState):
    """Initialize all application components"""
    try:
        logger.info("Initializing components...")
        
        # Create necessary directories
        create_directories()
        
        # Initialize utilities
        state.file_handler = FileHandler()
        logger.info("FileHandler initialized")
        
        # Initialize document parsing
        state.parser_factory = get_parser_factory()
        logger.info(f"ParserFactory initialized with support for: {', '.join(state.parser_factory.get_supported_extensions())}")
        
        # Initialize chunking
        state.chunking_selector = get_adaptive_selector()
        logger.info("AdaptiveChunkingSelector initialized")
        
        # Initialize embeddings
        state.embedder = get_embedder()
        state.embedding_cache = get_embedding_cache()
        logger.info(f"Embedder initialized: {state.embedder.get_model_info()}")
        
        # Initialize ingestion
        state.ingestion_router = get_ingestion_router()
        state.progress_tracker = get_progress_tracker()
        logger.info("Ingestion components initialized")
        
        # Initialize vector store
        state.index_builder = get_index_builder()
        state.metadata_store = get_metadata_store()
        logger.info("Vector store components initialized")
        
        # Check if indexes exist and load them
        if state.index_builder.is_index_built():
            logger.info("Existing indexes found - loading...")
            index_stats = state.index_builder.get_index_stats()
            logger.info(f"Indexes loaded: {index_stats}")
            state.is_ready = True
        
        # Initialize retrieval
        state.hybrid_retriever = get_hybrid_retriever()
        state.context_assembler = get_context_assembler()
        logger.info("Retrieval components initialized")
        
        # Initialize generation components
        state.response_generator = get_response_generator(
            model_name=settings.OLLAMA_MODEL,
            provider="ollama"
        )
        state.llm_client = get_llm_client(provider=LLMProvider.OLLAMA)
        logger.info(f"Generation components initialized: model={settings.OLLAMA_MODEL}")
        
        # Check LLM health
        if state.llm_client.check_health():
            logger.info("✓ LLM provider health check: PASSED")
        else:
            logger.warning("⚠ LLM provider health check: FAILED - Ensure Ollama is running")
            logger.warning("  Run: ollama serve (in a separate terminal)")
            logger.warning("  Run: ollama pull mistral (if model not downloaded)")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}", exc_info=True)
        raise


async def cleanup_components(state: AppState):
    """Cleanup components on shutdown"""
    try:
        logger.info("Starting cleanup...")
        logger.info("Cleanup complete")
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}", exc_info=True)


def create_directories():
    """Create necessary directories"""
    directories = [
        settings.UPLOAD_DIR,
        settings.VECTOR_STORE_DIR,
        settings.BACKUP_DIR,
        Path(settings.METADATA_DB_PATH).parent,
        settings.LOG_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directories created/verified")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML"""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    state = app.state.app
    
    # Check LLM health
    llm_healthy = False
    if state.llm_client:
        llm_healthy = state.llm_client.check_health()
    
    # Check retrieval components
    retrieval_healthy = state.hybrid_retriever is not None
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "vector_store": state.is_ready,
            "llm": llm_healthy,
            "embeddings": state.embedder is not None,
            "retrieval": retrieval_healthy,
            "generation": state.response_generator is not None,
            "hybrid_retriever": retrieval_healthy
        }
    }


@app.get("/api/system-info")
async def get_system_info():
    """Get system information and status"""
    state = app.state.app
    
    # Get LLM provider info
    llm_info = {}
    if state.llm_client:
        llm_info = state.llm_client.get_provider_info()
    
    return {
        "system_state": {
            "is_ready": state.is_ready,
            "processing_status": state.processing_status,
            "total_documents": len(state.uploaded_files),
            "active_sessions": len(state.active_sessions)
        },
        "configuration": {
            "inference_model": settings.OLLAMA_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "vector_weight": settings.VECTOR_WEIGHT,
            "bm25_weight": settings.BM25_WEIGHT,
            "temperature": settings.DEFAULT_TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS
        },
        "llm_provider": llm_info,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    state = app.state.app
    
    try:
        logger.info(f"Received {len(files)} files for upload")
        uploaded_info = []
        
        for file in files:
            try:
                # Validate file type
                if not state.parser_factory.is_supported(Path(file.filename)):
                    logger.warning(f"Unsupported file type: {file.filename}")
                    continue
                
                # Save file to upload directory
                file_path = settings.UPLOAD_DIR / FileHandler.generate_unique_filename(
                    file.filename, 
                    settings.UPLOAD_DIR
                )
                
                # Write file content
                content = await file.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # Get file metadata
                file_metadata = FileHandler.get_file_metadata(file_path)
                
                file_info = {
                    "filename": file_path.name,
                    "original_name": file.filename,
                    "size": file_metadata["size_bytes"],
                    "upload_time": datetime.now().isoformat(),
                    "file_path": str(file_path),
                    "status": "uploaded"
                }
                
                uploaded_info.append(file_info)
                state.uploaded_files.append(file_info)
                
                logger.info(f"Uploaded: {file.filename} -> {file_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to upload {file.filename}: {e}")
                continue
        
        return {
            "success": True,
            "message": f"Successfully uploaded {len(uploaded_info)} files",
            "files": uploaded_info
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start-processing")
async def start_processing():
    """Start processing uploaded documents"""
    state = app.state.app
    
    if not state.uploaded_files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if state.processing_status == "processing":
        raise HTTPException(status_code=400, detail="Processing already in progress")
    
    try:
        state.processing_status = "processing"
        state.processing_progress = {
            "status": "processing",
            "current_step": "Starting document processing...",
            "progress": 0,
            "processed": 0,
            "total": len(state.uploaded_files),
            "details": {}
        }
        
        logger.info("Starting document processing pipeline...")
        
        all_chunks = []
        
        # Process each file
        for idx, file_info in enumerate(state.uploaded_files):
            try:
                file_path = Path(file_info["file_path"])
                
                # Update progress - Parsing
                state.processing_progress["current_step"] = f"Parsing {file_info['original_name']}..."
                state.processing_progress["progress"] = int((idx / len(state.uploaded_files)) * 20)
                
                # Parse document
                logger.info(f"Parsing document: {file_path}")
                text, metadata = state.parser_factory.parse(
                    file_path,
                    extract_metadata=True,
                    clean_text=True
                )
                
                if not text:
                    logger.warning(f"No text extracted from {file_path}")
                    continue
                
                logger.info(f"Extracted {len(text)} characters from {file_path}")
                
                # Update progress - Chunking
                state.processing_progress["current_step"] = f"Chunking {file_info['original_name']}..."
                state.processing_progress["progress"] = int((idx / len(state.uploaded_files)) * 40) + 20
                
                # Chunk document
                logger.info(f"Chunking document: {metadata.document_id}")
                chunks = state.chunking_selector.chunk_text(
                    text=text,
                    metadata=metadata
                )
                
                logger.info(f"Created {len(chunks)} chunks for {metadata.document_id}")
                
                # Update progress - Embedding
                state.processing_progress["current_step"] = f"Generating embeddings for {file_info['original_name']}..."
                state.processing_progress["progress"] = int((idx / len(state.uploaded_files)) * 60) + 40
                
                # Generate embeddings for chunks
                logger.info(f"Generating embeddings for {len(chunks)} chunks...")
                chunks_with_embeddings = state.embedder.embed_chunks(
                    chunks=chunks,
                    batch_size=settings.EMBEDDING_BATCH_SIZE,
                    normalize=True
                )
                
                logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
                
                # Store chunks
                all_chunks.extend(chunks_with_embeddings)
                
                # Store processed document and chunks
                state.processed_documents[metadata.document_id] = {
                    "metadata": metadata,
                    "text": text,
                    "file_info": file_info,
                    "chunks_count": len(chunks_with_embeddings),
                    "processed_time": datetime.now().isoformat()
                }
                
                state.document_chunks[metadata.document_id] = chunks_with_embeddings
                
                # Update progress
                state.processing_progress["processed"] = idx + 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_info['original_name']}: {e}", exc_info=True)
                continue
        
        if not all_chunks:
            raise Exception("No chunks were successfully processed")
        
        # Update progress - Building indexes
        state.processing_progress["current_step"] = "Building vector and keyword indexes..."
        state.processing_progress["progress"] = 80
        
        # Build indexes (FAISS + BM25 + Metadata)
        logger.info(f"Building indexes for {len(all_chunks)} chunks...")
        index_stats = state.index_builder.build_indexes(
            chunks=all_chunks,
            rebuild=True
        )
        
        logger.info(f"Indexes built: {index_stats}")
        
        # Update progress - Indexing for hybrid retrieval
        state.processing_progress["current_step"] = "Indexing for hybrid retrieval..."
        state.processing_progress["progress"] = 95
        
        try:
            # Index chunks for hybrid retrieval
            state.hybrid_retriever.index_chunks(all_chunks)
            logger.info("Hybrid retrieval index ready")
        except Exception as e:
            logger.error(f"Hybrid retriever indexing failed: {e}")
            # Continue anyway - the system can still work with basic retrieval
            logger.warning("Hybrid retriever indexing failed, but processing will continue")
        
        # Mark as ready
        state.processing_status = "ready"
        state.is_ready = True
        state.processing_progress["status"] = "ready"
        state.processing_progress["current_step"] = "Processing complete"
        state.processing_progress["progress"] = 100
        
        logger.info(f"Processing complete. Processed {len(state.processed_documents)} documents with {len(all_chunks)} total chunks.")
        
        return {
            "success": True,
            "message": "Processing completed successfully",
            "status": "ready",
            "documents_processed": len(state.processed_documents),
            "total_chunks": len(all_chunks),
            "index_stats": index_stats
        }
        
    except Exception as e:
        state.processing_status = "error"
        state.processing_progress["status"] = "error"
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/processing-status")
async def get_processing_status():
    """Get current processing status"""
    state = app.state.app
    
    return {
        "status": state.processing_progress["status"],
        "progress": state.processing_progress["progress"],
        "current_step": state.processing_progress["current_step"],
        "processed_documents": state.processing_progress["processed"],
        "total_documents": state.processing_progress["total"],
        "details": state.processing_progress["details"]
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat queries with full RAG pipeline including generation"""
    state = app.state.app
    
    # Extract from request
    message = request.message
    session_id = request.session_id
    
    if not state.is_ready:
        raise HTTPException(
            status_code=400,
            detail="System not ready. Please upload and process documents first."
        )
    
    # Check if LLM is healthy
    if not state.llm_client.check_health():
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Please ensure Ollama is running (ollama serve) and the model is available (ollama pull mistral)."
        )
    
    try:
        logger.info(f"Chat query received: {message[:100]}...")
        
        import time
        start_time = time.time()
        
        # Get all chunks from processed documents
        all_chunks = []
        for doc_id, chunks in state.document_chunks.items():
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No indexed chunks available")
        
        logger.info(f"Searching across {len(all_chunks)} chunks...")
        
        # 1. RETRIEVAL: Perform hybrid retrieval
        retrieval_start = time.time()
        
        # Get configuration overrides if any
        config = state.config_overrides.get(session_id, {})
        top_k = config.get('retrieval_top_k', settings.TOP_K_RETRIEVE)
        
        try:
            retrieved_chunks = state.hybrid_retriever.retrieve(
                query=message,
                top_k=top_k,
                chunks=all_chunks
            )
        except Exception as retrieval_error:
            logger.error(f"Retrieval failed: {retrieval_error}")
            # Fallback: use simple keyword search
            logger.warning("Using fallback keyword search...")
            retrieved_chunks = state.hybrid_retriever._keyword_search(
                query=message,
                top_k=top_k,
                chunks=all_chunks
            )
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # Handle empty retrieval results
        if not retrieved_chunks:
            logger.warning("No relevant chunks found for query")
            # Return a helpful response instead of error
            return {
                "session_id": session_id or f"session_{datetime.now().timestamp()}",
                "response": "I couldn't find any relevant information in the uploaded documents to answer your question. Please try rephrasing your query or upload more relevant documents.",
                "sources": [],
                "metrics": {
                    "retrieval_time": int(retrieval_time),
                    "context_assembly_time": 0,
                    "generation_time": 0,
                    "total_time": int((time.time() - start_time) * 1000),
                    "chunks_retrieved": 0,
                    "chunks_used": 0,
                    "tokens_used": 0
                }
            }
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.0f}ms")
        
        # 2. CONTEXT ASSEMBLY: Assemble context from retrieved chunks
        context_start = time.time()
        try:
            context = state.context_assembler.assemble_context(
                chunks=retrieved_chunks,
                query=message,
                include_citations=True,
                format_for_llm=True
            )
        except Exception as context_error:
            logger.error(f"Context assembly failed: {context_error}")
            # Fallback: simple context assembly
            context_parts = []
            for chunk in retrieved_chunks[:5]:  # Use top 5 chunks
                context_parts.append(chunk.chunk.text)
            context = "\n\n".join(context_parts)
        
        context_time = (time.time() - context_start) * 1000
        
        logger.info(f"Assembled context in {context_time:.0f}ms")
        
        # 3. GENERATION: Generate response using LLM
        generation_start = time.time()
        
        # Get temperature from config overrides or use default
        temperature = config.get('temperature', settings.DEFAULT_TEMPERATURE)
        max_tokens = config.get('max_tokens', settings.MAX_TOKENS)
        
        try:
            # Generate response using the response generator
            query_response = await state.response_generator.generate_response(
                query=message,
                context=context,
                sources=retrieved_chunks,
                prompt_type=PromptType.QA,
                include_citations=True,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as generation_error:
            logger.error(f"Generation failed: {generation_error}")
            # Fallback response
            query_response = type('obj', (object,), {
                'answer': "I'm having trouble generating a response right now. Please try again or rephrase your question.",
                'tokens_used': 0
            })()
        
        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Generated response in {generation_time:.0f}ms (Total: {total_time:.0f}ms)")
        
        # Format sources for response
        sources = []
        for i, chunk_with_score in enumerate(retrieved_chunks[:5], 1):
            chunk = chunk_with_score.chunk
            source = {
                "rank": i,
                "score": chunk_with_score.score,
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "retrieval_method": chunk_with_score.retrieval_method
            }
            sources.append(source)
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.now().timestamp()}"
        
        response = {
            "session_id": session_id,
            "response": query_response.answer,
            "sources": sources,
            "metrics": {
                "retrieval_time": int(retrieval_time),
                "context_assembly_time": int(context_time),
                "generation_time": int(generation_time),
                "total_time": int(total_time),
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_used": len(sources),
                "tokens_used": query_response.tokens_used
            }
        }
        
        # Store in session
        if session_id not in state.active_sessions:
            state.active_sessions[session_id] = []
        
        state.active_sessions[session_id].append({
            "query": message,
            "response": query_response.answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
            "metrics": response["metrics"]
        })
        
        logger.info(f"✓ Chat response generated successfully in {total_time:.0f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/configuration")
async def get_configuration():
    """Get current configuration"""
    state = app.state.app
    
    # Check LLM health
    llm_healthy = False
    if state.llm_client:
        llm_healthy = state.llm_client.check_health()
    
    return {
        "configuration": {
            "inference_model": settings.OLLAMA_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunking_strategy": "adaptive",
            "chunk_size": settings.FIXED_CHUNK_SIZE,
            "chunk_overlap": settings.FIXED_CHUNK_OVERLAP,
            "retrieval_top_k": settings.TOP_K_RETRIEVE,
            "vector_weight": settings.VECTOR_WEIGHT,
            "bm25_weight": settings.BM25_WEIGHT,
            "temperature": settings.DEFAULT_TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS,
            "enable_reranking": settings.ENABLE_RERANKING,
            "is_ready": state.is_ready,
            "llm_healthy": llm_healthy
        }
    }


@app.post("/api/configuration")
async def update_configuration(
    temperature: float = Form(None),
    max_tokens: int = Form(None),
    retrieval_top_k: int = Form(None),
    vector_weight: float = Form(None),
    bm25_weight: float = Form(None),
    enable_reranking: bool = Form(None),
    session_id: str = Form(None)
):
    """Update system configuration (runtime parameters only)"""
    state = app.state.app
    
    try:
        updates = {}
        
        # Runtime parameters (no rebuild required)
        if temperature is not None:
            updates["temperature"] = temperature
        
        if max_tokens and max_tokens != settings.MAX_TOKENS:
            updates["max_tokens"] = max_tokens
        
        if retrieval_top_k and retrieval_top_k != settings.TOP_K_RETRIEVE:
            updates["retrieval_top_k"] = retrieval_top_k
        
        if vector_weight is not None and vector_weight != settings.VECTOR_WEIGHT:
            updates["vector_weight"] = vector_weight
            # Update hybrid retriever weights
            if bm25_weight is not None:
                state.hybrid_retriever.update_weights(vector_weight, bm25_weight)
        
        if bm25_weight is not None and bm25_weight != settings.BM25_WEIGHT:
            updates["bm25_weight"] = bm25_weight
        
        if enable_reranking is not None:
            updates["enable_reranking"] = enable_reranking
        
        # Store session-based config overrides
        if session_id:
            if session_id not in state.config_overrides:
                state.config_overrides[session_id] = {}
            state.config_overrides[session_id].update(updates)
        
        logger.info(f"Configuration updated: {updates}")
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "updates": updates
        }
        
    except Exception as e:
        logger.error(f"Configuration update error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics")
async def get_analytics():
    """Get system analytics and metrics"""
    state = app.state.app
    
    try:
        # Calculate basic statistics
        total_sessions = len(state.active_sessions)
        total_messages = sum(len(msgs) for msgs in state.active_sessions.values())
        total_docs = len(state.processed_documents)
        total_chunks = sum(len(chunks) for chunks in state.document_chunks.values())
        
        # Calculate average response time from session history
        all_response_times = []
        for session_messages in state.active_sessions.values():
            for msg in session_messages:
                if 'metrics' in msg and 'total_time' in msg['metrics']:
                    all_response_times.append(msg['metrics']['total_time'])
        
        avg_response_time = int(sum(all_response_times) / len(all_response_times)) if all_response_times else 0
        
        # Get index statistics
        index_stats = state.index_builder.get_index_stats() if state.is_ready else {}
        
        # PLACEHOLDER for Ragas metrics - will be populated when evaluation module is integrated
        ragas_metrics = {
            "answer_relevancy": None,
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None,
            "overall_score": None,
            "status": "not_configured",
            "message": "Evaluation module not yet configured"
        }
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "average_response_time": avg_response_time,
            "ragas_metrics": ragas_metrics,
            "index_stats": index_stats,
            "system_status": {
                "is_ready": state.is_ready,
                "processing_status": state.processing_status,
                "llm_healthy": state.llm_client.check_health() if state.llm_client else False
               }
           }

    except Exception as e:
        logger.error(f"Analytics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export-chat/{session_id}")
async def export_chat(session_id: str, format: str = "json"):
    """Export chat history"""
    state = app.state.app
    if session_id not in state.active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        chat_history = state.active_sessions[session_id]
        
        if format == "json":
            return JSONResponse(content={
                "session_id": session_id,
                "export_time": datetime.now().isoformat(),
                "total_messages": len(chat_history),
                "history": chat_history
            })
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            
            if chat_history:
                fieldnames = ["timestamp", "query", "response", "sources_count", "response_time_ms"]
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for entry in chat_history:
                    writer.writerow({
                        "timestamp": entry.get("timestamp", ""),
                        "query": entry.get("query", ""),
                        "response": entry.get("response", "")[:500],  # Truncate for CSV
                        "sources_count": len(entry.get("sources", [])),
                        "response_time_ms": entry.get("metrics", {}).get("total_time", 0)
                    })
            
            return JSONResponse(content={
                "csv": output.getvalue(),
                "session_id": session_id,
                "format": "csv"
            })
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'json' or 'csv'")
            
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    uvicorn.run("app:app",
                host=settings.HOST,
                port=settings.PORT,
                reload=settings.DEBUG,
                log_level="info"
               )



