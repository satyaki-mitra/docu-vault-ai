# AI Universal Knowledge Ingestion System - Setup Guide

## What We've Built So Far

### ✅ Completed Modules (Foundation Layer)

1. **config/settings.py** - Complete configuration management with environment variables
2. **config/logging_config.py** - Production-grade logging with colors and structured output
3. **config/models.py** - 20+ Pydantic models for type safety
4. **config/prompts.py** - Comprehensive prompt templates for RAG
5. **utils/file_handler.py** - File operations, validation, temp file management
6. **utils/text_cleaner.py** - Text cleaning and normalization
7. **utils/error_handler.py** - Custom exceptions and error handling
8. **requirements.txt** - All dependencies

## Quick Start

### Prerequisites

- Python 3.11+
- Ollama installed and running (for LLM inference)
- 8GB+ RAM (16GB recommended)
- (Optional) NVIDIA GPU with CUDA for faster embeddings

### Installation

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama model
ollama pull mistral:7b-instruct
```

### Environment Configuration

Create `.env` file in project root:

```env
# Application
DEBUG=true
HOST=0.0.0.0
PORT=8000

# File Upload
MAX_FILE_SIZE_MB=100
UPLOAD_DIR=data/uploads

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct
DEFAULT_TEMPERATURE=0.1

# Embeddings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu  # Change to 'cuda' if GPU available
EMBEDDING_BATCH_SIZE=32

# Storage
VECTOR_STORE_DIR=data/vector_store
METADATA_DB_PATH=data/metadata.db

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Retrieval
TOP_K_RETRIEVE=10
VECTOR_WEIGHT=0.6
BM25_WEIGHT=0.4
```

### Test the Foundation

```python
# test_foundation.py
from config.settings import get_settings
from config.logging_config import setup_logging, get_logger
from utils.file_handler import FileHandler
from utils.text_cleaner import TextCleaner

# Initialize
settings = get_settings()
setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)
logger = get_logger(__name__)

# Test configuration
print("=== Configuration Test ===")
print(f"Model: {settings.OLLAMA_MODEL}")
print(f"Max file size: {settings.MAX_FILE_SIZE_MB} MB")
print(f"Allowed types: {settings.ALLOWED_EXTENSIONS}")

# Test file handler
print("\n=== File Handler Test ===")
test_file = "test_document.pdf"
safe_name = FileHandler.safe_filename(test_file)
unique_name = FileHandler.generate_unique_filename(test_file)
print(f"Safe filename: {safe_name}")
print(f"Unique filename: {unique_name}")

# Test text cleaner
print("\n=== Text Cleaner Test ===")
dirty_text = "<p>This  is   a   <b>test</b>   text</p>"
clean_text = TextCleaner.clean(dirty_text)
print(f"Original: {dirty_text}")
print(f"Cleaned: {clean_text}")

logger.info("Foundation tests completed successfully!")
```

Run it:
```bash
python test_foundation.py
```

## Next Steps

We'll build these modules in order:

### Phase 1: Document Parsing (Next)
- ✅ `document_parser/pdf_parser.py` - Extract text from PDFs
- ✅ `document_parser/docx_parser.py` - Extract text from Word docs
- ✅ `document_parser/txt_parser.py` - Handle plain text files
- ✅ `document_parser/parser_factory.py` - Route to correct parser

### Phase 2: Chunking
- `chunking/token_counter.py`
- `chunking/base_chunker.py`
- `chunking/fixed_chunker.py`
- `chunking/adaptive_selector.py`

### Phase 3: Embeddings & Vector Storage
- `embeddings/model_loader.py`
- `embeddings/bge_embedder.py`
- `vector_store/faiss_manager.py`
- `vector_store/bm25_index.py`

### Phase 4: Retrieval
- `retrieval/vector_search.py`
- `retrieval/keyword_search.py`
- `retrieval/hybrid_retriever.py`

### Phase 5: Generation
- `generation/ollama_client.py`
- `generation/prompt_builder.py`
- `generation/response_generator.py`

### Phase 6: API & Frontend
- `app.py` - FastAPI application
- `frontend/index.html` - User interface

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         CONFIGURATION LAYER             │
│  settings.py | logging.py | models.py  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          UTILITIES LAYER                │
│  file_handler | text_cleaner | errors  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        DOCUMENT PARSING LAYER           │
│  pdf_parser | docx_parser | txt_parser │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         PROCESSING LAYER                │
│  chunking | embeddings | indexing       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         RETRIEVAL LAYER                 │
│  vector_search | bm25 | hybrid          │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        GENERATION LAYER                 │
│  ollama_client | prompt_builder         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            API LAYER                    │
│        FastAPI Application              │
└─────────────────────────────────────────┘
```

## Key Design Patterns Used

1. **Dependency Injection**: Settings injected via `get_settings()`
2. **Factory Pattern**: `parser_factory.py` will route to correct parser
3. **Strategy Pattern**: Adaptive chunking selects strategy based on content
4. **Singleton Pattern**: Single settings and logger instances
5. **Context Managers**: `TempFileManager`, `ErrorContext` for resource management
6. **Builder Pattern**: `PromptBuilder` for flexible prompt construction

## Testing Each Layer

After building each layer, test it independently:

```python
# Example: Test document parser after building it
from document_parser.pdf_parser import PDFParser

parser = PDFParser()
text = parser.parse("test.pdf")
print(f"Extracted {len(text)} characters")
```

## Troubleshooting

### Ollama not responding
```bash
# Check Ollama is running
ollama list

# Start Ollama
ollama serve
```

### Import errors
```bash
# Ensure you're in the project root and venv is activated
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### GPU not detected
```python
import torch
print(torch.cuda.is_available())  # Should be True if CUDA GPU present
```

## Ready to Continue?

The foundation is solid! We can now build:
1. Document parsers (PDF, DOCX, TXT)
2. The full chunking system
3. Embedding generation
4. Vector storage
5. Retrieval system
6. LLM generation
7. API endpoints
8. Frontend UI

Let me know when you're ready to proceed with the document parsers!
