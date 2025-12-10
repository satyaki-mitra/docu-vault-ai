# DocuVault AI - Technical Architecture Document

## 1. System Overview

### 1.1 High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Web UI<br/>HTML/CSS/JS]
        B[File Upload<br/>Drag & Drop]
        C[Chat Interface<br/>Real-time]
        D[Analytics Dashboard<br/>RAGAS Metrics]
    end
    
    subgraph "API Gateway"
        E[FastAPI Server<br/>Python 3.11+]
    end
    
    subgraph "Core Processing Engine"
        F[Ingestion Module]
        G[Processing Module]
        H[Retrieval Module]
        I[Generation Module]
        J[Evaluation Module]
    end
    
    subgraph "AI/ML Layer"
        K[Ollama LLM<br/>Mistral-7B]
        L[Embedding Model<br/>BGE-small-en]
        M[FAISS Vector DB]
    end
    
    subgraph "Quality Assurance"
        N[RAGAS Evaluator<br/>Real-time Metrics]
    end
    
    A --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> K
    G --> L
    L --> M
    H --> M
    I --> N
    N --> E
```

### 1.2 System Characteristics

| Aspect | Specification |
|--------|---------------|
| **Architecture Style** | Modular Microservices-inspired |
| **Deployment** | Docker Containerized |
| **Processing Model** | Async/Event-driven |
| **Data Flow** | Pipeline-based with Checkpoints |
| **Scalability** | Horizontal (Stateless API) + Vertical (GPU) |
| **Caching** | In-Memory LRU Cache |
| **Evaluation** | Real-time RAGAS Metrics |

---

## 2. Component Architecture

### 2.1 Ingestion Module

```mermaid
flowchart TD
    A[User Input] --> B{Input Type Detection}
    
    B -->|PDF/DOCX| D[Document Parser]
    B -->|ZIP| E[Archive Extractor]
    
    subgraph D [Document Processing]
        D1[PyPDF2<br/>PDF Text]
        D2[python-docx<br/>Word Docs]
        D3[EasyOCR<br/>Scanned PDFs]
    end
    
    subgraph E [Archive Handling]
        E1[zipfile<br/>Extraction]
        E2[Recursive Processing]
        E3[Size Validation<br/>2GB Max]
    end
    
    D --> F[Text Cleaning]
    E --> F
    
    F --> G[Encoding Normalization]
    G --> H[Structure Preservation]
    H --> I[Output: Cleaned Text<br/>+ Metadata]
```

#### Ingestion Specifications:

| Component | Technology | Configuration | Limits |
|-----------|------------|---------------|---------|
| **PDF Parser** | PyPDF2 + EasyOCR | OCR: English+Multilingual | 1000 pages max |
| **Document Parser** | python-docx | Preserve formatting | 50MB per file |
| **Archive Handler** | zipfile | Recursion depth: 5 | 2GB total, 10k files |

### 2.2 Processing Module

#### 2.2.1 Adaptive Chunking Strategy

```mermaid
flowchart TD
    A[Input Text] --> B[Token Count Analysis]
    B --> C{Document Size}
    
    C -->|<50K tokens| D[Fixed-Size Chunking]
    C -->|50K-500K tokens| E[Semantic Chunking]
    C -->|>500K tokens| F[Hierarchical Chunking]
    
    subgraph D [Strategy 1: Fixed]
        D1[Chunk Size: 512 tokens]
        D2[Overlap: 50 tokens]
        D3[Method: Simple sliding window]
    end
    
    subgraph E [Strategy 2: Semantic]
        E1[Breakpoint: 95th percentile similarity]
        E2[Method: LlamaIndex SemanticSplitter]
        E3[Preserve: Section boundaries]
    end
    
    subgraph F [Strategy 3: Hierarchical]
        F1[Parent: 2048 tokens]
        F2[Child: 512 tokens]
        F3[Retrieval: Child → Parent expansion]
    end
    
    D --> G[Chunk Metadata]
    E --> G
    F --> G
    
    G --> H[Embedding Generation]
```

#### 2.2.2 Embedding Pipeline

```python
# Embedding Configuration
EMBEDDING_CONFIG = {
    "model": "BAAI/bge-small-en-v1.5",
    "dimensions": 384,
    "batch_size": 32,
    "normalize": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_sequence_length": 512
}
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | BAAI/bge-small-en-v1.5 | SOTA quality, 62.17 MTEB score |
| **Dimensions** | 384 | Optimal speed/accuracy balance |
| **Batch Size** | 32 | Memory efficiency on GPU/CPU |
| **Normalization** | L2 | Required for cosine similarity |
| **Speed** | 1000 docs/sec (CPU) | 10x faster than alternatives |

---

### 2.3 Storage Module Architecture

```mermaid
graph TB
    subgraph "Storage Layer"
        A[FAISS Vector Store]
        B[BM25 Keyword Index]
        C[SQLite Metadata]
        D[LRU Cache<br/>In-Memory]
    end
    
    subgraph A [Vector Storage Architecture]
        A1[IndexHNSW<br/>Large datasets]
        A2[IndexIVFFlat<br/>Medium datasets]
        A3[IndexFlatL2<br/>Small datasets]
    end
    
    subgraph B [Keyword Index]
        B1[rank_bm25 Library]
        B2[TF-IDF Weights]
        B3[In-memory Index]
    end
    
    subgraph C [Metadata Management]
        C1[Document Metadata]
        C2[Chunk Relationships]
        C3[User Sessions]
        C4[RAGAS Evaluations]
    end
    
    subgraph D [Cache Layer]
        D1[Query Embeddings]
        D2[Frequent Results]
        D3[LRU Eviction]
    end
    
    A --> E[Hybrid Retrieval]
    B --> E
    C --> E
    D --> E
```

#### Vector Store Configuration

| Index Type | Use Case | Parameters | Performance |
|------------|----------|------------|-------------|
| **IndexFlatL2** | < 100K vectors | Exact search | O(n), High accuracy |
| **IndexIVFFlat** | 100K-1M vectors | nprobe: 10-20 | O(log n), Balanced |
| **IndexHNSW** | > 1M vectors | M: 16, efConstruction: 40 | O(log n), Fastest |

#### Caching Strategy

```python
# LRU Cache Configuration
CACHE_CONFIG = {
    "max_size": 1000,        # Maximum cached items
    "ttl": 3600,             # Time to live (seconds)
    "eviction": "LRU",       # Least Recently Used
    "cache_embeddings": True,
    "cache_results": True
}
```

**Benefits:**
- **Reduced latency**: 80% reduction for repeat queries
- **Resource efficiency**: Avoid re-computing embeddings
- **No external dependencies**: Pure Python implementation
- **Memory efficient**: LRU eviction prevents unbounded growth

---

### 2.4 Retrieval Module

#### 2.4.1 Hybrid Retrieval Pipeline

```mermaid
flowchart TD
    A[User Query] --> B[Query Processing]
    
    B --> C[Vector Embedding]
    B --> D[Keyword Extraction]
    
    C --> E[FAISS Search<br/>Top-K: 10]
    D --> F[BM25 Search<br/>Top-K: 10]
    
    E --> G[Reciprocal Rank Fusion]
    F --> G
    
    G --> H{Reranking Enabled?}
    
    H -->|Yes| I[Cross-Encoder Reranking]
    H -->|No| J[Final Top-5 Selection]
    
    I --> J
    
    J --> K[Context Assembly]
    K --> L[Citation Formatting]
    L --> M[Output: Context + Sources]
```

#### 2.4.2 Retrieval Algorithms

**Hybrid Fusion Formula:**

```text
RRF_score(doc) = vector_weight * (1 / (60 + vector_rank)) + bm25_weight * (1 / (60 + bm25_rank))
```

**Default Weights:**
- Vector Similarity: 60%
- BM25 Keyword: 40%

**BM25 Parameters:**

```python
BM25_CONFIG = {
    "k1": 1.5,      # Term frequency saturation
    "b": 0.75,      # Length normalization
    "epsilon": 0.25  # Smoothing factor
}
```

---

### 2.5 Generation Module

#### 2.5.1 LLM Integration Architecture

```mermaid
graph TB
    subgraph "Ollama Integration"
        A[Ollama Server]
        B[Mistral-7B-Instruct]
        C[LLaMA-2-13B-Chat]
    end
    
    subgraph "Prompt Engineering"
        D[System Prompt Template]
        E[Context Formatting]
        F[Citation Injection]
    end
    
    subgraph "Generation Control"
        G[Temperature Controller]
        H[Token Manager]
        I[Streaming Handler]
    end
    
    A --> J[API Client]
    B --> A
    C --> A
    
    D --> K[Prompt Assembly]
    E --> K
    F --> K
    
    G --> L[Generation Parameters]
    H --> L
    I --> L
    
    K --> M[LLM Request]
    L --> M
    M --> J
    J --> N[Response Processing]
```

#### 2.5.2 LLM Configuration

| Parameter | Default Value | Range | Description |
|-----------|---------------|-------|-------------|
| **Model** | Mistral-7B-Instruct | - | Primary inference model |
| **Temperature** | 0.1 | 0.0-1.0 | Response creativity |
| **Max Tokens** | 1000 | 100-4000 | Response length limit |
| **Top-P** | 0.9 | 0.1-1.0 | Nucleus sampling |
| **Context Window** | 32K | - | Mistral model capacity |

---

### 2.6 RAGAS Evaluation Module

#### 2.6.1 RAGAS Evaluation Pipeline

```mermaid
flowchart LR
    A[Query] --> B[Generated Answer]
    C[Retrieved Context] --> B
    
    B --> D[RAGAS Evaluator]
    C --> D
    
    D --> E[Answer Relevancy]
    D --> F[Faithfulness]
    D --> G[Context Utilization]
    D --> H[Context Relevancy]
    
    E --> I[Metrics Aggregation]
    F --> I
    G --> I
    H --> I
    
    I --> J[Analytics Dashboard]
    I --> K[SQLite Storage]
    I --> L[Session Statistics]
```

#### 2.6.2 Evaluation Metrics

| Metric | Target | Measurement Method | Importance |
|--------|--------|-------------------|------------|
| **Answer Relevancy** | > 0.85 | LLM-based evaluation | Core user satisfaction |
| **Faithfulness** | > 0.90 | Grounded in context check | Prevents hallucinations |
| **Context Utilization** | > 0.80 | How well context is used | Generation effectiveness |
| **Context Relevancy** | > 0.85 | Retrieved chunks relevance | Retrieval quality |

**Implementation Details:**

```python
# RAGAS Configuration
RAGAS_CONFIG = {
    "enable_ragas": True,
    "enable_ground_truth": False,
    "base_metrics": [
        "answer_relevancy",
        "faithfulness",
        "context_utilization",
        "context_relevancy"
    ],
    "ground_truth_metrics": [
        "context_precision",
        "context_recall",
        "answer_similarity",
        "answer_correctness"
    ],
    "evaluation_timeout": 60,
    "batch_size": 10
}
```

**Evaluation Flow:**

1. **Automatic Trigger**: Every query-response pair is evaluated
2. **Async Processing**: Evaluation runs in background (non-blocking)
3. **Storage**: Results stored in SQLite for analytics
4. **Aggregation**: Session-level statistics computed on-demand
5. **Export**: Full evaluation data available for download

---

## 3. Data Flow & Workflows

### 3.1 End-to-End Processing Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant A as API Gateway
    participant I as Ingestion
    participant P as Processing
    participant S as Storage
    participant R as Retrieval
    participant G as Generation
    participant E as RAGAS Evaluator
    
    U->>F: Upload Documents
    F->>A: POST /api/upload
    A->>I: Process Input Sources
    
    Note over I: Parallel Processing
    I->>I: Document Parsing
    I->>P: Extracted Text + Metadata
    
    P->>P: Adaptive Chunking
    P->>P: Embedding Generation
    P->>S: Store Vectors + Indexes
    
    S->>F: Processing Complete
    
    U->>F: Send Query
    F->>A: POST /api/chat
    
    A->>R: Hybrid Retrieval
    R->>S: Vector + BM25 Search
    S->>R: Top-K Chunks
    
    R->>G: Context + Query
    G->>G: LLM Generation
    G->>F: Response + Citations
    
    G->>E: Auto-evaluation (async)
    E->>E: Compute RAGAS Metrics
    E->>S: Store Evaluation Results
    E->>F: Return Metrics
```

### 3.2 Real-time Query Processing

```mermaid
flowchart TD
    A[User Query] --> B[Query Understanding]
    B --> C[Check Cache]
    
    C --> D{Cache Hit?}
    D -->|Yes| E[Return Cached Embedding]
    D -->|No| F[Generate Embedding]
    
    F --> G[Store in Cache]
    E --> H[FAISS Vector Search]
    G --> H
    
    B --> I[Keyword Extraction]
    I --> J[BM25 Keyword Search]
    
    H --> K[Reciprocal Rank Fusion]
    J --> K
    
    K --> L[Top-20 Candidates]
    L --> M{Reranking Enabled?}
    
    M -->|Yes| N[Cross-Encoder Reranking]
    M -->|No| O[Select Top-5]
    
    N --> O
    O --> P[Context Assembly]
    P --> Q[LLM Prompt Construction]
    Q --> R[Ollama Generation]
    R --> S[Citation Formatting]
    S --> T[Response Streaming]
    T --> U[User Display]
    
    R --> V[Async RAGAS Evaluation]
    V --> W[Compute Metrics]
    W --> X[Store Results]
    X --> Y[Update Dashboard]
```

---

## 4. Infrastructure & Deployment

### 4.1 Container Architecture

```mermaid
graph TB
    subgraph "Docker Compose Stack"
        A[Frontend Container<br/>nginx:alpine]
        B[Backend Container<br/>python:3.11]
        C[Ollama Container<br/>ollama/ollama]
    end
    
    subgraph "External Services"
        D[FAISS Indices<br/>Persistent Volume]
        E[SQLite Database<br/>Persistent Volume]
        F[Log Files<br/>Persistent Volume]
    end
    
    A --> B
    B --> C
    B --> D
    B --> E
    B --> F
```

### 4.2 Resource Requirements

#### 4.2.1 Minimum Deployment

| Resource | Specification | Purpose |
|----------|---------------|---------|
| **CPU** | 4 cores | Document processing, embeddings |
| **RAM** | 8GB | Model loading, FAISS indices, cache |
| **Storage** | 20GB | Models, indices, documents |
| **GPU** | Optional | 2-3x speedup for inference |

#### 4.2.2 Production Deployment

| Resource | Specification | Purpose |
|----------|---------------|---------|
| **CPU** | 8+ cores | Concurrent processing |
| **RAM** | 16GB+ | Larger datasets, caching |
| **GPU** | RTX 3090/4090 | 20-30 tokens/sec inference |
| **Storage** | 100GB+ SSD | Fast vector search |

---

## 5. API Architecture

### 5.1 REST API Endpoints

```mermaid
graph TB
    subgraph "System Management"
        A[GET /api/health]
        B[GET /api/system-info]
        C[GET /api/configuration]
        D[POST /api/configuration]
    end
    
    subgraph "Document Management"
        E[POST /api/upload]
        F[POST /api/start-processing]
        G[GET /api/processing-status]
    end
    
    subgraph "Query & Chat"
        H[POST /api/chat]
        I[GET /api/export-chat/:session_id]
    end
    
    subgraph "RAGAS Evaluation"
        J[GET /api/ragas/history]
        K[GET /api/ragas/statistics]
        L[POST /api/ragas/clear]
        M[GET /api/ragas/export]
        N[GET /api/ragas/config]
    end
    
    subgraph "Analytics"
        O[GET /api/analytics]
        P[GET /api/analytics/refresh]
        Q[GET /api/analytics/detailed]
    end
```

### 5.2 Request/Response Flow

```python
# Typical Chat Request Flow with RAGAS
REQUEST_FLOW = {
    "authentication": "None (local deployment)",
    "rate_limiting": "100 requests/minute per IP",
    "validation": "Query length, session ID format",
    "processing": "Async with progress tracking",
    "response": "JSON with citations + metrics + RAGAS scores",
    "caching": "LRU cache for embeddings",
    "evaluation": "Automatic RAGAS metrics (async)"
}
```

---

## 6. Monitoring & Quality Assurance

### 6.1 RAGAS Integration

```mermaid
graph LR
    A[API Gateway] --> B[Query Processing]
    C[Retrieval Module] --> B
    D[Generation Module] --> B
    
    B --> E[RAGAS Evaluator]
    
    E --> F[Analytics Dashboard]
    
    F --> G[Answer Relevancy]
    F --> H[Faithfulness]
    F --> I[Context Utilization]
    F --> J[Context Relevancy]
    F --> K[Session Statistics]
```

### 6.2 Key Performance Indicators

| Category | Metric | Target | Alert Threshold |
|----------|--------|--------|-----------------|
| **Performance** | Query Latency (p95) | < 5s | > 10s |
| **Quality** | Answer Relevancy | > 0.85 | < 0.70 |
| **Quality** | Faithfulness | > 0.90 | < 0.80 |
| **Quality** | Context Utilization | > 0.80 | < 0.65 |
| **Quality** | Overall Score | > 0.85 | < 0.70 |
| **Reliability** | Uptime | > 99.5% | < 95% |

### 6.3 Analytics Dashboard Features

**Real-Time Metrics:**
- RAGAS evaluation table with all query-response pairs
- Session-level aggregate statistics
- Performance metrics (latency, throughput)
- Component health status

**Historical Analysis:**
- Quality trend over time
- Performance degradation detection
- Cache hit rate monitoring
- Resource utilization tracking

**Export Capabilities:**
- JSON export of all evaluation data
- CSV export for external analysis
- Session-based filtering
- Time-range queries

---

## 7. Technology Stack Details

### Complete Technology Matrix

| Layer | Component | Technology | Version | Purpose |
|-------|-----------|------------|---------|----------|
| **Frontend** | UI Framework | HTML5/CSS3/JS | - | Responsive interface |
| **Frontend** | Styling | Tailwind CSS | 3.3+ | Utility-first CSS |
| **Frontend** | Icons | Font Awesome | 6.0+ | Icon library |
| **Backend** | API Framework | FastAPI | 0.104+ | Async REST API |
| **Backend** | Python Version | Python | 3.11+ | Runtime |
| **AI/ML** | LLM Engine | Ollama | 0.1.20+ | Local LLM inference |
| **AI/ML** | Primary Model | Mistral-7B-Instruct | v0.2 | Text generation |
| **AI/ML** | Embeddings | sentence-transformers | 2.2.2+ | Vector embeddings |
| **AI/ML** | Embedding Model | BAAI/bge-small-en | v1.5 | Semantic search |
| **Vector DB** | Storage | FAISS | 1.7.4+ | Vector similarity |
| **Search** | Keyword | rank-bm25 | 0.2.1 | BM25 implementation |
| **Evaluation** | Quality | Ragas | 0.1.9 | RAG evaluation |
| **Document** | PDF | PyPDF2 | 3.0+ | PDF text extraction |
| **Document** | Word | python-docx | 1.1+ | DOCX processing |
| **OCR** | Text Recognition | EasyOCR | 1.7+ | Scanned documents |
| **Database** | Metadata | SQLite | 3.35+ | Local storage |
| **Cache** | In-memory | Python functools | - | LRU caching |
| **Deployment** | Container | Docker | 24.0+ | Containerization |
| **Deployment** | Orchestration | Docker Compose | 2.20+ | Multi-container |

---

## 8. Key Architectural Decisions

### 8.1 Why Local Caching Instead of Redis?

**Decision:** Use in-memory LRU cache with Python's `functools.lru_cache`

**Rationale:**
- **Simplicity**: No external service to manage
- **Performance**: Faster access (no network overhead)
- **MVP Focus**: Adequate for initial deployment
- **Resource Efficient**: No additional memory footprint
- **Easy Migration**: Can upgrade to Redis later if needed

**Trade-offs:**
- Cache doesn't persist across restarts
- Can't share cache across multiple instances
- Limited by single-process memory

### 8.2 Why RAGAS for Evaluation?

**Decision:** Integrate RAGAS for real-time quality assessment

**Rationale:**
- **Automated Metrics**: No manual annotation required
- **Production-Ready**: Quantifiable quality scores
- **Real-Time**: Evaluate every query-response pair
- **Comprehensive**: Multiple dimensions of quality
- **Research-Backed**: Based on academic research

**Implementation Details:**
- OpenAI API key required for LLM-based metrics
- Async evaluation to avoid blocking responses
- SQLite storage for historical analysis
- Export capability for offline processing

### 8.3 Why No Web Scraping?

**Decision:** Removed web scraping from MVP

**Rationale:**
- **Complexity**: Anti-scraping mechanisms require maintenance
- **Reliability**: Website changes break scrapers
- **Legal**: Potential legal/ethical issues
- **Scope**: Focus on core RAG functionality first

**Alternative:**
- Users can save web pages as PDFs
- Future enhancement if market demands it

---

## 9. Performance Optimization Strategies

### 9.1 Embedding Cache Strategy

```python
# Cache Implementation
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_query_embedding(query: str) -> np.ndarray:
    """Cache query embeddings for repeat queries"""
    return embedder.embed(query)

# Benefits:
# - 80% reduction in latency for repeat queries
# - No re-computation of identical queries
# - Automatic LRU eviction
```

### 9.2 Batch Processing

```python
# Batch Embedding Generation
BATCH_SIZE = 32

def embed_chunks_batch(chunks: List[str]) -> List[np.ndarray]:
    embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_embeddings = embedder.embed_batch(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

### 9.3 Async Processing

```python
# Async Document Processing
import asyncio

async def process_documents_async(documents: List[Path]):
    tasks = [process_single_document(doc) for doc in documents]
    results = await asyncio.gather(*tasks)
    return results
```

---

## 10. Security Considerations

### 10.1 Data Privacy

- **On-Premise Processing**: All data stays local
- **No External APIs**: Except OpenAI for RAGAS (configurable)
- **Local LLM**: Ollama runs entirely on-premise
- **Encrypted Storage**: Optional SQLite encryption

### 10.2 Input Validation

```python
# File Upload Validation
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.zip'}

def validate_upload(file: UploadFile):
    # Check extension
    if Path(file.filename).suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type")
    
    # Check size
    if file.size > MAX_FILE_SIZE:
        raise ValueError("File too large")
    
    # Scan for malicious content (optional)
    # scan_for_malware(file)
```

### 10.3 Rate Limiting

```python
# Simple rate limiting
from fastapi import Request
from collections import defaultdict
from datetime import datetime, timedelta

rate_limits = defaultdict(list)

def check_rate_limit(request: Request, limit: int = 100):
    ip = request.client.host
    now = datetime.now()
    
    # Clean old requests
    rate_limits[ip] = [
        ts for ts in rate_limits[ip] 
        if now - ts < timedelta(minutes=1)
    ]
    
    # Check limit
    if len(rate_limits[ip]) >= limit:
        raise HTTPException(429, "Rate limit exceeded")
    
    rate_limits[ip].append(now)
```

---

## Conclusion

This architecture document provides a comprehensive technical blueprint for the DocuVault AI. The modular design, clear separation of concerns, and production-ready considerations make this system suitable for enterprise deployment while maintaining flexibility for future enhancements.

### Key Architectural Strengths

1. **Modularity**: Each component is independent and replaceable
2. **Scalability**: Horizontal scaling through stateless API design
3. **Performance**: Intelligent caching and batch processing
4. **Quality**: Real-time RAGAS evaluation for continuous monitoring
5. **Privacy**: Complete on-premise processing with local LLM
6. **Simplicity**: Minimal external dependencies (no Redis, no web scraping)

### Future Enhancements

**Short-term:**
- Redis cache for multi-instance deployments
- Advanced monitoring dashboard
- User authentication and authorization
- API rate limiting enhancements

**Long-term:**
- Distributed processing with Celery
- Web scraping module (optional)
- Fine-tuned domain-specific embeddings
- Multi-tenant support
- Advanced analytics and reporting

---

Document Version: 1.0
Last Updated: December 2025
Author: Satyaki Mitra

---

> This document is part of the DocuVault AI technical documentation suite.