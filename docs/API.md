# DocuVault AI- API Documentation

## Overview
The DocuVault AI is a production-grade RAG (Retrieval-Augmented Generation) platform that enables organizations to unlock knowledge from multiple document sources while maintaining complete data privacy and eliminating API costs.

**Base URL:** http://localhost:8000 (or your deployed domain)

**API Version:** v1.0.0

---

## Authentication
Currently, the API operates without authentication for local development. For production deployments, consider implementing:

- API Key Authentication
- JWT Tokens
- OAuth2

---

## Rate Limiting
- Default: 100 requests per minute per IP
- File Uploads: 10MB max per file, 50MB total per request
- Chat Endpoints: 30 requests per minute per session

---

## Response Format

All API responses follow this standard format:

```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Error responses:

```json
{
  "success": false,
  "error": "Error Type",
  "message": "Human-readable error message",
  "detail": {...},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## System Management Endpoints

### Get System Health

**GET** `/api/health`

Check system health and component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "vector_store": true,
    "llm": true,
    "embeddings": true,
    "retrieval": true,
    "generation": true
  },
  "details": {
    "overall": "healthy",
    "vector_store": true,
    "llm": true,
    "embeddings": true,
    "retrieval": true,
    "generation": true
  }
}
```

### Get System Information

**GET** `/api/system-info`

Get comprehensive system status and statistics.

**Response:**
```json
{
  "system_state": {
    "is_ready": true,
    "processing_status": "ready",
    "total_documents": 15,
    "active_sessions": 3
  },
  "configuration": {
    "inference_model": "mistral:7b",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "retrieval_top_k": 10,
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "temperature": 0.1,
    "enable_reranking": true
  },
  "llm_provider": {
    "provider": "ollama",
    "model": "mistral:7b",
    "status": "healthy"
  },
  "system_information": {
    "vector_store_status": "Ready (145 chunks)",
    "current_model": "mistral:7b",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "chunking_strategy": "adaptive",
    "system_uptime_seconds": 3600
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Document Management Endpoints

### Upload Files

**POST** `/api/upload`

Upload multiple documents for processing.

**Form Data:**
- `files`: List of files (PDF, DOCX, TXT, ZIP) - max 2GB total

**Supported Formats:**
- PDF Documents (.pdf)
- Microsoft Word (.docx, .doc)
- Text Files (.txt, .md)
- ZIP Archives (.zip) - automatic extraction

**Response:**
```json
{
  "success": true,
  "message": "Successfully uploaded 3 files",
  "files": [
    {
      "filename": "document_20240115_103000.pdf",
      "original_name": "quarterly_report.pdf",
      "size": 1542890,
      "upload_time": "2024-01-15T10:30:00Z",
      "file_path": "/uploads/document_20240115_103000.pdf",
      "status": "uploaded"
    }
  ]
}
```

### Start Processing

**POST** `/api/start-processing`

Start processing uploaded documents through the RAG pipeline.

**Pipeline Stages:**
1. Document parsing and text extraction
2. Adaptive chunking (fixed/semantic/hierarchical)
3. Embedding generation with BGE model
4. Vector indexing (FAISS + BM25)
5. Knowledge base compilation

**Response:**
```json
{
  "success": true,
  "message": "Processing completed successfully",
  "status": "ready",
  "documents_processed": 3,
  "total_chunks": 245,
  "chunking_statistics": {
    "adaptive": 120,
    "semantic": 80,
    "hierarchical": 45
  },
  "index_stats": {
    "total_chunks_indexed": 245,
    "vector_index_size": 245,
    "bm25_indexed": true,
    "metadata_stored": true
  }
}
```

### Get Processing Status

**GET** `/api/processing-status`

Monitor real-time processing progress.

**Response:**
```json
{
  "status": "processing",
  "progress": 65,
  "current_step": "Generating embeddings for quarterly_report.pdf...",
  "processed": 2,
  "total": 3,
  "details": {
    "chunks_processed": 156,
    "embeddings_generated": 156
  }
}
```

---

## Chat & Query Endpoints

### Chat with Documents

**POST** `/api/chat`

Query your knowledge base with natural language questions. Includes automatic RAGAS evaluation if enabled.

**Request Body (JSON):**
```json
{
  "message": "What were the Q3 revenue trends?",
  "session_id": "session_1705314600"
}
```

**Response:**
```json
{
  "session_id": "session_1705314600",
  "response": "Based on the Q3 financial report, revenue increased by 15% quarter-over-quarter, reaching $45 million. The growth was primarily driven by enterprise sales and new market expansion. [1][2]",
  "sources": [
    {
      "rank": 1,
      "score": 0.894,
      "document_id": "doc_1705300000_abc123",
      "chunk_id": "chunk_doc_1705300000_abc123_0",
      "text_preview": "Q3 Financial Highlights: Revenue growth of 15% QoQ reaching $45M...",
      "page_number": 7,
      "section_title": "Financial Performance",
      "retrieval_method": "hybrid"
    }
  ],
  "metrics": {
    "retrieval_time": 245,
    "generation_time": 3100,
    "total_time": 3345,
    "chunks_retrieved": 8,
    "chunks_used": 3,
    "tokens_used": 487
  },
  "ragas_metrics": {
    "answer_relevancy": 0.89,
    "faithfulness": 0.94,
    "context_utilization": 0.87,
    "context_relevancy": 0.91,
    "overall_score": 0.90,
    "context_precision": null,
    "context_recall": null,
    "answer_similarity": null,
    "answer_correctness": null
  }
}
```

**Note:** Ground truth metrics (context_precision, context_recall, answer_similarity, answer_correctness) are null unless ground truth is provided and `RAGAS_ENABLE_GROUND_TRUTH=True`.

### Export Chat History

**GET** `/api/export-chat/{session_id}`

Export conversation history for analysis or reporting.

**Parameters:**
- `session_id`: string (required) - Session identifier
- `format`: string (optional) - Export format: `json` (default) or `csv`

**Response (JSON):**
```json
{
  "session_id": "session_1705314600",
  "export_time": "2024-01-15T11:00:00Z",
  "total_messages": 5,
  "history": [
    {
      "query": "What was the Q3 revenue growth?",
      "response": "Revenue increased by 15% quarter-over-quarter...",
      "sources": [...],
      "timestamp": "2024-01-15T10:30:00Z",
      "metrics": {
        "total_time": 3345
      },
      "ragas_metrics": {
        "answer_relevancy": 0.89,
        "faithfulness": 0.94,
        "overall_score": 0.90
      }
    }
  ]
}
```

---

## RAGAS Evaluation Endpoints

### Get RAGAS History

**GET** `/api/ragas/history`

Get complete RAGAS evaluation history for the current session.

**Response:**
```json
{
  "success": true,
  "total_count": 25,
  "statistics": {
    "total_evaluations": 25,
    "avg_answer_relevancy": 0.876,
    "avg_faithfulness": 0.912,
    "avg_context_utilization": 0.845,
    "avg_context_relevancy": 0.889,
    "avg_overall_score": 0.881,
    "avg_retrieval_time_ms": 235,
    "avg_generation_time_ms": 3250,
    "avg_total_time_ms": 3485,
    "min_score": 0.723,
    "max_score": 0.967,
    "std_dev": 0.089,
    "session_start": "2024-01-15T09:00:00Z",
    "last_updated": "2024-01-15T11:00:00Z"
  },
  "history": [
    {
      "query": "What were the Q3 revenue trends?",
      "answer": "Revenue increased by 15%...",
      "contexts": ["Q3 Financial Highlights...", "Revenue breakdown..."],
      "timestamp": "2024-01-15T10:30:00Z",
      "answer_relevancy": 0.89,
      "faithfulness": 0.94,
      "context_utilization": 0.87,
      "context_relevancy": 0.91,
      "overall_score": 0.90,
      "retrieval_time_ms": 245,
      "generation_time_ms": 3100,
      "total_time_ms": 3345,
      "chunks_retrieved": 8
    }
  ]
}
```

### Get RAGAS Statistics

**GET** `/api/ragas/statistics`

Get aggregate RAGAS statistics for the current session.

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_evaluations": 25,
    "avg_answer_relevancy": 0.876,
    "avg_faithfulness": 0.912,
    "avg_context_utilization": 0.845,
    "avg_context_relevancy": 0.889,
    "avg_overall_score": 0.881,
    "avg_retrieval_time_ms": 235,
    "avg_generation_time_ms": 3250,
    "avg_total_time_ms": 3485,
    "min_score": 0.723,
    "max_score": 0.967,
    "std_dev": 0.089,
    "session_start": "2024-01-15T09:00:00Z",
    "last_updated": "2024-01-15T11:00:00Z"
  }
}
```

### Clear RAGAS History

**POST** `/api/ragas/clear`

Clear all RAGAS evaluation history and start a new session.

**Response:**
```json
{
  "success": true,
  "message": "RAGAS evaluation history cleared, new session started"
}
```

### Export RAGAS Data

**GET** `/api/ragas/export`

Export all RAGAS evaluation data as JSON.

**Response:** JSON file download containing:
```json
{
  "export_timestamp": "2024-01-15T11:00:00Z",
  "total_evaluations": 25,
  "statistics": {...},
  "evaluations": [...],
  "ground_truth_enabled": false
}
```

### Get RAGAS Configuration

**GET** `/api/ragas/config`

Get current RAGAS configuration settings.

**Response:**
```json
{
  "enabled": true,
  "ground_truth_enabled": false,
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

---

## Analytics Endpoints

### Get System Analytics

**GET** `/api/analytics`

Get comprehensive system analytics and performance metrics with caching.

**Response:**
```json
{
  "performance_metrics": {
    "avg_response_time": 3485,
    "min_response_time": 2100,
    "max_response_time": 8900,
    "total_queries": 127,
    "queries_last_hour": 23,
    "p95_response_time": 7200
  },
  "quality_metrics": {
    "answer_relevancy": 0.876,
    "faithfulness": 0.912,
    "context_precision": 0.845,
    "context_recall": null,
    "overall_score": 0.878,
    "avg_sources_per_query": 4.2,
    "queries_with_sources": 125,
    "confidence": "high",
    "metrics_available": true
  },
  "system_information": {
    "vector_store_status": "Ready (245 chunks)",
    "current_model": "mistral:7b",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "chunking_strategy": "adaptive",
    "system_uptime_seconds": 7200,
    "last_updated": "2024-01-15T11:00:00Z"
  },
  "health_status": {
    "overall": "healthy",
    "llm": true,
    "vector_store": true,
    "embeddings": true,
    "retrieval": true,
    "generation": true
  },
  "chunking_statistics": {
    "primary_strategy": "semantic",
    "total_chunks": 245,
    "strategies_used": {
      "fixed": 98,
      "semantic": 112,
      "hierarchical": 35
    }
  },
  "document_statistics": {
    "total_documents": 15,
    "total_chunks": 245,
    "uploaded_files": 15,
    "total_file_size_bytes": 52428800,
    "total_file_size_mb": 50.0,
    "avg_chunks_per_document": 16.3
  },
  "session_statistics": {
    "total_sessions": 8,
    "total_messages": 127,
    "avg_messages_per_session": 15.9
  },
  "index_statistics": {
    "total_chunks_indexed": 245,
    "vector_index_size": 245,
    "bm25_indexed": true
  },
  "calculated_at": "2024-01-15T11:00:00Z",
  "cache_info": {
    "from_cache": false,
    "next_refresh_in": 30
  }
}
```

### Refresh Analytics Cache

**GET** `/api/analytics/refresh`

Force refresh analytics cache and get fresh data.

**Response:**
```json
{
  "success": true,
  "message": "Analytics cache refreshed successfully",
  "data": {
    // Same structure as /api/analytics
  }
}
```

### Get Detailed Analytics

**GET** `/api/analytics/detailed`

Get detailed analytics including session breakdowns and component performance.

**Response:**
```json
{
  // All fields from /api/analytics, plus:
  "detailed_sessions": [
    {
      "session_id": "session_1705314600",
      "message_count": 12,
      "first_message": "2024-01-15T09:00:00Z",
      "last_message": "2024-01-15T10:45:00Z",
      "total_response_time": 38500,
      "avg_sources_per_query": 3.8
    }
  ],
  "component_performance": {
    "retrieval": {
      "avg_time_ms": 245,
      "cache_hit_rate": 0.23
    },
    "embeddings": {
      "model": "BAAI/bge-small-en-v1.5",
      "dimension": 384,
      "device": "cpu"
    }
  }
}
```

---

## Configuration Endpoints

### Get Current Configuration

**GET** `/api/configuration`

Retrieve current system configuration.

**Response:**
```json
{
  "configuration": {
    "inference_model": "mistral:7b",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "temperature": 0.1,
    "max_tokens": 1000,
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k_retrieve": 10,
    "enable_reranking": true,
    "is_ready": true,
    "llm_healthy": true
  },
  "health": {
    "overall": "healthy",
    "llm": true,
    "vector_store": true,
    "embeddings": true,
    "retrieval": true,
    "generation": true
  }
}
```

### Update Configuration

**POST** `/api/configuration`

Update system configuration parameters.

**Form Data:**
- `temperature`: float (0.0-1.0) - Generation temperature
- `max_tokens`: integer (100-4000) - Maximum response tokens
- `retrieval_top_k`: integer (1-50) - Number of chunks to retrieve
- `vector_weight`: float (0.0-1.0) - Weight for vector search
- `bm25_weight`: float (0.0-1.0) - Weight for keyword search
- `enable_reranking`: boolean - Enable cross-encoder reranking
- `session_id`: string (optional) - Session identifier for overrides

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "updates": {
    "temperature": 0.2,
    "retrieval_top_k": 15
  }
}
```

---

## Error Handling

### Common HTTP Status Codes

- **200** - Success
- **400** - Bad Request (invalid parameters)
- **404** - Resource Not Found
- **500** - Internal Server Error
- **503** - Service Unavailable (component not ready)

### Error Response Examples

#### RAGAS Evaluation Disabled:
```json
{
  "success": false,
  "error": "RAGASDisabled",
  "message": "RAGAS evaluation is not enabled. Set ENABLE_RAGAS=True in settings.",
  "detail": {
    "current_setting": "ENABLE_RAGAS=False"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### System Not Ready:
```json
{
  "success": false,
  "error": "SystemNotReady",
  "message": "System not ready. Please upload and process documents first.",
  "detail": {
    "is_ready": false,
    "documents_processed": 0
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### LLM Service Unavailable:
```json
{
  "success": false,
  "error": "LLMUnavailable",
  "message": "LLM service unavailable. Please ensure Ollama is running.",
  "detail": {
    "llm_healthy": false,
    "suggestion": "Run 'ollama serve' in a separate terminal"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Best Practices

### 1. File Upload

- Use chunked upload for large files (>100MB)
- Compress documents into ZIP archives for multiple files
- Ensure documents are text-extractable (not scanned images without OCR)

### 2. Query Optimization

- Be specific and contextual in questions
- Use natural language - no special syntax required
- Break complex questions into multiple simpler queries

### 3. Session Management

- Reuse `session_id` for conversation continuity
- Sessions automatically expire after 24 hours of inactivity
- Export important conversations for long-term storage

### 4. RAGAS Evaluation

- Ensure OpenAI API key is configured for RAGAS to work
- Monitor evaluation metrics to track system quality
- Use analytics endpoints to identify quality trends
- Export evaluation data regularly for offline analysis

### 5. Performance Monitoring

- Monitor response times and token usage
- Use analytics endpoint for system health checks
- Set up alerts for quality metric degradation
- Enable caching for frequently accessed embeddings

### 6. Configuration Management

- Test configuration changes with a few queries first
- Monitor RAGAS metrics after configuration updates
- Use session-based overrides for experimentation
- Document optimal configurations for different use cases

---

## SDK Examples

### Python Client

```python
import requests

class KnowledgeBaseClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        
    def upload_documents(self, file_paths):
        files = [('files', open(fpath, 'rb')) for fpath in file_paths]
        response = requests.post(f"{self.base_url}/api/upload", files=files)
        return response.json()
    
    def start_processing(self):
        response = requests.post(f"{self.base_url}/api/start-processing")
        return response.json()
    
    def query(self, question):
        data = {'message': question}
        if self.session_id:
            data['session_id'] = self.session_id
        response = requests.post(f"{self.base_url}/api/chat", json=data)
        result = response.json()
        if not self.session_id:
            self.session_id = result.get('session_id')
        return result
    
    def get_ragas_history(self):
        response = requests.get(f"{self.base_url}/api/ragas/history")
        return response.json()
    
    def get_analytics(self):
        response = requests.get(f"{self.base_url}/api/analytics")
        return response.json()

# Usage
client = KnowledgeBaseClient()

# Upload and process
client.upload_documents(['report.pdf', 'contract.docx'])
client.start_processing()

# Query
result = client.query("What are the key findings?")
print(result['response'])
print(f"Quality Score: {result['ragas_metrics']['overall_score']}")

# Get analytics
analytics = client.get_analytics()
print(f"Avg Response Time: {analytics['performance_metrics']['avg_response_time']}ms")
```

### JavaScript Client

```javascript
class KnowledgeBaseClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.sessionId = null;
    }
    
    async uploadDocuments(files) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        
        const response = await fetch(`${this.baseUrl}/api/upload`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }
    
    async startProcessing() {
        const response = await fetch(`${this.baseUrl}/api/start-processing`, {
            method: 'POST'
        });
        return await response.json();
    }
    
    async query(question) {
        const body = { message: question };
        if (this.sessionId) body.session_id = this.sessionId;
        
        const response = await fetch(`${this.baseUrl}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        
        const result = await response.json();
        if (!this.sessionId) this.sessionId = result.session_id;
        return result;
    }
    
    async getRagasHistory() {
        const response = await fetch(`${this.baseUrl}/api/ragas/history`);
        return await response.json();
    }
    
    async getAnalytics() {
        const response = await fetch(`${this.baseUrl}/api/analytics`);
        return await response.json();
    }
}

// Usage
const client = new KnowledgeBaseClient();

// Query
const result = await client.query("What are the revenue trends?");
console.log(result.response);
console.log(`Quality: ${result.ragas_metrics.overall_score}`);

// Get RAGAS history
const history = await client.getRagasHistory();
console.log(`Total evaluations: ${history.total_count}`);
console.log(`Avg relevancy: ${history.statistics.avg_answer_relevancy}`);
```

---

## Support & Troubleshooting

### For API issues:

- Check system health endpoint first
- Verify document processing status
- Review error messages and suggested actions
- Check component readiness flags

### For RAGAS issues:

- Ensure OpenAI API key is configured
- Check RAGAS is enabled in settings
- Monitor evaluation timeout settings
- Review logs for detailed error messages

### For quality issues:

- Monitor RAGAS evaluation metrics
- Adjust retrieval and generation parameters
- Review source citations for context relevance
- Consider document preprocessing improvements

---

> **This API provides a complete RAG solution with multi-format document ingestion, intelligent retrieval, local LLM generation, and comprehensive RAGAS-based quality evaluation.**

---