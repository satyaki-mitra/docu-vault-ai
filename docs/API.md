# AI Universal Knowledge Ingestion System - API Documentation

## Overview
The AI Universal Knowledge Ingestion System is a production-grade RAG (Retrieval-Augmented Generation) platform that enables organizations to unlock knowledge from multiple document sources while maintaining complete data privacy and eliminating API costs.

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
- All API responses follow this standard format:

```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z"
}

- Error responses:

```json
{
  "success": false,
  "error": "Error Type",
  "message": "Human-readable error message",
  "detail": {...},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## System Management Endpoints

### Get System Health

- GET `/api/health`

  - Check system health and component status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "vector_store": "healthy",
    "embedding_model": "healthy",
    "llm_service": "healthy"
  }
}
```

### Get System Information

- GET `/api/system-info`

  - Get comprehensive system status and statistics.

**Response**:
```json
{
  "system_state": {
    "is_ready": true,
    "processing_status": "ready",
    "total_documents": 15,
    "active_sessions": 3,
    "evaluation_ready": true,
    "generation_ready": true
  },
  "configuration": {
    "inference_model": "mistral:7b-instruct",
    "embedding_model": "BAAI/bge-small-en-v1.5"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Document Management Endpoints

### Upload Files

- POST `/api/upload`

  - Upload multiple documents for processing.

 - **Form Data**:

   - files: List of files (PDF, DOCX, TXT, ZIP) - max 2GB total

   - Supported Formats:

     - PDF Documents (.pdf)

     - Microsoft Word (.docx, .doc)

     - Text Files (.txt, .md)

     - ZIP Archives (.zip) - automatic extraction

**Response**:
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
      "status": "uploaded"
    }
  ]
}
```

### Start Processing

POST `/api/start-processing`

- Start processing uploaded documents through the RAG pipeline.

- Pipeline Stages:

  - Document parsing and text extraction

  - Adaptive chunking (fixed/semantic/hierarchical)

  - Embedding generation with BGE model

  - Vector indexing (FAISS + BM25)

  - Knowledge base compilation

**Response**:
```json
{
  "success": true,
  "message": "Processing completed successfully",
  "status": "ready",
  "documents_processed": 3,
  "total_chunks": 245,
  "index_stats": {
    "vector_index": "FAISS-HNSW",
    "keyword_index": "BM25",
    "total_vectors": 245,
    "index_size_mb": 45.2
  }
}
```

### Get Processing Status

- GET `/api/processing-status`

  - Monitor real-time processing progress.

**Response**:

```json
{
  "status": "processing",
  "progress": 65,
  "current_step": "Generating embeddings for quarterly_report.pdf...",
  "processed_documents": 2,
  "total_documents": 3,
  "details": {
    "chunks_processed": 156,
    "embeddings_generated": 156
  }
}
```

---

## Chat & Query Endpoints

### Chat with Documents

- POST `/api/chat`

  - Query your knowledge base with natural language questions.

**Form Data**:

  - message: string (required) - User question

  - session_id: string (optional) - Session identifier for conversation history

**Response**:
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
    "context_assembly_time": 45,
    "generation_time": 1250,
    "total_time": 1540,
    "chunks_retrieved": 8,
    "chunks_used": 3,
    "tokens_used": 487
  }
}
```

### Export Chat History

- GET `/api/export-chat/{session_id}`

  - Export conversation history for analysis or reporting.

  - Parameters:

    - session_id: string (required) - Session identifier

    - format: string (optional) - Export format: json (default) or csv

**Response (JSON)**:
```json
{
  "session_id": "session_1705314600",
  "history": [
    {
      "query": "What was the Q3 revenue growth?",
      "response": "Revenue increased by 15% quarter-over-quarter...",
      "sources": [...],
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

---

## Evaluation & Analytics Endpoints

### Evaluate Response Quality

- POST `/api/evaluate`

  - Evaluate a query-response pair using Ragas metrics with GPT-2 as judge LLM.

  - Form Data:

    - query: string (required) - Original user question

    - response: string (required) - Generated answer to evaluate

    - session_id: string (optional) - Session context for evaluation

**Response**:
```json
{
  "success": true,
  "evaluation_result": {
    "answer_relevancy": 0.87,
    "faithfulness": 0.92,
    "context_precision": 0.78,
    "context_recall": 0.81,
    "overall_score": 0.845,
    "evaluation_time_ms": 1245,
    "model_used": "GPT-2 (gpt2)"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Get Evaluation Metrics

- GET `/api/evaluation-metrics`

  - Get comprehensive system evaluation statistics.

**Response**:

```json
{
  "evaluation_status": "available",
  "metrics": {
    "average_overall_score": 0.823,
    "total_evaluations": 147,
    "recent_evaluations": 50,
    "metrics_breakdown": {
      "answer_relevancy": {
        "average": 0.834,
        "std_dev": 0.067,
        "min": 0.612,
        "max": 0.956,
        "count": 147
      },
      "faithfulness": {
        "average": 0.891,
        "std_dev": 0.045,
        "min": 0.723,
        "max": 0.978,
        "count": 147
      }
    },
    "judge_model": "GPT-2"
  }
}
```

### Get System Analytics

- GET `/api/analytics`

  - Get comprehensive system analytics and performance metrics.

**Response**:

```json
{
  "total_sessions": 45,
  "total_messages": 289,
  "total_documents": 15,
  "total_chunks": 1245,
  "average_response_time": 1845,
  "ragas_metrics": {
    "answer_relevancy": 0.834,
    "faithfulness": 0.891,
    "context_precision": 0.765,
    "context_recall": 0.812,
    "overall_score": 0.823,
    "evaluation_status": "active",
    "total_evaluations": 147,
    "judge_model": "GPT-2",
    "message": "Real-time evaluation with 147 samples"
  },
  "index_stats": {
    "vector_index": "FAISS-HNSW",
    "keyword_index": "BM25",
    "total_vectors": 1245,
    "index_size_mb": 228.7
  },
  "system_status": {
    "is_ready": true,
    "processing_status": "ready",
    "generation_ready": true,
    "evaluation_ready": true
  },
  "generation_info": {
    "model": "mistral:7b-instruct",
    "provider": "ollama",
    "temperature_controller": {
      "strategy": "adaptive",
      "base_temperature": 0.1
    }
  }
}
```

---

## Configuration Endpoints

### Get Current Configuration

- GET `/api/configuration`

  - Retrieve current system configuration.

**Response**:

```json
{
  "configuration": {
    "inference_model": "mistral:7b-instruct",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "chunking_strategy": "adaptive",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "retrieval_top_k": 10,
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "temperature": 0.1,
    "max_tokens": 1000,
    "enable_reranking": false,
    "is_ready": true,
    "evaluation_enabled": true
  }
}
```

### Update Configuration

- POST /api/configuration

  - Update system configuration parameters.

  - Form Data:

    - `inference_model`: string (optional) - LLM model name

    - `chunking_strategy`: string (optional) - "fixed", "semantic", "hierarchical"

    - `chunk_size`: integer (optional) - Token size for chunks (100-2000)

    - `chunk_overlap`: integer (optional) - Token overlap between chunks (0-200)

    - `retrieval_top_k`: integer (optional) - Number of chunks to retrieve (1-50)

    - `vector_weight`: float (optional) - Weight for vector search (0.0-1.0)

    - `bm25_weight`: float (optional) - Weight for keyword search (0.0-1.0)

    - `temperature`: float (optional) - Generation temperature (0.0-1.0)

    - `max_tokens`: integer (optional) - Maximum response tokens (100-4000)

    - `enable_reranking`: boolean (optional) - Enable cross-encoder reranking

    - `rebuild_knowledge_base`: boolean (optional) - Rebuild indexes with new config

**Response**:

```json
{
  "success": true,
  "message": "Configuration updated successfully. Knowledge base rebuilt successfully with new configuration.",
  "updates": {
    "chunk_size": 768,
    "temperature": 0.2
  },
  "rebuild_performed": true,
  "rebuild_message": "Knowledge base rebuilt successfully with new configuration.",
  "system_ready": true
}
```

---

## WebSocket Endpoints

### Real-time Updates

- WebSocket `/ws`

  - Establish WebSocket connection for real-time system updates.

  - Connection:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

  - Message Types:

    - system_message: General system notifications

    - progress_update: Processing progress updates

    - evaluation_update: Real-time evaluation results

**Example Messages**:

```json
{
  "type": "progress_update",
  "progress": 75,
  "current_step": "Building vector indexes...",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

```json
{
  "type": "evaluation_update",
  "session_id": "session_1705314600",
  "metrics": {
    "answer_relevancy": 0.87,
    "faithfulness": 0.92,
    "overall_score": 0.845
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Error Handling

### Common HTTP Status Codes

- 200 - Success

- 400 - Bad Request (invalid parameters)

- 404 - Resource Not Found

- 500 - Internal Server Error

- 503 - Service Unavailable (component not ready)


### Error Response Examples

#### Document Processing Error:

```json
{
  "success": false,
  "error": "ProcessingError",
  "message": "Failed to process document: Unsupported file format",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### System Not Ready:

```json
{
  "success": false,
  "error": "SystemNotReady",
  "message": "System not ready. Please upload and process documents first.",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Evaluation Service Unavailable:

```json
{
  "success": false,
  "error": "EvaluationUnavailable",
  "message": "Evaluation service not available",
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

- Reuse session_id for conversation continuity

- Sessions automatically expire after 24 hours of inactivity

- Export important conversations for long-term storage


### 4. Performance Monitoring

- Monitor response times and token usage

- Use analytics endpoint for system health checks

- Set up alerts for quality metric degradation


### 5. Evaluation Integration

- Enable auto-evaluation for quality assurance

- Monitor Ragas metrics for system improvement

- Use evaluation results to fine-tune configuration

---

## SDK Examples

### Python Client

```python
import requests

class KnowledgeBaseClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def upload_documents(self, file_paths):
        files = [('files', open(fpath, 'rb')) for fpath in file_paths]
        response = requests.post(f"{self.base_url}/api/upload", files=files)
        return response.json()
    
    def query(self, question, session_id=None):
        data = {'message': question}
        if session_id:
            data['session_id'] = session_id
        response = requests.post(f"{self.base_url}/api/chat", data=data)
        return response.json()
```

### JavaScript Client

```javascript
class KnowledgeBaseClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async query(question, sessionId = null) {
        const formData = new FormData();
        formData.append('message', question);
        if (sessionId) formData.append('session_id', sessionId);
        
        const response = await fetch(`${this.baseUrl}/api/chat`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    }
}
```

--- 

## Support & Troubleshooting

### For API issues:

- Check system health endpoint first

- Verify document processing status

- Review error messages and suggested actions

- Check component readiness flags


### For quality issues:

- Monitor Ragas evaluation metrics

- Adjust retrieval and generation parameters

- Review source citations for context relevance

- Consider document preprocessing improvements


> **This API provides a complete RAG solution with multi-source ingestion, intelligent retrieval, and comprehensive evaluation capabilities.**

---