# 🧠 Enterprise Knowledge Assistant

> A RAG-powered Q&A system that lets you ask questions about your company documents in natural language. Upload PDFs, Word docs, and text files — get cited, grounded answers backed by your actual documents.

**Tech Stack:** `Python` `FastAPI` `LangChain` `ChromaDB` `Google Gemini` `SQLAlchemy`

---

## What is RAG?

**RAG = Retrieval Augmented Generation**

Instead of asking a general-purpose AI (which might make things up), RAG:
1. **Retrieves** the most relevant chunks from YOUR documents
2. **Augments** the LLM prompt with that real context
3. **Generates** an answer grounded only in your actual documents

Every answer shows you exactly which document and section it came from. Nothing is invented.

---

## Project Structure

```
enterprise-knowledge-assistant/
├── main.py                              # FastAPI entry point
├── requirements.txt
├── data/sample_docs/                    # Sample documents for testing
│
└── app/
    ├── core/
    │   ├── config.py                    # Settings (Pydantic Settings)
    │   └── database.py                  # SQLAlchemy setup
    │
    ├── models/models.py                 # SQL tables: Document, QueryLog
    ├── schemas/schemas.py               # Pydantic request/response schemas
    │
    ├── services/
    │   ├── document_processor.py        # File reading + text chunking
    │   ├── vector_store.py              # ChromaDB operations (embed + search)
    │   └── rag_service.py               # Full RAG pipeline
    │
    └── api/routes/
        ├── documents.py                 # Upload, list, delete documents
        └── queries.py                   # Ask questions, analytics, feedback
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Add your Gemini API key (free at aistudio.google.com)
```

### 3. Run the server
```bash
python main.py
# Open: http://localhost:8001/docs
```

### 4. Try it end-to-end

**Upload a document:**
```bash
curl -X POST http://localhost:8001/api/v1/documents/upload-text \
  -F "content=TechCorp was founded in 2010. Employees get 21 days annual leave." \
  -F "source_name=handbook.txt" \
  -F "collection_name=hr_policies"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8001/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many leave days do employees get?", "collection_name": "hr_policies"}'
```

**Response:**
```json
{
  "question": "How many leave days do employees get?",
  "answer": "According to handbook.txt, employees receive 21 days of paid annual leave per year.",
  "sources": [
    {
      "document_filename": "handbook.txt",
      "chunk_text": "Employees get 21 days annual leave...",
      "similarity_score": 0.91
    }
  ],
  "had_relevant_context": true,
  "response_time_ms": 1243.5
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload PDF, DOCX, TXT, MD file |
| POST | `/api/v1/documents/upload-text` | Add plain text directly |
| GET | `/api/v1/documents/` | List documents in a collection |
| GET | `/api/v1/documents/collections` | List all collections |
| DELETE | `/api/v1/documents/{id}` | Delete a document |
| POST | `/api/v1/ask` | Ask a question (RAG pipeline) |
| POST | `/api/v1/feedback/{id}` | Submit answer feedback |
| GET | `/api/v1/analytics` | Platform analytics |

---

## Running Tests
```bash
pytest tests/ -v
```

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| **RAG Pipeline** | `rag_service.py` — retrieve → augment → generate |
| **Embeddings** | `vector_store.py` — Google text-embedding-004 via LangChain |
| **Vector DB** | `vector_store.py` — ChromaDB similarity search |
| **LangChain** | Document splitters, Chroma wrapper, ChatGoogleGenerativeAI |
| **FastAPI** | File uploads (UploadFile), async routes, dependency injection |
| **SQL** | Two-table schema: Document metadata + QueryLog |
| **Prompt Engineering** | Grounded RAG prompts that prevent hallucination |
| **Testing** | Mocked embeddings + LLM, 11 tests |

---

## Production Improvements (for interviews)

1. **Hybrid search** — combine semantic search (vector) with BM25 keyword search for better retrieval
2. **Re-ranking** — use a cross-encoder model to re-rank retrieved chunks before generation
3. **Streaming responses** — stream the LLM answer token-by-token using FastAPI's StreamingResponse
4. **Parent document retrieval** — retrieve small chunks but send larger parent chunks to LLM for more context
5. **Multi-modal** — add image extraction from PDFs using vision models
6. **Evaluation** — use RAGAs library to automatically evaluate retrieval quality and answer faithfulness
