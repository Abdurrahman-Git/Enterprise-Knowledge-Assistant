"""
WHAT THIS FILE DOES:
────────────────────
Tests for the Enterprise Knowledge Assistant.

TESTING CHALLENGE WITH RAG:
  RAG pipelines have multiple expensive external calls:
  1. Embedding API (Google) — called when uploading docs + asking questions
  2. LLM API (Gemini) — called when generating answers
  3. ChromaDB — local but still stateful

  STRATEGY: Mock the embedding and LLM calls entirely.
  Use a real (in-memory) ChromaDB for testing the vector storage logic.
  Use a real SQLite test DB for the metadata.

SKILL LEARNED — Mocking a full pipeline:
  When your system has multiple components, you mock at the "seam" —
  the boundary between your code and external services.
  Here: mock the Gemini API calls, test everything else for real.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock

os.environ["GEMINI_API_KEY"] = "fake-test-key"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from app.core.database import Base, get_db

# ── Test Database Setup ───────────────────────────────────────────────────────

TEST_DB_URL = "sqlite:///./test_knowledge.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_database():
    """Fresh database for every test."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def mock_vector_store():
    """
    Mock the vector store service to avoid real ChromaDB + Embedding API calls.
    
    WHAT WE'RE MOCKING:
      add_documents() — pretend we stored the chunks
      similarity_search() — return fake relevant chunks
      delete_document_chunks() — pretend we deleted them
    """
    with patch("app.api.routes.documents.vector_store_service") as mock_vs, \
         patch("app.api.routes.queries.vector_store_service") as mock_vs2, \
         patch("app.services.rag_service.vector_store_service") as mock_vs3:

        # When add_documents is called, pretend 3 chunks were stored
        mock_vs.add_documents.return_value = 3
        mock_vs2.add_documents.return_value = 3
        mock_vs3.add_documents.return_value = 3

        # When similarity_search is called, return fake relevant chunks
        from langchain_core.documents import Document as LCDocument
        fake_doc = LCDocument(
            page_content="The company was founded in 2010 and has 500 employees.",
            metadata={
                "filename": "company_info.txt",
                "document_id": "1",
                "chunk_index": 0,
                "page_number": 1,
                "collection_name": "default"
            }
        )
        mock_vs3.similarity_search.return_value = [(fake_doc, 0.87)]
        mock_vs.delete_document_chunks.return_value = 3

        yield mock_vs, mock_vs2, mock_vs3


@pytest.fixture
def mock_doc_processor():
    """Mock document processor to avoid real file parsing."""
    from langchain_core.documents import Document as LCDocument
    with patch("app.api.routes.documents.doc_processor") as mock_proc:
        fake_chunks = [
            LCDocument(
                page_content="This is chunk one of the test document.",
                metadata={"filename": "test.txt", "chunk_index": 0, "document_id": "1"}
            ),
            LCDocument(
                page_content="This is chunk two of the test document.",
                metadata={"filename": "test.txt", "chunk_index": 1, "document_id": "1"}
            ),
            LCDocument(
                page_content="This is chunk three of the test document.",
                metadata={"filename": "test.txt", "chunk_index": 2, "document_id": "1"}
            ),
        ]
        mock_proc.process_file.return_value = (fake_chunks, 1234)
        mock_proc.process_text_directly.return_value = (fake_chunks, 500)
        yield mock_proc


# ── Document Upload Tests ─────────────────────────────────────────────────────

class TestDocumentUpload:

    def test_upload_text_document_success(self, mock_doc_processor, mock_vector_store):
        """Test uploading plain text content."""
        response = client.post("/api/v1/documents/upload-text", data={
            "content": "This is our company handbook. We have 500 employees.",
            "source_name": "company_handbook.txt",
            "collection_name": "hr_policies",
            "description": "Company handbook 2024"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "company_handbook.txt"
        assert data["status"] == "ready"
        assert data["collection_name"] == "hr_policies"
        assert data["chunk_count"] == 3  # matches mock return value

    def test_upload_txt_file(self, mock_doc_processor, mock_vector_store):
        """Test uploading a .txt file."""
        file_content = b"This is a test document about company policies."
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("policy.txt", file_content, "text/plain")},
            data={"collection_name": "policies"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["file_type"] == "txt"

    def test_upload_unsupported_file_type_rejected(self):
        """Test that unsupported file types are rejected."""
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("script.exe", b"binary content", "application/octet-stream")},
            data={"collection_name": "default"}
        )
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]

    def test_list_documents_in_collection(self, mock_doc_processor, mock_vector_store):
        """Test listing documents after upload."""
        # Upload a document first
        client.post("/api/v1/documents/upload-text", data={
            "content": "Test content for listing.",
            "source_name": "test_doc.txt",
            "collection_name": "test_collection"
        })

        response = client.get("/api/v1/documents/?collection_name=test_collection")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 1
        assert data["documents"][0]["filename"] == "test_doc.txt"

    def test_list_empty_collection_returns_zero(self):
        """Test listing a collection with no documents."""
        response = client.get("/api/v1/documents/?collection_name=empty_collection")
        assert response.status_code == 200
        assert response.json()["total_documents"] == 0

    def test_delete_document(self, mock_doc_processor, mock_vector_store):
        """Test deleting a document."""
        # Upload first
        upload_resp = client.post("/api/v1/documents/upload-text", data={
            "content": "Document to be deleted.",
            "source_name": "delete_me.txt",
            "collection_name": "default"
        })
        doc_id = upload_resp.json()["id"]

        # Delete it
        delete_resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert delete_resp.status_code == 200
        assert "deleted" in delete_resp.json()["message"]


# ── RAG Query Tests ───────────────────────────────────────────────────────────

class TestRAGQueries:

    @patch("app.services.rag_service.rag_service.llm")
    def test_ask_question_success(self, mock_llm, mock_doc_processor, mock_vector_store):
        """Test the full RAG Q&A pipeline."""
        # First, upload a document so the collection exists
        client.post("/api/v1/documents/upload-text", data={
            "content": "The company was founded in 2010 and has 500 employees.",
            "source_name": "company_info.txt",
            "collection_name": "default"
        })

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "The company was founded in 2010 according to company_info.txt."
        mock_llm.invoke.return_value = mock_response

        # Ask a question
        response = client.post("/api/v1/ask", json={
            "question": "When was the company founded?",
            "collection_name": "default",
            "top_k": 3
        })

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "When was the company founded?"
        assert len(data["answer"]) > 0
        assert "sources" in data
        assert data["collection_name"] == "default"
        assert data["response_time_ms"] > 0

    def test_ask_question_empty_collection_returns_404(self):
        """Test that asking in a non-existent collection returns 404."""
        response = client.post("/api/v1/ask", json={
            "question": "What is our leave policy?",
            "collection_name": "nonexistent_collection"
        })
        assert response.status_code == 404
        assert "no ready documents" in response.json()["detail"].lower()

    def test_ask_question_too_short_rejected(self):
        """Test Pydantic validation: question must be at least 3 characters."""
        response = client.post("/api/v1/ask", json={
            "question": "Hi",  # Only 2 chars — below min_length=3
            "collection_name": "default"
        })
        assert response.status_code == 422  # Pydantic validation error

    @patch("app.services.rag_service.rag_service.llm")
    def test_query_is_logged_to_db(self, mock_llm, mock_doc_processor, mock_vector_store):
        """Test that queries are saved to the QueryLog table."""
        client.post("/api/v1/documents/upload-text", data={
            "content": "Test content for logging.",
            "source_name": "log_test.txt",
            "collection_name": "default"
        })

        mock_response = MagicMock()
        mock_response.content = "Here is the answer based on log_test.txt."
        mock_llm.invoke.return_value = mock_response

        client.post("/api/v1/ask", json={
            "question": "What is in the log test?",
            "collection_name": "default"
        })

        # Check analytics to confirm the query was logged
        analytics = client.get("/api/v1/analytics")
        assert analytics.status_code == 200
        assert analytics.json()["total_queries"] == 1

    def test_analytics_initial_state(self):
        """Test analytics with no data."""
        response = client.get("/api/v1/analytics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 0
        assert data["total_queries"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
