"""
WHAT THIS FILE DOES:
────────────────────
SQLAlchemy models (database tables) for the Knowledge Assistant.

WHY STORE METADATA IN SQL IF WE HAVE A VECTOR DB?
─────────────────────────────────────────────────
ChromaDB stores text chunks and vectors. But it's not ideal for:
  - "How many documents did we upload this week?"
  - "List all PDFs, sorted by upload date"
  - "Delete all documents from collection 'HR Policies'"
  - "What's the total word count of all our documents?"

SQL is perfect for these queries. So we use both:
  SQL = document registry and metadata
  ChromaDB = document content and semantic search

The `document_id` in ChromaDB chunks links back to the `Document.id` in SQL.
This is how the two databases stay in sync.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float
from app.core.database import Base


class Document(Base):
    """
    Tracks every document uploaded to the knowledge base.
    
    SQL TABLE: documents
    
    When you upload a PDF/TXT/DOCX:
    1. We save the metadata here (filename, size, etc.)
    2. We split the content into chunks
    3. We embed each chunk and store in ChromaDB
    4. The ChromaDB chunks reference this document's ID
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    
    # Original filename: "employee_handbook.pdf"
    filename = Column(String(500), nullable=False)
    
    # File type: "pdf", "txt", "docx", "md"
    file_type = Column(String(20), nullable=False)
    
    # File size in bytes
    file_size = Column(Integer, nullable=True)
    
    # The collection this doc belongs to (like folders)
    # e.g., "hr_policies", "technical_docs", "onboarding"
    collection_name = Column(String(200), nullable=False, default="default")
    
    # How many chunks was this document split into?
    # Important: if chunk_count > 0, the vectors exist in ChromaDB
    chunk_count = Column(Integer, default=0)
    
    # Total characters in the document
    total_characters = Column(Integer, nullable=True)
    
    # Processing status
    # "pending" → "processing" → "ready" → "failed"
    status = Column(String(50), default="pending")
    
    # Error message if processing failed
    error_message = Column(Text, nullable=True)
    
    # The embedding model used (important for reproducibility)
    embedding_model = Column(String(200), nullable=True)
    
    # Optional description/tags added by the user
    description = Column(Text, nullable=True)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class QueryLog(Base):
    """
    Logs every question asked to the knowledge assistant.
    
    SQL TABLE: query_logs
    
    WHY LOG QUERIES?
    - Analytics: what are users actually asking about?
    - Quality monitoring: are the answers good?
    - Debugging: if an answer is wrong, you can replay the query
    - Building a Q&A dataset for future fine-tuning
    
    This is a professional practice called "LLM observability" —
    companies like LangSmith and Langfuse are built around this idea.
    """
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # The user's question
    question = Column(Text, nullable=False)
    
    # The AI's answer
    answer = Column(Text, nullable=True)
    
    # Which collection was searched
    collection_name = Column(String(200), nullable=True)
    
    # How many chunks were retrieved from ChromaDB
    chunks_retrieved = Column(Integer, nullable=True)
    
    # The filenames of the source documents used
    source_documents = Column(Text, nullable=True)  # stored as comma-separated
    
    # How long the full RAG pipeline took (milliseconds)
    response_time_ms = Column(Float, nullable=True)
    
    # Confidence indicators
    # We ask the LLM: "did the retrieved context contain a good answer?"
    had_relevant_context = Column(Boolean, nullable=True)
    
    # Optional: user feedback (thumbs up/down)
    user_rating = Column(Integer, nullable=True)  # 1=good, -1=bad, None=no feedback
    
    created_at = Column(DateTime, default=datetime.utcnow)
