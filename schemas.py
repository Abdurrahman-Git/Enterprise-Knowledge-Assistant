"""
WHAT THIS FILE DOES:
────────────────────
Pydantic schemas — same pattern as Project 1, same separation of
API contract from database model.

NEW CONCEPTS HERE:

SourceChunk:
  When the RAG system answers a question, it tells you exactly WHICH
  parts of which documents it used. This is critical for enterprise use —
  you don't want to trust an AI answer without knowing where it came from.
  SourceChunk represents one retrieved document chunk with its metadata.

QueryResponse:
  The full answer to a user's question includes:
  - The answer text
  - The source chunks used (with filenames and page numbers)
  - Confidence indicators
  This transparency is what makes RAG trustworthy for business use.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


# ─── Document Schemas ─────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Returned after successfully uploading and processing a document."""
    id: int
    filename: str
    file_type: str
    collection_name: str
    chunk_count: int
    total_characters: int
    status: str
    message: str

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Full document metadata response."""
    id: int
    filename: str
    file_type: str
    file_size: Optional[int]
    collection_name: str
    chunk_count: int
    status: str
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Response when listing documents in a collection."""
    collection_name: str
    total_documents: int
    documents: List[DocumentResponse]


# ─── Query (RAG) Schemas ──────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """What the user sends when asking a question."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question to ask the knowledge base"
    )
    collection_name: str = Field(
        "default",
        description="Which document collection to search"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve"
    )
    # If True, include the raw retrieved chunks in the response
    # Useful for debugging; normally False in production
    include_sources: bool = True


class SourceChunk(BaseModel):
    """
    One retrieved document chunk used to answer the question.
    
    This is the key to RAG transparency — users can see exactly
    which part of which document the answer came from.
    """
    # Which document this chunk came from
    document_filename: str
    
    # The actual text content of this chunk
    chunk_text: str
    
    # How similar this chunk was to the question (0.0 to 1.0)
    # Higher = more relevant
    similarity_score: float
    
    # Page number if available (from PDF)
    page_number: Optional[int] = None
    
    # Position of this chunk within the document
    chunk_index: int


class QueryResponse(BaseModel):
    """
    The complete response to a user's question.
    
    WHY IS THIS RICHER THAN JUST THE ANSWER TEXT?
    Enterprise knowledge assistants need to be auditable. Legal, compliance,
    HR teams need to know WHERE the answer came from so they can verify it.
    Just returning text (like ChatGPT) isn't enough for business use.
    """
    question: str
    answer: str
    
    # The chunks that were retrieved from the vector DB
    sources: List[SourceChunk]
    
    # Did we find relevant context? If False, the answer might be a hallucination
    had_relevant_context: bool
    
    # Which collection was searched
    collection_name: str
    
    # Total time taken for the full RAG pipeline
    response_time_ms: float
    
    # If the LLM couldn't find an answer in the docs, it says so
    answer_found_in_docs: bool


# ─── Collection Management Schemas ───────────────────────────────────────────

class CollectionInfo(BaseModel):
    """Info about one document collection."""
    name: str
    document_count: int
    total_chunks: int


class CollectionsResponse(BaseModel):
    """List of all available collections."""
    collections: List[CollectionInfo]


# ─── Analytics Schemas ────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    """Dashboard analytics for the knowledge base."""
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_response_time_ms: Optional[float]
    most_queried_collections: List[str]
    recent_questions: List[str]
