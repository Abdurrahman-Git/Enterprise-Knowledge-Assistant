"""
WHAT THIS FILE DOES:
────────────────────
FastAPI routes for the RAG Q&A system and analytics.

The /ask endpoint is the main feature — it runs the full RAG pipeline
and returns a cited answer from the knowledge base.
"""

import time
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from app.core.database import get_db
from app.models.models import QueryLog, Document
from app.schemas.schemas import (
    QueryRequest, QueryResponse, AnalyticsSummary
)
from app.services.rag_service import rag_service
from app.services.vector_store import vector_store_service

router = APIRouter(tags=["Knowledge Q&A"])


@router.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Ask a question and get an AI answer grounded in your documents.
    
    This runs the full RAG pipeline:
      1. Embed the question
      2. Retrieve relevant chunks from ChromaDB
      3. Build context-grounded prompt
      4. Generate answer with Gemini
      5. Return answer + source citations
    
    The answer includes SOURCES so users know exactly which documents
    were used. This is what makes it suitable for enterprise use —
    every answer is traceable and auditable.
    """
    # Check that the collection exists and has documents
    doc_count = db.query(Document).filter(
        Document.collection_name == request.collection_name,
        Document.status == "ready",
        Document.is_active == True
    ).count()

    if doc_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection_name}' has no ready documents. "
                   f"Upload documents first via POST /documents/upload"
        )

    # Run the RAG pipeline
    try:
        response = rag_service.ask_question(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {str(e)}")

    # Log the query to SQL for analytics
    source_filenames = ", ".join(set(s.document_filename for s in response.sources))
    log = QueryLog(
        question=request.question,
        answer=response.answer,
        collection_name=request.collection_name,
        chunks_retrieved=len(response.sources),
        source_documents=source_filenames,
        response_time_ms=response.response_time_ms,
        had_relevant_context=response.had_relevant_context
    )
    db.add(log)
    db.commit()

    # If the user doesn't want sources in response, clear them
    if not request.include_sources:
        response.sources = []

    return response


@router.post("/ask/{collection_name}", response_model=QueryResponse)
def ask_question_in_collection(
    collection_name: str,
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Convenience endpoint: ask within a specific collection via URL path."""
    request.collection_name = collection_name
    return ask_question(request, db)


@router.get("/collections/{collection_name}/summary")
def get_collection_summary(collection_name: str):
    """
    Get an AI-generated summary of what a collection covers.
    
    Useful when users want to know what topics they can ask about.
    """
    try:
        summary = rag_service.generate_summary(collection_name)
        return {
            "collection_name": collection_name,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/{query_id}")
def submit_feedback(
    query_id: int,
    rating: int,
    db: Session = Depends(get_db)
):
    """
    Submit feedback on an answer (1=helpful, -1=not helpful).
    
    SKILL LEARNED — Feedback loops in AI systems:
      Storing user feedback lets you:
      1. Monitor answer quality over time
      2. Identify which questions the system struggles with
      3. Build a dataset for evaluating prompt improvements
      4. This is called "RLHF data" (Reinforcement Learning from Human Feedback)
         at a basic level
    """
    if rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 (good) or -1 (bad)")

    log = db.query(QueryLog).filter(QueryLog.id == query_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Query log not found")

    log.user_rating = rating
    db.commit()

    return {"message": "Feedback recorded", "query_id": query_id, "rating": rating}


@router.get("/analytics", response_model=AnalyticsSummary)
def get_analytics(db: Session = Depends(get_db)):
    """Platform-wide analytics dashboard."""
    total_docs = db.query(func.count(Document.id)).filter(
        Document.is_active == True, Document.status == "ready"
    ).scalar() or 0

    total_chunks = db.query(func.sum(Document.chunk_count)).filter(
        Document.is_active == True
    ).scalar() or 0

    total_queries = db.query(func.count(QueryLog.id)).scalar() or 0

    avg_response = db.query(func.avg(QueryLog.response_time_ms)).scalar()

    # Most queried collections
    collection_counts = db.query(
        QueryLog.collection_name,
        func.count(QueryLog.id).label("count")
    ).group_by(QueryLog.collection_name).order_by(func.count(QueryLog.id).desc()).limit(5).all()

    # Recent questions
    recent_logs = db.query(QueryLog.question).order_by(
        QueryLog.created_at.desc()
    ).limit(10).all()

    return AnalyticsSummary(
        total_documents=total_docs,
        total_chunks=int(total_chunks),
        total_queries=total_queries,
        avg_response_time_ms=round(avg_response, 1) if avg_response else None,
        most_queried_collections=[r.collection_name for r in collection_counts if r.collection_name],
        recent_questions=[r.question for r in recent_logs]
    )
