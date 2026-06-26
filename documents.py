"""
WHAT THIS FILE DOES:
────────────────────
FastAPI routes for document upload and management.

KEY FEATURE — File Upload with FastAPI:
  FastAPI handles multipart file uploads using UploadFile.
  When a user sends a file, FastAPI reads it and gives you:
    file.filename: "employee_handbook.pdf"
    file.content_type: "application/pdf"
    await file.read(): the raw bytes of the file

  We then pass those bytes to DocumentProcessor → VectorStoreService.
"""

import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.models.models import Document
from app.schemas.schemas import (
    DocumentUploadResponse, DocumentResponse,
    DocumentListResponse, CollectionsResponse, CollectionInfo
)
from app.services.document_processor import doc_processor
from app.services.vector_store import vector_store_service

router = APIRouter(prefix="/documents", tags=["Documents"])

# Allowed file types
ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "docx"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, TXT, DOCX, MD)"),
    collection_name: str = Form(default="default", description="Collection/category for this document"),
    description: Optional[str] = Form(default=None),
    db: Session = Depends(get_db)
):
    """
    Upload a document to the knowledge base.
    
    This endpoint:
    1. Reads the uploaded file
    2. Validates the file type
    3. Creates a DB record for the document
    4. Processes the text (splits into chunks)
    5. Embeds chunks and stores in ChromaDB
    6. Updates the DB record with chunk count and status
    
    SKILL LEARNED — async/await in FastAPI:
      File reading is an I/O operation. Using 'await file.read()' lets
      FastAPI handle other requests while waiting for the file to load.
      This is why FastAPI routes can be 'async def' — it's more efficient
      for I/O-bound operations like file reading and API calls.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type .{file_ext} not supported. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Step 1: Create a DB record immediately (status = "processing")
    doc_record = Document(
        filename=file.filename,
        file_type=file_ext,
        collection_name=collection_name,
        description=description,
        status="processing"
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    try:
        # Step 2: Read file bytes
        file_content = await file.read()
        doc_record.file_size = len(file_content)

        # Step 3: Process into chunks
        chunks, total_chars = doc_processor.process_file(
            file_content=file_content,
            filename=file.filename,
            collection_name=collection_name,
            document_id=doc_record.id
        )

        # Step 4: Store in ChromaDB (this calls the Embedding API)
        chunks_stored = vector_store_service.add_documents(
            documents=chunks,
            collection_name=collection_name
        )

        # Step 5: Update DB record with success
        doc_record.chunk_count = chunks_stored
        doc_record.total_characters = total_chars
        doc_record.status = "ready"
        doc_record.embedding_model = "models/text-embedding-004"

    except Exception as e:
        # If anything fails, mark as failed but don't lose the record
        doc_record.status = "failed"
        doc_record.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    db.commit()
    db.refresh(doc_record)

    return DocumentUploadResponse(
        id=doc_record.id,
        filename=doc_record.filename,
        file_type=doc_record.file_type,
        collection_name=doc_record.collection_name,
        chunk_count=doc_record.chunk_count,
        total_characters=doc_record.total_characters or 0,
        status=doc_record.status,
        message=f"Successfully processed {chunks_stored} chunks from {file.filename}"
    )


@router.post("/upload-text", response_model=DocumentUploadResponse)
def upload_text(
    content: str = Form(...),
    source_name: str = Form(..., description="Name for this text document"),
    collection_name: str = Form(default="default"),
    description: Optional[str] = Form(default=None),
    db: Session = Depends(get_db)
):
    """
    Add plain text directly to the knowledge base (no file needed).
    
    Useful for:
    - Adding FAQ content
    - Pasting documentation
    - Adding structured text (company policies typed out, etc.)
    """
    doc_record = Document(
        filename=source_name,
        file_type="txt",
        collection_name=collection_name,
        description=description,
        status="processing"
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    try:
        chunks, total_chars = doc_processor.process_text_directly(
            text=content,
            source_name=source_name,
            collection_name=collection_name,
            document_id=doc_record.id
        )

        chunks_stored = vector_store_service.add_documents(
            documents=chunks,
            collection_name=collection_name
        )

        doc_record.chunk_count = chunks_stored
        doc_record.total_characters = total_chars
        doc_record.file_size = len(content.encode())
        doc_record.status = "ready"
        doc_record.embedding_model = "models/text-embedding-004"

    except Exception as e:
        doc_record.status = "failed"
        doc_record.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    db.commit()
    db.refresh(doc_record)

    return DocumentUploadResponse(
        id=doc_record.id,
        filename=source_name,
        file_type="txt",
        collection_name=collection_name,
        chunk_count=doc_record.chunk_count,
        total_characters=total_chars,
        status="ready",
        message=f"Successfully processed {chunks_stored} chunks"
    )


@router.get("/", response_model=DocumentListResponse)
def list_documents(
    collection_name: str = "default",
    db: Session = Depends(get_db)
):
    """List all documents in a collection."""
    docs = db.query(Document).filter(
        Document.collection_name == collection_name,
        Document.is_active == True,
        Document.status == "ready"
    ).order_by(Document.created_at.desc()).all()

    return DocumentListResponse(
        collection_name=collection_name,
        total_documents=len(docs),
        documents=docs
    )


@router.get("/collections", response_model=CollectionsResponse)
def list_collections(db: Session = Depends(get_db)):
    """List all available document collections with stats."""
    from sqlalchemy import func

    # Group by collection_name and count documents
    results = db.query(
        Document.collection_name,
        func.count(Document.id).label("doc_count"),
        func.sum(Document.chunk_count).label("chunk_count")
    ).filter(Document.is_active == True).group_by(Document.collection_name).all()

    collections = [
        CollectionInfo(
            name=row.collection_name,
            document_count=row.doc_count,
            total_chunks=row.chunk_count or 0
        )
        for row in results
    ]

    return CollectionsResponse(collections=collections)


@router.delete("/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document and all its vector chunks.
    
    SKILL LEARNED — Cascade deletion across two databases:
      We need to delete from BOTH SQLite (metadata) AND ChromaDB (vectors).
      Order matters: delete vectors first, then metadata.
      If ChromaDB deletion fails, we still delete from SQL (best-effort).
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from ChromaDB first
    try:
        deleted_chunks = vector_store_service.delete_document_chunks(
            document_id=document_id,
            collection_name=doc.collection_name
        )
    except Exception:
        deleted_chunks = 0

    # Mark as inactive in SQL (soft delete)
    doc.is_active = False
    db.commit()

    return {
        "message": f"Document '{doc.filename}' deleted",
        "chunks_removed": deleted_chunks
    }
