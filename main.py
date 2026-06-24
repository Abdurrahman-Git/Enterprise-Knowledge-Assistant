"""
WHAT THIS FILE DOES:
────────────────────
Main FastAPI application — entry point for the Enterprise Knowledge Assistant.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import engine
from app.models.models import Base
from app.api.routes import documents, queries


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Enterprise Knowledge Assistant...")
    # Create SQL tables on startup
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables initialized")
    print(f"📁 Vector store directory: {settings.CHROMA_PERSIST_DIR}")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## Enterprise Knowledge Assistant — RAG-Powered Q&A System
    
    Upload your company documents and ask questions in natural language.
    Every answer is grounded in your documents with full source citations.
    
    ### Key Features
    * **Upload Documents** — PDF, TXT, DOCX, Markdown
    * **Semantic Search** — finds relevant content by meaning, not just keywords
    * **Cited Answers** — every answer shows which document it came from
    * **Collections** — organize documents by department or topic
    * **Analytics** — track what users are asking about
    
    ### How it works (RAG)
    1. Documents are split into chunks and embedded into vectors
    2. Vectors are stored in ChromaDB for fast similarity search
    3. User question is embedded and matched against stored vectors
    4. Top matching chunks are sent to Gemini as context
    5. Gemini generates an answer grounded only in retrieved context
    
    Built with: FastAPI · LangChain · ChromaDB · Google Gemini · SQLAlchemy
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1")
app.include_router(queries.router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "features": [
            "POST /api/v1/documents/upload — Upload a document",
            "POST /api/v1/documents/upload-text — Add text directly",
            "GET  /api/v1/documents/ — List documents",
            "POST /api/v1/ask — Ask a question",
            "GET  /api/v1/analytics — View analytics"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
