"""
WHAT THIS FILE DOES:
────────────────────
Settings for the Enterprise Knowledge Assistant.
Same Pydantic Settings pattern as Project 1 — loads from .env file.

NEW SETTINGS TO UNDERSTAND:

EMBEDDING_MODEL:
  An embedding model converts text into a list of numbers (a "vector").
  "models/text-embedding-004" is Google's embedding model.
  It produces vectors of 768 numbers for any piece of text.
  Similar texts produce similar vectors — this is how semantic search works.

CHROMA_PERSIST_DIR:
  ChromaDB stores its vector database as files in this folder.
  Unlike SQLite (one .db file), ChromaDB creates a directory with
  multiple files inside. './chroma_db' means it's in your project folder.

TOP_K_RESULTS:
  When a user asks a question, we search the vector DB and retrieve
  the top K most relevant document chunks. K=5 is a good default:
  enough context without overwhelming the LLM's prompt.

CHUNK_SIZE / CHUNK_OVERLAP:
  Documents are split into chunks before being stored.
  chunk_size=1000: each chunk is ~1000 characters
  chunk_overlap=200: chunks share 200 characters with their neighbors
  Overlap prevents answers from being split across chunk boundaries.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise Knowledge Assistant"
    DEBUG: bool = True

    # LLM settings
    GEMINI_API_KEY: str = "your-key-here"
    LLM_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Vector DB
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # RAG settings
    TOP_K_RESULTS: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
