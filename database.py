"""
WHAT THIS FILE DOES:
────────────────────
Database setup — same pattern as Project 1.

In this project we use TWO storage systems:
  1. SQLite (via SQLAlchemy) → stores document METADATA
     (filename, upload time, page count, who uploaded it, etc.)

  2. ChromaDB (vector database) → stores document CONTENT as vectors
     (the actual chunks of text + their embedding vectors)

WHY TWO DATABASES?
  SQL databases are great for structured data you want to filter/sort/count.
  Vector databases are built for one specific job: "find me text that's
  semantically similar to this query." A regular SQL LIKE query would need
  to scan every row. ChromaDB does this in milliseconds using math (cosine
  similarity between vectors).

  Think of it this way:
    SQLite answers: "Give me all PDFs uploaded in the last 7 days"
    ChromaDB answers: "Give me the 5 most relevant chunks for this question"
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

engine = create_engine(
    "sqlite:///./knowledge_base.db",
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
