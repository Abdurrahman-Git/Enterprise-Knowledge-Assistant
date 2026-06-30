"""
WHAT THIS FILE DOES:
────────────────────
Manages ChromaDB — the vector database that stores document embeddings
and enables semantic search.

THIS IS THE CORE OF RAG. Understand this file deeply.

─────────────────────────────────────────────────────────────────────────────
SKILL LEARNED — What is a Vector / Embedding?
─────────────────────────────────────────────────────────────────────────────
An embedding is a list of numbers (a vector) that represents the MEANING
of a piece of text. The embedding model produces these numbers.

Example:
  "Python programming language" → [0.23, -0.81, 0.45, 0.12, ...] (768 numbers)
  "Python snake in the jungle"  → [-0.62, 0.33, -0.54, 0.89, ...] (768 numbers)

Same word, but very different vectors because the meanings are different!
That's what makes embeddings powerful — similar MEANINGS → similar vectors.

To compare how similar two texts are, we compute "cosine similarity" between
their vectors. Score of 1.0 = identical meaning. Score of 0.0 = unrelated.

─────────────────────────────────────────────────────────────────────────────
SKILL LEARNED — What is a Vector Database?
─────────────────────────────────────────────────────────────────────────────
A vector database stores millions of these embedding vectors and can answer
"which of my stored vectors is most similar to this new query vector?"
extremely fast using Approximate Nearest Neighbor (ANN) algorithms.

Without a vector DB:
  To find the most relevant chunk from 10,000 chunks, you'd compute
  cosine similarity with all 10,000 — called a "full scan" — very slow.

With ChromaDB (vector DB):
  ChromaDB builds an index (like HNSW — Hierarchical Navigable Small Worlds)
  that lets it find the top 5 most similar vectors in milliseconds.

─────────────────────────────────────────────────────────────────────────────
SKILL LEARNED — Collections in ChromaDB
─────────────────────────────────────────────────────────────────────────────
ChromaDB organizes vectors into "collections" — similar to tables in SQL
or folders in a file system.

Our knowledge assistant uses one collection per "category" of documents:
  - "hr_policies" → all HR documents
  - "technical_docs" → engineering documentation
  - "default" → uncategorized documents

When a user asks a question, they can specify which collection to search.
This prevents HR questions from pulling in technical documentation.

─────────────────────────────────────────────────────────────────────────────
SKILL LEARNED — The Full RAG Flow (Read this carefully)
─────────────────────────────────────────────────────────────────────────────
INDEXING PHASE (done once when document is uploaded):
  1. Read document → extract text
  2. Split into chunks (DocumentProcessor)
  3. Embed each chunk with the embedding model → list of 768 numbers
  4. Store chunk text + vector + metadata in ChromaDB

RETRIEVAL PHASE (done on every user question):
  1. Embed the user's question → query vector
  2. ChromaDB finds the N chunks whose vectors are most similar to query vector
  3. Those N chunks are the "context"

GENERATION PHASE (done on every user question):
  4. Build a prompt: "Here is context: [chunk1][chunk2]... Answer: [question]"
  5. Send prompt to LLM (Gemini)
  6. LLM reads context and generates an answer grounded in the documents
  7. Return answer + sources to user

RAG = Retrieval Augmented Generation = steps 1-7
"""

import os
from typing import List, Optional
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings


class VectorStoreService:
    """
    Manages ChromaDB vector store operations.
    
    Handles:
    - Adding document chunks (with auto-embedding)
    - Semantic similarity search
    - Collection management
    - Deletion
    """

    def __init__(self):
        # GoogleGenerativeAIEmbeddings is the LangChain wrapper around
        # Google's text-embedding-004 model.
        # It takes text → calls Google API → returns list of 768 floats
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY
        )
        
        # Where ChromaDB persists its data on disk
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

    def _get_collection(self, collection_name: str) -> Chroma:
        """
        Get (or create) a ChromaDB collection.
        
        SKILL LEARNED — Chroma + LangChain integration:
          Chroma.from_existing_collection() loads an existing collection.
          If it doesn't exist yet, we use Chroma() to create it.
          
          LangChain's Chroma wrapper automatically:
          1. Embeds text using our embedding model when adding documents
          2. Embeds the query when searching
          3. Handles the ChromaDB API calls for us
          
          The collection_name lets us namespace documents (like SQL schemas).
        """
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )

    def add_documents(
        self,
        documents: List[LCDocument],
        collection_name: str
    ) -> int:
        """
        Embed and store document chunks in ChromaDB.
        
        WHAT HAPPENS INTERNALLY:
          1. LangChain calls self.embeddings.embed_documents([chunk1, chunk2, ...])
             → Google API returns a vector (list of 768 floats) for each chunk
          2. ChromaDB stores: text + vector + metadata for each chunk
          3. ChromaDB updates its HNSW index for fast future searches
        
        Returns: number of chunks successfully stored
        """
        if not documents:
            return 0

        vector_store = self._get_collection(collection_name)

        # Generate unique IDs for each chunk
        # Format: "docID_chunkIndex" — links ChromaDB entry to SQL document
        ids = [
            f"doc{doc.metadata.get('document_id', 'x')}_chunk{doc.metadata.get('chunk_index', i)}"
            for i, doc in enumerate(documents)
        ]

        # This single call: embeds all chunks + stores in ChromaDB
        # Under the hood: calls Google Embeddings API for each chunk
        vector_store.add_documents(documents=documents, ids=ids)
        
        return len(documents)

    def similarity_search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most relevant document chunks for a query.
        
        WHAT HAPPENS INTERNALLY:
          1. Embed the query: self.embeddings.embed_query(query)
             → Returns one vector (768 floats) for the question
          2. ChromaDB computes cosine similarity between query vector
             and every stored chunk vector
          3. Returns the top_k chunks with highest similarity scores
        
        Returns: list of (LangChain Document, similarity_score) tuples
                 score is between 0.0 (unrelated) and 1.0 (identical)
        
        SKILL LEARNED — similarity_search_with_score:
          This returns scores so we can tell users HOW relevant each chunk is.
          If all scores are below 0.3, the document collection probably
          doesn't contain a good answer to this question.
        """
        vector_store = self._get_collection(collection_name)
        
        # similarity_search_with_score returns (Document, score) pairs
        # Higher score = more relevant
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        return results  # List of (LCDocument, float) tuples

    def delete_document_chunks(self, document_id: int, collection_name: str) -> int:
        """
        Delete all chunks belonging to a specific document.
        
        ChromaDB supports filtering by metadata when deleting.
        We stored document_id in the metadata, so we can target all
        chunks from one document without affecting others.
        """
        vector_store = self._get_collection(collection_name)
        
        # Query to find all chunk IDs for this document
        results = vector_store.get(
            where={"document_id": str(document_id)}
        )
        
        if results and results.get("ids"):
            ids_to_delete = results["ids"]
            vector_store.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        
        return 0

    def get_collection_stats(self, collection_name: str) -> dict:
        """Get stats about a collection (how many chunks, etc.)."""
        try:
            vector_store = self._get_collection(collection_name)
            count = vector_store._collection.count()
            return {"collection_name": collection_name, "chunk_count": count}
        except Exception:
            return {"collection_name": collection_name, "chunk_count": 0}

    def list_collections(self) -> List[str]:
        """List all available collection names."""
        import chromadb
        client = chromadb.PersistentClient(path=self.persist_dir)
        return [col.name for col in client.list_collections()]


vector_store_service = VectorStoreService()
