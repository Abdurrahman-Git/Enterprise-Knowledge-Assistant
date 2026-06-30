"""
WHAT THIS FILE DOES:
────────────────────
The RAG Pipeline — connects retrieval (ChromaDB) to generation (Gemini).

This is the "intelligence" layer. It takes a user question, retrieves
relevant context, builds a grounded prompt, and returns a cited answer.

─────────────────────────────────────────────────────────────────────────────
SKILL LEARNED — LangChain's Role in RAG
─────────────────────────────────────────────────────────────────────────────
LangChain is a framework that provides:
  1. Standard interfaces: LangChain Documents, Chat Models, Embeddings
     all follow the same interface regardless of which provider you use.
     Switch from Gemini to OpenAI by changing one import.

  2. Chains: LangChain "chains" connect components.
     RetrievalQA chain = retriever + LLM + prompt template, all wired up.

  3. Prompt Templates: PromptTemplate / ChatPromptTemplate let you define
     prompts with variables that get filled in at runtime.

  4. Document loaders and text splitters built-in.

In this project we use LangChain for embeddings (GoogleGenerativeAIEmbeddings),
ChromaDB integration (Chroma), and prompt templates (ChatPromptTemplate).

─────────────────────────────────────────────────────────────────────────────
SKILL LEARNED — Prompt Engineering for RAG
─────────────────────────────────────────────────────────────────────────────
RAG prompts have a specific structure:

  SYSTEM: "You are a helpful assistant. Answer ONLY using the provided context.
           If the context doesn't contain the answer, say so. Do NOT make up info."

  CONTEXT: [Chunk 1 text]
            [Chunk 2 text]
            [Chunk 3 text]

  QUESTION: [User's actual question]

The critical constraint: "Answer ONLY using the provided context."
This prevents "hallucination" — the LLM making up plausible-sounding
but incorrect information. Without this constraint, the LLM would mix
its training knowledge with the retrieved documents, making answers
unreliable for enterprise use.
"""

import time
import json
from typing import List, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import settings
from app.services.vector_store import vector_store_service
from app.schemas.schemas import SourceChunk, QueryResponse


class RAGService:
    """
    The RAG (Retrieval Augmented Generation) pipeline.
    
    ask_question() is the main method. It runs the full pipeline:
      1. Retrieve relevant chunks from ChromaDB
      2. Build a grounded prompt with those chunks as context
      3. Call Gemini to generate a cited answer
      4. Return answer + sources
    """

    def __init__(self):
        # ChatGoogleGenerativeAI is LangChain's wrapper for Gemini chat models
        # It follows LangChain's standard ChatModel interface
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2  # Low temperature: factual, consistent answers
        )

    def ask_question(
        self,
        question: str,
        collection_name: str = "default",
        top_k: int = 5
    ) -> QueryResponse:
        """
        Full RAG pipeline: question → retrieve → generate → respond.
        
        This is the method that makes everything work together.
        """
        start_time = time.time()

        # ── STEP 1: RETRIEVE ──────────────────────────────────────────────────
        # Search ChromaDB for the most relevant chunks to this question
        search_results = vector_store_service.similarity_search(
            query=question,
            collection_name=collection_name,
            top_k=top_k
        )

        # Determine if we got useful results
        # A score > 0.3 indicates meaningful relevance
        had_relevant_context = any(
            score > 0.3
            for _, score in search_results
        ) if search_results else False

        # ── STEP 2: BUILD CONTEXT ─────────────────────────────────────────────
        # Format the retrieved chunks into a context string for the LLM
        context_parts = []
        source_chunks = []

        for rank, (doc, score) in enumerate(search_results):
            # Format each chunk with its source for the prompt
            chunk_header = f"[Source {rank+1}: {doc.metadata.get('filename', 'Unknown')}]"
            context_parts.append(f"{chunk_header}\n{doc.page_content}")

            # Build the SourceChunk for the API response
            source_chunks.append(SourceChunk(
                document_filename=doc.metadata.get("filename", "Unknown"),
                chunk_text=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                similarity_score=round(float(score), 3),
                page_number=doc.metadata.get("page_number"),
                chunk_index=doc.metadata.get("chunk_index", 0)
            ))

        context_text = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant documents found."

        # ── STEP 3: GENERATE ──────────────────────────────────────────────────
        # Build the RAG prompt and call Gemini
        answer, answer_found = self._generate_answer(
            question=question,
            context=context_text,
            had_relevant_context=had_relevant_context
        )

        # ── STEP 4: RESPOND ───────────────────────────────────────────────────
        elapsed_ms = round((time.time() - start_time) * 1000, 1)

        return QueryResponse(
            question=question,
            answer=answer,
            sources=source_chunks,
            had_relevant_context=had_relevant_context,
            collection_name=collection_name,
            response_time_ms=elapsed_ms,
            answer_found_in_docs=answer_found
        )

    def _generate_answer(
        self,
        question: str,
        context: str,
        had_relevant_context: bool
    ) -> Tuple[str, bool]:
        """
        Build the RAG prompt and call the LLM.
        
        SKILL LEARNED — The RAG Prompt Structure:
          
          The system message sets strict grounding rules:
          - Only answer from the provided context
          - If context doesn't have the answer, admit it
          - Cite which source you used
          
          This is called "grounded generation" and it's what makes
          RAG reliable for business use vs. a regular chatbot.
        """
        system_message = SystemMessage(content="""You are an intelligent enterprise knowledge assistant.
Your job is to answer questions accurately based ONLY on the provided document context.

STRICT RULES:
1. Answer ONLY using information from the provided context sections
2. If the context doesn't contain enough information, say: 
   "I couldn't find a clear answer to this in the available documents."
3. DO NOT use your general knowledge to fill gaps — that causes hallucination
4. Always mention which source document(s) you used in your answer
5. Be concise and direct — get to the point
6. If the question is asking for a list, use bullet points
7. If relevant, mention the page number from the source

Your answers must be trustworthy and auditable.""")

        user_message = HumanMessage(content=f"""DOCUMENT CONTEXT:
{context}

─────────────────────────────────────────────
QUESTION: {question}

Please answer based on the document context above.""")

        # Call the LLM with our structured messages
        # LangChain's ChatGoogleGenerativeAI accepts a list of messages
        response = self.llm.invoke([system_message, user_message])
        
        # Extract the text answer from LangChain's AIMessage response
        answer_text = response.content

        # Determine if the LLM found a useful answer or said "I don't know"
        not_found_phrases = [
            "couldn't find",
            "not in the",
            "no information",
            "not mentioned",
            "don't have information",
            "cannot find"
        ]
        answer_found = not any(phrase in answer_text.lower() for phrase in not_found_phrases)

        return answer_text, answer_found

    def generate_summary(self, collection_name: str) -> str:
        """
        Generate a summary of what topics are covered in a collection.
        
        SKILL LEARNED — Metadata-driven summaries:
          Instead of reading all documents, we sample a few chunks and ask
          the LLM to describe what the collection is about. Efficient and useful.
        """
        # Get a sample of chunks from this collection
        sample_results = vector_store_service.similarity_search(
            query="What topics and subjects are covered in these documents?",
            collection_name=collection_name,
            top_k=10
        )

        if not sample_results:
            return "No documents found in this collection."

        sample_text = "\n\n".join([doc.page_content[:300] for doc, _ in sample_results])

        system_msg = SystemMessage(content="You are a document analyst. Given sample excerpts from a document collection, write a concise 2-3 sentence summary of what topics the collection covers.")
        user_msg = HumanMessage(content=f"Sample excerpts from the '{collection_name}' collection:\n\n{sample_text}\n\nSummarize what this collection covers in 2-3 sentences.")

        response = self.llm.invoke([system_msg, user_msg])
        return response.content


rag_service = RAGService()
