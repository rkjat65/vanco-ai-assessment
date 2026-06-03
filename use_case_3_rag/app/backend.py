"""
FastAPI backend for the Hybrid RAG application.

Endpoints:
  POST /query          — ask a question, get grounded answer
  GET  /health         — health check
  GET  /debug/{query}  — retrieval debug trace
  GET  /graph/concepts — list graph concepts
  GET  /stats          — index statistics
"""

import sys
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from retrieval.hybrid_retriever import HybridRetriever
from generation.answer_generator import AnswerGenerator

app = FastAPI(
    title="Hybrid RAG — NCERT Physics",
    description="Grounded question answering from NCERT Class 12 Physics using hybrid retrieval",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_DIR = os.getenv("INDEX_DIR", "indexes/")

# Singleton instances (loaded on first request)
_retriever: Optional[HybridRetriever] = None
_generator: Optional[AnswerGenerator] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(index_dir=INDEX_DIR)
        _retriever.load()
    return _retriever


def get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


class QueryRequest(BaseModel):
    question: str
    top_n: int = 6
    show_retrieval_debug: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    sources: list[dict]
    input_tokens: int
    output_tokens: int
    chunks_used: int
    retrieval_debug: Optional[dict] = None


@app.get("/health")
def health():
    return {"status": "ok", "index_dir": INDEX_DIR}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main endpoint: retrieve relevant chunks and generate grounded answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        retriever = get_retriever()
        generator = get_generator()

        retrieval_result = retriever.retrieve(request.question, final_top_n=request.top_n)
        answer_data = generator.generate_with_retrieval_trace(
            request.question, retrieval_result
        )

        return QueryResponse(
            question=request.question,
            answer=answer_data["answer"],
            citations=answer_data["citations"],
            sources=answer_data.get("sources", []),
            input_tokens=answer_data.get("input_tokens", 0),
            output_tokens=answer_data.get("output_tokens", 0),
            chunks_used=answer_data.get("chunks_used", 0),
            retrieval_debug=answer_data.get("retrieval_debug") if request.show_retrieval_debug else None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/{query}")
def debug_retrieval(query: str, top_n: int = 10):
    """Show detailed retrieval trace for a query — useful for live demo explanation."""
    retriever = get_retriever()
    result = retriever.retrieve(query, final_top_n=top_n)
    return result["debug"]


@app.get("/graph/concepts")
def graph_concepts():
    """List all concept nodes in the knowledge graph."""
    retriever = get_retriever()
    concepts = [
        data.get("name")
        for _, data in retriever.graph_store.graph.nodes(data=True)
        if data.get("type") == "concept"
    ]
    return {"concepts": sorted(concepts)}


@app.get("/graph/related/{concept}")
def related_concepts(concept: str):
    """Get concepts related to a given concept."""
    retriever = get_retriever()
    related = retriever.graph_store.get_related_concepts(concept)
    return {"concept": concept, "related": related}


@app.get("/stats")
def stats():
    """Index statistics."""
    retriever = get_retriever()
    return {
        "vector_store_size": retriever.vector_store.index.ntotal if retriever.vector_store.index else 0,
        "bm25_chunks": len(retriever.bm25_index.chunk_dicts),
        "graph_stats": retriever.graph_store.stats(),
    }
