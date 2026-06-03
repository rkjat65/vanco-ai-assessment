"""
End-to-end ingestion pipeline: PDF → chunks → vector store → graph → BM25 index.

Usage:
    python ingestion/ingest_pipeline.py --pdf path/to/ncert_physics.pdf
"""

import argparse
import json
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pdf_parser import extract_blocks, save_blocks
from ingestion.chunker import all_chunks
from retrieval.vector_store import VectorStore
from retrieval.graph_store import GraphStore
from retrieval.keyword_search import BM25Index


def run_ingestion(pdf_path: str, output_dir: str = "indexes/"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("HYBRID RAG INGESTION PIPELINE")
    print(f"  PDF: {pdf_path}")
    print(f"  Output: {output_dir}")
    print("="*60)

    # Step 1: Parse PDF
    print("\n[1/5] Parsing PDF...")
    blocks = extract_blocks(pdf_path)
    save_blocks(blocks, str(output_path / "blocks.json"))
    print(f"  Extracted {len(blocks)} blocks")

    # Step 2: Chunk
    print("\n[2/5] Chunking (heading-aware)...")
    chunks = all_chunks(blocks)
    chunks_data = [c.to_dict() for c in chunks]
    with open(output_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Avg token count: {sum(c.token_count for c in chunks) / len(chunks):.0f}")

    # Step 3: Build vector store
    print("\n[3/5] Building vector store (FAISS + sentence-transformers)...")
    vector_store = VectorStore(index_dir=str(output_path / "faiss"))
    vector_store.build(chunks)
    print(f"  Vector store built with {len(chunks)} embeddings")

    # Step 4: Build knowledge graph
    print("\n[4/5] Building knowledge graph...")
    graph_store = GraphStore()
    graph_store.build_from_chunks(chunks, blocks)
    graph_store.save(str(output_path / "graph.pkl"))
    stats = graph_store.stats()
    print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"  Chapters: {stats['chapters']}")
    print(f"  Concepts: {stats['concepts']}")

    # Step 5: Build BM25 keyword index
    print("\n[5/5] Building BM25 keyword index...")
    bm25_index = BM25Index()
    bm25_index.build(chunks)
    bm25_index.save(str(output_path / "bm25.pkl"))
    print(f"  BM25 index built for {len(chunks)} chunks")

    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"  Indexes saved to: {output_dir}")
    print("="*60)

    return {
        "blocks": len(blocks),
        "chunks": len(chunks),
        "vector_store": str(output_path / "faiss"),
        "graph": str(output_path / "graph.pkl"),
        "bm25": str(output_path / "bm25.pkl"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to NCERT Physics PDF")
    parser.add_argument("--output", default="indexes/", help="Output directory for indexes")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"PDF not found: {args.pdf}")
        sys.exit(1)

    run_ingestion(args.pdf, args.output)
