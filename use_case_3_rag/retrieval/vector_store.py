"""
FAISS vector store for semantic retrieval.

Why FAISS over alternatives:
- Chroma/Weaviate: adds server complexity for single-PDF demo
- Pinecone/Qdrant: requires external service + API key
- FAISS: local, fast, well-tested, supports L2 and cosine similarity
- Trade-off: no built-in metadata filtering (handled at re-ranking stage)

Embedding model: all-MiniLM-L6-v2
- 384-dim embeddings; fast on CPU (vs. OpenAI ada-002 which requires API)
- Strong semantic similarity for physics terminology
- Trade-off vs. domain-specific model: generic but not physics-tuned
  → acceptable because physics vocabulary overlaps with general English
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install faiss-cpu sentence-transformers")

from ingestion.chunker import Chunk

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("TOP_K_SEMANTIC", 5))


class VectorStore:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        index_dir: str = "indexes/faiss",
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.chunks: list[Chunk] = []
        self.chunk_dicts: list[dict] = []

    def _load_model(self):
        if self.model is None:
            print(f"  Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def _embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        self._load_model()
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine similarity via dot product
        )

    def build(self, chunks: list[Chunk]):
        """Embed all chunks and build FAISS IVF index."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = chunks
        self.chunk_dicts = [c.to_dict() for c in chunks]

        texts = [c.text for c in chunks]
        print(f"  Embedding {len(texts)} chunks...")
        embeddings = self._embed(texts)

        dim = embeddings.shape[1]

        # IVF index: approximate nearest neighbor, scales to 100K+ chunks
        # For < 10K chunks (typical for a textbook), flat index is fine
        if len(chunks) < 10_000:
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vecs)
        else:
            nlist = min(256, len(chunks) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)

        self.index.add(embeddings)
        self._save()

    def _save(self):
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunk_dicts, f)
        print(f"  Saved FAISS index and chunk metadata to {self.index_dir}")

    def load(self):
        index_path = self.index_dir / "index.faiss"
        chunks_path = self.index_dir / "chunks.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}. Run ingest_pipeline.py first.")
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunk_dicts = pickle.load(f)
        self._load_model()
        print(f"  Loaded FAISS index ({self.index.ntotal} vectors)")

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Semantic search: embed query and return top-k chunks."""
        if self.index is None:
            self.load()

        query_emb = self._embed([query])
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_dicts):
                continue
            chunk = dict(self.chunk_dicts[idx])
            chunk["semantic_score"] = float(score)
            chunk["retrieval_source"] = "semantic"
            results.append(chunk)

        return results
