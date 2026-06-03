"""
BM25 keyword search for exact term matching.

Why keyword search alongside semantic search:
- Semantic search can miss exact formula names, chapter numbers, exact definitions
- "Coulomb's constant value" → BM25 finds the exact value; semantic may return
  related concepts instead
- Hybrid: semantic handles paraphrase/conceptual; BM25 handles lexical/exact

BM25 vs alternatives:
- Elasticsearch: production choice, but adds server dependency for demo
- PostgreSQL FTS: good if already using postgres
- Whoosh: pure Python, no server, good for demo
- rank_bm25: simplest, no server, perfect for single-document RAG demo

BM25 parameters:
- k1=1.5: term frequency saturation (prevents one rare term from dominating)
- b=0.75: document length normalization (penalizes very long chunks slightly)
"""

import os
import re
import pickle
import string
from pathlib import Path
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    raise ImportError("pip install rank-bm25 nltk")

from ingestion.chunker import Chunk

DEFAULT_TOP_K = int(os.getenv("TOP_K_KEYWORD", 5))


def _ensure_nltk():
    """Download NLTK resources if not present."""
    for resource in ["punkt", "stopwords", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


_ensure_nltk()

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "and", "or"}


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25: lowercase, remove punctuation, remove stopwords.
    Keeps physics keywords (e.g., 'c', 'v', 'e' are important in physics context).
    """
    text = text.lower()
    # Keep Greek letters written out
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = text.split()
    # Only remove stopwords for tokens longer than 1 char
    # (single-letter physics symbols like F, E, B should not be removed)
    filtered = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    return filtered if filtered else tokens


class BM25Index:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_dicts: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[Chunk]):
        self.chunk_dicts = [c.to_dict() for c in chunks]
        self.tokenized_corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=1.5, b=0.75)
        print(f"  BM25 index built for {len(chunks)} chunks")

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() or load() first.")

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = dict(self.chunk_dicts[idx])
            chunk["bm25_score"] = float(scores[idx])
            chunk["retrieval_source"] = "keyword"
            results.append(chunk)

        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_dicts": self.chunk_dicts,
                "tokenized_corpus": self.tokenized_corpus,
            }, f)
        print(f"  BM25 index saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_dicts = data["chunk_dicts"]
        self.tokenized_corpus = data["tokenized_corpus"]
        print(f"  BM25 index loaded ({len(self.chunk_dicts)} chunks)")
