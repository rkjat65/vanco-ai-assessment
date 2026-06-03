"""
Hybrid retriever: combines semantic, keyword, and graph retrieval
with reciprocal rank fusion (RRF) and optional cross-encoder reranking.

Retrieval flow for a query:
  1. Semantic search (FAISS cosine)  → top-K1 chunks
  2. Keyword search (BM25 Okapi)     → top-K2 chunks
  3. Graph traversal (NetworkX)      → top-K3 chunks
  4. Merge + deduplicate by chunk_id
  5. RRF scoring: rank-weighted fusion
  6. Optional: cross-encoder reranking for final top-N

Why Reciprocal Rank Fusion (RRF) over simple score normalization:
- Scores from different retrieval systems are incomparable (cosine vs BM25)
- RRF only uses rank position, not raw scores → scale-invariant
- Experimentally robust: consistently outperforms score normalization
- Formula: RRF(d) = Σ 1 / (k + rank_i(d)) where k=60 is a constant

Reranking:
- Uses FlashRank (lightweight cross-encoder) for final re-scoring
- Trade-off: adds ~50-100ms latency but significantly improves precision
- Alternative: cohere-rerank API (better but costs money)
"""

import os
from collections import defaultdict
from typing import Optional

from retrieval.vector_store import VectorStore
from retrieval.keyword_search import BM25Index
from retrieval.graph_store import GraphStore


RRF_K = 60  # RRF constant — empirically optimal around 60
DEFAULT_RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 6))


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Returns deduped list sorted by combined RRF score, highest first.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_data: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = chunk.get("chunk_id", "")
            if not cid:
                continue
            rrf_scores[cid] += 1.0 / (k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = chunk

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    merged = []
    for cid in sorted_ids:
        chunk = dict(chunk_data[cid])
        chunk["rrf_score"] = rrf_scores[cid]
        merged.append(chunk)

    return merged


class HybridRetriever:
    def __init__(
        self,
        index_dir: str = "indexes/",
        top_k_semantic: int = int(os.getenv("TOP_K_SEMANTIC", 5)),
        top_k_keyword: int = int(os.getenv("TOP_K_KEYWORD", 5)),
        top_k_graph: int = int(os.getenv("TOP_K_GRAPH", 3)),
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    ):
        self.index_dir = index_dir
        self.top_k_semantic = top_k_semantic
        self.top_k_keyword = top_k_keyword
        self.top_k_graph = top_k_graph
        self.rerank_top_n = rerank_top_n

        self.vector_store = VectorStore(index_dir=f"{index_dir}/faiss")
        self.bm25_index = BM25Index()
        self.graph_store = GraphStore()
        self._chunks_by_id: dict[str, dict] = {}
        self._loaded = False
        self._reranker = None

    def _try_load_reranker(self):
        """Load FlashRank cross-encoder for reranking (optional, degrades gracefully)."""
        try:
            from flashrank import Ranker
            self._reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
            print("  Cross-encoder reranker loaded")
        except Exception as e:
            print(f"  Reranker not available ({e}); using RRF scores only")
            self._reranker = None

    def load(self):
        """Load all indexes from disk."""
        if self._loaded:
            return
        print("Loading retrieval indexes...")
        self.vector_store.load()
        self.bm25_index.load(f"{self.index_dir}/bm25.pkl")
        self.graph_store.load(f"{self.index_dir}/graph.pkl")

        # Build chunk lookup by ID
        for chunk in self.bm25_index.chunk_dicts:
            self._chunks_by_id[chunk["chunk_id"]] = chunk

        self._try_load_reranker()
        self._loaded = True
        print("All indexes loaded.")

    def retrieve(
        self,
        query: str,
        final_top_n: Optional[int] = None,
    ) -> dict:
        """
        Full hybrid retrieval pipeline.

        Returns:
            {
                "chunks": [top N chunks with scores],
                "debug": {
                    "semantic_results": [...],
                    "keyword_results": [...],
                    "graph_results": [...],
                    "rrf_merged": [...],
                }
            }
        """
        self.load()
        final_top_n = final_top_n or self.rerank_top_n

        # ── Retrieve from all sources ────────────────────────────────────────
        semantic_results = self.vector_store.search(query, top_k=self.top_k_semantic)
        keyword_results  = self.bm25_index.search(query, top_k=self.top_k_keyword)
        graph_results    = self.graph_store.search(
            query, top_k=self.top_k_graph, chunks_by_id=self._chunks_by_id
        )

        # ── Merge with RRF ───────────────────────────────────────────────────
        merged = reciprocal_rank_fusion([
            semantic_results,
            keyword_results,
            graph_results,
        ])

        # ── Rerank (optional) ────────────────────────────────────────────────
        if self._reranker and len(merged) > 0:
            try:
                from flashrank import RerankRequest
                passages = [
                    {"id": c["chunk_id"], "text": c["text"]}
                    for c in merged[:20]  # rerank top 20 before selecting top_n
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                reranked = self._reranker.rerank(rerank_req)

                # Sort by rerank score and map back to chunk dicts
                id_to_chunk = {c["chunk_id"]: c for c in merged}
                final_chunks = []
                for item in sorted(reranked, key=lambda x: x["score"], reverse=True):
                    cid = item.get("id") or item.get("chunk_id", "")
                    if cid in id_to_chunk:
                        chunk = dict(id_to_chunk[cid])
                        chunk["rerank_score"] = item["score"]
                        final_chunks.append(chunk)
                final_chunks = final_chunks[:final_top_n]
            except Exception as e:
                print(f"  Reranking failed: {e}; using RRF order")
                final_chunks = merged[:final_top_n]
        else:
            final_chunks = merged[:final_top_n]

        return {
            "chunks": final_chunks,
            "debug": {
                "query": query,
                "semantic_count": len(semantic_results),
                "keyword_count": len(keyword_results),
                "graph_count": len(graph_results),
                "merged_count": len(merged),
                "final_count": len(final_chunks),
                "semantic_results": semantic_results,
                "keyword_results": keyword_results,
                "graph_results": graph_results,
            }
        }
