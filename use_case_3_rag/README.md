# Use Case 3: Hybrid RAG Application for NCERT Physics PDF

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                                │
│                                                                          │
│  NCERT Physics PDF                                                       │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐   heading-aware    ┌─────────────┐                     │
│  │  PDF Parser │──────chunking─────►│   Chunks    │                     │
│  │ (pdfplumber)│  section/formula   │ (max 400tok)│                     │
│  └─────────────┘                   └──────┬──────┘                     │
│                                           │                             │
│              ┌────────────────────────────┼───────────────────┐        │
│              │                            │                   │        │
│              ▼                            ▼                   ▼        │
│  ┌───────────────────┐     ┌──────────────────────┐  ┌────────────┐   │
│  │  FAISS Vector DB  │     │  NetworkX/Neo4j Graph │  │ BM25 Index │  │
│  │  (all-MiniLM-L6)  │     │  Chapters → Sections │  │  (keyword) │  │
│  │  384-dim cosine   │     │  Sections → Concepts │  │  rank-bm25 │  │
│  └───────────────────┘     │  Concepts ↔ Formulas │  └────────────┘  │
│                            └──────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                                   │
│                                                                          │
│  User Question                                                           │
│       │                                                                  │
│       ├──────────────────────────────────────────────────┐              │
│       │                        │                         │              │
│       ▼                        ▼                         ▼              │
│  Semantic Search           Keyword Search            Graph Traversal    │
│  (FAISS cosine)            (BM25 Okapi)             (concept nodes)     │
│  top-5 chunks              top-5 chunks              top-3 chunks       │
│       │                        │                         │              │
│       └────────────────────────┴─────────────────────────┘              │
│                                │                                        │
│                                ▼                                        │
│                   Reciprocal Rank Fusion (RRF)                          │
│                   score = Σ 1/(60 + rank_i)                             │
│                                │                                        │
│                                ▼                                        │
│              Cross-Encoder Reranking (FlashRank)                        │
│              ms-marco-MiniLM-L-12-v2                                    │
│                                │                                        │
│                                ▼                                        │
│              Top-6 Chunks + Citations                                   │
│                                │                                        │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Claude (Anthropic)                           │   │
│  │  System: "Answer ONLY from provided context. Cite sources."     │   │
│  │  Context: [6 retrieved chunks with page/section metadata]       │   │
│  │  → Grounded answer + citations                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                │                                        │
│                                ▼                                        │
│  Streamlit Frontend / FastAPI Response                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env: add your ANTHROPIC_API_KEY
```

### 3. Download NCERT Physics PDF
Download from: https://ncert.nic.in/textbook.php?leph1=0-8 (NCERT Class 12 Physics Part 1)
```bash
# Or use wget/curl
```

### 4. Run ingestion (one-time)
```bash
python ingestion/ingest_pipeline.py --pdf path/to/ncert_physics_part1.pdf
# This creates: indexes/faiss/, indexes/graph.pkl, indexes/bm25.pkl
```

### 5. Launch the app
```bash
# Frontend (recommended for demo)
streamlit run app/frontend.py

# Or backend API only
uvicorn app.backend:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 6. Evaluate
```bash
python evaluation/eval_rag.py --index-dir indexes/
```

---

## Design Decisions & Trade-offs

### Chunking Strategy
| Approach | Pros | Cons |
|---|---|---|
| **Heading-aware (chosen)** | Preserves semantic context, clean citations | Uneven chunk sizes |
| Fixed character split | Uniform sizes | Cuts mid-concept, bad citations |
| Sentence split | Clean units | Loses section context |
| Page-level | Simple | Too large for focused retrieval |

### Vector Store
| Option | Chosen? | Reason |
|---|---|---|
| **FAISS** | ✓ | Local, fast, no server required |
| Chroma | Alternative | Adds server complexity |
| Pinecone | No | Requires external API + cost |
| Weaviate | No | Heavy for single-PDF demo |

### Graph Database
| Option | Chosen? | Reason |
|---|---|---|
| **NetworkX** | ✓ (default) | In-memory, no server, Python native |
| **Neo4j** | ✓ (production) | Same interface, scales to millions of nodes |
| ArangoDB | No | Less common, extra setup |

### Embedding Model
| Model | Chosen? | Reason |
|---|---|---|
| **all-MiniLM-L6-v2** | ✓ | Fast CPU inference, 384-dim |
| OpenAI ada-002 | No | Costs money, requires API |
| bge-large | No | Slower, minimal gain for this domain |

### LLM for Generation
| Model | Chosen? | Reason |
|---|---|---|
| **Claude Sonnet** | ✓ | Best instruction following for strict grounding |
| GPT-4 | Alternative | Also good but more expensive |
| Llama 3 local | Alternative | Free but worse at citation format |

---

## Limitations & Improvement Plan

### Current Limitations
1. **Formula extraction**: Math symbols from PDF don't parse cleanly; LaTeX not preserved
2. **Graph quality**: Concept extraction uses keyword matching — misses implicit relationships
3. **Hallucination risk**: If retrieved chunks lack information but are adjacent, Claude may infer
4. **Latency**: ~2-4s total (1s retrieval + 2-3s generation)
5. **No multi-turn**: Each question is independent — no conversation memory

### Improvement Plan
1. **Better formula extraction**: Use pdfminer with math detection or MathOCR
2. **LLM-based graph extraction**: Use Claude to extract triplets (entity, relation, entity)
3. **Confidence calibration**: Add retrieval score threshold — if all scores low, say "not found"
4. **Streaming**: Stream Claude's response token-by-token for better UX
5. **Multi-turn context**: Maintain conversation history for follow-up questions
6. **Domain embeddings**: Fine-tune embeddings on physics text for better retrieval
7. **Cohere Rerank**: Replace FlashRank with Cohere rerank API for higher accuracy
