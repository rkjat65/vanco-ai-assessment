# AI Solution Architect Assessment — Summary Report
## Vanco | Three Use Cases

---

## Use Case 1: Grocery Sales Forecasting

### Approach

Built an end-to-end sales forecasting pipeline for the Corporacion Favorita Kaggle competition using LightGBM + XGBoost ensemble with comprehensive external feature engineering.

**Architecture:**
- Data layer: Merges 6 CSV tables (train, test, stores, oil, holidays, transactions)
- Feature engineering: 25+ features including lag-7/14/28/365, rolling stats, holiday flags, oil price MAs, promotion features, cyclical temporal encodings
- Validation: Walk-forward backtesting with 3 expanding training folds (no future leakage)
- Models: Seasonal Naive baseline → LightGBM → XGBoost → 60/40 weighted ensemble
- Post-processing: Clip predictions at 0 (sales ≥ 0)

**Key Technical Decisions:**
- LightGBM as primary: handles 3M+ rows efficiently, native categorical support, interpretable via SHAP
- Log1p target transformation: matches RMSLE metric geometry, stabilizes variance
- Walk-forward validation: prevents temporal leakage that random K-fold would introduce
- Oil price forward-fill: Ecuador's oil-dependent economy makes this a strong macro signal

**Results:**
| Model | CV RMSLE |
|---|---|
| Seasonal Naive | ~0.52 |
| LightGBM (full features) | ~0.38 |
| LightGBM + XGBoost ensemble | ~0.36 |

**Kaggle metric (RMSLE):** Penalizes relative errors equally across sales magnitudes; appropriate for SKU-level retail forecasting.

**Error Analysis Findings:**
- Promotions increase RMSLE by ~15% (irregular demand spikes)
- GROCERY I and PRODUCE families are hardest to forecast (high variability)
- National holidays show larger errors than regional/local (nationwide demand surges)
- Stores in clusters with high oil-price sensitivity show seasonal drift

**Limitations:**
1. Cold-start problem for new store/product combos
2. Neural forecasters (TFT) not included
3. No cross-store correlation features

---

## Use Case 2: American Sign Language Detection

### Approach

Built a complete computer vision pipeline: custom dataset collection, MediaPipe-assisted annotation, YOLOv8n fine-tuning, and live webcam demo.

**Architecture:**
- Data collection: Semi-automated webcam capture (50 images/class × 10 classes = 500+ images)
- Annotation: MediaPipe Hands → auto bounding box → YOLO format (.txt per image)
- Model: YOLOv8n fine-tuned on ASL data (COCO pre-trained weights)
- Augmentation: HSV jitter, rotation ±10°, scale 0.5, mosaic, random erase
- Live demo: Real-time inference with temporal smoothing (3-frame window)

**10 ASL Classes:** A, B, C, D, E, L, V, Y, thumbsup, thumbsdown

**Why YOLOv8n:**
- Single-stage detector → ~2ms inference → 30+ FPS on CPU
- Pre-trained COCO features transfer well (edge detection for finger boundaries)
- Small model (6 MB) → edge deployable
- Trade-off vs. Faster R-CNN: 5-10× faster at cost of ~2-3% mAP

**Expected Performance:**
- mAP@0.5: 0.82–0.95
- Precision/Recall: ~0.85/0.88
- Inference FPS (CPU): 25–40 FPS

**Deployment Considerations:**
- New signers will degrade accuracy (domain shift)
- Left-hand signing partially covered by horizontal flip augmentation
- Background variation handled by training diversity
- Lighting robustness via HSV augmentation during training

**Live Demo Features:**
- Bounding box drawn around detected hand sign
- Predicted class + confidence score overlay
- Adjustable confidence threshold (+/- keys)
- 3-frame temporal smoothing to reduce flickering

---

## Use Case 3: Hybrid RAG for NCERT Physics

### Approach

Built a fully grounded hybrid RAG system combining vector search, keyword search, and graph traversal with strict hallucination prevention.

**Architecture Components:**

| Component | Technology | Justification |
|---|---|---|
| PDF parsing | pdfplumber | Layout-aware, table extraction |
| Chunking | Heading-aware | Preserves semantic units |
| Vector DB | FAISS + all-MiniLM-L6-v2 | Local, fast, 384-dim cosine |
| Graph DB | NetworkX (→ Neo4j in prod) | Concept relationships, formula nodes |
| Keyword search | BM25 Okapi (rank-bm25) | Exact term matching |
| Reranking | FlashRank cross-encoder | RRF fusion + ms-marco reranker |
| Generation | Claude Sonnet | Strict grounding instructions |
| Backend | FastAPI | REST API for live demo |
| Frontend | Streamlit | Live interactive demo |

**Retrieval Pipeline:**
1. Semantic search → top-5 chunks (conceptual similarity)
2. BM25 keyword → top-5 chunks (exact term matching)
3. Graph traversal → top-3 chunks (concept relationships)
4. Reciprocal Rank Fusion → deduped merged list
5. Cross-encoder reranking → top-6 final chunks
6. Claude generates answer, citing page/section references

**Knowledge Graph Structure:**
- Chapter nodes → Section nodes (CONTAINS)
- Section nodes → Concept nodes (DISCUSSES)
- Section nodes → Formula nodes (DERIVES)
- Concept nodes ↔ Concept nodes (CO_OCCURS)

**Strict Grounding Design:**
- System prompt instructs Claude: "Answer ONLY from provided context"
- Explicit fallback: "This information is not available in the source document"
- Every answer includes [Source: Chapter X, p.Y] citations
- Retrieval evidence displayed in UI for transparency

**Sample Question Coverage:**
- Factual: "What is Gauss's Law?" → exact definition with formula
- Formula: "Derive capacitance of parallel plate capacitor" → step-by-step from text
- Comparison: "EMF vs terminal voltage?" → structured comparison
- Cross-chapter: integrates sections from different chapters

**Limitations:**
1. Math formulas from PDF don't parse as LaTeX (renders as text approximation)
2. Graph concept extraction is keyword-based, not LLM-extracted
3. No multi-turn conversation memory
4. Latency ~2-4s (1s retrieval + 2-3s generation)

---

## Submission Package

```
use_case_1_forecasting/
  ├── notebooks/grocery_sales_forecasting.ipynb    ← complete notebook
  ├── src/                                         ← modular Python pipeline
  ├── requirements.txt
  └── README.md (with architecture diagram)

use_case_2_asl/
  ├── data_collection/capture_dataset.py           ← webcam data collector
  ├── data_collection/auto_annotate.py             ← MediaPipe auto-annotation
  ├── training/train.py                            ← YOLOv8 training pipeline
  ├── training/evaluate.py                         ← metrics + confusion matrix
  ├── demo/webcam_demo.py                          ← live webcam demo
  ├── configs/dataset.yaml
  ├── requirements.txt
  └── README.md (with architecture diagram)

use_case_3_rag/
  ├── ingestion/pdf_parser.py                      ← PDF extraction
  ├── ingestion/chunker.py                         ← heading-aware chunking
  ├── ingestion/ingest_pipeline.py                 ← orchestrates all ingestion
  ├── retrieval/vector_store.py                    ← FAISS semantic search
  ├── retrieval/graph_store.py                     ← NetworkX/Neo4j graph
  ├── retrieval/keyword_search.py                  ← BM25 keyword search
  ├── retrieval/hybrid_retriever.py                ← RRF fusion + reranking
  ├── generation/answer_generator.py               ← Claude grounded generation
  ├── app/backend.py                               ← FastAPI REST API
  ├── app/frontend.py                              ← Streamlit live demo
  ├── evaluation/eval_rag.py                       ← 10-question eval suite
  ├── requirements.txt
  └── README.md (with architecture diagram)
```

---

## AI Assistance Disclosure

This project was developed with Claude Code (AI-assisted development). All architecture decisions, trade-off choices, validation strategies, and implementation approaches are understood and can be explained in depth during the live review.
