# Vanco AI Solution Architect Technical Assessment

Three complete use cases: Grocery Sales Forecasting, ASL Detection, and Hybrid RAG.

## Repository Structure

```
├── use_case_1_forecasting/   # Kaggle time-series forecasting pipeline
├── use_case_2_asl/           # American Sign Language detection (YOLO + webcam)
├── use_case_3_rag/           # Hybrid RAG application for NCERT Physics PDF
```

## Quick Start

### Use Case 1 — Grocery Sales Forecasting
```bash
cd use_case_1_forecasting
pip install -r requirements.txt
# Download Kaggle data first (see README inside)
jupyter notebook notebooks/grocery_sales_forecasting.ipynb
```

### Use Case 2 — ASL Detection
```bash
cd use_case_2_asl
pip install -r requirements.txt
# Step 1: Collect data
python data_collection/capture_dataset.py
# Step 2: Train model
python training/train.py
# Step 3: Live demo
python demo/webcam_demo.py
```

### Use Case 3 — Hybrid RAG
```bash
cd use_case_3_rag
pip install -r requirements.txt
cp .env.example .env   # Add your ANTHROPIC_API_KEY
# Step 1: Ingest PDF
python ingestion/ingest_pipeline.py --pdf path/to/physics.pdf
# Step 2: Launch app
streamlit run app/frontend.py
# Or API only:
uvicorn app.backend:app --reload
```

## Architecture Diagrams

See each use case README for detailed architecture diagrams.

## AI Assistance Disclosure

This project was developed with AI-assisted code generation (Claude). All architecture decisions, design choices, trade-offs, and implementation details are understood and can be explained in depth.
