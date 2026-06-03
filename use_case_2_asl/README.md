# Use Case 2: American Sign Language Detection

## Architecture Diagram

```
                        ┌──────────────────────────────────────────────────┐
                        │              DATA COLLECTION PIPELINE             │
                        │                                                   │
  Webcam Input ────────►│  capture_dataset.py                               │
  (live video)          │  - Manual trigger (SPACE) per frame               │
                        │  - 50 images per class                            │
                        │  - 10 ASL classes = 500+ raw images               │
                        └──────────────┬───────────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────────────────────┐
                        │            ANNOTATION PIPELINE                    │
                        │                                                   │
                        │  auto_annotate.py                                 │
                        │  ├── MediaPipe Hands → auto bbox detection        │
                        │  ├── Padding: 8% around hand landmarks            │
                        │  ├── YOLO format: class x_c y_c w h               │
                        │  └── 80/10/10 train/val/test split                │
                        └──────────────┬───────────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────────────────────┐
                        │             TRAINING PIPELINE                     │
                        │                                                   │
                        │  YOLOv8n (pre-trained COCO)                       │
                        │  ├── Fine-tune on ASL dataset                     │
                        │  ├── Augmentation: HSV, rotate, scale,            │
                        │  │   mosaic, mixup, random erase                  │
                        │  ├── Early stopping (patience=20)                 │
                        │  └── Best weights saved by mAP@0.5                │
                        └──────────────┬───────────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────────────────────┐
                        │          EVALUATION                               │
                        │  mAP@0.5, mAP@0.5:0.95, precision, recall        │
                        │  Confusion matrix per class                       │
                        │  Inference latency benchmark (target: <30ms)      │
                        └──────────────┬───────────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────────────────────────┐
                        │            LIVE WEBCAM DEMO                       │
                        │  webcam_demo.py                                   │
                        │  ├── Real-time inference at 30+ FPS               │
                        │  ├── Bounding box overlay                         │
                        │  ├── Predicted sign + confidence score            │
                        │  ├── Temporal smoothing (3-frame window)          │
                        │  └── Adjustable confidence threshold              │
                        └──────────────────────────────────────────────────┘
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# For annotation with auto hand detection:
pip install mediapipe

# 2. Collect data (will guide you through each sign)
python data_collection/capture_dataset.py --count 50 --output data/raw/

# 3. Auto-annotate with YOLO bounding boxes
python data_collection/auto_annotate.py --input data/raw/ --output data/

# 4. Train model
python training/train.py --model yolov8n --epochs 100

# 5. Evaluate
python training/evaluate.py --model runs/train/asl_detector/weights/best.pt

# 6. Live demo
python demo/webcam_demo.py --model runs/train/asl_detector/weights/best.pt
```

## Dataset

| Property | Value |
|---|---|
| Classes | 10 ASL signs (A, B, C, D, E, L, V, Y, thumbsup, thumbsdown) |
| Images per class | 50 (minimum 20 required) |
| Annotation format | YOLO (.txt per image) |
| Split | 80% train / 10% val / 10% test |
| Bounding box | Hand region detected via MediaPipe Hands |

**Why these 10 classes:**
- Selected to maximize visual distinctiveness (A vs E vs B are dissimilar)
- Avoids ambiguous pairs like J/Z (require motion, not static frames)
- Covers most common gestures including thumbsup/thumbsdown for usability

## Model Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | YOLOv8n | Single-stage, 2ms inference, 2.1M params |
| Pre-training | COCO weights | Transfers edge/texture features |
| Input size | 640×640 | Balance accuracy vs. latency |
| Optimizer | AdamW | Better than SGD for small datasets |
| Label smoothing | 0.1 | Prevents overconfidence on 50-image classes |
| Horizontal flip | 0.5 | Covers both left-hand and mirrored webcam |
| No vertical flip | 0.0 | ASL signs are orientation-specific |

## Expected Performance

| Metric | Expected Range |
|---|---|
| mAP@0.5 | 0.82–0.95 |
| Precision | 0.85–0.95 |
| Recall | 0.80–0.92 |
| Inference FPS (CPU) | 25–40 FPS |
| Inference FPS (GPU) | 60–100 FPS |

## Deployment Constraints

- **CPU deployment:** YOLOv8n runs at ~25 FPS on modern CPU, acceptable for demo
- **Model size:** ~6 MB — deployable on edge devices
- **ONNX export:** Available for cross-platform deployment (`python training/train.py --export`)
- **New signers:** Model may degrade; diversity in training data is critical
- **Lighting robustness:** HSV augmentation during training provides partial robustness

## Robustness Limitations & Improvements

1. **Signer independence:** Collect data from multiple signers for generalization
2. **Background variation:** More diverse backgrounds in training data
3. **Lighting:** Histogram equalization as preprocessing for low-light conditions
4. **Similar signs:** A/E and similar pairs may confuse the model — add more examples
5. **Occlusion:** Random erasing augmentation partially addresses this
