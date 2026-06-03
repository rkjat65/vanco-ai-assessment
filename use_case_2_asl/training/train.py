"""
YOLOv8 training pipeline for ASL hand sign detection.

Why YOLOv8:
- Single-stage detector: fast inference (suitable for real-time webcam)
- Pre-trained on COCO: strong low-level feature extraction
- Excellent small-object detection with the nano/small variants
- Native data augmentation (mosaic, mixup, HSV, flips)
- Export to ONNX/TensorRT for deployment optimization

Trade-offs vs. alternatives:
  Faster R-CNN: higher accuracy but too slow for real-time webcam (5-10 FPS)
  SSD: comparable speed but less accurate on small objects
  EfficientDet: good accuracy/speed but harder to fine-tune
  YOLOv8n: best speed/accuracy for real-time hand sign demo

Usage:
    python train.py --model yolov8n --epochs 100 --imgsz 640
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)


AUGMENTATION_CONFIG = {
    # Brightness/contrast changes — simulates different lighting conditions
    "hsv_h": 0.015,      # hue jitter
    "hsv_s": 0.7,        # saturation jitter
    "hsv_v": 0.4,        # value (brightness) jitter
    # Geometric augmentations
    "degrees": 10.0,     # rotation (hand signs can tilt slightly)
    "translate": 0.1,
    "scale": 0.5,        # scaling (different distances from camera)
    "shear": 2.0,
    "perspective": 0.0001,
    # Flips
    "flipud": 0.0,       # no vertical flip (ASL signs are orientation-specific)
    "fliplr": 0.5,       # horizontal flip (mirror self vs. facing)
    # Advanced
    "mosaic": 1.0,       # combines 4 images — improves generalization
    "mixup": 0.1,        # blends 2 images
    "copy_paste": 0.0,
    # Blur simulates out-of-focus webcam
    "blur": 0.01,
    "erasing": 0.4,      # random erase simulates partial occlusion
}


def validate_dataset(data_yaml: str) -> bool:
    """Check that the dataset has the minimum required images per class."""
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config.get("path", "."))
    train_img_dir = dataset_path / config.get("train", "images/train")

    if not train_img_dir.exists():
        print(f"ERROR: Training image directory not found: {train_img_dir}")
        return False

    images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    n_classes = config.get("nc", 0)

    print(f"Dataset validation:")
    print(f"  Training images: {len(images)}")
    print(f"  Classes: {n_classes}")
    print(f"  Expected min: {n_classes * 20} images ({n_classes} classes × 20 each)")

    return len(images) >= n_classes * 20


def train_model(
    data_yaml: str = "configs/dataset.yaml",
    model_size: str = "yolov8n",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    project: str = "runs/train",
    name: str = "asl_detector",
    resume: bool = False,
):
    """
    Train YOLOv8 on the custom ASL dataset.

    Model size options:
    - yolov8n (nano)   2.1M params  ~2ms inference    ← recommended for webcam demo
    - yolov8s (small)  11M params   ~3ms inference    ← better accuracy
    - yolov8m (medium) 25M params   ~5ms inference    ← if GPU available
    """
    print(f"\n{'='*60}")
    print("YOLOv8 ASL Detection Training")
    print(f"  Model:     {model_size}.pt")
    print(f"  Epochs:    {epochs}")
    print(f"  Image size:{imgsz}px")
    print(f"  Batch:     {batch_size}")
    print(f"  Device:    {device}")
    print(f"{'='*60}\n")

    # Validate dataset
    if not resume and not validate_dataset(data_yaml):
        print("\nWARNING: Dataset does not meet minimum requirements.")
        print("Proceeding anyway — collect more data for better accuracy.\n")

    # Load pre-trained YOLOv8
    # Pre-trained on COCO gives strong general feature extraction
    # Fine-tuning on ASL-specific data adapts the head to our classes
    model = YOLO(f"{model_size}.pt")

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        resume=resume,
        patience=20,           # early stopping: stop if no improvement for 20 epochs
        save=True,
        save_period=10,        # save checkpoint every 10 epochs
        val=True,
        verbose=True,
        # Augmentation parameters
        **AUGMENTATION_CONFIG,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Regularization
        label_smoothing=0.1,   # reduces overconfidence on small dataset
        # Anchors
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    return results


def evaluate_model(model_path: str, data_yaml: str):
    """Run evaluation: mAP@0.5, mAP@0.5:0.95, precision, recall per class."""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, verbose=True)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")

    return metrics


def export_model(model_path: str, export_format: str = "onnx"):
    """Export trained model for deployment."""
    model = YOLO(model_path)
    model.export(format=export_format, dynamic=True, simplify=True)
    print(f"Model exported to {export_format} format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for ASL Detection")
    parser.add_argument("--data",    default="configs/dataset.yaml")
    parser.add_argument("--model",   default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m"])
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  default="auto")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--eval",    action="store_true", help="Evaluate best model after training")
    parser.add_argument("--export",  action="store_true", help="Export to ONNX after training")
    args = parser.parse_args()

    results = train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        resume=args.resume,
    )

    best_model = "runs/train/asl_detector/weights/best.pt"

    if args.eval and os.path.exists(best_model):
        evaluate_model(best_model, args.data)

    if args.export and os.path.exists(best_model):
        export_model(best_model, "onnx")
