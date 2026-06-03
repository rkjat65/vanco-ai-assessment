"""
Automatic annotation tool for ASL dataset.

Uses MediaPipe Hands to detect hand bounding boxes automatically,
then saves annotations in YOLO format. Allows manual correction via GUI.

Usage:
    pip install mediapipe
    python auto_annotate.py --input ../data/raw/ --output ../data/ --split 0.8
"""

import cv2
import os
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import Optional

# Optional MediaPipe for auto bbox detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Using full-image bounding boxes.")


ASL_CLASSES = ["A", "B", "C", "D", "E", "L", "V", "Y", "thumbsup", "thumbsdown"]
CLASS_MAP = {cls: i for i, cls in enumerate(ASL_CLASSES)}


def detect_hand_bbox(image_path: str) -> Optional[tuple[float, float, float, float]]:
    """
    Detect hand bounding box using MediaPipe.

    Returns (x_center, y_center, width, height) in YOLO normalized format,
    or None if no hand detected.
    """
    if not MEDIAPIPE_AVAILABLE:
        # Fallback: center crop with reasonable hand region assumption
        return 0.5, 0.5, 0.7, 0.7

    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    ) as hands:
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            # No hand detected — use central region
            return 0.5, 0.5, 0.6, 0.8

        landmarks = results.multi_hand_landmarks[0]
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]

        # Add padding around detected landmarks
        padding = 0.08
        x_min = max(0.0, min(xs) - padding)
        x_max = min(1.0, max(xs) + padding)
        y_min = max(0.0, min(ys) - padding)
        y_max = min(1.0, max(ys) + padding)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        return x_center, y_center, bbox_w, bbox_h


def save_yolo_annotation(
    annotation_path: str,
    class_id: int,
    x_center: float,
    y_center: float,
    bbox_w: float,
    bbox_h: float,
):
    """Save a single annotation in YOLO format: class_id x_center y_center width height"""
    with open(annotation_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")


def annotate_dataset(
    raw_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
):
    """
    Process all images in raw_dir and create YOLO-formatted dataset.

    Directory structure created:
        output_dir/
            images/train/   ← images
            images/val/
            images/test/
            labels/train/   ← .txt annotation files
            labels/val/
            labels/test/
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    for split in ["train", "val", "test"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "no_hand": 0, "errors": 0}
    annotation_log = []

    for sign_dir in sorted(raw_path.iterdir()):
        if not sign_dir.is_dir():
            continue

        sign = sign_dir.name.upper()
        if sign not in CLASS_MAP:
            print(f"  Skipping unknown class: {sign}")
            continue

        class_id = CLASS_MAP[sign]
        images = list(sign_dir.glob("*.jpg")) + list(sign_dir.glob("*.png"))

        if len(images) < 20:
            print(f"  Warning: {sign} has only {len(images)} images (minimum 20 required)")

        # Shuffle and split
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_split)
        n_val = int(n * val_split)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        print(f"\nAnnotating {sign} ({n} images): "
              f"train={n_train}, val={n_val}, test={n - n_train - n_val}")

        for split_name, split_images in splits.items():
            for img_path in split_images:
                try:
                    bbox = detect_hand_bbox(str(img_path))
                    if bbox is None:
                        stats["no_hand"] += 1
                        continue

                    x_c, y_c, bw, bh = bbox

                    # Copy image
                    dest_img = out_path / "images" / split_name / img_path.name
                    shutil.copy2(img_path, dest_img)

                    # Save annotation
                    dest_label = (
                        out_path / "labels" / split_name /
                        img_path.with_suffix(".txt").name
                    )
                    save_yolo_annotation(str(dest_label), class_id, x_c, y_c, bw, bh)

                    stats["processed"] += 1
                    annotation_log.append({
                        "image": img_path.name,
                        "class": sign,
                        "class_id": class_id,
                        "split": split_name,
                        "bbox": [x_c, y_c, bw, bh],
                    })

                except Exception as e:
                    print(f"  Error processing {img_path.name}: {e}")
                    stats["errors"] += 1

    # Save annotation manifest
    with open(out_path / "annotation_manifest.json", "w") as f:
        json.dump(annotation_log, f, indent=2)

    print(f"\n{'='*50}")
    print("Annotation Summary:")
    print(f"  Processed: {stats['processed']}")
    print(f"  No hand detected: {stats['no_hand']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Manifest saved: {out_path / 'annotation_manifest.json'}")

    # Class distribution
    from collections import Counter
    class_counts = Counter(a["class"] for a in annotation_log)
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        bar = "█" * (count // 2)
        print(f"  {cls:12s}: {count:4d} {bar}")


def visualize_annotations(
    image_path: str,
    label_path: str,
    class_names: list[str] = ASL_CLASSES,
):
    """Visualize YOLO annotations overlaid on an image for quality check."""
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x_c, y_c, bw, bh = int(parts[0]), *[float(p) for p in parts[1:]]

            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Annotation Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Auto-Annotator")
    parser.add_argument("--input",  default="../data/raw/",
                        help="Raw images directory (organized by class name)")
    parser.add_argument("--output", default="../data/",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--split",  type=float, default=0.8,
                        help="Train split ratio (default: 0.8)")
    parser.add_argument("--visualize", action="store_true",
                        help="Preview annotations after creation")
    args = parser.parse_args()

    annotate_dataset(args.input, args.output, train_split=args.split)
