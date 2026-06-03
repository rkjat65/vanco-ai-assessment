"""
Live webcam demo for ASL hand sign detection.

Displays real-time bounding boxes, predicted sign labels, and confidence scores.
Designed to meet the live demo requirements for the Vanco assessment.

Usage:
    python webcam_demo.py --model ../runs/train/asl_detector/weights/best.pt

Controls:
    Q / ESC  - quit
    S        - save current frame as screenshot
    +/-      - increase/decrease confidence threshold
    P        - pause/resume
"""

import cv2
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    sys.exit(1)

ASL_CLASSES = ["A", "B", "C", "D", "E", "L", "V", "Y", "thumbsup", "thumbsdown"]

# Color palette per class for distinct bounding box colors
CLASS_COLORS = [
    (255, 80,  80),   # A - red
    (255, 160, 0),    # B - orange
    (255, 255, 0),    # C - yellow
    (0,   200, 0),    # D - green
    (0,   200, 200),  # E - cyan
    (0,   100, 255),  # L - blue
    (140, 0,   255),  # V - purple
    (255, 0,   200),  # Y - pink
    (255, 140, 60),   # thumbsup
    (80,  80,  255),  # thumbsdown
]


class ASLDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.frame_times = []
        self.detection_history = []  # for temporal smoothing

    def predict(self, frame: np.ndarray) -> list[dict]:
        """Run inference and return list of detections."""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": cls_id,
                    "class_name": ASL_CLASSES[cls_id] if cls_id < len(ASL_CLASSES) else "unknown",
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        return detections

    def smooth_prediction(self, detections: list[dict], window: int = 3) -> str:
        """
        Temporal smoothing: return the most common prediction over last N frames.
        Reduces flickering between similar signs.
        """
        if detections:
            self.detection_history.append(detections[0]["class_name"])
        else:
            self.detection_history.append(None)

        if len(self.detection_history) > window:
            self.detection_history.pop(0)

        non_null = [h for h in self.detection_history if h is not None]
        if not non_null:
            return ""

        from collections import Counter
        return Counter(non_null).most_common(1)[0][0]


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    smoothed_pred: str,
    fps: float,
    conf_threshold: float,
    paused: bool,
) -> np.ndarray:
    """Draw bounding boxes, labels, confidence, and HUD on the frame."""
    h, w = frame.shape[:2]

    # Draw each detection
    for det in detections:
        cls_id = det["class_id"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Label background
        label = f"{det['class_name']}  {conf:.1%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 14), (x1 + lw + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Confidence bar below bbox
        bar_len = int((x2 - x1) * conf)
        cv2.rectangle(frame, (x1, y2 + 4), (x2, y2 + 12), (60, 60, 60), -1)
        cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_len, y2 + 12), color, -1)

    # ── HUD overlay ────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # FPS
    fps_color = (0, 200, 0) if fps >= 25 else (0, 150, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

    # Confidence threshold
    cv2.putText(frame, f"Conf: {conf_threshold:.2f}  [+/-]", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Smoothed prediction (large, prominent)
    if smoothed_pred:
        cv2.putText(frame, f"Sign: {smoothed_pred}", (w // 2 - 80, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 100), 3)

    # Controls hint
    hint = "Q=quit  S=screenshot  P=pause"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, hint, (w - hw - 10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 70, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    return frame


def run_demo(
    model_path: str,
    conf_threshold: float = 0.4,
    camera_id: int = 0,
    output_dir: str = "demo_screenshots/",
):
    """Main demo loop."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {model_path}")
    detector = ASLDetector(model_path, conf_threshold)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    print("Webcam demo started. Controls: Q=quit  S=screenshot  P=pause  +/-=threshold")

    paused = False
    frame_count = 0
    fps = 0.0
    t_prev = time.time()
    smoothed = ""

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error")
                break
            frame = cv2.flip(frame, 1)

        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ────────────────────────────────────────────────────
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break

        if key == ord("s") or key == ord("S"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(output_dir) / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"  Screenshot saved: {path}")

        if key == ord("p") or key == ord("P"):
            paused = not paused

        if key == ord("+") or key == ord("="):
            detector.conf_threshold = min(0.95, detector.conf_threshold + 0.05)
            print(f"  Confidence threshold: {detector.conf_threshold:.2f}")

        if key == ord("-"):
            detector.conf_threshold = max(0.05, detector.conf_threshold - 0.05)
            print(f"  Confidence threshold: {detector.conf_threshold:.2f}")

        if paused:
            cv2.imshow("ASL Detection Demo", frame)
            continue

        # ── Inference ───────────────────────────────────────────────────────
        detections = detector.predict(frame)
        smoothed = detector.smooth_prediction(detections)

        # ── FPS calculation ─────────────────────────────────────────────────
        frame_count += 1
        t_now = time.time()
        if t_now - t_prev >= 0.5:
            fps = frame_count / (t_now - t_prev)
            frame_count = 0
            t_prev = t_now

        # ── Draw and display ─────────────────────────────────────────────────
        display = draw_detections(
            frame.copy(), detections, smoothed, fps,
            detector.conf_threshold, paused
        )
        cv2.imshow("ASL Detection Demo — Vanco Assessment", display)

    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Live Webcam Demo")
    parser.add_argument("--model", default="runs/train/asl_detector/weights/best.pt",
                        help="Path to trained YOLOv8 model")
    parser.add_argument("--conf",  type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID (default: 0)")
    parser.add_argument("--output", default="demo_screenshots/",
                        help="Directory for saved screenshots")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        print("Train first: python training/train.py")
        sys.exit(1)

    run_demo(args.model, args.conf, args.camera, args.output)
