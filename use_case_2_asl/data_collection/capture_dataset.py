"""
Custom ASL dataset collection tool using webcam.

Usage:
    python capture_dataset.py --sign A --count 50 --output ../data/raw/A/

Controls:
    SPACE  - capture current frame
    Q      - quit current sign collection
    R      - reset (discard last frame)
"""

import cv2
import os
import argparse
import time
from pathlib import Path
from datetime import datetime


ASL_CLASSES = ["A", "B", "C", "D", "E", "L", "V", "Y", "thumbsup", "thumbsdown"]

# Collection tips displayed on screen for each sign
SIGN_TIPS = {
    "A": "Fist with thumb to the side",
    "B": "Four fingers up, thumb across palm",
    "C": "Curved hand like letter C",
    "D": "Index up, other fingers curved",
    "E": "Fingers bent down, thumb under",
    "L": "L-shape: index up, thumb out",
    "V": "Peace sign: index + middle up",
    "Y": "Thumb and pinky out",
    "thumbsup": "Thumbs up gesture",
    "thumbsdown": "Thumbs down gesture",
}


def show_overlay(frame, sign: str, captured: int, target: int, tip: str):
    """Draw instructional overlay on the webcam frame."""
    h, w = frame.shape[:2]

    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Sign: {sign} ({tip})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Captured: {captured}/{target} | SPACE=capture  Q=quit",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Progress bar
    progress = int((captured / max(target, 1)) * (w - 20))
    cv2.rectangle(frame, (10, 80), (w - 10, 95), (100, 100, 100), -1)
    cv2.rectangle(frame, (10, 80), (10 + progress, 95), (0, 200, 0), -1)

    return frame


def capture_sign(sign: str, target_count: int, output_dir: str, preview_delay: float = 0.1):
    """
    Interactively capture images for a single ASL sign.

    Captures are triggered manually (SPACE) to ensure quality control.
    Auto-capture mode is available with --auto flag for faster collection.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = len(list(output_path.glob("*.jpg")))
    captured = existing

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera connection.")

    tip = SIGN_TIPS.get(sign, "")
    print(f"\nCollecting sign: {sign}")
    print(f"  Tip: {tip}")
    print(f"  Target: {target_count} images | Existing: {existing}")
    print("  Press SPACE to capture, Q to quit\n")

    last_capture_time = 0

    while captured < target_count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Mirror for natural webcam feel
        frame = cv2.flip(frame, 1)

        display = show_overlay(frame.copy(), sign, captured, target_count, tip)
        cv2.imshow(f"ASL Capture: {sign}", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            print(f"  Stopped early at {captured} images")
            break

        if key == ord(" "):
            # Minimum interval between captures to ensure diversity
            now = time.time()
            if now - last_capture_time < preview_delay:
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = output_path / f"{sign}_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            captured += 1
            last_capture_time = now
            print(f"  Saved [{captured}/{target_count}]: {filename.name}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nCollection complete: {captured} images for '{sign}' in {output_path}")
    return captured


def capture_all_signs(output_base: str, count_per_sign: int = 50):
    """Sequentially collect images for all ASL classes."""
    summary = {}
    for sign in ASL_CLASSES:
        print(f"\n{'='*50}")
        print(f"Next: Sign '{sign}' — {SIGN_TIPS.get(sign, '')}")
        input("  Press ENTER when ready...")

        n = capture_sign(
            sign=sign,
            target_count=count_per_sign,
            output_dir=os.path.join(output_base, sign),
        )
        summary[sign] = n

    print("\n\n===== Collection Summary =====")
    for sign, n in summary.items():
        status = "OK" if n >= 20 else "INSUFFICIENT (<20)"
        print(f"  {sign}: {n} images  [{status}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Dataset Collector")
    parser.add_argument("--sign", type=str, default=None,
                        help="Single sign to capture (e.g., A). Omit to collect all.")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of images to capture per sign")
    parser.add_argument("--output", type=str, default="../data/raw/",
                        help="Output base directory")
    args = parser.parse_args()

    if args.sign:
        capture_sign(args.sign.upper(), args.count,
                     os.path.join(args.output, args.sign.upper()))
    else:
        capture_all_signs(args.output, args.count)
