"""Live camera face recognition script.

Usage:
  python live.py                # Webcam locale (index 0)
  python live.py --source auto  # Auto-detect RealSense D435i sur Jetson
  python live.py --source 2     # Camera index specifique
Press 'q' to quit.
"""

import argparse
import platform

import cv2
import numpy as np
from models.face_engine import FaceEngine
from db.vector_store import FaceVectorStore
from pathlib import Path
from PIL import Image

FACES_DIR = Path("faces")
IS_LINUX = platform.system() == "Linux"


def load_reference_faces(engine: FaceEngine, store: FaceVectorStore):
    if not FACES_DIR.exists():
        return
    existing_names = {f["name"] for f in store.list_faces()}
    for img_path in sorted(FACES_DIR.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        name = img_path.stem
        if name in existing_names:
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        faces = engine.detect_faces(image)
        if faces:
            store.add(name, faces[0]["embedding"])
            print(f"  Registered {name}")


def find_realsense_index() -> int | None:
    """Auto-detect RealSense RGB stream on /dev/video0-5."""
    for index in range(6):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            print(f"  RealSense RGB found on /dev/video{index}")
            return index
    return None


def open_camera(source: str) -> cv2.VideoCapture:
    """Open camera by index. On Linux use V4L2 + RealSense config."""
    index = int(source)
    if IS_LINUX:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def main():
    parser = argparse.ArgumentParser(description="Live face recognition")
    parser.add_argument(
        "--source", "-s", default="0",
        help="Camera index (0), 'auto' for RealSense auto-detect",
    )
    args = parser.parse_args()

    print("Loading InsightFace model...")
    engine = FaceEngine()
    store = FaceVectorStore(data_dir="data")

    print("Loading reference faces...")
    load_reference_faces(engine, store)
    print(f"{len(store.list_faces())} faces in database")

    source = args.source
    if source == "auto":
        print("Auto-detecting RealSense...")
        idx = find_realsense_index()
        if idx is None:
            print("No RealSense RGB stream found on /dev/video0-5")
            return
        source = str(idx)

    print(f"Opening camera index {source}...")
    cap = open_camera(source)
    print("Camera ready. Press 'q' to quit.")

    frame_count = 0
    cached_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_count % 3 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = engine.detect_faces(rgb)
            cached_results = []
            for face in faces:
                matches = store.search(face["embedding"])
                best = matches[0] if matches else {"name": "inconnu", "confidence": 0.0}
                bbox = [int(v) for v in face["bbox"]]
                cached_results.append({
                    "name": best["name"],
                    "confidence": best["confidence"],
                    "bbox": bbox,
                })

        for res in cached_results:
            x1, y1, x2, y2 = res["bbox"]
            name = res["name"]
            conf = res["confidence"]
            is_known = name != "inconnu" and name != "unknown"

            color = (0, 200, 0) if is_known else (0, 0, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
