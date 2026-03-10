"""Live camera face recognition script.

Usage: python live.py
Press 'q' to quit.
"""

import cv2
import numpy as np
from models.face_engine import FaceEngine
from db.vector_store import FaceVectorStore
from pathlib import Path
from PIL import Image

FACES_DIR = Path("faces")


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


def main():
    print("Loading InsightFace model...")
    engine = FaceEngine()
    store = FaceVectorStore(data_dir="data")

    print("Loading reference faces...")
    load_reference_faces(engine, store)
    print(f"{len(store.list_faces())} faces in database")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera ready. Press 'q' to quit.")

    frame_count = 0
    cached_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect every 3 frames for performance
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

        # Draw results
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
