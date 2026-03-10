import asyncio
import platform
import threading
import time

import cv2
import numpy as np
from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/stream", tags=["stream"])

_lock = threading.Lock()
_streams: dict[str, dict] = {}

# Auto-detect: on Linux (Jetson) use V4L2 + RealSense config
IS_LINUX = platform.system() == "Linux"


def _open_camera(source: str) -> cv2.VideoCapture:
    """Open camera source. On Linux/Jetson: V4L2 + RealSense D435i config."""
    index = int(source)

    if IS_LINUX:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        # macOS / Windows
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap


def _find_realsense_index() -> int | None:
    """Auto-detect RealSense RGB stream index on Linux."""
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
            return index
    return None


def _recognize_loop(source: str, engine, store):
    """Background thread: read frames, run recognition, store latest result."""
    state = _streams[source]

    if source == "auto":
        # Auto-detect RealSense on Jetson
        idx = _find_realsense_index()
        if idx is None:
            print("No RealSense RGB stream found")
            state["running"] = False
            return
        print(f"RealSense RGB found on /dev/video{idx}")
        cap = _open_camera(str(idx))
    else:
        cap = _open_camera(source)

    if not cap.isOpened():
        print(f"Cannot open camera source: {source}")
        state["running"] = False
        return

    state["cap"] = cap
    frame_count = 0
    cached_results = []

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
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

        # Draw annotations
        annotated = frame.copy()
        for res in cached_results:
            x1, y1, x2, y2 = res["bbox"]
            name = res["name"]
            conf = res["confidence"]
            is_known = name != "inconnu" and name != "unknown"

            color = (0, 200, 0) if is_known else (0, 0, 220)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{name} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        with _lock:
            state["frame"] = annotated
            state["results"] = cached_results

        frame_count += 1

    cap.release()
    with _lock:
        state["running"] = False


def _ensure_stream(source: str, engine, store):
    """Start the capture thread if not already running."""
    with _lock:
        if source in _streams and _streams[source]["running"]:
            return
        _streams[source] = {
            "cap": None,
            "thread": None,
            "frame": None,
            "results": [],
            "running": True,
        }

    t = threading.Thread(target=_recognize_loop, args=(source, engine, store), daemon=True)
    _streams[source]["thread"] = t
    t.start()


@router.get("/mjpeg")
async def mjpeg_stream(
    request: Request,
    source: str = Query(default="auto", description="Camera index (0-5) or 'auto' for RealSense auto-detect"),
):
    """MJPEG stream with face recognition overlays."""
    engine = request.app.state.face_engine
    store = request.app.state.vector_store

    _ensure_stream(source, engine, store)

    async def generate():
        while True:
            with _lock:
                state = _streams.get(source)
                if not state or not state["running"]:
                    break
                frame = state["frame"]

            if frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )

            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/faces")
async def stream_faces(
    request: Request,
    source: str = Query(default="auto"),
):
    """Get current detected faces from the active stream."""
    with _lock:
        state = _streams.get(source)
        if not state or not state["running"]:
            return {"faces": [], "active": False}
        return {"faces": state["results"], "active": True}


@router.post("/stop")
async def stop_stream(
    source: str = Query(default="auto"),
):
    """Stop an active stream."""
    with _lock:
        state = _streams.get(source)
        if state:
            state["running"] = False
    return {"message": "Stream stopped"}


@router.get("/detect-cameras")
async def detect_cameras():
    """List available camera indices (Linux/V4L2 only)."""
    cameras = []
    for index in range(6):
        if IS_LINUX:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            is_color = ret and frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3
            cameras.append({"index": index, "color": is_color})
            cap.release()
    return {"cameras": cameras}
