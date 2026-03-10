import io
import json

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

router = APIRouter()


@router.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    """WebSocket for real-time face recognition.

    Client sends: binary JPEG frame
    Server responds: JSON with detected faces
    """
    await websocket.accept()

    engine = websocket.app.state.face_engine
    store = websocket.app.state.vector_store

    try:
        while True:
            # Receive binary JPEG data
            data = await websocket.receive_bytes()

            # Decode image
            image = np.array(Image.open(io.BytesIO(data)).convert("RGB"))

            # Detect and recognize
            faces = engine.detect_faces(image)
            results = []
            for face in faces:
                matches = store.search(face["embedding"])
                best = matches[0] if matches else {"name": "unknown", "confidence": 0.0}
                results.append({
                    "name": best["name"],
                    "confidence": round(best["confidence"], 4),
                    "bbox": [int(v) for v in face["bbox"]],
                })

            await websocket.send_text(json.dumps({"faces": results}))

    except WebSocketDisconnect:
        pass
