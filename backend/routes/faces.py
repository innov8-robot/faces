import io

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

router = APIRouter(prefix="/api/faces", tags=["faces"])


def _load_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(image)


@router.post("/register")
async def register_face(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
):
    engine = request.app.state.face_engine
    store = request.app.state.vector_store

    image = _load_image(await file.read())
    faces = engine.detect_faces(image)

    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    face = faces[0]
    face_id = store.add(name, face["embedding"])

    return {
        "id": face_id,
        "name": name,
        "message": f"Face registered for {name}",
        "faces_detected": len(faces),
    }


@router.post("/register-multiple")
async def register_multiple_faces(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
):
    engine = request.app.state.face_engine
    store = request.app.state.vector_store

    image = _load_image(await file.read())
    faces = engine.detect_faces(image)

    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    ids = []
    for face in faces:
        face_id = store.add(name, face["embedding"])
        ids.append(face_id)

    return {
        "ids": ids,
        "name": name,
        "count": len(ids),
        "message": f"Registered {len(ids)} face(s) for {name}",
    }


@router.post("/recognize")
async def recognize_faces(
    request: Request,
    file: UploadFile = File(...),
):
    engine = request.app.state.face_engine
    store = request.app.state.vector_store

    image = _load_image(await file.read())
    faces = engine.detect_faces(image)

    results = []
    for face in faces:
        matches = store.search(face["embedding"])
        best = matches[0] if matches else {"name": "unknown", "confidence": 0.0}
        results.append({
            "name": best["name"],
            "confidence": round(best["confidence"], 4),
            "bbox": face["bbox"],
        })

    return {"faces": results, "count": len(results)}


@router.get("/")
async def list_faces(request: Request):
    store = request.app.state.vector_store
    return {"faces": store.list_faces()}


@router.delete("/{face_id}")
async def delete_face(request: Request, face_id: str):
    store = request.app.state.vector_store
    success = store.delete(face_id)
    if not success:
        raise HTTPException(status_code=404, detail="Face not found")
    return {"message": "Face deleted", "id": face_id}
