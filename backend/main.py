import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from db.vector_store import FaceVectorStore
from models.face_engine import FaceEngine
from routes.faces import router as faces_router
from routes.stream import router as stream_router

FACES_DIR = Path("faces")


def _load_reference_faces(engine: FaceEngine, store: FaceVectorStore):
    """Auto-register faces from the faces/ directory using filename as name."""
    if not FACES_DIR.exists():
        return

    existing_names = {f["name"] for f in store.list_faces()}

    for img_path in sorted(FACES_DIR.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        name = img_path.stem
        if name in existing_names:
            print(f"  Skipping {name} (already registered)")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        faces = engine.detect_faces(image)
        if not faces:
            print(f"  No face found in {img_path.name}")
            continue

        store.add(name, faces[0]["embedding"])
        print(f"  Registered {name} from {img_path.name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading InsightFace model...")
    app.state.face_engine = FaceEngine()
    app.state.vector_store = FaceVectorStore(data_dir="data")
    print("Loading reference faces from faces/...")
    _load_reference_faces(app.state.face_engine, app.state.vector_store)
    print("Ready!")
    yield
    # Shutdown
    app.state.vector_store.save()


app = FastAPI(title="Face Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(faces_router)
app.include_router(stream_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
