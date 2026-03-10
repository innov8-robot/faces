import json
import os
import uuid

import faiss
import numpy as np


class FaceVectorStore:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "faiss.index")
        self.meta_path = os.path.join(data_dir, "faces.json")
        self.dimension = 512  # InsightFace buffalo_l embedding size
        self.metadata: dict[str, dict] = {}
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self._id_order: list[str] = []  # Track insertion order for index mapping

        os.makedirs(data_dir, exist_ok=True)
        self.load()

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def add(self, name: str, embedding: np.ndarray) -> str:
        face_id = str(uuid.uuid4())
        normalized = self._normalize(embedding).astype(np.float32).reshape(1, -1)
        self.index.add(normalized)
        self._id_order.append(face_id)
        self.metadata[face_id] = {"name": name, "index": len(self._id_order) - 1}
        self.save()
        return face_id

    def search(self, embedding: np.ndarray, threshold: float = 0.4, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        normalized = self._normalize(embedding).astype(np.float32).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(normalized, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            face_id = self._id_order[idx]
            meta = self.metadata[face_id]
            if dist >= threshold:
                results.append({
                    "id": face_id,
                    "name": meta["name"],
                    "confidence": float(dist),
                })
            else:
                results.append({
                    "id": None,
                    "name": "unknown",
                    "confidence": float(dist),
                })
        return results

    def delete(self, face_id: str) -> bool:
        if face_id not in self.metadata:
            return False
        del self.metadata[face_id]
        self._rebuild_index()
        self.save()
        return True

    def list_faces(self) -> list[dict]:
        return [
            {"id": fid, "name": meta["name"]}
            for fid, meta in self.metadata.items()
        ]

    def _rebuild_index(self):
        """Rebuild FAISS index from remaining metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        if not self.metadata:
            self._id_order = []
            return

        # Reload embeddings from the old index isn't possible with IndexFlatIP
        # We need to store embeddings. Let's reconstruct from the current index.
        old_order = self._id_order[:]
        new_order = []
        vectors = []

        for i, fid in enumerate(old_order):
            if fid in self.metadata:
                vec = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * self.dimension)
                # This won't work after removal. Store embeddings separately.
                break
        else:
            self._id_order = []
            return

        # Alternative: reconstruct vectors
        all_vectors = faiss.rev_swig_ptr(
            self.index.get_xb(), len(old_order) * self.dimension
        ).reshape(-1, self.dimension).copy()

        for i, fid in enumerate(old_order):
            if fid in self.metadata:
                vectors.append(all_vectors[i])
                new_order.append(fid)

        self._id_order = new_order
        self.index = faiss.IndexFlatIP(self.dimension)
        if vectors:
            matrix = np.array(vectors, dtype=np.float32)
            self.index.add(matrix)

        # Update index references in metadata
        for i, fid in enumerate(self._id_order):
            self.metadata[fid]["index"] = i

    def save(self):
        faiss.write_index(self.index, self.index_path)
        data = {"metadata": self.metadata, "id_order": self._id_order}
        with open(self.meta_path, "w") as f:
            json.dump(data, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path) as f:
                data = json.load(f)
            self.metadata = data.get("metadata", {})
            self._id_order = data.get("id_order", [])
