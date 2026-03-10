import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        faces = self.app.get(image)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            results.append({
                "bbox": bbox,
                "embedding": face.normed_embedding,
                "det_score": float(face.det_score),
            })
        return results

    def get_embeddings(self, image: np.ndarray) -> list[np.ndarray]:
        faces = self.app.get(image)
        return [face.normed_embedding for face in faces]
