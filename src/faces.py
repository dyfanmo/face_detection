import numpy as np
from deepface import DeepFace

from src.config import DETECTOR, MIN_DETECTION_CONFIDENCE, MIN_FACE_SIZE, PADDING


def detect_faces(frame: np.ndarray) -> list:
    """Detects all faces in a frame using RetinaFace. Returns empty list if no faces found."""
    results = DeepFace.extract_faces(
        img_path=frame,
        detector_backend=DETECTOR,
        enforce_detection=False,
    )

    faces = []
    for result in results:
        region = result.get("facial_area")
        if region is None:
            continue
        faces.append(
            {
                "x": region["x"],
                "y": region["y"],
                "w": region["w"],
                "h": region["h"],
                "confidence": result.get("confidence", 0.0),
            }
        )
    return faces


def filter_faces(faces: list) -> list:
    """Filters faces by minimum confidence and minimum size. Returns only faces that pass both checks."""
    return [
        f for f in faces
        if f["confidence"] >= MIN_DETECTION_CONFIDENCE
        and is_face_large_enough(f)
    ]


def crop_face(frame: np.ndarray, bbox: dict, padding: int = PADDING) -> np.ndarray | None:
    """Crops a face region from a frame using a bounding box dict with x, y, w, h keys. Applies padding.
    Returns None if the resulting crop has no pixels."""
    height, width = frame.shape[:2]

    x = max(0, bbox["x"] - padding)
    y = max(0, bbox["y"] - padding)
    crop_x_end = min(width, bbox["x"] + bbox["w"] + padding)
    crop_y_end = min(height, bbox["y"] + bbox["h"] + padding)

    crop = frame[y:crop_y_end, x:crop_x_end]
    if crop.size == 0:
        return None
    return crop


def get_largest_face(faces: list) -> dict | None:
    """Returns the face with the largest bounding box area. Returns None if the list is empty."""
    if not faces:
        return None
    return max(faces, key=lambda f: f["w"] * f["h"])


def is_face_large_enough(face: dict) -> bool:
    """Returns True if the face bounding box meets the minimum size requirement."""
    return face["w"] >= MIN_FACE_SIZE and face["h"] >= MIN_FACE_SIZE
