import logging

import numpy as np
from deepface import DeepFace

from src.config import DETECTOR, MIN_DETECTION_CONFIDENCE, MIN_FACE_SIZE, PADDING
from src.exceptions import CropFailedError, NoFacesDetectedError
from src.models import FaceBBox

logger = logging.getLogger(__name__)


def detect_faces(frame_image: np.ndarray) -> list[FaceBBox]:
    """Detects all faces in a frame using RetinaFace. Returns empty list if no faces found."""
    results = DeepFace.extract_faces(
        img_path=frame_image,
        detector_backend=DETECTOR,
        enforce_detection=False,
    )

    faces = []
    for result in results:
        region = result.get("facial_area")
        if region is None:
            continue
        faces.append(
            FaceBBox(
                x=region["x"],
                y=region["y"],
                w=region["w"],
                h=region["h"],
                confidence=result.get("confidence", 0.0),
            )
        )
    return faces


def filter_faces(faces: list[FaceBBox]) -> list[FaceBBox]:
    """Filters faces by minimum confidence and minimum size. Returns only faces that pass both checks."""
    return [f for f in faces if f.confidence >= MIN_DETECTION_CONFIDENCE and is_face_large_enough(f)]


def crop_face(frame_image: np.ndarray, bbox: FaceBBox, padding: int = PADDING) -> np.ndarray:
    """Crops a face region from a frame using a FaceBBox. Applies padding.
    Raises CropFailedError if the resulting crop has no pixels."""
    height, width = frame_image.shape[:2]

    x = max(0, bbox.x - padding)
    y = max(0, bbox.y - padding)
    crop_x_end = min(width, bbox.x + bbox.w + padding)
    crop_y_end = min(height, bbox.y + bbox.h + padding)

    crop = frame_image[y:crop_y_end, x:crop_x_end]
    if crop.size == 0:
        raise CropFailedError(f"Crop at ({bbox.x}, {bbox.y}) produced an empty array")
    return crop


def get_largest_face(faces: list[FaceBBox]) -> FaceBBox | None:
    """Returns the face with the largest bounding box area. Returns None if the list is empty."""
    if not faces:
        return None
    return max(faces, key=lambda f: f.w * f.h)


def is_face_large_enough(face: FaceBBox) -> bool:
    """Returns True if the face bounding box meets the minimum size requirement."""
    return face.w >= MIN_FACE_SIZE and face.h >= MIN_FACE_SIZE


def extract_dominant_face_crop(frame_image: np.ndarray) -> np.ndarray:
    """Detects faces, selects the largest, and returns a cropped numpy array.
    Raises NoFacesDetectedError if no faces are detected or the crop fails."""
    detected_faces = detect_faces(frame_image)

    if not detected_faces:
        raise NoFacesDetectedError("no face detected")

    largest_face = get_largest_face(detected_faces)

    if largest_face is None:
        raise NoFacesDetectedError("could not determine largest face")

    try:
        return crop_face(frame_image, largest_face)
    except CropFailedError as e:
        raise NoFacesDetectedError(f"could not crop face region: {e}") from e
