import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import MIN_FACE_SIZE
from src.faces import crop_face, filter_faces, get_largest_face, is_face_large_enough


def make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_get_largest_face_returns_biggest():
    """Returns the face with the largest bounding box area."""
    faces = [
        {"x": 0, "y": 0, "w": 50, "h": 50, "confidence": 0.9},
        {"x": 100, "y": 100, "w": 100, "h": 100, "confidence": 0.8},
        {"x": 200, "y": 200, "w": 30, "h": 30, "confidence": 0.7},
    ]
    assert get_largest_face(faces)["w"] == 100


def test_get_largest_face_returns_none_on_empty_list():
    """Returns None when no faces are provided."""
    assert get_largest_face([]) is None


def test_crop_face_returns_correct_shape():
    """Cropped region matches the expected dimensions given bbox and no padding."""
    crop = crop_face(make_frame(), {"x": 100, "y": 100, "w": 100, "h": 100}, padding=0)
    assert crop.shape[:2] == (100, 100)


def test_crop_face_returns_none_on_empty_crop():
    """Returns None when the resulting crop has no pixels."""
    assert crop_face(make_frame(), {"x": 640, "y": 480, "w": 0, "h": 0}, padding=0) is None


def test_is_face_large_enough_returns_true():
    """Returns True when face meets minimum size requirement."""
    assert is_face_large_enough({"x": 0, "y": 0, "w": MIN_FACE_SIZE, "h": MIN_FACE_SIZE, "confidence": 0.9}) is True


def test_is_face_large_enough_returns_false_when_too_small():
    """Returns False when face is below minimum size requirement."""
    assert (
        is_face_large_enough({"x": 0, "y": 0, "w": MIN_FACE_SIZE - 1, "h": MIN_FACE_SIZE - 1, "confidence": 0.9})
        is False
    )


def test_filter_faces_removes_low_confidence():
    """Removes faces below the minimum confidence threshold."""
    faces = [
        {"x": 0, "y": 0, "w": 100, "h": 100, "confidence": 0.95},
        {"x": 0, "y": 0, "w": 100, "h": 100, "confidence": 0.5},
    ]
    assert len(filter_faces(faces)) == 1
