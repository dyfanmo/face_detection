import numpy as np
import pytest

from src.config import FACENET_INPUT_SIZE, MIN_FACE_SIZE
from src.data_models import FaceBBox
from src.exceptions import CropFailedError
from src.faces import crop_face, filter_faces, get_largest_face, is_face_large_enough, normalise_crop_size


def make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_bbox(x: int = 0, y: int = 0, w: int = 100, h: int = 100, confidence: float = 0.95) -> FaceBBox:
    return FaceBBox(x=x, y=y, w=w, h=h, confidence=confidence)


def test_get_largest_face_returns_biggest():
    """Returns the face with the largest bounding box area."""
    faces = [make_bbox(w=50, h=50), make_bbox(w=100, h=100), make_bbox(w=30, h=30)]
    assert get_largest_face(faces).w == 100


def test_get_largest_face_returns_none_on_empty_list():
    """Returns None when no faces are provided."""
    assert get_largest_face([]) is None


def test_crop_face_returns_correct_shape():
    """Cropped region matches the expected dimensions given bbox and no padding."""
    crop = crop_face(make_frame(), make_bbox(x=100, y=100, w=100, h=100), padding=0)
    assert crop.shape[:2] == (100, 100)


def test_crop_face_raises_on_empty_crop():
    """Raises CropFailedError when the resulting crop has no pixels."""
    with pytest.raises(CropFailedError):
        crop_face(make_frame(), make_bbox(x=640, y=480, w=0, h=0), padding=0)


def test_is_face_large_enough_returns_true():
    """Returns True when face meets minimum size requirement."""
    assert is_face_large_enough(make_bbox(w=MIN_FACE_SIZE, h=MIN_FACE_SIZE)) is True


def test_is_face_large_enough_returns_false_when_too_small():
    """Returns False when face is below minimum size requirement."""
    assert is_face_large_enough(make_bbox(w=MIN_FACE_SIZE - 1, h=MIN_FACE_SIZE - 1)) is False


def test_filter_faces_removes_low_confidence():
    """Removes faces below the minimum confidence threshold."""
    faces = [make_bbox(confidence=0.95), make_bbox(confidence=0.5)]
    assert len(filter_faces(faces)) == 1


def test_normalise_crop_size_upscales_small_image():
    """Resizes a crop smaller than FACENET_INPUT_SIZE to at least the minimum size."""
    small_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    result = normalise_crop_size(small_crop)
    assert result.shape[0] >= FACENET_INPUT_SIZE
    assert result.shape[1] >= FACENET_INPUT_SIZE


def test_normalise_crop_size_does_not_change_large_image():
    """Does not resize a crop already larger than FACENET_INPUT_SIZE."""
    large_crop = np.zeros((200, 200, 3), dtype=np.uint8)
    result = normalise_crop_size(large_crop)
    assert result.shape == large_crop.shape
