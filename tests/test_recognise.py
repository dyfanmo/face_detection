from unittest.mock import patch

import numpy as np

from src.config import FACENET_INPUT_SIZE
from src.models import FaceBBox, FacePrediction
from src.recognise import deduplicate_predictions, normalise_crop_size, parse_character_name, recognise_face


def make_crop() -> np.ndarray:
    return np.zeros((160, 160, 3), dtype=np.uint8)


def make_bbox() -> FaceBBox:
    return FaceBBox(x=0, y=0, w=160, h=160, confidence=0.95)


def make_embedding() -> list[float]:
    """Returns a valid non-zero embedding vector."""
    rng = np.random.default_rng(42)
    return rng.random(512).tolist()


def test_parse_character_name_strips_index():
    """Strips trailing index suffix from a reference filename."""
    assert parse_character_name("Harry Potter_9.jpg") == "Harry Potter"


def test_parse_character_name_handles_prof():
    """Handles character names with dots and multiple words."""
    assert parse_character_name("Prof. Severus Snape_1.jpg") == "Prof. Severus Snape"


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


def test_deduplicate_predictions_keeps_lowest_distance():
    """Keeps only the best (lowest distance) prediction per character."""
    bbox = make_bbox()
    predictions = [
        FacePrediction(character="Harry Potter", distance=0.3, is_match=True, bbox=bbox),
        FacePrediction(character="Harry Potter", distance=0.2, is_match=True, bbox=bbox),
        FacePrediction(character="Hermione Granger", distance=0.15, is_match=True, bbox=bbox),
    ]
    result = deduplicate_predictions(predictions)
    assert len(result) == 2
    harry = next(r for r in result if r.character == "Harry Potter")
    assert harry.distance == 0.2


@patch("src.recognise.DeepFace.represent")
def test_recognise_face_returns_correct_character(mock_represent):
    """Returns FacePrediction with correct character when distance is below threshold."""
    embedding = make_embedding()
    mock_represent.return_value = [{"embedding": embedding}]
    result = recognise_face(make_crop(), {"Harry Potter": [embedding]}, make_bbox())
    assert result.character == "Harry Potter"
    assert result.is_match


@patch("src.recognise.DeepFace.represent")
def test_recognise_face_raises_when_no_results(mock_represent):
    """Raises FaceRecognitionError when DeepFace cannot represent the crop."""
    from src.exceptions import FaceRecognitionError

    mock_represent.return_value = []
    import pytest

    with pytest.raises(FaceRecognitionError):
        recognise_face(make_crop(), {"Harry Potter": [make_embedding()]}, make_bbox())
