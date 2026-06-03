from unittest.mock import patch

import numpy as np

from src.models import FaceBBox, FacePrediction
from src.pipeline import identify_faces


def make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_face_bbox() -> FaceBBox:
    return FaceBBox(x=100, y=100, w=100, h=100, confidence=0.95)


def make_prediction(character: str = "Harry Potter", distance: float = 0.1) -> FacePrediction:
    return FacePrediction(character=character, distance=distance, is_match=True, bbox=make_face_bbox())


@patch("src.pipeline.recognise_face")
@patch("src.pipeline.crop_face")
@patch("src.pipeline.filter_faces")
@patch("src.pipeline.detect_faces")
def test_identify_faces_returns_correct_structure(mock_detect, mock_filter, mock_crop, mock_recognise):
    """Returns predicted characters, face detections, and counts."""
    face_bbox = make_face_bbox()
    crop = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_detect.return_value = [face_bbox]
    mock_filter.return_value = [face_bbox]
    mock_crop.return_value = crop
    mock_recognise.return_value = make_prediction()

    predicted_characters, face_detections, faces_detected, faces_passed = identify_faces(make_frame(), {})

    assert faces_detected == 1
    assert faces_passed == 1
    assert len(face_detections) == 1
    assert len(predicted_characters) == 1
    assert predicted_characters[0].character == "Harry Potter"


@patch("src.pipeline.recognise_face")
@patch("src.pipeline.crop_face")
@patch("src.pipeline.filter_faces")
@patch("src.pipeline.detect_faces")
def test_identify_faces_excludes_unconfident_predictions(mock_detect, mock_filter, mock_crop, mock_recognise):
    """Excludes predictions where is_match is False."""
    face_bbox = make_face_bbox()
    crop = np.zeros((100, 100, 3), dtype=np.uint8)

    mock_detect.return_value = [face_bbox]
    mock_filter.return_value = [face_bbox]
    mock_crop.return_value = crop
    mock_recognise.return_value = FacePrediction(character="Harry Potter", distance=0.5, is_match=False, bbox=face_bbox)

    predicted_characters, _, _, _ = identify_faces(make_frame(), {})
    assert predicted_characters == []
