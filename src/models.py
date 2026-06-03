"""Dataclasses for the face detection and recognition pipeline."""

from dataclasses import dataclass


@dataclass
class FaceBBox:
    """Bounding box and confidence score for a detected face."""

    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass
class FacePrediction:
    """Recognition result for a single detected face."""

    character: str
    distance: float
    is_match: bool
    bbox: FaceBBox
