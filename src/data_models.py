"""Dataclasses for the face detection and recognition pipeline."""

from dataclasses import dataclass

import numpy as np


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
    is_confident_match: bool
    bbox: FaceBBox


@dataclass
class FrameAnalysis:
    """Results of running the full identification pipeline on a single frame."""

    predicted_characters: list[FacePrediction]
    face_detections: list[tuple[np.ndarray, FaceBBox, FacePrediction | None]]
    faces_detected: int
    faces_passed: int


@dataclass
class EvaluationResult:
    """A single row in the evaluation results — one per expected character or false positive."""

    frame_number: int
    true_character: str | None
    predicted_character: str | None
    distance: float | None
    is_confident_match: bool
    character_found: bool
    bbox_x: int | None
    bbox_y: int | None
    bbox_w: int | None
    bbox_h: int | None
