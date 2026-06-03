import logging
import math
import os

import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

from src.config import DISTANCE_THRESHOLD, FACENET_INPUT_SIZE, RECOGNITION_MODEL
from src.exceptions import FaceRecognitionError, ReferenceLoadError
from src.models import FaceBBox, FacePrediction

logger = logging.getLogger(__name__)


def parse_character_name(filename: str) -> str:
    """Strips the index suffix from a reference filename to get the character name.
    e.g. 'Harry Potter_9.jpg' -> 'Harry Potter'"""
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    return "_".join(filename_without_extension.split("_")[:-1])


def normalise_crop_size(crop: np.ndarray) -> np.ndarray:
    """Resizes a face crop to meet the minimum input size required by Facenet512."""
    h, w = crop.shape[:2]
    if h < FACENET_INPUT_SIZE or w < FACENET_INPUT_SIZE:
        scale = FACENET_INPUT_SIZE / min(w, h)
        new_width = int(w * scale)
        new_height = int(h * scale)
        crop = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return crop


def build_reference_embeddings(references_dir: str) -> dict[str, list[list[float]]]:
    """Pre-computes Facenet512 embeddings for all reference images.
    Returns a dict mapping character name to a list of embeddings.
    Call once at startup — avoids reloading on every recognition call.
    Raises ReferenceLoadError if no reference images are found."""
    images = [f for f in os.listdir(references_dir) if f.endswith(".jpg")]

    if not images:
        raise ReferenceLoadError(f"No reference images found in {references_dir}")

    embeddings: dict[str, list[list[float]]] = {}

    for filename in sorted(images):
        image_path = os.path.join(references_dir, filename)
        character = parse_character_name(filename)

        deepface_output = DeepFace.represent(
            img_path=image_path,
            model_name=RECOGNITION_MODEL,
            enforce_detection=False,
        )

        if deepface_output:
            embeddings.setdefault(character, []).append(deepface_output[0]["embedding"])

    logger.info(f"Built embeddings for {len(embeddings)} characters from {len(images)} reference images")
    return embeddings


def recognise_face(
    face_crop: np.ndarray, ref_embeddings: dict[str, list[list[float]]], bbox: FaceBBox
) -> FacePrediction:
    """Compares a face crop against pre-computed reference embeddings.
    Returns a FacePrediction with the best matching character, distance, and is_match flag.
    Raises FaceRecognitionError if the crop cannot be represented.

    is_match=True means distance is below DISTANCE_THRESHOLD — confident identification.
    is_match=False means a best guess was found but confidence is too low.
    """
    face_crop = normalise_crop_size(face_crop)

    deepface_output = DeepFace.represent(
        img_path=face_crop,
        model_name=RECOGNITION_MODEL,
        enforce_detection=False,
    )

    if not deepface_output:
        raise FaceRecognitionError("DeepFace could not represent the face crop")

    crop_embedding = deepface_output[0]["embedding"]

    best_character = None
    best_distance = float("inf")

    # find the reference embedding with the smallest cosine distance to the crop
    for character, reference_embeddings in ref_embeddings.items():
        for embedding in reference_embeddings:
            distance = cosine(crop_embedding, embedding)
            if math.isnan(distance):
                continue
            if distance < best_distance:
                best_distance = distance
                best_character = character

    if best_character is None:
        raise FaceRecognitionError("No valid reference embeddings to compare against")

    return FacePrediction(
        character=best_character,
        distance=round(float(best_distance), 4),
        is_match=best_distance < DISTANCE_THRESHOLD,
        bbox=bbox,
    )


def deduplicate_predictions(predictions: list[FacePrediction]) -> list[FacePrediction]:
    """Keeps only the best (lowest distance) prediction per character.
    Prevents the same character appearing multiple times from different face crops."""
    best_predictions: dict[str, FacePrediction] = {}
    for prediction in predictions:
        if (
            prediction.character not in best_predictions
            or prediction.distance < best_predictions[prediction.character].distance
        ):
            best_predictions[prediction.character] = prediction
    return list(best_predictions.values())
