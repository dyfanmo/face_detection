import logging

import numpy as np

from src.exceptions import CropFailedError, FaceRecognitionError
from src.faces import crop_face, detect_faces, filter_faces
from src.models import FaceBBox, FacePrediction, FrameAnalysis
from src.recognise import deduplicate_predictions, recognise_face

logger = logging.getLogger(__name__)


def identify_faces(
    frame_image: np.ndarray,
    ref_embeddings: dict[str, list[list[float]]],
) -> FrameAnalysis:
    """Detects, filters, crops, recognises and deduplicates faces in a single frame.
    Returns a FrameAnalysis containing predicted characters, face detections, and counts."""
    detected_faces = detect_faces(frame_image)
    filtered_faces = filter_faces(detected_faces)

    face_detections: list[tuple[np.ndarray, FaceBBox, FacePrediction | None]] = []
    for face_bbox in filtered_faces:
        try:
            crop = crop_face(frame_image, face_bbox)
            prediction = recognise_face(crop, ref_embeddings, face_bbox)
            face_detections.append((crop, face_bbox, prediction))
        except CropFailedError as e:
            logger.debug(f"Crop failed for face at ({face_bbox.x}, {face_bbox.y}): {e}")
        except FaceRecognitionError as e:
            logger.debug(f"Recognition failed for face at ({face_bbox.x}, {face_bbox.y}): {e}")
            face_detections.append((crop, face_bbox, None))

    confident_predictions = [
        prediction
        for _, _, prediction in face_detections
        if prediction is not None and prediction.is_match
    ]

    return FrameAnalysis(
        predicted_characters=deduplicate_predictions(confident_predictions),
        face_detections=face_detections,
        faces_detected=len(detected_faces),
        faces_passed=len(filtered_faces),
    )
