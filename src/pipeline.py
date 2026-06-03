import numpy as np

from src.faces import crop_face, detect_faces, filter_faces
from src.recognise import deduplicate_predictions, recognise_face


def identify_faces(frame_image: np.ndarray, ref_embeddings: dict) -> tuple:
    """Detects, filters, crops, recognises and deduplicates faces in a single frame.

    Returns:
        recognised_faces: list of confident deduplicated matches with character, distance, match, bbox
        face_detections: list of (crop, face, result) tuples for all filtered faces — used for debug
        faces_detected: total faces detected before filtering
        faces_passed: faces that passed confidence and size filters
    """
    detected_faces = detect_faces(frame_image)
    filtered_faces = filter_faces(detected_faces)

    face_detections = []
    for face_bbox in filtered_faces:
        crop = crop_face(frame_image, face_bbox)
        if crop is None:
            continue
        result = recognise_face(crop, ref_embeddings)
        face_detections.append((crop, face_bbox, result))

    confident_predictions = [
        {**recognition_result, "bbox": face_bbox}
        for _, face_bbox, recognition_result in face_detections
        if recognition_result is not None and recognition_result["is_match"]
    ]

    return deduplicate_predictions(confident_predictions), face_detections, len(detected_faces), len(filtered_faces)
