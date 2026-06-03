import os

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1
BOX_THICKNESS = 2

LABEL_COLOURS = {
    "Harry Potter": (255, 200, 0),
    "Hermione Granger": (0, 200, 255),
    "Ron Weasley": (0, 100, 255),
    "Prof. Severus Snape": (180, 0, 255),
    "Prof. McGonagall": (0, 255, 150),
}
DEFAULT_COLOUR = (200, 200, 200)
DEBUG_COLOUR = (0, 0, 220)
UNKNOWN_COLOUR = (180, 180, 180)


def load_image(path: str) -> np.ndarray | None:
    """Loads an image from disk. Returns None if the file cannot be read."""
    image = cv2.imread(path)
    return image if image is not None and image.size > 0 else None


def save_image(image: np.ndarray, path: str) -> None:
    """Saves an image to disk, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def draw_label(frame: np.ndarray, bbox: dict, character: str, colour: tuple | None = None) -> None:
    """Draws a bounding box and character name label on a frame.
    Uses LABEL_COLOURS by default — pass colour to override."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    colour = colour or LABEL_COLOURS.get(character, DEFAULT_COLOUR)

    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, BOX_THICKNESS)

    text_size, baseline = cv2.getTextSize(character, FONT, FONT_SCALE, FONT_THICKNESS)
    text_y = max(y - 8, text_size[1] + 4)

    cv2.rectangle(
        frame,
        (x, text_y - text_size[1] - 4),
        (x + text_size[0] + 4, text_y + baseline),
        colour,
        -1,
    )
    cv2.putText(
        frame,
        character,
        (x + 2, text_y - 2),
        FONT,
        FONT_SCALE,
        (0, 0, 0),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def visualise_frame(
    frame: np.ndarray,
    frame_predictions: list,
    frame_number: int,
    output_dir: str,
    ground_truth: list | None = None,
    detected_faces: list | None = None,
) -> None:
    """Draws bounding boxes on a frame and saves it to output_dir.

    Colour coding:
      - Correct match     → character colour
      - False positive    → red
      - Unrecognised face → grey "Unknown" (requires detected_faces)

    ground_truth is required to identify false positives.
    detected_faces is a list of face bbox dicts for all filtered detections —
    required to draw Unknown boxes for faces that were not confidently recognised.
    """
    confident_positions = {
        (prediction["bbox"]["x"], prediction["bbox"]["y"]) for prediction in frame_predictions if prediction.get("bbox")
    }

    if detected_faces:
        for face_bbox in detected_faces:
            if (face_bbox["x"], face_bbox["y"]) not in confident_positions:
                draw_label(frame, face_bbox, "Unknown", colour=UNKNOWN_COLOUR)

    for prediction in frame_predictions:
        if not prediction.get("bbox"):
            continue
        prediction_is_false_positive = ground_truth is not None and prediction["character"] not in ground_truth
        colour = DEBUG_COLOUR if prediction_is_false_positive else None
        draw_label(frame, prediction["bbox"], prediction["character"], colour=colour)

    save_image(frame, os.path.join(output_dir, f"frame_{frame_number:04d}.jpg"))
