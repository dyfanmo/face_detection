import os
from typing import Generator

import cv2
import numpy as np

from src.exceptions import FrameExtractionError


class VideoReader:
    def __init__(self, path: str):
        """Opens a video file and reads its properties. Raises FileNotFoundError if path does not exist."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self._cap = cv2.VideoCapture(path)
        self.path = path
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract_frame(self, frame_number: int) -> np.ndarray:
        """Extracts and returns a single frame by frame number.
        Raises FrameExtractionError if the frame cannot be read."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame_image = self._cap.read()
        if not success:
            raise FrameExtractionError(f"Could not extract frame {frame_number}")
        return frame_image

    def frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Generator that yields every frame in the video as a tuple of (frame_number, frame_image)."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_number = 0
        while True:
            success, frame_image = self._cap.read()
            if not success:
                break
            yield frame_number, frame_image
            frame_number += 1

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._cap.release()
