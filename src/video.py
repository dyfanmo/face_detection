from typing import Generator

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str):
        """Opens a video file and reads its properties."""
        self._cap = cv2.VideoCapture(path)
        self.path = path
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract_frame(self, frame_number: int) -> np.ndarray | None:
        """Extracts and returns a single frame by frame number. Returns None if extraction fails."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame_image = self._cap.read()
        return frame_image if success else None

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
