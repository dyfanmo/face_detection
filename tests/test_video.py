import pytest

from src.exceptions import FrameExtractionError
from src.video import VideoReader


def test_video_raises_on_invalid_path():
    """Raises FileNotFoundError when video path does not exist."""
    with pytest.raises(FileNotFoundError):
        VideoReader("nonexistent.mp4")
