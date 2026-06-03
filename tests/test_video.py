import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.video import VideoReader


def test_video_raises_on_invalid_path():
    """Raises FileNotFoundError when video path does not exist."""
    with pytest.raises(FileNotFoundError):
        VideoReader("nonexistent.mp4")
