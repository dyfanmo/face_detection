import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.video import VideoReader


def test_video_properties_default_on_invalid_path():
    """fps and frame_count default to 0 when path does not exist."""
    with VideoReader("nonexistent.mp4") as video:
        assert video.fps == 0
        assert video.frame_count == 0
