import os
import sys
import tempfile

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.visualisation import draw_label, load_image, save_image


def make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_bbox() -> dict:
    return {"x": 100, "y": 100, "w": 100, "h": 100}


def test_save_image_writes_file_to_disk():
    """Saves an image to the specified path, creating parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "subdir", "test.jpg")
        save_image(make_frame(), path)
        assert os.path.exists(path)


def test_load_image_returns_none_for_missing_file():
    """Returns None when the file does not exist."""
    assert load_image("nonexistent.jpg") is None


def test_load_image_returns_array_for_valid_file():
    """Returns a numpy array when the file exists and is readable."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test.jpg")
        save_image(make_frame(), path)
        result = load_image(path)
        assert result is not None
        assert isinstance(result, np.ndarray)


def test_draw_label_does_not_raise():
    """Drawing a label on a frame completes without error."""
    draw_label(make_frame(), make_bbox(), "Harry Potter")
