import os
import sys
import tempfile

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.labels import load_frame_numbers, load_labels, remove_overlapping_frames


def test_load_labels_normalises_frame_number_to_int():
    """frame_number column is loaded as int not float."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("frame_number,character_name\n100,Harry Potter\n200,Hermione Granger\n")
        path = f.name
    labels = load_labels(path)
    assert labels["frame_number"].dtype == int
    os.unlink(path)


def test_load_frame_numbers_returns_list_of_ints():
    """Returns a list of integer frame numbers."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("frame_number,character_name\n100,Harry Potter\n200,Hermione Granger\n")
        path = f.name
    result = load_frame_numbers(path)
    assert result == [100, 200]
    os.unlink(path)


def test_remove_overlapping_frames_removes_shared_frames():
    """Removes rows whose frame_number appears in the reference labels."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as ref_f:
        ref_f.write("frame_number,character_name\n100,Harry Potter\n")
        ref_path = ref_f.name

    labels = pd.DataFrame([
        {"frame_number": 100, "character_name": "Harry Potter"},
        {"frame_number": 200, "character_name": "Hermione Granger"},
    ])
    result = remove_overlapping_frames(ref_path, labels)
    assert len(result) == 1
    assert result.iloc[0]["frame_number"] == 200
    os.unlink(ref_path)


def test_remove_overlapping_frames_returns_unchanged_when_no_overlap():
    """Returns the labels unchanged when no frames overlap with reference."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as ref_f:
        ref_f.write("frame_number,character_name\n999,Harry Potter\n")
        ref_path = ref_f.name

    labels = pd.DataFrame([
        {"frame_number": 100, "character_name": "Harry Potter"},
        {"frame_number": 200, "character_name": "Hermione Granger"},
    ])
    result = remove_overlapping_frames(ref_path, labels)
    assert len(result) == 2
    os.unlink(ref_path)
