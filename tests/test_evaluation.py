import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    build_result_row,
    evaluate_frame,
    find_matching_prediction,
    is_false_positive,
    is_missed_prediction,
    print_summary,
)


def make_prediction(character: str, distance: float = 0.1) -> dict:
    return {
        "character": character,
        "distance": distance,
        "is_match": True,
        "bbox": {"x": 10, "y": 10, "w": 50, "h": 50},
    }


def test_is_missed_prediction_returns_true_when_not_found():
    """Returns True when expected character was not found."""
    assert is_missed_prediction({"true_character": "Harry Potter", "found": False}) is True


def test_is_false_positive_returns_true_when_no_true_character():
    """Returns True when true_character is None."""
    assert is_false_positive({"true_character": None}) is True


def test_find_matching_prediction_returns_correct_prediction():
    """Returns the prediction dict matching the expected character."""
    predictions = [make_prediction("Harry Potter"), make_prediction("Hermione Granger")]
    result = find_matching_prediction("Harry Potter", predictions)
    assert result is not None
    assert result["character"] == "Harry Potter"


def test_find_matching_prediction_returns_none_when_not_found():
    """Returns None when no prediction matches the expected character."""
    assert find_matching_prediction("Harry Potter", [make_prediction("Hermione Granger")]) is None


def test_build_result_row_correct_prediction():
    """Builds a correct result row for a matched prediction."""
    prediction = make_prediction("Harry Potter", distance=0.15)
    row = build_result_row(100, "Harry Potter", prediction, True)
    assert row["frame_number"] == 100
    assert row["true_character"] == "Harry Potter"
    assert row["predicted_character"] == "Harry Potter"
    assert row["distance"] == 0.15
    assert row["found"] is True
    assert row["bbox_x"] == 10


def test_build_result_row_missed_prediction():
    """Builds a result row with None values when no prediction was made."""
    row = build_result_row(100, "Harry Potter", None, False)
    assert row["predicted_character"] is None
    assert row["distance"] is None
    assert row["found"] is False
    assert row["bbox_x"] is None


def test_evaluate_frame_correct_match():
    """Returns found=True when predicted character matches expected."""
    results = evaluate_frame(100, ["Harry Potter"], [make_prediction("Harry Potter")])
    assert len(results) == 1
    assert results[0]["found"] is True


def test_evaluate_frame_missed_prediction():
    """Returns found=False when expected character is not in predictions."""
    results = evaluate_frame(100, ["Harry Potter"], [])
    assert len(results) == 1
    assert results[0]["found"] is False


def test_evaluate_frame_false_positive():
    """Adds a false positive row when prediction does not match any expected character."""
    results = evaluate_frame(100, ["Harry Potter"], [make_prediction("Ron Weasley")])
    assert len(results) == 2
    fp = next(r for r in results if r["true_character"] is None)
    assert fp["predicted_character"] == "Ron Weasley"


def test_print_summary_outputs_recall(capsys):
    """Prints correct recall percentage."""
    df = pd.DataFrame([
        {"true_character": "Harry Potter", "found": True},
        {"true_character": "Hermione Granger", "found": False},
    ])
    print_summary(df, "data/results/test.csv")
    assert "50.0%" in capsys.readouterr().out
