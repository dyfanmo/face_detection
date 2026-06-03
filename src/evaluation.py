import pandas as pd


def is_missed_prediction(row: dict) -> bool:
    """Returns True if the result row is a missed ground truth prediction."""
    return bool(row["true_character"]) and not row["found"]


def is_false_positive(row: dict) -> bool:
    """Returns True if the result row is a false positive prediction."""
    return row["true_character"] is None


def find_matching_prediction(expected_character: str, predicted_characters: list) -> dict | None:
    """Returns the prediction dict for the expected character if found, otherwise None."""
    for prediction in predicted_characters:
        if prediction["character"] == expected_character:
            return prediction
    return None


def evaluate_frame(
    frame_number: int,
    expected_characters: list,
    predicted_characters: list,
) -> list:
    """Compares predicted characters against expected characters for a single frame.
    Returns a list of result rows — one per expected character plus any false positives."""
    frame_results = []
    found_character_names = {r["character"] for r in predicted_characters}

    for expected_character in expected_characters:
        matching_prediction = find_matching_prediction(expected_character, predicted_characters)
        is_match = expected_character in found_character_names
        frame_results.append(build_result_row(frame_number, expected_character, matching_prediction, is_match))

        status = "✓" if is_match else "✗"
        distance_str = f"distance: {matching_prediction['distance']}" if matching_prediction else "no prediction"
        print(f"  {status} {expected_character} — {distance_str}")

    for prediction in predicted_characters:
        if prediction["character"] not in expected_characters:
            frame_results.append(build_result_row(frame_number, None, prediction, False))
            print(f"  ✗ false positive — {prediction['character']} distance: {prediction['distance']}")

    return frame_results


def build_result_row(
    frame_number: int,
    expected_character: str | None,
    prediction: dict | None,
    is_match: bool,
) -> dict:
    """Builds a result row dict for the evaluation results CSV.
    Used for both expected character rows and false positive rows."""
    bbox = prediction.get("bbox") if prediction else None
    return {
        "frame_number": frame_number,
        "true_character": expected_character,
        "predicted_character": prediction["character"] if prediction else None,
        "distance": prediction["distance"] if prediction else None,
        "match": prediction["is_match"] if prediction else False,
        "found": is_match,
        "bbox_x": bbox["x"] if bbox else None,
        "bbox_y": bbox["y"] if bbox else None,
        "bbox_w": bbox["w"] if bbox else None,
        "bbox_h": bbox["h"] if bbox else None,
    }


def print_summary(evaluation_results_df: pd.DataFrame, output_path: str) -> None:
    """Prints evaluation summary — ground truth found, false positives, recall."""
    correctly_identified = evaluation_results_df["found"].sum()
    total_expected = evaluation_results_df["true_character"].notna().sum()
    false_positive_count = evaluation_results_df["true_character"].isna().sum()

    print(f"\nSaved {len(evaluation_results_df)} rows to {output_path}")
    print(f"Ground truth found:    {correctly_identified} / {total_expected}")
    print(f"False positives:       {false_positive_count}")
    if total_expected > 0:
        print(f"Recall:                {correctly_identified / total_expected * 100:.1f}%")
