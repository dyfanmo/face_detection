import logging

import pandas as pd

from src.data_models import EvaluationResult, FacePrediction

logger = logging.getLogger(__name__)


def is_missed_prediction(row: EvaluationResult) -> bool:
    """Returns True if the result row is a missed ground truth prediction."""
    return bool(row.true_character) and not row.character_found


def is_false_positive(row: EvaluationResult) -> bool:
    """Returns True if the result row is a false positive prediction."""
    return row.true_character is None


def find_matching_prediction(
    expected_character: str,
    predicted_characters: list[FacePrediction],
) -> FacePrediction | None:
    """Returns the FacePrediction for the expected character if found, otherwise None."""
    for prediction in predicted_characters:
        if prediction.character == expected_character:
            return prediction
    return None


def evaluate_frame(
    frame_number: int,
    expected_characters: list[str],
    predicted_characters: list[FacePrediction],
) -> list[EvaluationResult]:
    """Compares predicted characters against expected characters for a single frame.
    Returns a list of EvaluationResult — one per expected character plus any false positives."""
    frame_results = []
    found_character_names = {p.character for p in predicted_characters}

    for expected_character in expected_characters:
        matching_prediction = find_matching_prediction(expected_character, predicted_characters)
        character_found = expected_character in found_character_names
        frame_results.append(build_result_row(frame_number, expected_character, matching_prediction, character_found))

        status = "✓" if character_found else "✗"
        distance_str = f"distance: {matching_prediction.distance}" if matching_prediction else "no prediction"
        logger.info(f"  {status} {expected_character} — {distance_str}")

    for prediction in predicted_characters:
        if prediction.character not in expected_characters:
            frame_results.append(build_result_row(frame_number, None, prediction, False))
            logger.info(f"  ✗ false positive — {prediction.character} distance: {prediction.distance}")

    return frame_results


def build_result_row(
    frame_number: int,
    expected_character: str | None,
    prediction: FacePrediction | None,
    character_found: bool,
) -> EvaluationResult:
    """Builds an EvaluationResult for a single expected character or false positive."""
    bbox = prediction.bbox if prediction else None
    return EvaluationResult(
        frame_number=frame_number,
        true_character=expected_character,
        predicted_character=prediction.character if prediction else None,
        distance=prediction.distance if prediction else None,
        is_confident_match=prediction.is_confident_match if prediction else False,
        character_found=character_found,
        bbox_x=bbox.x if bbox else None,
        bbox_y=bbox.y if bbox else None,
        bbox_w=bbox.w if bbox else None,
        bbox_h=bbox.h if bbox else None,
    )


def print_summary(evaluation_results_df: pd.DataFrame, output_path: str) -> None:
    """Logs evaluation summary — ground truth found, false positives, recall."""
    if evaluation_results_df.empty:
        logger.warning("No frames were evaluated — check that the video path is correct.")
        return

    correctly_identified = evaluation_results_df["character_found"].sum()
    total_expected = evaluation_results_df["true_character"].notna().sum()
    false_positive_count = evaluation_results_df["true_character"].isna().sum()

    logger.info(f"Saved {len(evaluation_results_df)} rows to {output_path}")
    logger.info(f"Ground truth found:    {correctly_identified} / {total_expected}")
    logger.info(f"False positives:       {false_positive_count}")
    if total_expected > 0:
        logger.info(f"Recall:                {correctly_identified / total_expected * 100:.1f}%")
