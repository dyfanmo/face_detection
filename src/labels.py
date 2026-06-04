import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def parse_character_name(filename: str) -> str:
    """Strips the index suffix from a reference filename to get the character name.
    e.g. 'Harry Potter_9.jpg' -> 'Harry Potter'"""
    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    return "_".join(filename_without_extension.split("_")[:-1])


def load_labels(csv_path: str) -> pd.DataFrame:
    """Loads a labels CSV and normalises frame_number to int."""
    labels = pd.read_csv(csv_path)
    labels["frame_number"] = labels["frame_number"].astype(int)
    return labels


def load_frame_numbers(csv_path: str) -> list[int]:
    """Loads a labels CSV and returns the frame_number column as a list of ints."""
    return load_labels(csv_path)["frame_number"].tolist()


def remove_overlapping_frames(reference_path: str, labels: pd.DataFrame) -> pd.DataFrame:
    """Removes any rows from labels whose frame_number also appears in the reference labels.
    Logs the removed frame numbers if any overlap is found."""
    reference_frame_numbers = load_frame_numbers(reference_path)
    is_overlapping = labels["frame_number"].isin(reference_frame_numbers)
    overlapping_frames = labels.loc[is_overlapping, "frame_number"].tolist()

    if overlapping_frames:
        logger.info(
            f"Removed {len(overlapping_frames)} overlapping frame(s) from test labels: {sorted(overlapping_frames)}"
        )
        labels = labels[~is_overlapping].reset_index(drop=True)

    return labels
