import argparse
import logging
import os
from dataclasses import asdict

import pandas as pd

from src.config import REFERENCES_DIR, configure_environment, setup_logging
from src.evaluation import evaluate_frame, is_false_positive, is_missed_prediction, print_summary
from src.exceptions import FrameExtractionError
from src.labels import load_labels, remove_overlapping_frames
from src.pipeline import identify_faces
from src.recognise import build_reference_embeddings
from src.video import VideoReader
from src.visualisation import visualise_frame

logger = logging.getLogger(__name__)

VIDEO_PATH = "data/nimbus.mp4"
REFERENCE_LABELS = "data/labels/reference_labels.csv"
OUTPUT_PATH = "data/results/evaluation_results.csv"
DEBUG_DIR = "data/debug_frames"


def evaluate(
    video_path: str,
    labels_path: str,
    references_dir: str,
    output_path: str,
    visualise_dir: str | None = None,
    debug: bool = False,
) -> None:
    """Evaluates recognition performance against labelled ground truth frames.
    Automatically removes any test frames that overlap with reference labels.

    --visualise  saves all evaluated frames with bounding boxes — false positives in red.
    --debug      saves only failed frames to data/debug_frames/ with Unknown boxes
                 drawn for every detected face that was not confidently recognised.
    """
    labels = load_labels(labels_path)
    labels = remove_overlapping_frames(REFERENCE_LABELS, labels)
    ref_embeddings = build_reference_embeddings(references_dir)

    labelled_frames = labels.groupby("frame_number").agg(characters=("character_name", list)).to_dict("index")

    logger.info(f"Processing {len(labelled_frames)} unique frames")
    evaluation_results = []

    with VideoReader(video_path) as video:
        for frame_number, frame_info in sorted(labelled_frames.items()):
            try:
                frame_image = video.extract_frame(frame_number)
            except FrameExtractionError as e:
                logger.warning(f"frame {frame_number} — {e}, skipping")
                continue

            frame_analysis = identify_faces(frame_image, ref_embeddings)
            expected_characters: list[str] = frame_info["characters"]

            logger.info(
                f"frame {frame_number} — {frame_analysis.faces_detected} detected, "
                f"{frame_analysis.faces_passed} passed filters (ground truth: {expected_characters})"
            )

            frame_results = evaluate_frame(frame_number, expected_characters, frame_analysis.predicted_characters)
            evaluation_results.extend(frame_results)

            if visualise_dir:
                visualise_frame(
                    frame_image, frame_analysis.predicted_characters, frame_number, visualise_dir,
                    ground_truth=expected_characters
                )

            if debug:
                has_miss = any(is_missed_prediction(r) for r in frame_results)
                has_fp = any(is_false_positive(r) for r in frame_results)
                if has_miss or has_fp:
                    visualise_frame(
                        frame_image,
                        frame_analysis.predicted_characters,
                        frame_number,
                        DEBUG_DIR,
                        ground_truth=expected_characters,
                        detected_faces=[face_bbox for _, face_bbox, _ in frame_analysis.face_detections],
                    )

    evaluation_results_df = pd.DataFrame([asdict(r) for r in evaluation_results])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    evaluation_results_df.to_csv(output_path, index=False)
    print_summary(evaluation_results_df, output_path)


def main() -> None:
    configure_environment()
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate face recognition performance against labelled ground truth")
    parser.add_argument("--video", default=VIDEO_PATH, help="Path to video file")
    parser.add_argument("--labels", required=True, help="Path to test labels CSV")
    parser.add_argument("--references", default=REFERENCES_DIR, help="Path to reference images directory")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path to save evaluation results CSV")
    parser.add_argument("--visualise", metavar="DIR", default=None, help="Save annotated frames to this directory")
    parser.add_argument(
        "--debug", action="store_true", help="Save failed frames with Unknown boxes to data/debug_frames/"
    )
    args = parser.parse_args()

    evaluate(args.video, args.labels, args.references, args.output, visualise_dir=args.visualise, debug=args.debug)


if __name__ == "__main__":
    main()
