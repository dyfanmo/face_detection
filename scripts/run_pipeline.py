import argparse
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import FRAME_SAMPLE_RATE, REFERENCES_DIR, configure_environment
from src.pipeline import identify_faces
from src.recognise import build_reference_embeddings
from src.video import VideoReader
from src.visualisation import draw_label

VIDEO_PATH = "data/nimbus.mp4"
OUTPUT_PATH = "data/videos/nimbus_output.mp4"


def run_pipeline(video_path: str, references_dir: str, output_path: str) -> None:
    """Processes the full video — detects and recognises faces on sampled frames,
    carries labels forward to non-sampled frames, writes annotated output video."""
    ref_embeddings = build_reference_embeddings(references_dir)
    carried_predictions = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with VideoReader(video_path) as video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, video.fps, (video.width, video.height))

        print(f"Video: {video.width}x{video.height} @ {video.fps}fps — {video.frame_count} frames")
        print(f"Sampling every {FRAME_SAMPLE_RATE} frames")
        print(f"Output: {output_path}\n")

        for frame_number, frame_image in video.frames():
            if frame_number % FRAME_SAMPLE_RATE == 0:
                predicted_characters, _, _, _ = identify_faces(frame_image, ref_embeddings)
                carried_predictions = predicted_characters

            for prediction in carried_predictions:
                draw_label(frame_image, prediction["bbox"], prediction["character"])

            out.write(frame_image)

        out.release()
        print(f"\nDone — {video.frame_count} frames written to {output_path}")


def main() -> None:
    configure_environment()
    parser = argparse.ArgumentParser(description="Run face detection and recognition pipeline on full video")
    parser.add_argument("--video", default=VIDEO_PATH, help="Path to input video")
    parser.add_argument("--references", default=REFERENCES_DIR, help="Path to reference images directory")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Path to save output video")
    args = parser.parse_args()

    run_pipeline(args.video, args.references, args.output)


if __name__ == "__main__":
    main()
