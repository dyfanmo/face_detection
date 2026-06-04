import argparse
import logging
import os
from collections import defaultdict

from src.config import REFERENCE_LABELS, REFERENCES_DIR, VIDEO_PATH, configure_environment, setup_logging
from src.exceptions import CropFailedError, FrameExtractionError, NoFacesDetectedError
from src.faces import extract_dominant_face_crop
from src.labels import load_labels
from src.video import VideoReader
from src.visualisation import save_image

logger = logging.getLogger(__name__)


def extract_references(video_path: str, labels_path: str, output_dir: str) -> None:
    """Extracts tight face crops from labelled frames and saves as reference images."""
    labels = load_labels(labels_path)
    logger.info(f"Extracting references from {len(labels)} labels")

    reference_counts: defaultdict[str, int] = defaultdict(int)

    with VideoReader(video_path) as video:
        for _, row in labels.iterrows():
            character: str = row["character_name"]
            frame_number: int = row["frame_number"]

            try:
                frame_image = video.extract_frame(frame_number)
                crop_image = extract_dominant_face_crop(frame_image)
            except (FrameExtractionError, NoFacesDetectedError, CropFailedError) as e:
                logger.warning(f"{character} frame {frame_number} — {e}, skipping")
                continue

            reference_counts[character] += 1
            character_image_number = reference_counts[character]

            output_path = os.path.join(output_dir, f"{character}_{character_image_number}.jpg")
            save_image(crop_image, output_path)
            logger.info(f"Saved {character}_{character_image_number} → {output_path} (frame {frame_number})")

    logger.info(f"Done — {sum(reference_counts.values())} reference images saved")
    for character, count in sorted(reference_counts.items()):
        logger.info(f"  {character}: {count} reference(s)")


def main() -> None:
    configure_environment()
    setup_logging()
    parser = argparse.ArgumentParser(description="Extract tight face crop reference images from labelled video frames")
    parser.add_argument("--video", default=VIDEO_PATH, help="Path to video file")
    parser.add_argument("--labels", default=REFERENCE_LABELS, help="Path to reference labels CSV")
    parser.add_argument("--output", default=REFERENCES_DIR, help="Output directory for reference images")
    args = parser.parse_args()

    extract_references(args.video, args.labels, args.output)


if __name__ == "__main__":
    main()
