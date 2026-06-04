import argparse
import logging
import os

from src.config import REFERENCES_DIR, configure_environment, setup_logging
from src.exceptions import CropFailedError, ImageLoadError, NoFacesDetectedError
from src.faces import extract_dominant_face_crop
from src.labels import parse_character_name
from src.visualisation import load_image

logger = logging.getLogger(__name__)


def validate_references(references_dir: str) -> None:
    """Runs face detection on each reference image and reports whether a face was found."""
    images = [f for f in os.listdir(references_dir) if f.endswith(".jpg")]

    if not images:
        logger.warning(f"No reference images found in {references_dir}")
        return

    logger.info(f"Validating {len(images)} reference images")

    all_valid = True
    for filename in sorted(images):
        character = parse_character_name(filename)

        try:
            reference_image = load_image(os.path.join(references_dir, filename))
            extract_dominant_face_crop(reference_image)
            logger.info(f"PASS  {character} — face detected successfully")
        except (ImageLoadError, NoFacesDetectedError, CropFailedError) as e:
            logger.warning(f"FAIL  {character} — {e}")
            all_valid = False

    if all_valid:
        logger.info("All references valid")
    else:
        logger.warning("Some references failed — re-extract those characters")


def main() -> None:
    configure_environment()
    setup_logging()
    parser = argparse.ArgumentParser(description="Validate that DeepFace can detect a face in each reference image")
    parser.add_argument("--references", default=REFERENCES_DIR, help="Path to reference images directory")
    args = parser.parse_args()

    validate_references(args.references)


if __name__ == "__main__":
    main()
