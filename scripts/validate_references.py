import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import REFERENCES_DIR, configure_environment
from src.faces import detect_faces, get_largest_face
from src.recognise import parse_character_name
from src.visualisation import load_image


def validate_references(references_dir: str) -> None:
    """Runs face detection on each reference image and reports whether a face was found."""
    images = [f for f in os.listdir(references_dir) if f.endswith(".jpg")]

    if not images:
        print(f"No reference images found in {references_dir}")
        return

    print(f"Validating {len(images)} reference images\n")

    all_valid = True
    for filename in sorted(images):
        character = parse_character_name(filename)
        reference_image = load_image(os.path.join(references_dir, filename))

        if reference_image is None:
            print(f"FAIL  {character} — could not load image")
            all_valid = False
            continue

        faces = detect_faces(reference_image)

        if not faces:
            print(f"FAIL  {character} — no face detected")
            all_valid = False
            continue

        largest_face = get_largest_face(faces)

        if largest_face:
            size = f"{largest_face['w']}x{largest_face['h']}px"
            conf = f"{largest_face['confidence']:.2f}"
            print(f"PASS  {character} — detected face {size} (confidence {conf})")
        else:
            print(f"FAIL  {character} — could not determine largest face")
            all_valid = False

    print()
    print("All references valid" if all_valid else "Some references failed — re-extract those characters")


def main() -> None:
    configure_environment()
    parser = argparse.ArgumentParser(description="Validate that DeepFace can detect a face in each reference image")
    parser.add_argument("--references", default=REFERENCES_DIR, help="Path to reference images directory")
    args = parser.parse_args()

    validate_references(args.references)


if __name__ == "__main__":
    main()
