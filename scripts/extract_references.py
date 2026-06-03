import argparse
import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import REFERENCES_DIR, configure_environment
from src.faces import crop_face, detect_faces, get_largest_face
from src.labels import load_labels
from src.video import VideoReader
from src.visualisation import save_image


def extract_references(video_path: str, labels_path: str, output_dir: str) -> None:
    """Extracts tight face crops from labelled frames and saves as reference images."""
    labels = load_labels(labels_path)
    print(f"Extracting references from {len(labels)} labels\n")

    reference_counts: defaultdict = defaultdict(int)

    with VideoReader(video_path) as video:
        for _, row in labels.iterrows():
            character = row["character_name"]
            frame_number = row["frame_number"]
            frame_image = video.extract_frame(frame_number)

            if frame_image is None:
                print(f"SKIP    {character} frame {frame_number} — could not extract frame")
                continue

            detected_faces = detect_faces(frame_image)

            if not detected_faces:
                print(f"SKIP    {character} frame {frame_number} — no face detected")
                continue

            largest_face = get_largest_face(detected_faces)

            if largest_face is None:
                print(f"SKIP    {character} frame {frame_number} — could not determine largest face")
                continue

            crop_image = crop_face(frame_image, largest_face)
            if crop_image is None:
                print(f"SKIP    {character} frame {frame_number} — could not crop face")
                continue

            reference_counts[character] += 1
            character_image_number = reference_counts[character]

            output_path = os.path.join(output_dir, f"{character}_{character_image_number}.jpg")
            save_image(crop_image, output_path)
            print(f"SAVED   {character}_{character_image_number} → {output_path} (frame {frame_number})")

    print(f"\nDone — {sum(reference_counts.values())} reference images saved")
    for character, count in sorted(reference_counts.items()):
        print(f"  {character}: {count} reference(s)")


def main() -> None:
    configure_environment()
    parser = argparse.ArgumentParser(description="Extract tight face crop reference images from labelled video frames")
    parser.add_argument("--video", default="data/nimbus.mp4", help="Path to video file")
    parser.add_argument("--labels", required=True, help="Path to reference labels CSV")
    parser.add_argument("--output", default=REFERENCES_DIR, help="Output directory for reference images")
    args = parser.parse_args()

    extract_references(args.video, args.labels, args.output)


if __name__ == "__main__":
    main()
