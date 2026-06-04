import logging
import os


def configure_environment() -> None:
    """Suppresses TensorFlow and oneDNN logging. Call once at the start of each script."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def setup_logging(level: int = logging.INFO) -> None:
    """Configures root logger with a consistent format. Call once at the start of each script."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# Model configuration
DETECTOR = "retinaface"
RECOGNITION_MODEL = "Facenet512"
DISTANCE_THRESHOLD = 0.35
MIN_DETECTION_CONFIDENCE = 0.9
MIN_FACE_SIZE = 80
FACENET_INPUT_SIZE = 160
PADDING = 20
FRAME_SAMPLE_RATE = 5

# Project paths
VIDEO_PATH = "data/videos/nimbus.mp4"
REFERENCES_DIR = "data/reference_frames"
REFERENCE_LABELS = "data/labels/reference_labels.csv"
EVALUATION_OUTPUT_DIR = "data/results"
PIPELINE_OUTPUT_PATH = "data/videos/nimbus_output.mp4"
DEBUG_DIR = "data/debug_frames"
