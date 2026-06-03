import os


def configure_environment() -> None:
    """Suppresses TensorFlow and oneDNN logging. Call once at the start of each script."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


DETECTOR = "retinaface"
RECOGNITION_MODEL = "Facenet512"
DISTANCE_THRESHOLD = 0.35
MIN_DETECTION_CONFIDENCE = 0.9
MIN_FACE_SIZE = 80
FACENET_INPUT_SIZE = 160
PADDING = 20
FRAME_SAMPLE_RATE = 5
REFERENCES_DIR = "data/reference_frames"
