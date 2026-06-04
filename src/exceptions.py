class CropFailedError(RuntimeError):
    """Raised when a face region cannot be cropped from a frame."""


class ImageLoadError(RuntimeError):
    """Raised when an image file cannot be loaded from disk."""


class FrameExtractionError(RuntimeError):
    """Raised when a frame cannot be extracted from the video."""


class NoFacesDetectedError(RuntimeError):
    """Raised when no faces are detected in a frame or image."""


class FaceRecognitionError(RuntimeError):
    """Raised when a face crop cannot be represented by the recognition model."""


class ReferenceLoadError(RuntimeError):
    """Raised when reference embeddings cannot be built from the references directory."""
