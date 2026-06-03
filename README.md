# Nimbus — Face Detection and Recognition Pipeline

Face detection and recognition pipeline for video analysis using DeepFace (RetinaFace + Facenet512).

## Setup

```bash
uv venv
uv sync
```

## Running tests

```bash
uv run pytest tests/
```

## Extracting reference images

Extracts one reference face image per character from isolated label windows in the video.

```bash
uv run python scripts/extract_references.py --video data/nimbus.mp4 --labels data/nimbus_labels_only.csv
```

Reference images are saved to `references/`.
