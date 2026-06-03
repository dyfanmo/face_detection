# Face Detection and Recognition Pipeline

A face detection and recognition pipeline for identifying characters in video footage. Built using RetinaFace for detection and Facenet512 for recognition via DeepFace.

## Results

| Test Set | Frames | Recall | False Positives |
|---|---|---|---|
| Single face | 33 | 93.9% | 1 |
| Multi face | 27 | 77.8% | 0 |

## Project Structure

```
src/
  config.py          — constants, logging setup, environment configuration
  exceptions.py      — custom exceptions for the pipeline
  models.py          — dataclasses: FaceBBox, FacePrediction, FrameAnalysis, EvaluationResult
  video.py           — video I/O and frame extraction
  faces.py           — face detection, filtering, cropping
  recognise.py       — face recognition, embeddings, cosine similarity
  pipeline.py        — full identify_faces pipeline
  visualisation.py   — drawing, annotating, saving frames
  labels.py          — loading and cleaning label CSVs
  evaluation.py      — evaluation logic and reporting

scripts/
  extract_references.py   — extract reference face crops from video
  validate_references.py  — validate reference images are usable
  evaluate.py             — evaluate recognition against labelled frames
  run_pipeline.py         — run recognition on full video

tests/
  test_config.py
  test_evaluation.py
  test_faces.py
  test_labels.py
  test_pipeline.py
  test_recognise.py
  test_video.py
  test_visualisation.py

data/
  labels/
    reference_labels.csv        — frame labels for reference extraction
    one_face_test_labels.csv    — single face evaluation labels
    multi_face_test_labels.csv  — multi face evaluation labels
```

## Labelling Tool

Ground truth labels were generated using [video_annotator](https://github.com/dyfanmo/video_annotator) — a PyQt5 frame labelling tool built specifically for this project. It exports CSV files with exact frame numbers, character names, and optional notes.

The label CSVs in `data/labels/` were produced by loading `nimbus.mp4` into the tool and using the `F` key to capture frames at precise moments.

## Setup

### 1. Install uv

```bash
pip install uv
```

### 2. Clone the repository

```bash
git clone git@github.com:dyfanmo/face_detection.git
cd face_detection
```

### 3. Install dependencies

```bash
uv sync --extra dev
```

This installs all dependencies including dev tools (pytest, ruff, mypy, vulture). uv will handle the correct Python version automatically.

### 4. Obtain the video file

The source video `nimbus.mp4` is not included in the repository. Place it anywhere accessible on your machine — the path is passed as an argument to each script.

### 5. Extract reference frames

Generate face crop reference images for each character:

```bash
uv run python scripts/extract_references.py \
  --video path/to/nimbus.mp4 \
  --labels data/labels/reference_labels.csv \
  --output data/reference_frames
```

### 6. Validate references

Confirm DeepFace can detect a face in each reference image:

```bash
uv run python scripts/validate_references.py
```

---

## Running the Pipeline

Runs face recognition across the full video and produces an annotated output video with bounding boxes drawn around recognised characters:

```bash
uv run python scripts/run_pipeline.py \
  --video path/to/nimbus.mp4 \
  --output data/videos/output.mp4
```

The pipeline samples every 5 frames for recognition and carries predictions forward to non-sampled frames to keep boxes stable.

---

## Running Evaluation

Evaluates recognition performance against manually labelled ground truth frames.

**Single face evaluation:**

```bash
uv run python scripts/evaluate.py \
  --video path/to/nimbus.mp4 \
  --labels data/labels/one_face_test_labels.csv \
  --visualise data/test_frames/one_face
```

**Multi face evaluation:**

```bash
uv run python scripts/evaluate.py \
  --video path/to/nimbus.mp4 \
  --labels data/labels/multi_face_test_labels.csv \
  --visualise data/test_frames/multi_face
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--visualise DIR` | Save annotated frames to directory — false positives drawn in red |
| `--debug` | Save only failed frames to `data/debug_frames/` with Unknown boxes for unrecognised faces |

---

## Code Quality

```bash
uv run ruff check src/ scripts/ tests/                                          # lint
uv run mypy src/                                                                 # typecheck
uv run pytest tests/                                                             # tests
uv run pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70    # coverage
```

---

## Known Issues

**TensorFlow warnings on startup** — CPU instruction and oneDNN warnings are printed by TensorFlow's C++ layer before Python initialises. They do not affect results and cannot be suppressed via Python. Set these environment variables before running to hide them:

```bash
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
```

**H.264 stream warning** — `nimbus.mp4` contains a minor H.264 stream error that triggers an OpenCV decoder warning at certain frames. This does not affect frame extraction or evaluation results.

**First run is slow** — DeepFace downloads model weights on first use. Subsequent runs use the cached models.
