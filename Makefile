.PHONY: lint format typecheck test coverage check-data evaluate-one evaluate-multi check

lint:
	uv run ruff check src/ scripts/ tests/

format:
	uv run ruff format src/ scripts/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/

coverage:
	uv run pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70

check-data:
	uv run python scripts/check_label_overlap.py

evaluate-one:
	uv run python scripts/evaluate.py --labels data/labels/one_face_test_labels.csv --visualise data/test_frames/one_face

evaluate-multi:
	uv run python scripts/evaluate.py --labels data/labels/multi_face_test_labels.csv --visualise data/test_frames/multi_face

check: lint typecheck coverage check-data
