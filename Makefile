.PHONY: lint format typecheck test coverage evaluate-one evaluate-multi check

PYTHON = powershell -File ./run.ps1 uv run python

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

evaluate-one:
	$(PYTHON) scripts/evaluate.py --labels data/labels/one_face_test_labels.csv --visualise data/test_frames/one_face

evaluate-multi:
	$(PYTHON) scripts/evaluate.py --labels data/labels/multi_face_test_labels.csv --visualise data/test_frames/multi_face

check: lint typecheck coverage
