.PHONY: install test lint format clean

install:
	pip install -e .[dev]

test:
	pytest

lint:
	black --check pipelines/ml
	ruff check pipelines/ml
	mypy pipelines/ml

ruff:
	ruff check pipelines/ml

mypy:
	mypy pipelines/ml

format:
	black .
	ruff --fix .

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache
