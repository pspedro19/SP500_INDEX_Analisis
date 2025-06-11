.PHONY: install test lint format clean

install:
	pip install -e .[dev]

test:
	pytest

lint:
	black --check pipelines/ml src
	ruff check --exit-zero pipelines/ml src
	mypy pipelines/ml src

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
