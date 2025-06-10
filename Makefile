.PHONY: install test lint format clean

install:
	pip install -r requirement.txt

test:
	pytest

lint:
	black --check .
	isort --check-only .

format:
	black .
	isort .

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache
