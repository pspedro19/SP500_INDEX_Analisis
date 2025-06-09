.PHONY: install test lint format clean

install:
	pip install -r requirement.txt

test:
	pytest -q

lint:
	flake8 src tests

format:
	black src tests

clean:
	find . -type f -name '*.pyc' -delete
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
