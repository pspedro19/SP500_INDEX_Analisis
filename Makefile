.PHONY: install test lint format clean

PYTHON := python3
SOURCES := src tests

install:
	$(PYTHON) -m pip install -r requirement.txt

test:
	$(PYTHON) -m pytest -q

lint:
	flake8 $(SOURCES)
	mypy $(SOURCES)

format:
	isort $(SOURCES)
	black $(SOURCES)

clean:
	find . -type f -name '*.pyc' -delete
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
