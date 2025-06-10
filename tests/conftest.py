import pytest


def pytest_addoption(parser):
    try:
        parser.addoption("--cov", action="append", default=[])
    except ValueError:
        pass
    try:
        parser.addoption("--cov-report", action="append", default=[])
    except ValueError:
        pass
