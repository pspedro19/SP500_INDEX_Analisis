import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import add_numbers


def test_add_numbers_unit():
    assert add_numbers(1, 4) == 5
