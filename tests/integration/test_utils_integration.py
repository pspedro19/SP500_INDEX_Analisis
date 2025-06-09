import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import add_numbers


def test_add_numbers_integration():
    assert add_numbers(10, 5) == 15
