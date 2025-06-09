from src.utils import add_numbers


def test_add_numbers_integration():
    assert add_numbers(10, 5) == 15
