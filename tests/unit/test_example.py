"""
test_example.py

Description:
This module includes example pytest unit tests.

Author:
Ollie Tooth (oliver.tooth@noc.ac.uk)
"""
import pytest


# Example function to demonstrate unit testing:
def get_filepaths(example: str) -> None:
    if not isinstance(example, str):
        raise TypeError("`example` must be a string.")
    return None


# Example unit test class:
class TestExample():
    @pytest.mark.parametrize("invalid_example", [123, None, [], {}])
    def test_example_TypeError(self, invalid_example: str):
        with pytest.raises(TypeError, match="`example` must be a string."):
            get_filepaths(example=invalid_example)
