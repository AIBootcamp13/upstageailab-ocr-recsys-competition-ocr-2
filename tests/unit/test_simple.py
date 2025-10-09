"""
Simple test file to verify pytest setup in the OCR project.
"""


def test_basic_arithmetic():
    """Test basic arithmetic operations."""
    assert 2 + 2 == 4
    assert 5 * 6 == 30
    assert 10 - 4 == 6


def test_string_operations():
    """Test basic string operations."""
    text = "Hello, OCR Project!"
    assert "OCR" in text
    assert text.startswith("Hello")
    assert len(text) > 10


def test_list_operations():
    """Test basic list operations."""
    numbers = [1, 2, 3, 4, 5]
    assert len(numbers) == 5
    assert sum(numbers) == 15
    assert 3 in numbers


if __name__ == "__main__":
    # This allows running the test directly as a script for debugging
    test_basic_arithmetic()
    test_string_operations()
    test_list_operations()
    print("All tests passed!")
