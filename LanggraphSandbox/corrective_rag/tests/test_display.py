"""
test_display.py: Unit tests for terminal display, divider lines, and header formatting.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exception import MyException
from src.utils.display import (
    get_terminal_width,
    print_divider,
    print_header,
)


def test_get_terminal_width_returns_positive_integer() -> None:
    """Test get_terminal_width returns a valid positive integer column width."""
    width: int = get_terminal_width(fallback=80)
    assert isinstance(width, int)
    assert width > 0


def test_get_terminal_width_uses_mocked_terminal_size() -> None:
    """Test get_terminal_width retrieves columns from shutil.get_terminal_size."""
    mock_size: MagicMock = MagicMock(columns=120, lines=40)
    with patch("shutil.get_terminal_size", return_value=mock_size):
        width: int = get_terminal_width()
        assert width == 120


def test_get_terminal_width_invalid_fallback_raises_exception() -> None:
    """Test get_terminal_width raises MyException when fallback is non-positive."""
    with pytest.raises(MyException):
        get_terminal_width(fallback=0)

    with pytest.raises(MyException):
        get_terminal_width(fallback=-10)


def test_print_divider_outputs_expected_character_count(capsys: pytest.CaptureFixture[str]) -> None:
    """Test print_divider outputs a string containing the exact divider width."""
    print_divider(width=50, char="-", color="grey")
    captured: str = capsys.readouterr().out
    assert "-" * 50 in captured


def test_print_divider_raises_exception_on_empty_char() -> None:
    """Test print_divider raises MyException when char argument is empty."""
    with pytest.raises(MyException):
        print_divider(width=50, char="")


def test_print_header_outputs_centered_title_and_divider(capsys: pytest.CaptureFixture[str]) -> None:
    """Test print_header prints the title centered within the specified width."""
    title: str = "Test Section"
    width: int = 60
    print_header(title=title, width=width, text_color="blue", line_color="grey")
    captured: str = capsys.readouterr().out

    assert title in captured
    assert "_" * width in captured


def test_print_header_raises_exception_on_none_title() -> None:
    """Test print_header raises MyException when title is None."""
    with pytest.raises(MyException):
        print_header(title=None)  # type: ignore[arg-type]
