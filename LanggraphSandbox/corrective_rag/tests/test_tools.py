"""
test_tools.py: Unit tests for external tools, including web search.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.tools import (
    simplify_search_query,
    web_search,
)
from src.exception import MyException


def _mock_ddgs(return_value: Any = None, side_effect: Any = None) -> tuple[Any, MagicMock]:
    """Helper to patch DDGS context manager and its text method."""
    mock_instance = MagicMock()
    mock_instance.__enter__.return_value = mock_instance
    if side_effect is not None:
        mock_instance.text.side_effect = side_effect
    else:
        mock_instance.text.return_value = return_value
    patcher = patch("src.core.tools.DDGS", return_value=mock_instance)
    return patcher, mock_instance


def test_web_search_success() -> None:
    """Test web_search tool returns the expected dictionary structure on successful invocation."""
    mock_body: str = "Python is a high-level programming language."
    patcher, _ = _mock_ddgs(return_value=[{"body": mock_body}])
    with patcher:
        result: dict[str, Any] = web_search.invoke({"search_query": "Python"})

        assert result.get("status") == "success"
        assert result.get("query") == "Python"
        assert result.get("response") == mock_body


def test_web_search_uses_configured_region() -> None:
    """Test web_search invokes DDGS.text with configured region and max_results."""
    patcher, mock_instance = _mock_ddgs(return_value=[{"body": "Search result"}])
    with patcher:
        result: dict[str, Any] = web_search.invoke({"search_query": "Test query"})

        mock_instance.text.assert_called_with("Test query", region="us-en", max_results=5, backend="auto")
        assert result.get("status") == "success"


def test_web_search_raises_custom_exception_on_failure() -> None:
    """Test web_search wraps unhandled exceptions in MyException."""
    patcher, _ = _mock_ddgs(side_effect=RuntimeError("Search service unavailable"))
    with patcher, pytest.raises(MyException):
        web_search.invoke({"search_query": "Failing query"})


def test_web_search_handles_no_results_exception() -> None:
    """Test web_search returns no_results status when DDGS raises No results found exception."""
    patcher, _ = _mock_ddgs(side_effect=Exception("No results found."))
    with patcher:
        result: dict[str, Any] = web_search.invoke({"search_query": "obscure query"})

        assert result.get("status") == "no_results"
        assert result.get("query") == "obscure query"
        assert result.get("response") == ""


def test_web_search_handles_no_good_result_string() -> None:
    """Test web_search returns no_results status when DDGS returns fallback string."""
    fallback_response: str = "No good DuckDuckGo Search Result was found"
    patcher, _ = _mock_ddgs(return_value=[{"body": fallback_response}])
    with patcher:
        result: dict[str, Any] = web_search.invoke({"search_query": "obscure query"})

        assert result.get("status") == "no_results"
        assert result.get("query") == "obscure query"
        assert result.get("response") == ""


def test_web_search_handles_empty_string_response() -> None:
    """Test web_search returns no_results status when DDGS returns empty results."""
    patcher, _ = _mock_ddgs(return_value=[])
    with patcher:
        result: dict[str, Any] = web_search.invoke({"search_query": "empty query"})

        assert result.get("status") == "no_results"
        assert result.get("query") == "empty query"
        assert result.get("response") == ""


def test_web_search_handles_dns_error_gracefully() -> None:
    """Test web_search returns network_error status and empty response on DNSError."""
    dns_error_msg: str = "DNSError: no records found for Query wt.wikipedia.org"
    patcher, _ = _mock_ddgs(side_effect=Exception(dns_error_msg))
    with patcher:
        result: dict[str, Any] = web_search.invoke({"search_query": "What is jev?"})

        assert result.get("status") == "network_error"
        assert result.get("query") == "What is jev?"
        assert result.get("response") == ""
        assert dns_error_msg in str(result.get("error"))


def test_web_search_handles_timeout_error_gracefully() -> None:
    """Test web_search returns network_error status and empty response on timeout."""
    patcher, _ = _mock_ddgs(side_effect=Exception("ReadTimeoutError: connection timed out"))
    with patcher:
        result: dict[str, Any] = web_search.invoke({"search_query": "timeout test"})

        assert result.get("status") == "network_error"
        assert result.get("query") == "timeout test"
        assert result.get("response") == ""


def test_simplify_search_query_transformations() -> None:
    """Test simplify_search_query strips conversational prefixes and question words."""
    assert simplify_search_query("How does the Jev model work?") == "Jev model"
    assert simplify_search_query("What is reinforcement learning?") == "reinforcement learning"
    assert simplify_search_query("Explain the backpropagation algorithm") == "backpropagation algorithm"
    assert simplify_search_query("Why does dropout prevent overfitting?") == "dropout prevent overfitting"
    assert simplify_search_query("Transformer architecture") == "Transformer architecture"


def test_web_search_retries_with_simplified_query() -> None:
    """Test web_search falls back to simplified keyword query when raw conversational query fails."""
    mock_instance = MagicMock()
    mock_instance.__enter__.return_value = mock_instance

    def mock_text(query: str, **kwargs: Any) -> list[dict[str, str]]:
        if query == "How does the Jev model work?":
            raise Exception("No results found.")
        if query == "Jev model":
            return [{"body": "The Jev model is a system-one decision model."}]
        return []

    mock_instance.text.side_effect = mock_text

    with patch("src.core.tools.DDGS", return_value=mock_instance):
        result: dict[str, Any] = web_search.invoke({"search_query": "How does the Jev model work?"})

        assert result.get("status") == "success"
        assert result.get("query") == "Jev model"
        assert "The Jev model is a system-one decision model." in str(result.get("response"))
