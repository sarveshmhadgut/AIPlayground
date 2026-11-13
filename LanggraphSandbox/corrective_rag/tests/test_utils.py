"""
test_utils.py: Unit tests for helper utilities, text formatting, and UUID generation.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.documents.base import Document

ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exception import ValidationError
from src.utils.helpers import (
    build_scoring_rubric,
    format_documents,
    generate_uuid,
    load_yaml_file,
)


def test_build_scoring_rubric_dynamic() -> None:
    """Test dynamic generation of scoring rubric with custom runtime thresholds."""
    rubric: str = build_scoring_rubric(hit_min_threshold=8.0, miss_max_threshold=4.0)
    assert "8.0 to 10.0 (Hit / High Relevance)" in rubric
    assert "4.0 to 7.9 (Ambiguous / Partial Relevance)" in rubric
    assert "0.0 to 3.9 (Miss / Irrelevant)" in rubric


def test_generate_uuid_deterministic() -> None:
    """Test deterministic UUID5 generation from consistent text input."""
    uuid_1: str = generate_uuid(name="sample document text chunk")
    uuid_2: str = generate_uuid(name="sample document text chunk")
    uuid_diff: str = generate_uuid(name="different text")

    assert uuid_1 == uuid_2
    assert uuid_1 != uuid_diff


def test_generate_uuid_hex_format() -> None:
    """Test UUID generation as 32-character hex without hyphens."""
    uuid_hex: str = generate_uuid(name="test", as_hex=True)
    assert len(uuid_hex) == 32
    assert "-" not in uuid_hex


def test_generate_uuid_missing_name_raises_validation_error() -> None:
    """Test generate_uuid raises ValidationError when neither name nor text is provided."""
    with pytest.raises(ValidationError):
        generate_uuid()


def test_format_documents() -> None:
    """Test format_documents concatenates document contents with spaces."""
    documents: list[Document] = [
        Document(page_content="First page content."),
        Document(page_content="Second page content."),
    ]
    result: str = format_documents(documents)
    assert result == "First page content. Second page content."


def test_load_yaml_file_missing_returns_empty_dict(tmp_path: Path) -> None:
    """Test load_yaml_file returns an empty dictionary when the file does not exist."""
    missing_path = tmp_path / "non_existent.yml"
    result = load_yaml_file(missing_path)
    assert result == {}
