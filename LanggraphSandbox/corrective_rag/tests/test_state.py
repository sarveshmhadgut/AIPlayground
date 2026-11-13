"""
test_state.py: Unit tests for state schemas, document models, and LangGraph State.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.state import (
    DocumentGrade,
    EvaluationSchema,
    State,
)


def test_document_grade_valid() -> None:
    """Test valid DocumentGrade instantiation and constraints."""
    grade = DocumentGrade(document_index=1, score=8.5)
    assert grade.document_index == 1
    assert grade.score == 8.5


def test_document_grade_out_of_bounds() -> None:
    """Test DocumentGrade out-of-bounds score and index validation."""
    with pytest.raises(PydanticValidationError):
        DocumentGrade(document_index=0, score=5.0)

    with pytest.raises(PydanticValidationError):
        DocumentGrade(document_index=1, score=10.5)

    with pytest.raises(PydanticValidationError):
        DocumentGrade(document_index=1, score=-0.1)


def test_evaluation_schema() -> None:
    """Test EvaluationSchema serialization with document grades."""
    evaluations: list[DocumentGrade] = [
        DocumentGrade(document_index=1, score=9.0),
        DocumentGrade(document_index=2, score=2.0),
    ]
    batch = EvaluationSchema(evaluations=evaluations)
    assert len(batch.evaluations) == 2
    assert batch.evaluations[0].score == 9.0


def test_state_typed_dict_keys() -> None:
    """Test State TypedDict annotations contain required keys."""
    annotations = State.__annotations__
    required_keys = {
        "query",
        "retrieved_documents",
        "hit_documents",
        "miss_documents",
        "ambiguous_documents",
        "web_search_context",
        "rag_context",
        "response",
    }
    assert required_keys.issubset(annotations.keys())
