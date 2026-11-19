"""
test_routing.py: Unit tests for conditional routing edge decisions.
"""

import sys
from pathlib import Path

from langchain_core.documents.base import Document

ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.routing import (
    route_ingestion,
    route_web_search,
)
from src.core.state import State


def test_route_ingestion_skip() -> None:
    """Test route_ingestion defaults to skip_ingestion when use_existing is True."""
    state: State = {
        "query": "test query",
        "retrieved_documents": [],
        "hit_documents": [],
        "miss_documents": [],
        "ambiguous_documents": [],
        "web_search_context": "",
        "rag_context": "",
        "response": "",
    }
    decision = route_ingestion(state, config={"configurable": {"use_existing": True}})
    assert decision == "skip_ingestion"


def test_route_ingestion_initiate() -> None:
    """Test route_ingestion routes to initiate_ingestion when use_existing is False."""
    state: State = {
        "query": "test query",
        "retrieved_documents": [],
        "hit_documents": [],
        "miss_documents": [],
        "ambiguous_documents": [],
        "web_search_context": "",
        "rag_context": "",
        "response": "",
    }
    decision = route_ingestion(state, config={"configurable": {"use_existing": False}})
    assert decision == "initiate_ingestion"


def test_route_web_search_with_hits() -> None:
    """Test route_web_search branches to refinement when only hit_documents are present."""
    document = Document(page_content="Relevant document content.", metadata={"score": 8.5})
    state: State = {
        "query": "test query",
        "retrieved_documents": [document],
        "hit_documents": [document],
        "miss_documents": [],
        "ambiguous_documents": [],
        "web_search_context": "",
        "rag_context": "Relevant document content.",
        "response": "",
    }
    route = route_web_search(state)
    assert route == "refinement"


def test_route_web_search_without_hits() -> None:
    """Test route_web_search branches to search_web when hit_documents is empty."""
    document = Document(page_content="Irrelevant document content.", metadata={"score": 2.0})
    state: State = {
        "query": "test query",
        "retrieved_documents": [document],
        "hit_documents": [],
        "miss_documents": [document],
        "ambiguous_documents": [],
        "web_search_context": "",
        "rag_context": "",
        "response": "",
    }
    route = route_web_search(state)
    assert route == "search_web"


def test_route_web_search_empty_retrieval() -> None:
    """Test route_web_search branches to search_web when no documents are retrieved."""
    state: State = {
        "query": "test query",
        "retrieved_documents": [],
        "hit_documents": [],
        "miss_documents": [],
        "ambiguous_documents": [],
        "web_search_context": "",
        "rag_context": "",
        "response": "",
    }
    route = route_web_search(state)
    assert route == "search_web"
