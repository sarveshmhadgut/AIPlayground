"""
test_graph.py: Unit tests for workflow compilation and graph structure.
"""

import sys
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

ROOT: Path = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.graph import (
    get_workflow,
    workflow,
)


def test_workflow_compilation() -> None:
    """Test compiled workflow is an instance of CompiledStateGraph."""
    assert isinstance(workflow, CompiledStateGraph)

    fresh_workflow = get_workflow()
    assert isinstance(fresh_workflow, CompiledStateGraph)


def test_workflow_nodes_presence() -> None:
    """Test all expected nodes are registered in the compiled workflow."""
    raw_graph = workflow.get_graph()
    node_names = set(raw_graph.nodes.keys())

    expected_nodes = {
        "ingestion",
        "retrieval",
        "evaluation",
        "web_search",
        "generation",
    }
    assert expected_nodes.issubset(node_names)
