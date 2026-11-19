"""
graph.py: LangGraph workflow assembly, compilation, and execution interface.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.nodes import (
    evaluation_node,
    generation_node,
    ingestion_node,
    rag_query_rewriting_node,
    refinement_node,
    retrieval_node,
    set_rag_engine,
    web_query_rewriting_node,
    web_search_node,
)
from src.core.rag import RAGEngine
from src.core.routing import (
    route_ingestion,
    route_web_search,
)
from src.core.state import State
from src.exception import MyException
from src.logger import logging
from src.utils.serialization import save_workflow


def get_workflow(
    rag_engine: RAGEngine | None = None,
    save_workflow_file: bool = False,
) -> CompiledStateGraph:
    """Build and compile the Corrective RAG LangGraph workflow.

    Args:
        rag_engine (RAGEngine | None): Optional custom RAGEngine instance. Defaults to centralized RAG_ENGINE.
        save_workflow_file (bool): Whether to serialize workflow to files/workflow_viewer.pkl. Defaults to False.

    Returns:
        CompiledStateGraph: Compiled runnable workflow graph.

    Raises:
        MyException: If workflow assembly or compilation fails.
    """
    try:
        if rag_engine is not None:
            set_rag_engine(rag_engine)

        graph: StateGraph = StateGraph(State)

        graph.add_node("ingestion", ingestion_node)
        graph.add_node("rag_query_rewriting", rag_query_rewriting_node)
        graph.add_node("retrieval", retrieval_node)
        graph.add_node("evaluation", evaluation_node)
        graph.add_node("web_query_rewriting", web_query_rewriting_node)
        graph.add_node("web_search", web_search_node)
        graph.add_node("refinement", refinement_node)
        graph.add_node("generation", generation_node)

        graph.add_conditional_edges(
            START,
            route_ingestion,
            {
                "initiate_ingestion": "ingestion",
                "skip_ingestion": "rag_query_rewriting",
            },
        )

        graph.add_edge("ingestion", "rag_query_rewriting")
        graph.add_edge("rag_query_rewriting", "retrieval")
        graph.add_edge("retrieval", "evaluation")

        graph.add_conditional_edges(
            "evaluation",
            route_web_search,
            {
                "refinement": "refinement",
                "search_web": "web_query_rewriting",
            },
        )
        graph.add_edge("web_query_rewriting", "web_search")
        graph.add_edge("web_search", "refinement")
        graph.add_edge("refinement", "generation")
        graph.add_edge("generation", END)

        compiled_workflow: CompiledStateGraph = graph.compile()
        if save_workflow_file:
            save_workflow(compiled_workflow)
        return compiled_workflow

    except Exception as e:
        logging.error(f"Failed to assemble workflow: {e}")
        raise MyException(e, sys) from e


try:
    load_dotenv()

    workflow: CompiledStateGraph = get_workflow()

except Exception as e:
    logging.info(f"Failed to initialize graph module dependencies: {e}")
    raise MyException(e, sys) from e

__all__: list[str] = [
    "get_workflow",
    "workflow",
]
