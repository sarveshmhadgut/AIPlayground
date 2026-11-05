"""
graph.py: LangGraph State definition, nodes, and workflow compilation.
"""

import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.rag import RAGEngine
from src.utils import (
    FILES_DIRPATH,
    PARAMS_FILEPATH,
    PERSIST_DIRPATH,
    PROMPTS_FILEPATH,
    format_documents,
    load_yaml_file,
    save_workflow,
)

__all__: list[str] = [
    "RAG_ENGINE",
    "State",
    "get_workflow",
    "ingestion_decision",
    "ingestion_node",
    "invocation_node",
    "retrieval_node",
    "workflow",
]

PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)
PROMPTS: dict[str, Any] = load_yaml_file(PROMPTS_FILEPATH)

LLM_CONFIG: dict[str, Any] = PARAMS.get("llm", {})
RAG_PROMPTS: dict[str, Any] = PROMPTS.get("rag_generation", {})

SYSTEM_PROMPT: str = str(RAG_PROMPTS.get("system", ""))
USER_PROMPT: str = str(RAG_PROMPTS.get("user", "{query}"))

MODEL: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(**LLM_CONFIG)
PARSER: StrOutputParser = StrOutputParser()
RAG_ENGINE: RAGEngine = RAGEngine()


class State(TypedDict):
    """Schema representing the LangGraph workflow state.

    Attributes:
        query (str): The user's input query.
        context (str): Retrieved context documents as a formatted string.
        response (str): Final generated response from the LLM.
    """

    query: str
    context: str
    response: str


def ingestion_node(
    state: State,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Execute document ingestion pipeline on the centralized RAGEngine.

    Args:
        state (State): Current graph state.
        config (RunnableConfig): Runtime configuration providing optional path overrides.

    Returns:
        dict[str, Any]: Empty state update.
    """
    configurable: dict[str, Any] = config.get("configurable", {})
    files_dirpath: Path | str = configurable.get("files_dirpath", FILES_DIRPATH)
    persist_dirpath: Path | str = configurable.get("persist_dirpath", PERSIST_DIRPATH)

    RAG_ENGINE.update_paths(
        files_dirpath=files_dirpath,
        persist_dirpath=persist_dirpath,
    )
    RAG_ENGINE.run_ingestion_pipeline()
    return {}


def retrieval_node(state: State) -> dict[str, Any]:
    """Retrieve relevant context for the query using the centralized RAGEngine.

    Args:
        state (State): Current graph state containing the query.

    Returns:
        dict[str, Any]: State update containing formatted context.
    """
    retriever: VectorStoreRetriever = RAG_ENGINE.get_retriever()
    retrieved_documents: list[Document] = retriever.invoke(state["query"])
    return {"context": format_documents(retrieved_documents)}


def ingestion_decision(
    state: State,
    config: RunnableConfig,
) -> Literal["skip_ingestion", "initiate_ingestion"]:
    """Determine whether to skip or initiate document ingestion using RunnableConfig.

    Args:
        state (State): Current graph state.
        config (RunnableConfig): Runtime configuration containing configurable parameters.

    Returns:
        Literal["skip_ingestion", "initiate_ingestion"]: Route decision for the next node.
    """
    configurable: dict[str, Any] = config.get("configurable", {})
    use_existing: bool = configurable.get("use_existing", True)
    return "skip_ingestion" if use_existing else "initiate_ingestion"


def invocation_node(state: State) -> dict[str, Any]:
    """Node that invokes the LLM chain with context and query.

    Args:
        state (State): Current graph state containing context and query.

    Returns:
        dict[str, Any]: State update containing the model response.
    """
    prompt: ChatPromptTemplate = ChatPromptTemplate(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]
    )
    chain: RunnableSerializable[dict[str, Any], str] = prompt | MODEL | PARSER
    model_response: str = chain.invoke(
        {"context": state.get("context", ""), "query": state["query"]}
    )
    return {"response": model_response}


def get_workflow(
    rag_engine: RAGEngine | None = None,
    save_workflow_file: bool = False,
) -> CompiledStateGraph:
    """Build and compile the LangGraph workflow.

    Args:
        rag_engine (RAGEngine | None): Optional custom RAGEngine instance. Defaults to centralized RAG_ENGINE.
        save_workflow_file (bool): Whether to serialize workflow to files/workflow_viewer.pkl. Defaults to False.

    Returns:
        CompiledStateGraph: Compiled runnable workflow graph.
    """
    global RAG_ENGINE
    if rag_engine is not None:
        RAG_ENGINE = rag_engine

    graph: StateGraph = StateGraph(State)

    graph.add_node("ingestion", ingestion_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("invocation", invocation_node)

    graph.add_conditional_edges(
        START,
        ingestion_decision,
        {
            "initiate_ingestion": "ingestion",
            "skip_ingestion": "retrieval",
        },
    )
    graph.add_edge("ingestion", "retrieval")
    graph.add_edge("retrieval", "invocation")
    graph.add_edge("invocation", END)

    compiled_workflow: CompiledStateGraph = graph.compile()
    if save_workflow_file:
        save_workflow(compiled_workflow)
    return compiled_workflow


workflow: CompiledStateGraph = get_workflow()
