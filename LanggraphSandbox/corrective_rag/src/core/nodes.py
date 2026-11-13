"""
nodes.py: Execution node functions for the Corrective RAG LangGraph workflow.
"""

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.rag import RAGEngine
from src.core.state import (
    DocumentGrade,
    EvaluationSchema,
    State,
)
from src.core.tools import web_search
from src.exception import MyException
from src.logger import logging
from src.utils.display import get_terminal_width
from src.utils.helpers import build_scoring_rubric, load_yaml_file

try:
    load_dotenv()

    WIDTH: int = get_terminal_width()

    CONFIG_DIRPATH: Path = (ROOT / "config").resolve()
    PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / "params.yml").resolve()
    PROMPTS_FILEPATH: Path = (CONFIG_DIRPATH / "prompts.yml").resolve()

    PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)
    PROMPTS: dict[str, Any] = load_yaml_file(PROMPTS_FILEPATH)

    PATHS_CONFIG: dict[str, Any] = PARAMS.get("paths", {})
    FILES_DIRNAME: str = str(PATHS_CONFIG.get("files_dir", "files"))
    PERSIST_DIRNAME: str = str(PATHS_CONFIG.get("persist_directory", "chroma_store"))

    DOCUMENTS_DIRPATH: Path = (ROOT / FILES_DIRNAME / "documents").resolve()
    PERSIST_DIRPATH: Path = (ROOT / PERSIST_DIRNAME).resolve()

    TRACING_CONFIG: dict[str, Any] = PARAMS.get("tracing", {})
    if TRACING_CONFIG.get("enabled", True):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault(
            "LANGCHAIN_PROJECT",
            str(TRACING_CONFIG.get("project", "corrective-rag")),
        )

    LLM_CONFIG: dict[str, Any] = PARAMS.get("llm", {})

    EVALUATION_PARAMS: dict[str, Any] = PARAMS.get("evaluation", {})
    MISS_MAX_THRESHOLD: float = float(EVALUATION_PARAMS.get("miss_max_threshold", 3.5))
    HIT_MIN_THRESHOLD: float = float(EVALUATION_PARAMS.get("hit_min_threshold", 7.5))

    EVALUATION_SYSTEM_PROMPT: str = str(PROMPTS.get("document_evaluation", {}).get("system", ""))
    EVALUATION_USER_PROMPT: str = str(PROMPTS.get("document_evaluation", {}).get("user", "{query}"))

    EVALUATION_PROMPT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("system", EVALUATION_SYSTEM_PROMPT),
            ("user", EVALUATION_USER_PROMPT),
        ]
    )

    GENERATION_SYSTEM_PROMPT: str = str(PROMPTS.get("generation", {}).get("system", ""))
    GENERATION_USER_PROMPT: str = str(PROMPTS.get("generation", {}).get("user", "{query}"))

    GENERATION_PROMPT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATION_SYSTEM_PROMPT),
            ("user", GENERATION_USER_PROMPT),
        ]
    )

    MODEL: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(**LLM_CONFIG)
    PARSER: StrOutputParser = StrOutputParser()
    RAG_ENGINE: RAGEngine = RAGEngine()

except Exception as e:
    logging.info(f"Failed to initialize nodes module dependencies: {e}")
    raise MyException(e, sys) from e


def set_rag_engine(rag_engine: RAGEngine) -> None:
    """Set the centralized RAGEngine instance across the core pipeline.

    Args:
        rag_engine (RAGEngine): The active RAGEngine instance to configure.
    """
    global RAG_ENGINE
    RAG_ENGINE = rag_engine


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

    Raises:
        MyException: If document ingestion fails.
    """
    try:
        configurable: dict[str, Any] = config.get("configurable", {})
        files_dirpath: Path | str = configurable.get("files_dirpath", DOCUMENTS_DIRPATH)
        persist_dirpath: Path | str = configurable.get("persist_dirpath", PERSIST_DIRPATH)

        RAG_ENGINE.update_paths(
            files_dirpath=files_dirpath,
            persist_dirpath=persist_dirpath,
        )
        RAG_ENGINE.run_ingestion_pipeline()
        return {}
    except Exception as e:
        logging.error(f"Failed to execute ingestion_node: {e}")
        raise MyException(e, sys) from e


def retrieval_node(
    state: State,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Retrieve relevant context for the query using the centralized RAGEngine.

    Args:
        state (State): Current graph state containing the query.
        config (RunnableConfig | None): Runtime configuration for tracing propagation.

    Returns:
        dict[str, Any]: State update containing formatted context.

    Raises:
        MyException: If vector retrieval fails.
    """
    try:
        retriever: VectorStoreRetriever = RAG_ENGINE.get_retriever()
        retrieved_documents: list[Document] = retriever.invoke(
            state.get("query", ""),
            config={"run_name": "vectorstore_retrieval"},
        )
        return {
            "retrieved_documents": retrieved_documents,
        }
    except Exception as e:
        logging.error(f"Failed to execute retrieval_node: {e}")
        raise MyException(e, sys) from e


def evaluation_node(
    state: State,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Grade all retrieved documents in a single batch into hits, misses, and ambiguous groups.

    Args:
        state (State): Current graph state containing query and retrieved documents.
        config (RunnableConfig): Runtime configuration providing grading thresholds.

    Returns:
        dict[str, Any]: State update containing hit, miss, and ambiguous document groups.

    Raises:
        MyException: If document grading fails.
    """
    try:
        retrieved_documents: list[Document] = state.get("retrieved_documents", [])
        if not retrieved_documents:
            return {
                "hit_documents": [],
                "miss_documents": [],
                "ambiguous_documents": [],
                "rag_context": "",
            }

        structured_evaluation_model: Runnable[Any, Any] = MODEL.with_structured_output(schema=EvaluationSchema)
        evaluation_chain: Runnable[dict[str, Any], Any] = (
            EVALUATION_PROMPT_TEMPLATE | structured_evaluation_model
        ).with_config(run_name="document_evaluation")

        configurable: dict[str, Any] = config.get("configurable", {}) if config else {}
        hit_min_threshold: float = float(configurable.get("hit_min_threshold", HIT_MIN_THRESHOLD))
        miss_max_threshold: float = float(configurable.get("miss_max_threshold", MISS_MAX_THRESHOLD))

        # Dynamically compile scoring rubric using runtime configurable thresholds
        scoring_rubric: str = build_scoring_rubric(
            hit_min_threshold=hit_min_threshold,
            miss_max_threshold=miss_max_threshold,
        )

        numbered_chunks: list[str] = [
            f"[Document {index}]\n{getattr(document, 'page_content', str(document))}"
            for index, document in enumerate(retrieved_documents, start=1)
        ]
        documents_text: str = "\n\n".join(numbered_chunks)

        result: EvaluationSchema = evaluation_chain.invoke(
            input={
                "query": state.get("query", ""),
                "documents_text": documents_text,
                "scoring_rubric": scoring_rubric,
            },
            config=config,
        )

        evaluations: list[DocumentGrade] = getattr(result, "evaluations", []) or []
        score_map: dict[int, float] = {
            item.document_index: float(item.score)
            for item in evaluations
            if getattr(item, "document_index", None) is not None
        }

        hit_documents: list[Document] = []
        miss_documents: list[Document] = []
        ambiguous_documents: list[Document] = []

        # Calibrate documents into 3-tier relevance buckets based on rubric score thresholds
        for index, document in enumerate(retrieved_documents, start=1):
            score: float = score_map.get(index, 0.0)
            document.metadata["score"] = score

            if score >= hit_min_threshold:
                hit_documents.append(document)
            elif score <= miss_max_threshold:
                miss_documents.append(document)
            else:
                ambiguous_documents.append(document)

        # Filter out misses; retain hits and ambiguous documents as usable RAG context
        rag_context: str = "\n\n".join([document.page_content for document in (hit_documents + ambiguous_documents)])
        return {
            "hit_documents": hit_documents,
            "miss_documents": miss_documents,
            "ambiguous_documents": ambiguous_documents,
            "rag_context": rag_context,
        }

    except Exception as e:
        logging.error(f"Failed to execute evaluation_node: {e}")
        raise MyException(e, sys) from e


def web_search_node(
    state: State,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Execute web search to augment context when retrieved documents are insufficient.

    Args:
        state (State): Current graph state containing the user query and context.
        config (RunnableConfig | None): Runtime configuration for tracing propagation.

    Returns:
        dict[str, Any]: State update containing updated context with web search results.

    Raises:
        MyException: If web search execution fails.
    """
    try:
        query: str = state.get("query", "")
        search_result: dict[str, Any] = web_search.invoke(
            {"search_query": query},
            config={"run_name": "duckduckgo_web_search"},
        )
        web_content: str = str(search_result.get("response", "")).strip()

        return {"web_search_context": web_content}

    except Exception as e:
        logging.error(f"Failed to execute web_search_node: {e}")
        raise MyException(e, sys) from e


def generation_node(
    state: State,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Node that generates the final response using RAG context, web search context, and query.

    Args:
        state (State): Current graph state containing rag_context, web_search_context, and query.
        config (RunnableConfig | None): Runtime configuration for tracing propagation.

    Returns:
        dict[str, Any]: State update containing the model response.

    Raises:
        MyException: If generation chain fails.
    """
    try:
        generation_chain: Runnable[dict[str, Any], str] = (GENERATION_PROMPT_TEMPLATE | MODEL | PARSER).with_config(
            run_name="llm_generation"
        )

        model_response: str = generation_chain.invoke(
            {
                "rag_context": state.get("rag_context", ""),
                "web_search_context": state.get("web_search_context", ""),
                "query": state.get("query", ""),
            },
            config=config,
        )
        return {"response": model_response}

    except Exception as e:
        logging.error(f"Failed to execute generation_node: {e}")
        raise MyException(e, sys) from e


__all__: list[str] = [
    "evaluation_node",
    "generation_node",
    "ingestion_node",
    "retrieval_node",
    "set_rag_engine",
    "web_search_node",
]
