"""
routing.py: Conditional edge decision functions for LangGraph workflow branching.
"""

import sys
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.state import State
from src.exception import MyException
from src.logger import logging
from src.utils.helpers import load_yaml_file

try:
    load_dotenv()

    CONFIG_DIRPATH: Path = (ROOT / "config").resolve()
    PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / "params.yml").resolve()
    PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)

    PATHS_CONFIG: dict[str, Any] = PARAMS.get("paths", {})
    DEFAULT_USE_EXISTING: bool = bool(PATHS_CONFIG.get("use_existing_store", True))

except Exception as e:
    logging.info(f"Failed to initialize routing module dependencies: {e}")
    raise MyException(e, sys) from e


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

    Raises:
        MyException: If routing evaluation fails.
    """
    try:
        configurable: dict[str, Any] = config.get("configurable", {})
        # Skip vector store re-indexing if persistent Chroma store is available and requested
        use_existing: bool = configurable.get("use_existing", DEFAULT_USE_EXISTING)
        return "skip_ingestion" if use_existing else "initiate_ingestion"

    except Exception as e:
        logging.error(f"Failed to evaluate ingestion_decision: {e}")
        raise MyException(e, sys) from e


def route_web_search(state: State) -> Literal["generation", "search_web"]:
    """Determine routing path based on evaluated document relevance.

    Args:
        state (State): Current graph state containing evaluated document groups.

    Returns:
        Literal["generation", "search_web"]: Route to web search if miss or ambiguous documents exist, else generation.

    Raises:
        MyException: If routing evaluation fails.
    """
    try:
        # Core CRAG rule: If any document is classified as miss or ambiguous,
        # fall back to web search to retrieve fresh external context; otherwise generate directly.
        if state.get("miss_documents") or state.get("ambiguous_documents"):
            return "search_web"
        return "generation"

    except Exception as e:
        logging.error(f"Failed to evaluate route_web_search: {e}")
        raise MyException(e, sys) from e


__all__: list[str] = [
    "ingestion_decision",
    "route_web_search",
]
