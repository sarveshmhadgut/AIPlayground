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


def route_ingestion(
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
        logging.error(f"Failed to evaluate route_ingestion: {e}")
        raise MyException(e, sys) from e


def route_web_search(state: State) -> Literal["refinement", "search_web"]:
    """Determine routing path based on evaluated document relevance.

    Args:
        state (State): Current graph state containing evaluated document groups.

    Returns:
        Literal["refinement", "search_web"]: Route to web search if miss or ambiguous documents exist
            or no hit documents are available, else refinement.

    Raises:
        MyException: If routing evaluation fails.
    """
    try:
        # Core CRAG rule: If there are no high-relevance hit documents, or if any document
        # is classified as miss or ambiguous, fall back to web search; otherwise refine directly.
        if not state.get("hit_documents") or state.get("miss_documents") or state.get("ambiguous_documents"):
            return "search_web"
        return "refinement"

    except Exception as e:
        logging.error(f"Failed to evaluate route_web_search: {e}")
        raise MyException(e, sys) from e


__all__: list[str] = [
    "route_ingestion",
    "route_web_search",
]
