"""
tools.py: External tool definitions for web search and retrieval augmentation.
"""

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore[no-redef,assignment]
from langchain_core.tools import tool

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exception import MyException
from src.logger import logging
from src.utils.display import get_terminal_width
from src.utils.helpers import load_yaml_file

try:
    load_dotenv()

    WIDTH: int = get_terminal_width()

    CONFIG_DIRPATH: Path = (ROOT / "config").resolve()
    PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / "params.yml").resolve()
    PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)

    WEB_SEARCH_CONFIG: dict[str, Any] = PARAMS.get("web_search", {})
    REGION: str = str(WEB_SEARCH_CONFIG.get("region", "us-en"))
    MAX_RESULTS: int = int(WEB_SEARCH_CONFIG.get("max_results", 5))

    NO_RESULTS_INDICATORS: tuple[str, ...] = (
        "no results found",
        "no good duckduckgo search result was found",
    )

    TRANSIENT_NETWORK_INDICATORS: tuple[str, ...] = (
        "dnserror",
        "connectionerror",
        "connecterror",
        "connection refused",
        "timeout",
        "ratelimit",
        "429",
        "too many requests",
        "name resolution",
        "no records found",
        "ssl error",
        "max retries exceeded",
    )

except Exception as e:
    logging.info(f"Failed to initialize web search module dependencies: {e}")
    raise MyException(e, sys) from e


@tool(name_or_callable="web_search")
def web_search(search_query: str) -> dict[str, Any]:
    """Search the web for information related to a query using DuckDuckGo.

    Args:
        search_query (str): The search query to look up on the web.

    Returns:
        dict[str, Any]: A dictionary containing the query, execution status, and response content.

    Raises:
        MyException: If the web search process fails.
    """
    try:
        logging.info(f"Executing web_search tool for {search_query}...")

        web_search_config: dict[str, Any] = PARAMS.get("web_search", {})
        region: str = str(web_search_config.get("region", REGION))
        max_results: int = int(web_search_config.get("max_results", MAX_RESULTS))

        with DDGS() as ddgs:
            raw_results: Any = ddgs.text(search_query, region=region, max_results=max_results)

        if isinstance(raw_results, str):
            response_text: str = raw_results.strip()
        elif isinstance(raw_results, list):
            snippets: list[str] = [
                str(item.get("body", item)).strip() if isinstance(item, dict) else str(item).strip()
                for item in raw_results
                if item
            ]
            response_text = "\n\n".join(snippet for snippet in snippets if snippet)
        else:
            response_text = str(raw_results).strip() if raw_results is not None else ""

        # Gracefully treat empty search results or negative indicator strings as empty context
        if not response_text or any(indicator in response_text.lower() for indicator in NO_RESULTS_INDICATORS):
            logging.info(f"No web search results found for query: {search_query}")
            return {
                "status": "no_results",
                "query": search_query,
                "response": "",
            }

        logging.info(f"Finished executing web_search tool for {search_query}.")
        return {
            "status": "success",
            "query": search_query,
            "response": response_text,
        }

    except Exception as e:
        error_message: str = str(e).strip().lower()
        if any(indicator in error_message for indicator in NO_RESULTS_INDICATORS):
            logging.info(f"No web search results found for query '{search_query}': {e}")
            return {
                "status": "no_results",
                "query": search_query,
                "response": "",
            }

        # Graceful network degradation: catch transient DNS, timeouts, or rate limits without crashing the graph
        if any(indicator in error_message for indicator in TRANSIENT_NETWORK_INDICATORS):
            logging.warning(
                f"Web search encountered transient network failure for query '{search_query}': {e}. "
                "Gracefully returning empty search context."
            )
            return {
                "status": "network_error",
                "query": search_query,
                "response": "",
                "error": str(e),
            }

        logging.error(f"Failed to execute web_search tool: {e}")
        raise MyException(e, sys) from e


__all__: list[str] = [
    "web_search",
]
