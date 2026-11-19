"""
tools.py: External tool definitions for web search and retrieval augmentation.
"""

import re
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


QUESTION_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:how\s+(?:does|do|can|is|are)\s+(?:the\s+)?|what\s+(?:is|are)\s+(?:the\s+)?|"
        r"why\s+(?:is|are|do|does)\s+(?:the\s+)?|tell\s+me\s+about\s+(?:the\s+)?|"
        r"explain\s+(?:the\s+)?|describe\s+(?:the\s+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:\s+work|\s+function|\s+operate|\s+meaning)?\s*\?*$", re.IGNORECASE),
)


def simplify_search_query(query: str) -> str:
    """Simplify a conversational natural language query into concise search keywords.

    Args:
        query (str): The raw conversational query text.

    Returns:
        str: Cleaned search keywords optimized for search engines.
    """
    clean_query: str = query.strip()
    for pattern in QUESTION_PREFIX_PATTERNS:
        clean_query = pattern.sub("", clean_query).strip()
    clean_query = re.sub(r"[^\w\s-]", " ", clean_query)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()
    return clean_query if clean_query else query


def _query_ddgs(
    search_query: str,
    region: str,
    max_results: int,
    backend: str = "auto",
) -> str:
    """Execute raw DDGS query and extract text snippets.

    Args:
        search_query (str): The search query.
        region (str): Region code.
        max_results (int): Max search results.
        backend (str): Search backend (e.g. 'auto', 'lite').

    Returns:
        str: Extracted search snippets joined by newlines.
    """
    with DDGS() as ddgs:
        raw_results: Any = ddgs.text(
            search_query,
            region=region,
            max_results=max_results,
            backend=backend,
        )

    if isinstance(raw_results, str):
        return raw_results.strip()
    if isinstance(raw_results, list):
        snippets: list[str] = [
            str(item.get("body", item.get("description", item))).strip()
            if isinstance(item, dict)
            else str(item).strip()
            for item in raw_results
            if item
        ]
        return "\n\n".join(snippet for snippet in snippets if snippet)
    return str(raw_results).strip() if raw_results is not None else ""


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

        candidate_queries: list[str] = [search_query]
        simplified: str = simplify_search_query(search_query)
        if simplified and simplified.lower() != search_query.lower():
            candidate_queries.append(simplified)

        response_text: str = ""
        successful_query: str = search_query
        last_network_error: Exception | None = None

        for query_candidate in candidate_queries:
            for backend in ("auto", "lite"):
                try:
                    response_text = _query_ddgs(
                        search_query=query_candidate,
                        region=region,
                        max_results=max_results,
                        backend=backend,
                    )
                    if response_text and not any(
                        indicator in response_text.lower() for indicator in NO_RESULTS_INDICATORS
                    ):
                        successful_query = query_candidate
                        last_network_error = None
                        break
                except Exception as ex:
                    err_lower: str = str(ex).lower()
                    if any(indicator in err_lower for indicator in NO_RESULTS_INDICATORS):
                        continue
                    if any(indicator in err_lower for indicator in TRANSIENT_NETWORK_INDICATORS):
                        last_network_error = ex
                        continue
                    raise ex

            if response_text and not any(indicator in response_text.lower() for indicator in NO_RESULTS_INDICATORS):
                break

        # If network failures occurred and no successful response was obtained, report network_error
        if not response_text and last_network_error is not None:
            logging.warning(
                f"Web search encountered transient network failure for query '{search_query}': {last_network_error}. "
                "Gracefully returning empty search context."
            )
            return {
                "status": "network_error",
                "query": search_query,
                "response": "",
                "error": str(last_network_error),
            }

        # Gracefully treat empty search results or negative indicator strings as empty context
        if not response_text or any(indicator in response_text.lower() for indicator in NO_RESULTS_INDICATORS):
            logging.info(f"No web search results found for query: {search_query}")
            return {
                "status": "no_results",
                "query": search_query,
                "response": "",
            }

        logging.info(f"Finished executing web_search tool for {successful_query}.")
        return {
            "status": "success",
            "query": successful_query,
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
    "simplify_search_query",
    "web_search",
]
