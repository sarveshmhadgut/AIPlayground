"""
artifacts.py: Run artifact persistence, structured directory exports, catalog generation, and retention pruning.
"""

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exception import MyException
from src.logger import logging
from src.utils.helpers import load_yaml_file

try:
    load_dotenv()

    CONFIG_DIRPATH: Path = (ROOT / "config").resolve()
    PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / "params.yml").resolve()
    PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)

    PATHS_CONFIG: dict[str, Any] = PARAMS.get("paths", {})
    FILES_DIRNAME: str = str(PATHS_CONFIG.get("files_dir", "files"))
    ARTIFACTS_DIRNAME: str = str(PATHS_CONFIG.get("artifacts_dir", "artifacts"))

    FILES_DIRPATH: Path = (ROOT / FILES_DIRNAME).resolve()
    ARTIFACTS_DIRPATH: Path = (FILES_DIRPATH / ARTIFACTS_DIRNAME).resolve()
    MAX_ARTIFACTS_RUNS: int = int(PATHS_CONFIG.get("max_artifacts_runs", 50))
    CATALOG_FILENAME: str = "catalog.md"

except Exception as e:
    logging.info(f"Failed to initialize artifacts module dependencies: {e}")
    raise MyException(e, sys) from e


def sanitize_query_slug(query: str, max_length: int = 50) -> str:
    """Sanitize query string into a filesystem-safe slug truncated to max_length.

    Args:
        query (str): The raw user query text.
        max_length (int): Maximum length for the slug string. Defaults to 50.

    Returns:
        str: Cleaned, lowercase slug consisting of alphanumeric characters and underscores.
    """
    clean_query: str = re.sub(r"[^\w\s-]", "", query).strip()
    slug: str = re.sub(r"[-\s]+", "_", clean_query).strip("_").lower()
    truncated_slug: str = slug[:max_length].rstrip("_")
    return truncated_slug if truncated_slug else "query"


def prune_old_artifacts(
    artifacts_dirpath: Path | str = ARTIFACTS_DIRPATH,
    max_runs: int = MAX_ARTIFACTS_RUNS,
) -> list[str]:
    """Ensure only the latest max_runs run directories are retained in artifacts_dirpath.

    Deletes the oldest run directories exceeding max_runs sorted by filesystem creation
    or modification time.

    Args:
        artifacts_dirpath (Path | str): Path to the artifacts root directory.
        max_runs (int): Maximum number of run directories to retain. Defaults to MAX_ARTIFACTS_RUNS.

    Returns:
        list[str]: Names of deleted directories.
    """
    target_dirpath: Path = Path(artifacts_dirpath).resolve()
    if not target_dirpath.exists() or not target_dirpath.is_dir():
        return []

    run_directories: list[Path] = [
        item for item in target_dirpath.iterdir() if item.is_dir() and not item.name.startswith(".")
    ]

    # Sort directories chronologically (oldest first) based on modification time and name
    run_directories.sort(key=lambda item: (item.stat().st_mtime, item.name))

    deleted_names: list[str] = []
    excess_count: int = len(run_directories) - max_runs
    if excess_count > 0:
        directories_to_delete: list[Path] = run_directories[:excess_count]
        for directory in directories_to_delete:
            try:
                shutil.rmtree(directory)
                deleted_names.append(directory.name)
                logging.info(f"Pruned old run artifact: {directory.name}")
            except OSError as error:
                logging.warning(f"Failed to prune old artifact directory {directory}: {error}")

    return deleted_names


def generate_artifacts_catalog(
    artifacts_dirpath: Path | str = ARTIFACTS_DIRPATH,
    catalog_filename: str = CATALOG_FILENAME,
) -> Path:
    """Generate a catalog.md index catalog listing all runs with the newest at the top.

    Args:
        artifacts_dirpath (Path | str): Root directory storing run artifacts.
        catalog_filename (str): Filename for the catalog markdown table. Defaults to CATALOG_FILENAME.

    Returns:
        Path: Path to the generated catalog.md file.
    """
    base_dirpath: Path = Path(artifacts_dirpath).resolve()
    base_dirpath.mkdir(parents=True, exist_ok=True)

    run_directories: list[Path] = [
        item for item in base_dirpath.iterdir() if item.is_dir() and not item.name.startswith(".")
    ]

    # Sort directories strictly newest first (descending by mtime and name)
    run_directories.sort(key=lambda item: (item.stat().st_mtime, item.name), reverse=True)

    catalog_lines: list[str] = [
        "# Corrective RAG Artifacts Catalog",
        "",
        f"*Total Runs Retained: {len(run_directories)} / {MAX_ARTIFACTS_RUNS}*",
        f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "| # | Timestamp | Run Directory | Query | Routing | Response Preview |",
        "|---|---|---|---|---|---|",
    ]

    for index, directory in enumerate(run_directories, start=1):
        metadata_filepath: Path = directory / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_filepath.exists():
            try:
                with open(metadata_filepath, encoding="utf-8") as f:
                    metadata = json.load(f)
            except (
                json.JSONDecodeError,
                OSError,
            ):
                metadata = {}

        query_text: str = str(metadata.get("query", directory.name)).replace("\n", " ")[:60]
        routing: str = str(metadata.get("routing_decision", "unknown"))
        timestamp_str: str = str(
            metadata.get("timestamp", datetime.fromtimestamp(directory.stat().st_mtime).isoformat())
        )

        # Extract brief response preview
        response_filepath: Path = directory / "response" / "response.md"
        response_preview: str = ""
        if response_filepath.exists():
            try:
                content: str = response_filepath.read_text(encoding="utf-8")
                parts: list[str] = content.split("Response:\n", 1)
                if len(parts) > 1:
                    response_preview = parts[1].strip().replace("\n", " ")[:80]
            except OSError:
                response_preview = ""

        formatted_time: str = timestamp_str[:19].replace("T", " ")
        catalog_entry: str = (
            f"| {index} | {formatted_time} | [`{directory.name}`](./{directory.name}/) "
            f"| {query_text} | `{routing}` | {response_preview}... |"
        )
        catalog_lines.append(catalog_entry)

    catalog_filepath: Path = base_dirpath / catalog_filename
    with open(catalog_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(catalog_lines) + "\n")

    return catalog_filepath


def save_run_artifacts(
    query: str,
    state: dict[str, Any],
    artifacts_dirpath: Path | str = ARTIFACTS_DIRPATH,
    max_runs: int = MAX_ARTIFACTS_RUNS,
) -> Path:
    """Save execution artifacts for a workflow run into structured directories.

    Creates:
      1) contexts/contexts.md (Query, RAG Context, Web Search Context)
      2) response/response.md (Query, Response)
      3) metadata.json (Run telemetry, routing decision, document counts)
      4) evaluations/ (evaluations.json and evaluations.md with scores and classifications)
      5) catalog.md (Catalog index with newest runs ordered at the very top)

    Directories are named timestamp_query[50] (e.g. 20261112_213015_query_slug)
    so chronologically recent runs group clearly and sort newest at the top.

    Also prunes the artifacts directory to retain only the latest max_runs runs.

    Args:
        query (str): The user query submitted to the workflow.
        state (dict[str, Any]): The final LangGraph state dictionary.
        artifacts_dirpath (Path | str): Base directory for run artifacts.
        max_runs (int): Maximum number of runs to retain. Defaults to MAX_ARTIFACTS_RUNS.

    Returns:
        Path: Absolute path to the newly created run artifact directory.
    """
    base_dirpath: Path = Path(artifacts_dirpath).resolve()
    base_dirpath.mkdir(parents=True, exist_ok=True)

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug: str = sanitize_query_slug(query=query, max_length=50)
    # Order with timestamp first so folders are grouped chronologically
    run_dirname: str = f"{timestamp}_{slug}"
    run_dirpath: Path = base_dirpath / run_dirname
    run_dirpath.mkdir(parents=True, exist_ok=True)

    rag_context: str = str(state.get("rag_context", "")).strip()
    rewritten_rag_query: str = str(state.get("rewritten_rag_query", "")).strip()
    rewritten_query: str = str(state.get("rewritten_query", "")).strip()
    web_search_context: str = str(state.get("web_search_context", "")).strip()
    refined_context: str = str(state.get("refined_context", "")).strip()
    response_text: str = str(state.get("response", "")).strip()

    # Persist input query and combined contexts for offline trace inspection
    contexts_dirpath: Path = run_dirpath / "contexts"
    contexts_dirpath.mkdir(parents=True, exist_ok=True)
    contexts_content: str = (
        "Query:\n"
        f"{query}\n\n"
        "Rewritten RAG Query:\n"
        f"{rewritten_rag_query or 'No RAG query rewriting performed.'}\n\n"
        "RAG Context:\n"
        f"{rag_context or 'No RAG context retrieved.'}\n\n"
        "Rewritten Web Query:\n"
        f"{rewritten_query or 'No web query rewriting required or performed.'}\n\n"
        "Web Search Context:\n"
        f"{web_search_context or 'No web search context required or retrieved.'}\n\n"
        "Refined Context:\n"
        f"{refined_context or 'No refined context generated.'}\n"
    )
    with open(contexts_dirpath / "contexts.md", "w", encoding="utf-8") as f:
        f.write(contexts_content)

    # Persist the final generated answer
    response_dirpath: Path = run_dirpath / "response"
    response_dirpath.mkdir(parents=True, exist_ok=True)
    response_content: str = f"Query: {query}\n\nResponse:\n{response_text or 'No response generated.'}\n"
    with open(response_dirpath / "response.md", "w", encoding="utf-8") as f:
        f.write(response_content)

    # Record run telemetry including routing outcome, model, and document counts
    hit_documents: list[Document] = state.get("hit_documents", [])
    ambiguous_documents: list[Document] = state.get("ambiguous_documents", [])
    miss_documents: list[Document] = state.get("miss_documents", [])
    retrieved_documents: list[Document] = state.get("retrieved_documents", [])

    routing_decision: str = (
        "generation" if bool(hit_documents) else ("search_web" if bool(web_search_context) else "generation")
    )

    metadata: dict[str, Any] = {
        "query": query,
        "rewritten_rag_query": rewritten_rag_query,
        "rewritten_query": rewritten_query,
        "timestamp": datetime.now().isoformat(),
        "run_dirname": run_dirname,
        "routing_decision": routing_decision,
        "model": PARAMS.get("llm", {}).get("model", "gemini-3.8-flash"),
        "counts": {
            "retrieved_documents": len(retrieved_documents),
            "hit_documents": len(hit_documents),
            "ambiguous_documents": len(ambiguous_documents),
            "miss_documents": len(miss_documents),
        },
    }
    with open(run_dirpath / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Persist calibrated document scores and classification breakdown
    evaluations_dirpath: Path = run_dirpath / "evaluations"
    evaluations_dirpath.mkdir(parents=True, exist_ok=True)

    evaluated_records: list[dict[str, Any]] = []
    markdown_rows: list[str] = [
        "| Index | Score | Classification | Source | Page | Content Snippet |",
        "|---|---|---|---|---|---|",
    ]

    for index, document in enumerate(retrieved_documents, start=1):
        score: float = float(document.metadata.get("score", 0.0))
        classification: str = (
            "hit" if document in hit_documents else ("ambiguous" if document in ambiguous_documents else "miss")
        )
        source: str = str(document.metadata.get("source", "unknown"))
        page: Any = document.metadata.get("page", "N/A")
        snippet: str = document.page_content.replace("\n", " ")[:120]

        record: dict[str, Any] = {
            "index": index,
            "score": score,
            "classification": classification,
            "source": source,
            "page": page,
            "content": document.page_content,
        }
        evaluated_records.append(record)
        markdown_rows.append(
            f"| {index} | {score:.1f} | {classification.upper()} | {Path(source).name} | {page} | {snippet}... |"
        )

    with open(evaluations_dirpath / "evaluations.json", "w", encoding="utf-8") as f:
        json.dump(evaluated_records, f, indent=2)

    with open(evaluations_dirpath / "evaluations.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_rows) + "\n")

    # Prune historical runs exceeding retention limit to prevent disk bloat
    prune_old_artifacts(artifacts_dirpath=base_dirpath, max_runs=max_runs)

    # Rebuild catalog.md index with runs ordered newest-first for rapid developer inspection
    generate_artifacts_catalog(artifacts_dirpath=base_dirpath)

    return run_dirpath


__all__: list[str] = [
    "generate_artifacts_catalog",
    "prune_old_artifacts",
    "sanitize_query_slug",
    "save_run_artifacts",
]
