"""
helpers.py: Helper utilities for YAML loading, text formatting, and UUID generation.
"""

import sys
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import yaml
from langchain_core.documents import Document

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exception import MyException, ValidationError
from src.logger import logging

CONFIG_DIRPATH: Path = (ROOT / "config").resolve()
PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / "params.yml").resolve()


def build_scoring_rubric(hit_min_threshold: float, miss_max_threshold: float) -> str:
    """Build a dynamic scoring rubric string based on runtime threshold values.

    Args:
        hit_min_threshold (float): Minimum score for hit / high relevance.
        miss_max_threshold (float): Maximum score for miss / irrelevant.

    Returns:
        str: Formatted multi-line scoring rubric.
    """
    ambiguous_upper: float = round(hit_min_threshold - 0.1, 1)
    miss_upper: float = round(miss_max_threshold - 0.1, 1)
    hit_description: str = (
        "The document directly addresses the query or provides key factual context "
        "needed to formulate an accurate answer."
    )
    ambiguous_description: str = (
        "The document is tangentially related or contains partial clues, but is "
        "insufficient on its own to fully answer the query."
    )
    miss_description: str = "The document is completely off-topic, unrelated, or provides zero utility for the query."
    return (
        "Scoring Rubric (0.0 to 10.0):\n"
        f"  - {hit_min_threshold:.1f} to 10.0 (Hit / High Relevance): {hit_description}\n"
        f"  - {miss_max_threshold:.1f} to {ambiguous_upper:.1f} (Ambiguous / Partial Relevance): "
        f"{ambiguous_description}\n"
        f"  - 0.0 to {miss_upper:.1f} (Miss / Irrelevant): {miss_description}"
    )


def load_yaml_file(filepath: Path | str = PARAMS_FILEPATH) -> dict[str, Any]:
    """Safely load a YAML configuration file with defensive fallbacks.

    Args:
        filepath (Path | str): Path to the YAML file.

    Returns:
        dict[str, Any]: Loaded dictionary configuration or empty dict on failure.
    """
    path: Path = Path(filepath)

    try:
        with open(path, encoding="utf-8") as f:
            content: Any = yaml.safe_load(f)
            return content if isinstance(content, dict) else {}
    except (yaml.YAMLError, OSError) as e:
        logging.error(f"Error parsing YAML {path}: {e}")
        return {}


def format_documents(documents: list[Document]) -> str:
    """Format a list of documents into a single string for RAG context.

    Args:
        documents (list[Document]): List of documents to format.

    Returns:
        str: Concatenated text content.
    """
    return " ".join([document.page_content for document in documents])


def generate_uuid(
    name: str | None = None,
    namespace: UUID | str = NAMESPACE_URL,
    as_hex: bool = False,
    *,
    text: str | None = None,
) -> str:
    """Generate a deterministic UUID5 string based on name/text and namespace.

    Args:
        name (str | None, optional): Name or content used to generate deterministic UUID5.
            Defaults to None.
        namespace (UUID | str, optional): Namespace identifier for UUID5.
            Defaults to NAMESPACE_URL.
        as_hex (bool, optional): If True, returns a 32-character hexadecimal string without
            hyphens. If False, returns standard 36-character hyphenated UUID. Defaults to False.
        text (str | None, optional): Keyword-only alias for name. Defaults to None.

    Returns:
        str: Generated deterministic UUID5 string in hexadecimal or standard format.

    Raises:
        ValidationError: If name or text is omitted, arguments are invalid, or namespace is malformed.
        MyException: If UUID generation encounters an unexpected error.
    """
    try:
        logging.info("Generating UUID...")

        target_name: str | None = text if name is None else name
        if target_name is None or not isinstance(target_name, str):
            raise ValidationError(
                f"Expected 'name' or 'text' to be a string for UUID5 generation, got {type(target_name).__name__}",
                sys,
            )

        if not isinstance(as_hex, bool):
            raise ValidationError(
                f"Expected 'as_hex' to be a boolean, got {type(as_hex).__name__}",
                sys,
            )

        target_namespace: UUID
        if isinstance(namespace, UUID):
            target_namespace = namespace
        elif isinstance(namespace, str):
            try:
                target_namespace = UUID(namespace)
            except Exception as exc:
                raise ValidationError(
                    f"Invalid namespace UUID string '{namespace}': {exc}",
                    sys,
                ) from exc
        else:
            raise ValidationError(
                f"Expected 'namespace' to be a UUID or str, got {type(namespace).__name__}",
                sys,
            )

        uuid_obj: UUID = uuid5(target_namespace, target_name)
        result_id: str = uuid_obj.hex if as_hex else str(uuid_obj)
        logging.info(f"Generated UUID successfully: {result_id[:8]}...")
        return result_id

    except MyException:
        raise
    except Exception as e:
        logging.info(f"Failed to generate UUID: {e}")
        raise MyException(e, sys) from e


__all__: list[str] = [
    "build_scoring_rubric",
    "format_documents",
    "generate_uuid",
    "load_yaml_file",
]
