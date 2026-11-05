"""
utils.py: Helper utilities for configuration loading, document ingestion, splitting, and console formatting.
"""

import pickle
import sys
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import dill
import yaml
from langchain_core.documents import Document
from rich.console import Console
from rich.theme import Theme

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.exception import MyException, ValidationError
from src.logger import logging

__all__: list[str] = [
    "CONFIG_DIRNAME",
    "CONFIG_DIRPATH",
    "DOCUMENTS_DIRNAME",
    "DOCUMENTS_DIRPATH",
    "FILES_DIRNAME",
    "FILES_DIRPATH",
    "PARAMS",
    "PARAMS_FILENAME",
    "PARAMS_FILEPATH",
    "PERSIST_DIRNAME",
    "PERSIST_DIRPATH",
    "PROMPTS_FILENAME",
    "PROMPTS_FILEPATH",
    "WORKFLOWS_DIRNAME",
    "WORKFLOWS_DIRPATH",
    "WORKFLOW_DILL_FILENAME",
    "WORKFLOW_DILL_FILEPATH",
    "WORKFLOW_PICKLE_FILENAME",
    "WORKFLOW_PICKLE_FILEPATH",
    "WorkflowViewer",
    "console",
    "format_documents",
    "generate_uuid",
    "load_workflow",
    "load_yaml_file",
    "save_workflow",
]

CONFIG_DIRNAME: str = "config"
CONFIG_DIRPATH: Path = (ROOT / CONFIG_DIRNAME).resolve()

PARAMS_FILENAME: str = "params.yml"
PROMPTS_FILENAME: str = "prompts.yml"

PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / PARAMS_FILENAME).resolve()
PROMPTS_FILEPATH: Path = (CONFIG_DIRPATH / PROMPTS_FILENAME).resolve()


theme = Theme(
    {
        "json.key": "#c1a2ff",
        "json.string": "#FCCEA1",
        "json.number": "#A1C4FD",
        "json.boolean": "#A6F5D8",
        "json.null": "#ffb3ba",
    }
)

console = Console(
    theme=theme,
    force_terminal=True,
    force_jupyter=False,
    color_system="truecolor",
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
        with open(path, "r", encoding="utf-8") as f:
            content: Any = yaml.safe_load(f)
            return content if isinstance(content, dict) else {}
    except (yaml.YAMLError, OSError) as e:
        console.print(f"[red]Error parsing YAML {path}: {e}[/red]")
        return {}


PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)

PATHS_CONFIG: dict[str, Any] = PARAMS.get("paths", {})
PERSIST_DIRNAME: str = str(PATHS_CONFIG.get("persist_directory", "chroma_store"))
FILES_DIRNAME: str = str(PATHS_CONFIG.get("files_dir", "files"))

PERSIST_DIRPATH: Path = (ROOT / PERSIST_DIRNAME).resolve()
FILES_DIRPATH: Path = (ROOT / FILES_DIRNAME).resolve()

WORKFLOWS_DIRNAME: str = "workflows"
DOCUMENTS_DIRNAME: str = "documents"
WORKFLOWS_DIRPATH: Path = (FILES_DIRPATH / WORKFLOWS_DIRNAME).resolve()
DOCUMENTS_DIRPATH: Path = (FILES_DIRPATH / DOCUMENTS_DIRNAME).resolve()

WORKFLOW_PICKLE_FILENAME: str = "workflow_viewer.pkl"
WORKFLOW_DILL_FILENAME: str = "workflow.dill"
WORKFLOW_PICKLE_FILEPATH: Path = (
    WORKFLOWS_DIRPATH / WORKFLOW_PICKLE_FILENAME
).resolve()
WORKFLOW_DILL_FILEPATH: Path = (WORKFLOWS_DIRPATH / WORKFLOW_DILL_FILENAME).resolve()


def format_documents(documents: list[Document]) -> str:
    """Format a list of documents into a single string for RAG context.

    Args:
        documents (list[Document]): List of documents to format.

    Returns:
        str: Concatenated text content.
    """
    return " ".join([document.page_content for document in documents])


class WorkflowViewer:
    """Lightweight pickle-compatible workflow visualizer for Jupyter notebooks.

    Attributes:
        png_bytes (bytes): Pre-rendered PNG diagram bytes.
        mermaid_syntax (str): Raw Mermaid syntax string.
    """

    def __init__(self, png_bytes: bytes, mermaid_syntax: str = "") -> None:
        self.png_bytes: bytes = png_bytes
        self.mermaid_syntax: str = mermaid_syntax

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
        """Mime bundle used by Jupyter to render the graphical flowchart automatically."""
        return {
            "text/plain": self.mermaid_syntax or "<WorkflowViewer>",
            "image/png": self.png_bytes,
        }


def save_workflow(
    workflow: Any,
    filepath: Path | str = WORKFLOW_PICKLE_FILEPATH,
    save_full_dill: bool = True,
) -> Path:
    """Serialize the workflow so it can be loaded and displayed in Jupyter notebooks.

    Saves a lightweight, pickle-serializable WorkflowViewer containing the pre-rendered
    flowchart image bytes, and optionally saves the complete CompiledStateGraph via dill.

    Args:
        workflow (Any): CompiledStateGraph workflow instance.
        filepath (Path | str): Path where the pickle file should be saved.
        save_full_dill (bool): Whether to also save the full runnable graph to workflow.dill.

    Returns:
        Path: Path to the saved viewer pickle file.

    Raises:
        MyException: If serialization fails.
    """
    try:
        raw_graph: Any = (
            workflow.get_graph() if hasattr(workflow, "get_graph") else workflow
        )
        png_bytes: bytes = (
            raw_graph.draw_mermaid_png()
            if hasattr(raw_graph, "draw_mermaid_png")
            else b""
        )
        mermaid_syntax: str = (
            raw_graph.draw_mermaid() if hasattr(raw_graph, "draw_mermaid") else ""
        )

        viewer: WorkflowViewer = WorkflowViewer(
            png_bytes=png_bytes, mermaid_syntax=mermaid_syntax
        )
        target_filepath: Path = Path(filepath).resolve()
        target_filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(target_filepath, "wb") as f:
            pickle.dump(viewer, f)

        if save_full_dill and dill is not None:
            dill_filepath: Path = target_filepath.parent / WORKFLOW_DILL_FILENAME
            with open(dill_filepath, "wb") as f:
                dill.dump(workflow, f)

        return target_filepath
    except Exception as e:
        logging.info(f"Failed to serialize workflow: {e}")
        raise MyException(e, sys) from e


def load_workflow(filepath: Path | str = WORKFLOW_PICKLE_FILEPATH) -> Any:
    """Load a serialized workflow viewer or compiled graph from disk.

    Args:
        filepath (Path | str): Path to the serialized .pkl or .dill file.

    Returns:
        Any: Deserialized WorkflowViewer or CompiledStateGraph.

    Raises:
        ValidationError: If file does not exist.
        MyException: If deserialization fails.
    """
    try:
        path: Path = Path(filepath).resolve()
        if not path.exists():
            raise ValidationError(f"Workflow file not found: {path}", sys)

        if path.suffix == ".dill" and dill is not None:
            with open(path, "rb") as f:
                return dill.load(f)
        with open(path, "rb") as f:
            return pickle.load(f)
    except MyException:
        raise
    except Exception as e:
        logging.info(f"Failed to load workflow from {filepath}: {e}")
        raise MyException(e, sys) from e


def generate_uuid(
    name: str | None = None,
    namespace: UUID | str = NAMESPACE_URL,
    as_hex: bool = False,
    *,
    text: str | None = None,
) -> str:
    """
    Generate a UUID string, either random (UUID4) or deterministic (UUID5).

    When a name or text is provided, a deterministic UUID5 is generated using the
    specified namespace. When omitted, a random UUID4 is generated.

    Args:
        name (str | None, optional): Name or content used to generate deterministic UUID5.
            Defaults to None.
        namespace (UUID | str, optional): Namespace identifier for UUID5.
            Defaults to NAMESPACE_URL.
        as_hex (bool, optional): If True, returns a 32-character hexadecimal string without
            hyphens. If False, returns standard 36-character hyphenated UUID. Defaults to False.
        text (str | None, optional): Keyword-only alias for name. Defaults to None.

    Returns:
        str: Generated UUID string in hexadecimal or standard format.

    Raises:
        ValidationError: If arguments are of invalid types or namespace string is malformed.
        MyException: If UUID generation encounters an unexpected error.
    """
    try:
        logging.info("Generating UUID...")

        target_name: str | None = text if name is None else name
        if target_name is not None and not isinstance(target_name, str):
            raise ValidationError(
                f"Expected 'name' or 'text' to be a string, got {type(target_name).__name__}",
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

        uuid_obj: UUID
        if target_name is not None:
            uuid_obj = uuid5(target_namespace, target_name)
        else:
            uuid_obj = uuid4()

        result_id: str = uuid_obj.hex if as_hex else str(uuid_obj)
        logging.info(f"Generated UUID successfully: {result_id[:8]}...")
        return result_id

    except MyException:
        raise
    except Exception as e:
        logging.info(f"Failed to generate UUID: {e}")
        raise MyException(e, sys) from e
