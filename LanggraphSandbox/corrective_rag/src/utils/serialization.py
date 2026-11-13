"""
serialization.py: Workflow graph serialization, persistence, and Jupyter visualizer.
"""

import pickle
import sys
from pathlib import Path
from typing import Any

import dill

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.exception import MyException, ValidationError
from src.logger import logging
from src.utils.helpers import load_yaml_file

try:
    load_dotenv()

    CONFIG_DIRPATH: Path = (ROOT / "config").resolve()
    PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / "params.yml").resolve()
    PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)

    PATHS_CONFIG: dict[str, Any] = PARAMS.get("paths", {})
    FILES_DIRNAME: str = str(PATHS_CONFIG.get("files_dir", "files"))
    FILES_DIRPATH: Path = (ROOT / FILES_DIRNAME).resolve()
    WORKFLOWS_DIRPATH: Path = (FILES_DIRPATH / "workflows").resolve()

    WORKFLOW_PICKLE_FILENAME: str = "workflow_viewer.pkl"
    WORKFLOW_DILL_FILENAME: str = "workflow.dill"
    WORKFLOW_PICKLE_FILEPATH: Path = (WORKFLOWS_DIRPATH / WORKFLOW_PICKLE_FILENAME).resolve()
    WORKFLOW_DILL_FILEPATH: Path = (WORKFLOWS_DIRPATH / WORKFLOW_DILL_FILENAME).resolve()

except Exception as e:
    logging.info(f"Failed to initialize serialization module dependencies: {e}")
    raise MyException(e, sys) from e


class WorkflowViewer:
    """Lightweight pickle-compatible workflow visualizer for Jupyter notebooks.

    Attributes:
        png_bytes (bytes): Pre-rendered PNG diagram bytes.
        mermaid_syntax (str): Raw Mermaid syntax string.
    """

    def __init__(self, png_bytes: bytes, mermaid_syntax: str = "") -> None:
        """Initialize the WorkflowViewer with diagram bytes and syntax.

        Args:
            png_bytes (bytes): Pre-rendered PNG diagram bytes.
            mermaid_syntax (str): Raw Mermaid syntax string. Defaults to empty string.
        """
        self.png_bytes: bytes = png_bytes
        self.mermaid_syntax: str = mermaid_syntax

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
        """Mime bundle used by Jupyter to render the graphical flowchart automatically."""
        bundle: dict[str, Any] = {
            "text/plain": self.mermaid_syntax or "<WorkflowViewer>",
        }
        if self.png_bytes and len(self.png_bytes) > 0:
            bundle["image/png"] = self.png_bytes
        elif self.mermaid_syntax:
            bundle["text/html"] = (
                f'<div class="mermaid">\n{self.mermaid_syntax}\n</div>\n'
                '<script type="module">'
                'import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";'
                "mermaid.initialize({startOnLoad: true});"
                "</script>"
            )
        return bundle


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
        raw_graph: Any = workflow.get_graph() if hasattr(workflow, "get_graph") else workflow
        png_bytes: bytes = b""
        if hasattr(raw_graph, "draw_mermaid_png"):
            try:
                png_bytes = raw_graph.draw_mermaid_png()
            # Gracefully handle network timeouts from external Mermaid rendering services (e.g. mermaid.ink)
            except (ValueError, OSError, ConnectionError, RuntimeError) as render_error:
                logging.warning(f"Could not render Mermaid PNG via external API: {render_error}")
        mermaid_syntax: str = raw_graph.draw_mermaid() if hasattr(raw_graph, "draw_mermaid") else ""

        viewer: WorkflowViewer = WorkflowViewer(png_bytes=png_bytes, mermaid_syntax=mermaid_syntax)
        target_filepath: Path = Path(filepath).resolve()
        target_filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(target_filepath, "wb") as f:
            pickle.dump(viewer, f)

        # Optionally persist the live executable CompiledStateGraph using dill
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


__all__: list[str] = [
    "WorkflowViewer",
    "load_workflow",
    "save_workflow",
]
