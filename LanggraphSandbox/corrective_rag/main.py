"""
main.py: Orchestrates ingestion, retrieval, and LangGraph workflow execution.
"""

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph
from rich.markdown import Markdown
from rich.prompt import Prompt

ROOT: Path = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.graph import get_workflow
from src.exception import MyException
from src.logger import logging
from src.utils.artifacts import save_run_artifacts
from src.utils.display import console, get_terminal_width, print_divider, print_header
from src.utils.helpers import load_yaml_file

try:
    load_dotenv()

    WIDTH: int = get_terminal_width()

    CONFIG_DIRPATH: Path = (ROOT / "config").resolve()
    PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / "params.yml").resolve()
    PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)

    PATHS_CONFIG: dict[str, Any] = PARAMS.get("paths", {})
    FILES_DIRNAME: str = str(PATHS_CONFIG.get("files_dir", "files"))
    PERSIST_DIRNAME: str = str(PATHS_CONFIG.get("persist_directory", "chroma_store"))

    DOCUMENTS_DIRPATH: Path = (ROOT / FILES_DIRNAME / "documents").resolve()
    PERSIST_DIRPATH: Path = (ROOT / PERSIST_DIRNAME).resolve()

    EVALUATION_CONFIG: dict[str, Any] = PARAMS.get("evaluation", {})
    MISS_MAX_THRESHOLD: float = float(EVALUATION_CONFIG.get("miss_max_threshold", 3.5))
    HIT_MIN_THRESHOLD: float = float(EVALUATION_CONFIG.get("hit_min_threshold", 7.5))
    DEFAULT_USE_EXISTING: bool = bool(PATHS_CONFIG.get("use_existing_store", True))

    LLM_CONFIG: dict[str, Any] = PARAMS.get("llm", {})
    LLM_MODEL: str = str(LLM_CONFIG.get("model", "gemini-3.8-flash"))

    DEFAULT_QUERY: str = "How does the Jev model work?"

except Exception as e:
    logging.info(f"Failed to initialize main module dependencies: {e}")
    raise MyException(e, sys) from e


def main() -> None:
    """Run the Corrective RAG (CRAG) ingestion and workflow pipeline.

    Initializes the compiled LangGraph workflow, invokes RAG retrieval, document
    evaluation, optional web search, and LLM generation with a user query, and
    displays the query, retrieved contexts, and the generated response.
    """

    try:
        query: str = Prompt.ask(
            "[bold yellow]Enter query[/bold yellow]",
            default=DEFAULT_QUERY,
            console=console,
        ).strip()
    except EOFError:
        query = DEFAULT_QUERY
    except KeyboardInterrupt:
        query = DEFAULT_QUERY

    print_header("Query", text_color="yellow", line_color="grey")
    console.print(f"{query}\n" if query else "\n[dim italic]Query not found.[/dim italic]\n")

    workflow: CompiledStateGraph = get_workflow(save_workflow_file=True)
    with console.status("Generating Response..."):
        response: dict[str, Any] = workflow.invoke(
            {"query": query},
            config={
                "run_name": "corrective_rag_run",
                "metadata": {
                    "pipeline": "corrective_rag",
                    "model": LLM_MODEL,
                },
                "configurable": {
                    "use_existing": DEFAULT_USE_EXISTING,
                    "files_dirpath": DOCUMENTS_DIRPATH,
                    "persist_dirpath": PERSIST_DIRPATH,
                    "miss_max_threshold": MISS_MAX_THRESHOLD,
                    "hit_min_threshold": HIT_MIN_THRESHOLD,
                },
            },
        )

    # Render workflow execution outputs to terminal
    rag_context_text: str = str(response.get("rag_context", "")).strip()

    print_header("RAG Context")
    if rag_context_text:
        console.print(f"\n{rag_context_text}\n")
    else:
        console.print("[dim italic]No RAG context retrieved.[/dim italic]\n")

    web_search_context_text: str = str(response.get("web_search_context", "")).strip()

    print_header("Web Search Context")
    if web_search_context_text:
        console.print(f"\n{web_search_context_text}\n")
    else:
        console.print("[dim italic]No web search context required or retrieved.[/dim italic]\n")

    response_text: str = str(response.get("response", "")).strip()

    print_header("Generated Response")
    if response_text:
        console.print(Markdown(response_text))
    else:
        console.print("[dim italic]No response generated.[/dim italic]\n")
    print_divider()

    # Persist execution artifacts, trace contexts, metadata, and update catalog.md
    save_run_artifacts(query=query, state=response)


if __name__ == "__main__":
    main()

__all__: list[str] = [
    "main",
]
