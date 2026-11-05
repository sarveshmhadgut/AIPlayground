"""
main.py: Orchestrates ingestion, retrieval, and LangGraph workflow execution.
"""

import sys
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph

ROOT: Path = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    console,
    get_workflow,
)


def main() -> None:
    """Run the Corrective RAG (CRAG) ingestion and workflow pipeline.

    Initializes the compiled LangGraph workflow, invokes context retrieval and LLM
    generation with a user query, and formats the output into structured JSON on
    the terminal console.
    """
    console.rule("[bold green]Corrective RAG (CRAG) Pipeline[/bold green]")

    query: str = "What are autoencoders?"
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    workflow: CompiledStateGraph = get_workflow(save_workflow_file=True)

    with console.status(
        "[bold cyan]Executing CRAG workflow pipeline...[/bold cyan]",
        spinner="dots",
    ):
        workflow_response: dict[str, Any] = workflow.invoke(
            {"query": query},
            config={
                "configurable": {
                    "use_existing": True,
                }
            },
        )

    console.rule("[bold green]Response[/bold green]")
    console.print_json(data=workflow_response)


if __name__ == "__main__":
    main()
