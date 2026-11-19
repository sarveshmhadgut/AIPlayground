"""
langgraph_utils: Shared utilities, terminal formatting, and runtime configurations.
"""

import os
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from termcolor import colored


def display_conversation(messages: list[BaseMessage]) -> None:
    """Format and print a conversation history with styled LLM and Human headers.

    Args:
        messages (list[BaseMessage]): List of conversation messages to render.
    """
    for message in messages:
        if isinstance(message, AIMessage):
            print(colored("LLM:", (193, 162, 255), attrs=["bold"]))
        else:
            print(colored("\nHuman:", (222, 207, 166), attrs=["bold"]))

        console.print(Markdown(f"{message.content}\n"))


def get_config(user_id: str, thread_id: str) -> RunnableConfig:
    """Generate a standard RunnableConfig with configurable parameters and metadata.

    Args:
        user_id (str): Unique user identifier.
        thread_id (str): Unique conversation thread identifier.

    Returns:
        RunnableConfig: Configured LangChain runnable configuration.
    """
    return RunnableConfig(
        {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
            "metadata": {
                "thread_id": thread_id,
                "user_id": user_id,
                "environment": os.getenv("APP_ENV", "default"),
                "app": "ltm_postgres",
            },
            "run_name": "ltm_postgres_run",
        }
    )


def get_console() -> Console:
    """Create and return a Rich Console configured with the project's standard theme."""
    theme = Theme(
        {
            "json.key": "#c1a2ff",
            "json.string": "#FCCEA1",
            "json.number": "#A1C4FD",
            "json.boolean": "#A6F5D8",
            "json.null": "#ffb3ba",
        }
    )
    return Console(
        theme=theme,
        force_terminal=True,
        force_jupyter=False,
        color_system="truecolor",
        width=120,
    )


console: Console = get_console()

__all__: list[str] = [
    "console",
    "display_conversation",
    "get_config",
    "get_console",
]
