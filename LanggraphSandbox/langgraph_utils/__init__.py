import os
from typing import List
from rich.theme import Theme
from termcolor import colored
from rich.console import Console
from rich.markdown import Markdown
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


def display_conversation(messages: List[BaseMessage]):
    for message in messages:
        print(colored("LLM:", (193, 162, 255), ["bold"])) if isinstance(
            message, AIMessage
        ) else print(colored("\nHuman:", (222, 207, 166), ["bold"]))

        console.print(Markdown(f"{message.content}\n"))


def get_config(user_id, thread_id):
    return RunnableConfig(
        {
            "configurable": {"thread_id": thread_id, "user_id": user_id},
            "metadata": {
                "thread_id": thread_id,
                "user_id": user_id,
                "environment": os.getenv("APP_ENV", "default"),
                "app": "ltm_postgres",
            },
        },
        run_name="ltm_postgres_run",
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


console = get_console()
