# Imports & Setups
import os
import sqlite3
from pathlib import Path
from typing import Annotated, List, TypedDict
from uuid import uuid1

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langsmith import traceable
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from tools import available_tools

from langgraph_utils import console

load_dotenv()
PARAMS_CONFIGS = yaml.safe_load(Path("configs/params.yaml").read_text())
PROMPTS_CONFIGS = yaml.safe_load(Path("configs/prompts.yaml").read_text())

os.environ["LANGSMITH_PROJECT"] = PARAMS_CONFIGS["LANGSMITH_PROJECT"]
os.environ["LANGSMITH_TRACING_V2"] = str(PARAMS_CONFIGS["LANGCHAIN_TRACING_V2"]).lower()

# Model
model = ChatGoogleGenerativeAI(**PARAMS_CONFIGS["llm"])
bound_model = model.bind_tools(available_tools)


# State
class ToolState(TypedDict):
    messages: Annotated[List, add_messages]


# Node & Utility Functions
@traceable(name="set_system_instructions")
def set_system_instructions(state: ToolState):
    if any(isinstance(message, SystemMessage) for message in state["messages"]):
        return {}

    return {"messages": [SystemMessage(content=PROMPTS_CONFIGS["chat"]["system"])]}


@traceable(name="chat")
def chat(state: ToolState, config):

    with console.status(
        status="Thinking...",
        spinner="dots",
        refresh_per_second=30,
    ):
        res = bound_model.invoke(
            input=state["messages"],
            config=config,
        )
    return {"messages": [res]}


@traceable(name="approve_tool")
def approve_tools(state: ToolState):
    required_tools = [
        tool_call["name"] for tool_call in state["messages"][-1].tool_calls
    ]

    approved = interrupt(
        {
            "type": "approval",
            "reason": "The model requires tool(s) approval.",
            "required_tools": required_tools,
        }
    )
    if approved:
        return Command(goto="tools")

    return Command(goto=END)


# HITL chatbot function
def invoke_with_hitl(input_, config):
    state = workflow.invoke(input_, config=config)

    while "__interrupt__" in state:
        interrupt = state["__interrupt__"][0].value
        # required_tools = ", ".join(interrupt["required_tools"])

        console.print(
            "\n",
            Panel.fit(
                JSON.from_data(interrupt),
                title="Interupt!",
            ),
        )

        approved = input("Approve execution? [y/n]: ").strip().lower() == "y"
        state = workflow.invoke(
            Command(resume=approved),
            config=config,
        )

    return state


# Tools Node
tools = ToolNode(available_tools)

# Init Graph
graph = StateGraph(ToolState)

# Add Nodes
graph.add_node("set_system_instructions", set_system_instructions)
graph.add_node("chat", chat)
graph.add_node("approve_tools", approve_tools)
graph.add_node("tools", tools)

# Add Edges
graph.add_edge(START, "set_system_instructions")
graph.add_edge("set_system_instructions", "chat")
graph.add_conditional_edges(
    "chat", tools_condition, {END: END, "tools": "approve_tools"}
)

graph.add_edge("approve_tools", "tools")
graph.add_edge("tools", "chat")

# SQLite Setup
db_dir = Path(__file__).parent / "db"
db_dir.mkdir(exist_ok=True)

database = Path(__file__).parent / "db/chatbot_with_hitl.sqlite"
conn = sqlite3.connect(database=database, check_same_thread=False)

# Compilation
checkpointer = SqliteSaver(conn=conn)
workflow = graph.compile(checkpointer=checkpointer)

# Config
thread_id = str(uuid1())

config = RunnableConfig(
    configurable={
        "thread_id": thread_id,
    },
    metadata={
        "thread_id": thread_id,
        "environment": os.getenv("APP_ENV", "default"),
        "app": "chatbot_with_hitl",
    },
    run_name="hitl_turn",
)

# Execution
user_query = "Convert 500 GBP to INR using the latest exchange rate. Find the current prices of both the Apple Polishing Cloth and the Apple AirTag in India. Then calculate how many of each I could buy with the converted amount and tell me which gives me the higher quantity."

final_state = invoke_with_hitl(
    input_={"messages": [HumanMessage(content=user_query)]},
    config=config,
)

console.print(
    "\n\n",
    Markdown(final_state["messages"][-1].content[0]["text"]),
)
