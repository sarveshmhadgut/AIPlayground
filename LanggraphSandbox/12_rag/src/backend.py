# imports & setups
import yaml
import operator
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable
from utils.tools import available_tools
from utils.config import get_conn, get_llm
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
PARAMS_CONFIGS = yaml.safe_load(
    (Path(__file__).parent.parent / "configs/params.yaml").read_text()
)

# model
model = get_llm(params=PARAMS_CONFIGS["llm"])
bound_model = model.bind_tools(available_tools)


# state
class MessageState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# utility functions
@traceable(name="chat_node")
def chat(state: MessageState, config: RunnableConfig):
    res = bound_model.invoke(state["messages"], config)
    return {"messages": [res]}


tools = ToolNode(available_tools)

# graph
graph = StateGraph(MessageState)
graph.add_node("chat", chat)
graph.add_node("tools", tools)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

# checkpointer
conn = get_conn(db_path=PARAMS_CONFIGS["files"]["chat_db_filepath"])
checkpointer = SqliteSaver(conn=conn)

# workflow
workflow = graph.compile(checkpointer=checkpointer)
