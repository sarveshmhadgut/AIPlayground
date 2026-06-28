# imports & setups
import yaml
import operator
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable
from utils.config import get_conn, get_llm
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, AIMessage

params_config = yaml.safe_load(Path("configs/params.yaml").read_text())
load_dotenv()
# model
model = get_llm(params=params_config["llm"])


# state
class MessageState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


@traceable(name="chat_node")
def chat(state: MessageState, config: RunnableConfig):
    res = model.invoke(state["messages"], config)
    return {"messages": [AIMessage(content=res)]}


# graph
graph = StateGraph(MessageState)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# checkpointer
conn = get_conn(db_path=params_config["files"]["chat_db_filepath"])
checkpointer = SqliteSaver(conn=conn)

# workflow
workflow = graph.compile(checkpointer=checkpointer)
