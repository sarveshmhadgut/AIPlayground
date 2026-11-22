# imports & setups
import operator
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from utils.config import get_conn, get_llm

load_dotenv()
PARAMS_CONFIGS = yaml.safe_load((Path(__file__).parent / "configs/params.yaml").read_text())
# model
model = get_llm(params=PARAMS_CONFIGS["llm"])


# state
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


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
conn = get_conn(db_path=PARAMS_CONFIGS["files"]["chat_db_filepath"])
checkpointer = SqliteSaver(conn=conn)

# workflow
workflow = graph.compile(checkpointer=checkpointer)
