import os
import yaml
import sqlite3
import operator
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage

params_configs = yaml.safe_load((Path(__file__).parent / "configs/params.yaml").read_text())
load_dotenv()

os.makedirs("db", exist_ok=True)
conn = sqlite3.connect("db/flaude.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


def load_db_conversations():
    threads = list({checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpointer.list(None)})

    return threads


model = ChatGoogleGenerativeAI(**params_configs["llm"])
parser = StrOutputParser()


class MessageState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


def chat(state: MessageState, config: RunnableConfig):
    res = (model | parser).invoke(state["messages"], config)
    return {"messages": [AIMessage(content=res)]}


graph = StateGraph(MessageState)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)


workflow = graph.compile(checkpointer=checkpointer)
