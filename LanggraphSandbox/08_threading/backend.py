# imports & setups
import yaml
import operator
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage

params_configs = yaml.safe_load((Path(__file__).parent / "configs/params.yaml").read_text())
load_dotenv()

# model & parser
model = ChatGoogleGenerativeAI(**params_configs["llm"])
parser = StrOutputParser()


# state
class MessageState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


def chat(state: MessageState, config: RunnableConfig):
    res = (model | parser).invoke(state["messages"], config)
    return {"messages": [AIMessage(content=res)]}


# graph
graph = StateGraph(MessageState)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# workflow
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)
