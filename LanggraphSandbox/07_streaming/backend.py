# imports & setups
import operator
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

PARAMS_CONFIGS = yaml.safe_load((Path(__file__).parent / "configs/params.yaml").read_text())
load_dotenv()

# model & parser
model = ChatGoogleGenerativeAI(**PARAMS_CONFIGS["llm"])
parser = StrOutputParser()


# state
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# node and utility functions
def chat(state: MessagesState):
    res = (model | parser).invoke(state["messages"])
    return {"messages": [AIMessage(content=res)]}


# init graph
graph = StateGraph(MessagesState)

# add nodes
graph.add_node("chat", chat)

# add edges
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# compilation
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)
