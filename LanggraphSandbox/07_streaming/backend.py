# imports & setups
import yaml
import operator
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage

PARAMS_CONFIGS = yaml.safe_load(
    (Path(__file__).parent / "configs/params.yaml").read_text()
)
load_dotenv()

# model & parser
model = ChatGoogleGenerativeAI(**PARAMS_CONFIGS["llm"])
parser = StrOutputParser()


# state
class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


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
