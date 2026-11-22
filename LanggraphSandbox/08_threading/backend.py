# imports & setups
import operator
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

PARAMS_CONFIGS = yaml.safe_load((Path(__file__).parent / "configs/params.yaml").read_text())
load_dotenv()

# model & parser
model = ChatGoogleGenerativeAI(**PARAMS_CONFIGS["llm"])
parser = StrOutputParser()


# state
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


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
