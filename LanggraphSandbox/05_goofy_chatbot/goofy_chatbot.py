# Imports & Setups
import operator
import os
from pathlib import Path
from typing import Annotated, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from termcolor import colored

load_dotenv()

PROMPTS_CONFIGS = yaml.safe_load(Path("configs/prompts.yaml").read_text())
PARAMS_CONFIGS = yaml.safe_load(Path("configs/params.yaml").read_text())

os.environ["LANGSMITH_PROJECT"] = PARAMS_CONFIGS["LANGSMITH_PROJECT"]
os.environ["LANGSMITH_TRACING_V2"] = str(PARAMS_CONFIGS["LANGCHAIN_TRACING_V2"]).lower()


# Model
def get_llm():
    model = ChatGoogleGenerativeAI(
        model=PARAMS_CONFIGS["llm"]["model"],
        temperature=PARAMS_CONFIGS["llm"]["temperature"],
        max_output_tokens=PARAMS_CONFIGS["llm"]["max_output_tokens"],
    )

    parser = StrOutputParser()
    chain = model | parser

    return chain


llm = get_llm()


# state
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]
    exit: bool


# node and helper functions
def set_system_instructions(state: ChatState):
    system_message = PROMPTS_CONFIGS["system_instructions"]

    with open("files/messages.txt", "a") as f:
        f.write(f"(('system'), '{system_message}')\n\n")

    return {"messages": [SystemMessage(system_message)]}


def chat(state: ChatState):
    print(colored(text="You: ", color="green"), end="")
    user_message = input()

    state["messages"].append(HumanMessage(content=user_message))
    with open("files/messages.txt", "a") as f:
        f.write(f"(('user'), '{user_message}')\n\n")

    if user_message in ["exit", "bye", "goodbye", "seeya"]:
        return {"exit": True}

    ai_message = llm.invoke(state["messages"])
    print(colored("Goofy: ", color="blue"), ai_message, end="\n\n")

    state["messages"].append(AIMessage(content=ai_message))
    with open("files/messages.txt", "a") as f:
        f.write(f"(('ai'), '{ai_message}')\n\n")

    return {"exit": False}


def route_chatting(state: ChatState):
    if state["exit"]:
        return "exit"

    return "continue"


# init graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node("set_system_instructions", set_system_instructions)
graph.add_node("chat", chat)

# add edges
graph.add_edge(START, "set_system_instructions")
graph.add_edge("set_system_instructions", "chat")
graph.add_conditional_edges("chat", route_chatting, {"exit": END, "continue": "chat"})

# compilation
workflow = graph.compile()

# execution
final_state = workflow.invoke({})

workflow_viz = workflow.get_graph()
os.makedirs("files", exist_ok=True)
workflow_viz.draw_mermaid_png(output_file_path="files/workflow.png")
