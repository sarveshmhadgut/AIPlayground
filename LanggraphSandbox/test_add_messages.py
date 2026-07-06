from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START

class ToolState(TypedDict):
    messages: Annotated[list, add_messages]

def my_node(state: ToolState):
    print("State inside node:", state)
    return {}

graph = StateGraph(ToolState)
graph.add_node("my_node", my_node)
graph.add_edge(START, "my_node")
workflow = graph.compile()

print("Testing with string...")
workflow.invoke({"messages": "Convert 123 usd to inr"})

