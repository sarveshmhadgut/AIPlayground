import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


class ToolState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


try:
    res = model.invoke(["Hello!"])
    print("invoke with string list passed:", res)
except Exception as e:
    print("Exception with string list:", repr(e))

try:
    res = model.invoke([("user", "Hello!")])
    print("invoke with tuple list passed:", res)
except Exception as e:
    print("Exception with tuple list:", repr(e))
