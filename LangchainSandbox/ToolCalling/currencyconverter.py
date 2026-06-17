import os
import dotenv
import requests
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.tools import InjectedToolArg
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from rich.theme import Theme
from rich.console import Console

load_dotenv()
EXCHANGE_RATE_KEY = os.getenv("EXCHANGE_RATE_KEY")


class LlmSchema(BaseModel):
    base_currency: str = Field(
        description="Three-letter ISO currency code representing the source currency (e.g., USD, KWD)."
    )
    target_currency: str = Field(
        description="Three-letter ISO currency code representing the destination currency (e.g., INR, IRR)."
    )
    conversion_rate: float = Field(
        description="Exchange rate indicating how many units of the target currency equal one unit of the base currency"
    )
    base_value: float = Field(description="Amount in the base currency that needs to be converted.")
    target_value: float = Field(
        description="Equivalent amount in the target currency after applying the conversion rate."
    )


theme = Theme(
    {
        "json.key": "#c1a2ff",
        "json.string": "#FCCEA1",
        "json.number": "#A1C4FD",
        "json.boolean": "#A6F5D8",
        "json.null": "#ffb3ba",
    }
)

console = Console(
    theme=theme,
    force_terminal=True,
    force_jupyter=False,
    color_system="truecolor",
)


# * tools
@tool
def get_conversion_rate(base_currency: str, target_currency: str) -> float:
    """
    MUST be used to get real-time, up-to-date currency exchange rates.
    Do NOT guess or approximate rates.
    Always call this tool when asked for current conversion rates.
    """
    URL: str = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_KEY}/pair/{base_currency}/{target_currency}"

    res: requests.Response = requests.get(URL)
    res.raise_for_status()

    return res.json()["conversion_rate"]


@tool
def convert_currency(
    base_value: float,
    conversion_rate: Annotated[float, InjectedToolArg],
) -> float:
    """
    MUST be used to fetch real-time currency exchange rates.
    Never approximate or guess.
    The model is NOT allowed to perform multiplication.
    """
    return base_value * conversion_rate


# * model init
model1 = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.0)
chat_history: list = [
    HumanMessage("Find the conversion rate between KWD and IRR, and based on that convert 18 KWD to IRR"),
]

# * tool binding
tool_binded_model = model1.bind_tools([get_conversion_rate, convert_currency])

# * tool calling
with console.status("Fetching model response...", spinner="dots"):
    res: AIMessage = tool_binded_model.invoke(chat_history)
chat_history.append(res)

conversion_rate: float = 0.0
converted_val: float = 0.0

for tool_call in res.tool_calls:
    if tool_call["name"] == "get_conversion_rate":
        with console.status("Invoking tool : get_conversion_rate...", spinner="dots"):
            tool_message1: ToolMessage = get_conversion_rate.invoke(tool_call)
        conversion_rate: float = float(tool_message1.content)
        chat_history.append(tool_message1)

    elif tool_call["name"] == "convert_currency":
        tool_call["args"]["conversion_rate"] = conversion_rate

        with console.status("Invoking tool : convert_currency...", spinner="dots"):
            tool_message2: ToolMessage = convert_currency.invoke(tool_call)
        converted_val: float = float(tool_message2.content)
        chat_history.append(tool_message2)

    else:
        print("Invalid tool call")

# * final output
parser: PydanticOutputParser[LlmSchema] = PydanticOutputParser(pydantic_object=LlmSchema)
parser_instructions: str = parser.get_format_instructions()

model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.0)
chain = model | parser

chat_history.append(HumanMessage(content=f"Extract the result into structured JSON.\n{parser_instructions}"))
with console.status("Fetching model response...", spinner="dots"):
    res: LlmSchema = chain.invoke(chat_history)

console.print_json(data=res.model_dump())
