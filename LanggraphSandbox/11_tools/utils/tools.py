import os
import yaml
import requests
from pathlib import Path
from sympy import sympify
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

PARAMS_CONFIGS = yaml.safe_load(
    (Path(__file__).parent.parent / "configs/params.yaml").read_text()
)

load_dotenv()
EXCHANGE_RATE_KEY = os.getenv("EXCHANGE_RATE_KEY")


@tool(name_or_callable="math_eval")
def math_eval(expression: str):
    """
    Evaluate a mathematical expression and return the result.
    Args:
        expression: Mathematical expression to evaluate.
    Returns:
        A dictionary containing the evaluation result or an error message.
    """
    try:
        res = sympify(expression)
        return {
            "status": "success",
            "expression": expression,
            "result": res,
        }

    except Exception as e:
        return {
            "status": "failure",
            "expression": expression,
            "error": str(e),
        }


@tool(name_or_callable="web_search")
def web_search(search_query: str):
    """
    Search the web for information related to a query.
    Args:
        search_query: Search query.
    Returns:
        A dictionary containing the search results.
    """
    try:
        engine = DuckDuckGoSearchRun()
        res = engine.invoke(search_query)
        return {
            "status": "success",
            "query": search_query,
            "response": res,
        }

    except Exception as e:
        return {
            "status": "failure",
            "search_query": search_query,
            "error": str(e),
        }


@tool(name_or_callable="file_search")
def file_search(filename: str):
    """
    Search for a file in the current directory tree and return its contents.
    Args:
        filename: Name of the file to locate.
    Returns:
        A dictionary containing the file contents if found, otherwise None.
    """
    try:
        current = Path.cwd()
        contents = []
        for filepath in current.rglob(filename):
            if filepath.is_file():
                contents.append(f"{filepath.name}\n{filepath.read_text()}\n")

        return {
            "status": "success",
            "filename": filename,
            "contents": "\n\n".join(contents)
            if contents
            else "No matching files found.",
        }

    except Exception:
        return {
            "status": "failure",
            "filename": filename,
            "error": "File not found",
        }


@tool(name_or_callable="get_conversion_rate")
def get_conversion_rate(base_currency: str, target_currency: str):
    """
    MUST be used to get real-time, up-to-date currency exchange rates.
    Do NOT guess or approximate rates.
    Always call this tool when asked for current conversion rates.
    """
    try:
        if not EXCHANGE_RATE_KEY:
            raise ValueError("EXCHANGE_RATE_KEY not found")

        URL = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_KEY}/pair/{base_currency}/{target_currency}"

        res = requests.get(URL, timeout=10)
        res.raise_for_status()

        data = res.json()
        if data["result"] != "success":
            raise ValueError(data.get("error-type", "Unknown API error"))

        return {
            "status": "success",
            "base_currency": base_currency,
            "target_currency": target_currency,
            "conversion_rate": data["conversion_rate"],
        }

    except Exception as e:
        return {
            "status": "failure",
            "base_currency": base_currency,
            "target_currency": target_currency,
            "error": str(e),
        }


available_tools = [
    math_eval,
    web_search,
    file_search,
    get_conversion_rate,
]
