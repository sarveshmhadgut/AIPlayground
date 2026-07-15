import os
import yaml
import requests
from pathlib import Path
from sympy import sympify
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()
PARAMS_CONFIGS = yaml.safe_load(Path("configs/params.yaml").read_text())


@tool(name_or_callable="math_eval")
def math_eval(expression: str):
    """
    Evaluate a mathematical expression.

    Args:
        expression: The mathematical expression to evaluate.

    Returns:
        A dictionary containing:
            - status: "success" or "failure".
            - expression: The input expression.
            - result: The evaluated result when successful.
            - error: The error message when evaluation fails.
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
        search_query: The search query.

    Returns:
        A dictionary containing:
            - status: "success" or "failure".
            - query: The search query.
            - response: The search results when successful.
            - error: The error message when the search fails.
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
    Search for a file in the current working directory and its subdirectories.

    Args:
        filename: The name of the file to search for.

    Returns:
        A dictionary containing:
            - status: "success" or "failure".
            - filename: The requested filename.
            - contents: The contents of all matching files, or a message if none are found.
            - error: The error message when the search fails.
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
    Retrieve the current exchange rate between two currencies.
    This tool must be used whenever an up-to-date currency conversion or exchange rate is required. Never estimate or approximate exchange rates.

    Args:
        base_currency: The source currency code (e.g., "GBP").
        target_currency: The target currency code (e.g., "INR").

    Returns:
        A dictionary containing:
            - status: "success" or "failure".
            - base_currency: The source currency code.
            - target_currency: The target currency code.
            - conversion_rate: The current exchange rate when successful.
            - error: The error message when the request fails.
    """
    try:
        EXCHANGE_RATE_KEY = os.environ.get("EXCHANGE_RATE_KEY", "")
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


available_tools = [math_eval, web_search, file_search, get_conversion_rate]
