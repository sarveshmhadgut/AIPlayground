import os
import yaml
import sqlite3
from pathlib import Path
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

params_configs = yaml.safe_load(Path("configs/params.yaml").read_text())
prompt_configs = yaml.safe_load(Path("configs/prompts.yaml").read_text())


# primitives
def load_css(filepath):
    if filepath.exists():
        return f"<style>{filepath}</style>"
    return ""


def get_llm(params):
    model = ChatGoogleGenerativeAI(**params)
    parser = StrOutputParser()
    return model | parser


# database
def get_conn(db_path):
    conn = sqlite3.connect(database=db_path, check_same_thread=False)
    return conn


# runnable
def get_runnable_config(thread_id, thread_name):
    return RunnableConfig(
        configurable={
            "thread_id": thread_id,
        },
        metadata={
            "thread_name": thread_name,
            "environment": os.getenv("APP_ENV", "default"),
            "app": "flaude",
        },
        run_name="flaude_turn",
    )
