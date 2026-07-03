import os
import yaml
import sqlite3
from pathlib import Path
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


PARAMS_CONFIGS = yaml.safe_load((Path(__file__).parent.parent / "configs/params.yaml").read_text())
PROMPTS_CONFIGS = yaml.safe_load((Path(__file__).parent.parent / "configs/prompts.yaml").read_text())


# primitives
def load_css(filepath):
    if filepath.exists():
        return f"<style>{filepath.read_text()}</style>"
    return ""


def get_llm(params):
    return ChatGoogleGenerativeAI(**params)


def get_embeddings(params):
    return GoogleGenerativeAIEmbeddings(**params)


# database
def get_conn(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        database=db_path,
        check_same_thread=False,
    )
    return conn


# runnable
def get_runnable_config(thread_id, thread_name):
    return RunnableConfig(
        configurable={
            "thread_id": thread_id,
        },
        metadata={
            "thread_id": thread_id,
            "thread_name": thread_name,
            "environment": os.getenv("APP_ENV", "default"),
            "app": "flaude",
        },
        run_name="flaude_turn",
    )
