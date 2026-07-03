# imports & setups
import os
import yaml
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

PARAMS_CONFIGS = yaml.safe_load((Path(__file__).parent / "configs/params.yaml").read_text())
PROMPTS_CONFIGS = yaml.safe_load((Path(__file__).parent / "configs/prompts.yaml").read_text())
load_dotenv()

# model & parser
model = ChatGoogleGenerativeAI(**PARAMS_CONFIGS["llm"])
parser = StrOutputParser()

# cursor
os.makedirs("db", exist_ok=True)
conn = sqlite3.connect("db/flaude_mapping.db", check_same_thread=False)
cursor = conn.cursor()

# init table
cursor.execute(PROMPTS_CONFIGS["create_table"])
conn.commit()


def generate_title(thread_id, conversation_history):
    if not conversation_history:
        return "new conversation"

    prompt = ChatPromptTemplate(
        [
            ("system", PROMPTS_CONFIGS["generate_title"]),
            ("user", "Conversation:\n\t{conversation_history}"),
        ]
    )

    chain = prompt | model | parser
    res = chain.invoke({"conversation_history": conversation_history})
    cursor.execute(
        PROMPTS_CONFIGS["insert_row"],
        (thread_id, res),
    )
    conn.commit()
    return res


def load_thread_mapping():
    cursor.execute(PROMPTS_CONFIGS["load_rows"])
    thread_mappings = dict(cursor.fetchall())
    return thread_mappings
