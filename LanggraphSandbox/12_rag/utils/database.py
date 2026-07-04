import os
import streamlit as st
from pathlib import Path
from utils.rag import ingestion_pipeline
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.config import get_conn, PARAMS_CONFIGS, PROMPTS_CONFIGS


def load_conversations():
    conn = get_conn(PARAMS_CONFIGS["files"]["chat_db_filepath"])
    checkpointer = SqliteSaver(conn=conn)
    threads = list({checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpointer.list(None)})
    return threads


def load_thread_mapping():
    conn = get_conn(db_path=PARAMS_CONFIGS["files"]["mapping_db_filepath"])
    cursor = conn.cursor()

    cursor.execute(PROMPTS_CONFIGS["create_table"])
    cursor.execute(PROMPTS_CONFIGS["load_rows"])
    thread_mappings = dict(cursor.fetchall())

    conn.commit()
    return thread_mappings


def save_row(thread_id, thread_name):
    conn = get_conn(db_path=PARAMS_CONFIGS["files"]["mapping_db_filepath"])
    cursor = conn.cursor()
    cursor.execute(PROMPTS_CONFIGS["create_table"])
    cursor.execute(
        PROMPTS_CONFIGS["insert_row"],
        (thread_id, thread_name),
    )
    conn.commit()


def save_file(file):
    if file:
        FILES_DIRPATH = Path(__file__).parent.parent / PARAMS_CONFIGS["files"]["files_dirname"]
        os.makedirs(FILES_DIRPATH, exist_ok=True)

        filepath = FILES_DIRPATH / file.name

        with open(filepath, "wb") as f:
            f.write(file.getbuffer())

        with st.status(f"Processing `{file.name}` ...", expanded=True) as status:
            results = ingestion_pipeline(filepath=str(filepath), **PARAMS_CONFIGS["ingestion_pipeline"])

            if results:
                status.update(label=f"Successfully processed `{file.name}`", state="complete", expanded=False)
                try:
                    os.remove(filepath)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {e}")
            else:
                status.update(label=f"Error processing `{file.name}`!", state="error")
