from langgraph.checkpoint.sqlite import SqliteSaver
from utils.config import get_conn, params_configs, prompt_configs


def load_conversations():
    conn = get_conn(params_configs["files"]["chat_db_filepath"])
    checkpointer = SqliteSaver(conn=conn)
    threads = list({checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpointer.list(None)})
    return threads


def load_thread_mapping():
    conn = get_conn(db_path=params_configs["files"]["mapping_db_filepath"])
    cursor = conn.cursor()

    cursor.execute(prompt_configs["create_table"])
    cursor.execute(prompt_configs["load_rows"])
    thread_mappings = dict(cursor.fetchall())

    conn.commit()
    return thread_mappings


def save_row(thread_id, thread_name):
    conn = get_conn(db_path=params_configs["files"]["mapping_db_filepath"])
    cursor = conn.cursor()
    cursor.execute(prompt_configs["create_table"])
    cursor.execute(
        prompt_configs["insert_row"],
        (thread_id, thread_name),
    )
    conn.commit()
