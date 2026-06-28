import uuid
import streamlit as st


def new_conversation():
    thread_id = str(uuid.uuid1())
    st.session_state["current_thread"] = thread_id

    if thread_id not in st.session_state["threads"]:
        st.session_state["threads"].append(thread_id)
    st.session_state["messages"] = []
    st.rerun()


def init_session(load_conversations, load_thread_mapping):
    if "threads" not in st.session_state:
        st.session_state["threads"] = load_conversations()

    if "thread_mapping" not in st.session_state:
        st.session_state["thread_mapping"] = load_thread_mapping()

    if "current_thread" not in st.session_state:
        new_conversation()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
