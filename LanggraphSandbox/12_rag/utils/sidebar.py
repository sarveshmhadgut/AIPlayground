import streamlit as st
from utils.rag import get_retriever
from utils.database import save_file
from src.backend import workflow as llm
from utils.config import get_runnable_config
from langchain_core.messages import HumanMessage, AIMessage


def load_conversation_history(thread_id):
    thread_name = st.session_state["thread_mapping"].get(thread_id, f"Conversation {thread_id[:8]}")
    state = llm.get_state(
        config=get_runnable_config(
            thread_id=thread_id,
            thread_name=thread_name,
        )
    )
    return state.values.get("messages", [])


def render_conversations():
    for thread in reversed(st.session_state["threads"]):
        title = st.session_state["thread_mapping"].get(thread, "")

        if title and st.sidebar.button(
            title,
            icon=":material/chat:",
            key=f"btn_{thread}",
            use_container_width=True,
        ):
            st.session_state["current_thread"] = thread
            conversation_history = load_conversation_history(thread_id=st.session_state["current_thread"])

            retriever = get_retriever(thread)
            vector_store = retriever.vectorstore

            if vector_store._collection.count() > 0:
                db_data = vector_store.get(limit=1)
                filepath = db_data["metadatas"][0].get("source", "Unknown Document")
                st.session_state["metadatas"][thread] = {"filepath": filepath}

            previous_messages = []

            for message in conversation_history:
                if isinstance(message, HumanMessage):
                    previous_messages.append({"role": "user", "content": str(message.content)})

                elif isinstance(message, AIMessage):
                    content = message.content
                    content = (
                        content
                        if isinstance(content, str)
                        else "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                        if content
                        else ""
                    )
                    if content:
                        previous_messages.append({"role": "assistant", "content": content})

            st.session_state["messages"] = previous_messages
            st.rerun()


def render_active_documents():
    st.sidebar.header("Active Documents")

    current_thread = st.session_state["current_thread"]
    if current_thread in st.session_state.get("metadatas", {}):
        metadata = st.session_state["metadatas"][current_thread]

        filename = metadata["filepath"].split("/")[-1]
        st.sidebar.button(
            filename,
            icon=":material/description:",
            key=f"active_doc_{current_thread}",
            use_container_width=True,
        )
    else:
        st.sidebar.caption("No documents loaded for this conversation.")


def render_sidebar():
    render_conversations()
    render_active_documents()
