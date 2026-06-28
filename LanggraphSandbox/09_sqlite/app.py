# imports & setups
import uuid
import streamlit as st
from pathlib import Path
from langchain_core.messages import HumanMessage
from backend import load_db_conversations, workflow as llm
from chat_title_generator import generate_title, load_thread_mapping

css_path = Path(__file__).parent / "styles/styles.css"
style_css = f"<style>{css_path.read_text()}</style>" if css_path.exists() else ""
import os

st.set_page_config(page_title="Flaude", layout="centered")
if style_css:
    st.markdown(style_css, unsafe_allow_html=True)
st.markdown("<h1>Flaude</h1>", unsafe_allow_html=True)


# utility functions
def append_thread(thread_id):
    if thread_id not in st.session_state["threads"]:
        st.session_state["threads"].append(thread_id)


def display_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.write(message["content"])


def new_conversation():
    thread_id = str(uuid.uuid1())
    st.session_state["current_thread"] = thread_id

    append_thread(st.session_state["current_thread"])
    st.session_state["messages"] = []
    st.rerun()


def load_conversation_history(thread_id):
    state = llm.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# session params
if "threads" not in st.session_state:
    st.session_state["threads"] = load_db_conversations()

if "thread_mapping" not in st.session_state:
    st.session_state["thread_mapping"] = load_thread_mapping()

if "current_thread" not in st.session_state:
    new_conversation()

if "messages" not in st.session_state:
    st.session_state["messages"] = []


config = {"configurable": {"thread_id": st.session_state["current_thread"]}}

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.write(message["content"])

# sidebar
if st.sidebar.button("New Conversation", key="btn_new_chat", type="primary", use_container_width=True):
    new_conversation()

# conversations
st.sidebar.header("Conversations")
for thread in reversed(st.session_state["threads"]):
    title = st.session_state["thread_mapping"].get(thread, "")
    if title:
        if st.sidebar.button(title, key=f"btn_{thread}", use_container_width=True):
            st.session_state["current_thread"] = thread
    
            conversation_history = load_conversation_history(thread_id=st.session_state["current_thread"])
    
            previous_messages = []
            for message in conversation_history:
                if isinstance(message, HumanMessage):
                    role = "user"
                else:
                    role = "assistant"
    
                previous_messages.append({"role": role, "content": message.content})
    
            st.session_state["messages"] = previous_messages
            st.rerun()

user_input = st.chat_input("Ask Flaude")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in llm.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
            if metadata.get("ls_integration") == "langchain_chat_model"
        )

    st.session_state["messages"].append({"role": "assistant", "content": ai_message})
    if not st.session_state["thread_mapping"].get(st.session_state["current_thread"]):
        new_title = generate_title(
            thread_id=st.session_state["current_thread"], conversation_history=st.session_state["messages"][:2]
        )

        st.session_state["thread_mapping"][st.session_state["current_thread"]] = new_title
        st.rerun()
