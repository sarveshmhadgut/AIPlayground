import streamlit as st
from langsmith import traceable, get_current_run_tree
from backend import workflow as llm
from utils.config import get_runnable_config
from chat_title_generator import generate_title
from langchain_core.messages import HumanMessage, AIMessageChunk


def display_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.write(message["content"])


def stream_chat(user_input, config, stream_mode="messages"):
    for message, metadata in llm.stream(
        input={"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode=stream_mode,
    ):
        if isinstance(message, AIMessageChunk):
            content = message.content
            yield (
                content
                if isinstance(content, str)
                else "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
                if content
                else ""
            )


@traceable(name="flaude_run")
def handle_input(user_input):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True
        )

    thread_id = st.session_state["current_thread"]
    thread_name = st.session_state["thread_mapping"].get(
        thread_id, f"Conversation {thread_id[:8]}"
    )

    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.add_metadata({"thread_id": thread_id})

    config = get_runnable_config(thread_id=thread_id, thread_name=thread_name)
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            stream_chat(user_input=user_input, config=config, stream_mode="messages")
        )

    st.session_state["messages"].append({"role": "assistant", "content": ai_message})
    if not st.session_state["thread_mapping"].get(st.session_state["current_thread"]):
        new_title = generate_title(
            thread_id=st.session_state["current_thread"],
            conversation_history=st.session_state["messages"][:2],
        )

        st.session_state["thread_mapping"][st.session_state["current_thread"]] = (
            new_title
        )
        return True
    return False
