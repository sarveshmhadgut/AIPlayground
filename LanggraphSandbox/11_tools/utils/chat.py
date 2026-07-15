import streamlit as st
from src.backend import workflow as llm
from utils.config import get_runnable_config
from src.chat_title_generator import generate_title
from langsmith import traceable, get_current_run_tree
from langchain_core.messages import HumanMessage, ToolMessage, AIMessageChunk


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


def stream_chat(user_input, config, stream_mode="messages", status_holder=None):
    for message_chunk, metadata in llm.stream(
        input={"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode=stream_mode,
    ):
        if isinstance(message_chunk, AIMessageChunk):
            content = message_chunk.content
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

        if isinstance(message_chunk, ToolMessage):
            tool_name = getattr(message_chunk, "name", "tool")
            if status_holder["box"] is None:
                status_holder["box"] = st.status(
                    f"Using `{tool_name}` …", expanded=True
                )
            else:
                status_holder["box"].update(
                    label=f"Using `{tool_name}` ...",
                    state="running",
                    expanded=True,
                )


@traceable(name="flaude_run")
def handle_input(user_input):
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-message">{user_input}</div>',
            unsafe_allow_html=True,
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
        tool_block = {"box": None}
        ai_message = st.write_stream(
            stream_chat(
                user_input=user_input,
                config=config,
                stream_mode="messages",
                status_holder=tool_block,
            )
        )

        if tool_block["box"] is not None:
            tool_block["box"].update(
                label="Tool finished",
                state="complete",
                expanded=False,
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
