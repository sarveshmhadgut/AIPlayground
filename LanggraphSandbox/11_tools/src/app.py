# imports & setups
import os
import yaml
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PARAMS_CONFIGS = yaml.safe_load(
    (Path(__file__).parent.parent / "configs/params.yaml").read_text()
)

os.environ["LANGSMITH_PROJECT"] = PARAMS_CONFIGS["LANGSMITH_PROJECT"]
os.environ["LANGSMITH_TRACING_V2"] = str(PARAMS_CONFIGS["LANGCHAIN_TRACING_V2"]).lower()

from utils.config import load_css
from utils.sidebar import render_sidebar
from utils.chat import display_messages, handle_input
from utils.session import new_conversation, init_session
from utils.database import load_thread_mapping, load_conversations

style_css = load_css(filepath=Path(PARAMS_CONFIGS["files"]["css_filepath"]))

# title and header
st.set_page_config(page_title="Flaude", layout="centered")
if style_css:
    st.markdown(style_css, unsafe_allow_html=True)
st.markdown("<h1>Flaude</h1>", unsafe_allow_html=True)

# init params
init_session(
    load_conversations=load_conversations, load_thread_mapping=load_thread_mapping
)

# display current conversation messages
display_messages(st.session_state["messages"])

# new chat button
if st.sidebar.button(
    "New Conversation",
    icon=":material/add:",
    key="btn_new_chat",
    type="primary",
    use_container_width=True,
):
    new_conversation()

# render sidebar
st.sidebar.header("Conversations")
render_sidebar()

# input field
user_input = st.chat_input("Ask Flaude")
if user_input:
    need_rerun = handle_input(user_input=user_input)
    if need_rerun:
        st.rerun()
