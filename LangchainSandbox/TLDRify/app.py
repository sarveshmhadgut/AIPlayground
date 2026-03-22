import json
import urllib.request
import streamlit as st
from typing import Any
from model import ChatModel
from store import ChromaStore
from splitter import TranscriptSplitter
from loader import YouTubeTranscriptLoader
from prompt_generator import PromptGenerator
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

with open("css.txt") as f:
    streamlit_css = "".join(f.readlines())

st.set_page_config(page_title="TLDRify", layout="centered")
st.markdown(streamlit_css, unsafe_allow_html=True)


@st.cache_resource
def initialize_pipeline(video_id: str) -> tuple[Any, Any]:
    """
    Construct the Langchain pipeline for querying a YouTube transcript.

    Args:
        video_id (str): The video ID to analyze.

    Returns:
        tuple[Any, Any]: The configured Retriever instance and the Langchain query pipeline.
    """
    try:
        loader = YouTubeTranscriptLoader()
        transcript = loader.fetch(video_id=video_id)

        splitter = TranscriptSplitter()
        chunks = splitter.split(transcript)

        chroma_store = ChromaStore(persist_directory=f"store_{video_id}")
        chroma_store.add_docs(docs=chunks)

        retriever = chroma_store.retriever
        model = ChatModel(model="gemini-3.1-pro-preview", temperature=0).model

        prompt = PromptGenerator().prompt
        parser = StrOutputParser()

        chain = prompt | model | parser
        return retriever, chain

    except Exception as e:
        st.error(f"Critical error initializing pipeline dependencies: {e}")
        st.stop()


if "video_id" not in st.session_state:
    st.session_state.video_id = "9M_QK4stCJU"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data
def get_video_title(video_id: str) -> str:
    """
    Fetch the video title from the official YouTube oEmbed API.

    Args:
        video_id (str): The video ID to look up.

    Returns:
        str: The video title, or a placeholder if the request fails.
    """
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return data.get("title", f"Video {video_id}")

    except Exception as e:
        print(f"Error quietly suppressing oEmbed title grab fail: {e}")
        return f"Video {video_id}"


st.title("YouTube TLDRify")

with st.sidebar:
    st.header("Video ID")

    video_id_input = st.text_input(
        "YouTube Video ID",
        value=st.session_state.video_id,
        label_visibility="collapsed",
        placeholder="Enter YouTube Video ID",
    )

    if st.button("Load Video", use_container_width=True):
        st.session_state.video_id = video_id_input
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

video_title = get_video_title(st.session_state.video_id)
thumbnail_url = (
    f"https://img.youtube.com/vi/{st.session_state.video_id}/maxresdefault.jpg"
)

st.markdown(
    f"""
    <a href="https://www.youtube.com/watch?v={st.session_state.video_id}" target="_blank" class="video-banner-link">
        <div class="video-banner-container">
            <div class="video-banner-bg" style="background-image: url('{thumbnail_url}');"></div>
            <div class="video-banner-content">
                <h3 style="color: white !important;">{video_title}</h3>
            </div>
        </div>
    </a>
    """,
    unsafe_allow_html=True,
)

retriever, chain = initialize_pipeline(st.session_state.video_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message"></div>{msg["content"]}',
                unsafe_allow_html=True,
            )

        else:
            st.markdown(msg["content"])

user_input: str = st.chat_input("Ask something about the video...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-message"></div>{user_input}',
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant"):
        with st.spinner(""):
            docs: list[Document] = retriever.invoke(user_input)
            context = "\n".join(doc.page_content for doc in docs)

            response = chain.invoke(
                {
                    "context": context,
                    "chat_history": st.session_state.chat_history,
                    "user_message": user_input,
                }
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    if "does not contain information" not in response:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response))

    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-9:]
