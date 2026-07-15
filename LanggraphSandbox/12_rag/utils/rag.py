import yaml
import streamlit as st
from pathlib import Path
from langsmith import traceable
from langchain_chroma import Chroma
from utils.config import get_embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PARAMS_CONFIGS = yaml.safe_load(
    (Path(__file__).parent.parent / "configs/params.yaml").read_text()
)
VECTOR_DB_PATH = PARAMS_CONFIGS["files"]["vector_db_path"]


def get_retriever(thread_id):
    if thread_id in st.session_state["retrievers"]:
        return st.session_state["retrievers"].get(thread_id, {})

    embeddings = get_embeddings(params=PARAMS_CONFIGS["embeddings"])
    vector_store = Chroma(
        collection_name=thread_id,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    retriever = vector_store.as_retriever(**PARAMS_CONFIGS["retriever"])
    st.session_state["retrievers"][thread_id] = retriever

    return retriever


@traceable(name="ingestion_pipeline")
def ingestion_pipeline(filepath, chunk_size, chunk_overlap):
    try:
        current_thread = st.session_state["current_thread"]
        loader = PyPDFLoader(file_path=filepath)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents=docs)

        embeddings = get_embeddings(params=PARAMS_CONFIGS["embeddings"])
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH,
            collection_name=current_thread,
        )

        retriever = vector_store.as_retriever(**PARAMS_CONFIGS["retriever"])

        st.session_state["retrievers"][current_thread] = retriever
        st.session_state["metadatas"][current_thread] = {
            "filepath": str(filepath),
            "docs": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filepath": str(filepath),
            "docs": len(docs),
            "chunks": len(chunks),
        }

    except Exception as e:
        print(str(e))
