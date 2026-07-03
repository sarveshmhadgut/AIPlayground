import yaml
from pathlib import Path
from langsmith import traceable
from langchain_chroma import Chroma
from utils.config import get_embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PARAMS_CONFIGS = yaml.safe_load((Path(__file__).parent.parent / "configs/params.yaml").read_text())
VECTOR_DB_PATH = PARAMS_CONFIGS["files"]["vector_db_path"]


def get_retriever(persist_directory, embeddings):
    if Path(VECTOR_DB_PATH).exists():
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

    else:
        vector_store = Chroma.from_documents(
            persist_directory=persist_directory,
            embedding=embeddings,
        )

    retriever = vector_store.as_retriever(**PARAMS_CONFIGS["retriever"])
    return retriever


@traceable(name="ingestion_pipeline")
def ingestion_pipeline(filepath, chunk_size, chunk_overlap):
    loader = PyPDFLoader(file_path=filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents=docs)

    embeddings = get_embeddings(**PARAMS_CONFIGS["embeddings"])
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )

    return {
        "filepath": filepath,
        "docs": len(docs),
        "chunks": len(chunks),
    }
