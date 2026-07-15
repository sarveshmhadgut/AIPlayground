from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()


class ChromaStore:
    """Manages the local ChromaDB vector store for transcript embeddings."""

    def __init__(self, persist_directory: str = "chroma_vector_store") -> None:
        """
        Initialize the ChromaDB vector store and Google embeddings.

        Args:
            persist_directory (str): The local directory to save the ChromaDB database.
        """
        self.persist_directory: str = persist_directory
        self.embeddings: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )
        self.store: Chroma = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.retriever = self.store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def add_docs(self, docs: list[Document]) -> None:
        """
        Add document chunks to the ChromaDB vector store.

        Args:
            docs (list[Document]): The list of Langchain Documents to insert.
        """
        try:
            if not docs:
                return
            self.store.add_documents(documents=docs)

        except Exception as e:
            print(f"Error persisting documents to vector store: {e}")

    def get_docs(self, include: List[str] | None = None) -> Dict[str, Any]:
        """
        Fetch documents and metadata from the ChromaDB store.

        Args:
            include (List[str] | None): Attributes to return. Defaults to ["documents", "metadatas"].

        Returns:
            Dict[str, Any]: The requested database items.
        """
        try:
            if include is None:
                include = ["documents", "metadatas"]
            return self.store.get(include=include)

        except Exception as e:
            print(f"Failed to extract document map from Vector database: {e}")
            return {}
