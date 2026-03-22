from langchain_community import BM25Retriever
from langchain_core.documents import Document


class Retriever:
    """Uses a BM25 sparse retriever to search document collections."""

    def __init__(self, docs: list[Document]) -> None:
        """
        Initialize the BM25 retriever with the loaded documents.

        Args:
            docs (list[Document]): A list of initialized Langchain Documents.
        """
        try:
            self.retriever: BM25Retriever = BM25Retriever.from_documents(
                documents=docs
            )
            self.retriever.k = 5

        except Exception as e:
            print(f"Error initializing BM25 retriever: {e}")
