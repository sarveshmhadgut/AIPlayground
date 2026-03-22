from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TranscriptSplitter:
    """Splits large transcript strings into smaller Langchain Documents."""

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> None:
        """
        Initialize the text splitter.

        Args:
            chunk_size (int): Max characters per chunk.
            chunk_overlap (int): Number of characters to overlap between chunks.
        """
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        )

    def split(self, text: str) -> List[Document]:
        """
        Split the transcript text into Langchain Document chunks.

        Args:
            text (str): The full transcript text.

        Returns:
            List[Document]: The split document chunks.
        """
        try:
            if not text:
                return []

            chunks: List[str] = self.splitter.split_text(text)
            docs: List[Document] = [
                Document(page_content=snippet) for snippet in chunks
            ]
            return docs

        except Exception as e:
            print(f"Error during transcript splitting sequence: {e}")
            return []
