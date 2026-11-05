"""
rag.py: Vector store management, embeddings, RAG chains, and class-based RAGEngine.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any

import pymupdf
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT: Path = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exception import (
    ConfigurationError,
    DocumentParsingError,
    EmbeddingError,
    MyException,
    ValidationError,
)
from src.logger import logging
from src.utils import (
    FILES_DIRPATH,
    PARAMS,
    PERSIST_DIRPATH,
    console,
    generate_uuid,
)

__all__: list[str] = [
    "EMBEDDING_MODEL",
    "PARAMS",
    "RAGEngine",
]

load_dotenv()

try:
    SPLIT_CONFIG: dict[str, Any] = PARAMS.get("document_splitting", {})
    EMBEDDINGS_CONFIG: dict[str, Any] = PARAMS.get("embeddings", {})
    RETRIEVER_CONFIG: dict[str, Any] = PARAMS.get("retriever", {})

    EMBEDDING_MODEL: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(
        **EMBEDDINGS_CONFIG
    )
except Exception as e:
    logging.info(f"Failed to load RAG configuration or models: {e}")
    raise ConfigurationError(e, sys) from e


class RAGEngine:
    """Encapsulates vector store lifecycle, embeddings, ingestion, and retriever access."""

    def __init__(
        self,
        files_dirpath: Path | str = FILES_DIRPATH,
        persist_dirpath: Path | str = PERSIST_DIRPATH,
        embedding_model: GoogleGenerativeAIEmbeddings = EMBEDDING_MODEL,
        use_existing: bool = True,
    ) -> None:
        """Initialize the RAGEngine with directory paths and models.

        Args:
            files_dirpath (Path | str): Path to the source documents directory.
            persist_dirpath (Path | str): Path for vector store persistence.
            embedding_model (GoogleGenerativeAIEmbeddings): Embeddings model instance.
            use_existing (bool): Whether to load an existing persisted vector store.

        Raises:
            ValidationError: If configuration paths are invalid.
        """
        try:
            logging.info("Initializing RAGEngine...")
            self.files_dirpath: Path = Path(files_dirpath)
            self.persist_dirpath: Path = Path(persist_dirpath)
            self.use_existing: bool = use_existing

            self.embedding_model: GoogleGenerativeAIEmbeddings = embedding_model
            self._vector_store: Chroma | None = None
            logging.info("RAGEngine initialized successfully.")

        except Exception as e:
            logging.info(f"Failed to initialize RAGEngine: {e}")
            raise ValidationError(e, sys) from e

    @property
    def vector_store(
        self,
    ) -> Chroma:
        """Lazily load or return the active Chroma vector store.

        Returns:
            Chroma: Active vector store instance.
        """
        if self._vector_store is None:
            self._vector_store = self.get_vector_store()
        return self._vector_store

    def update_paths(
        self,
        files_dirpath: Path | str | None = None,
        persist_dirpath: Path | str | None = None,
    ) -> None:
        """Update source files and persistence directory paths dynamically.

        Args:
            files_dirpath (Path | str | None): Optional new source files directory path.
            persist_dirpath (Path | str | None): Optional new persistence directory path.
        """
        if files_dirpath is not None:
            self.files_dirpath = Path(files_dirpath)
        if persist_dirpath is not None:
            self.persist_dirpath = Path(persist_dirpath)

    def _ingest_documents(
        self,
        dirpath: Path | str,
    ) -> list[Document]:
        """Ingest PDF documents from a directory using PyMuPDF with metadata and deterministic IDs.

        Args:
            dirpath (Path | str): Directory containing PDF files.

        Returns:
            list[Document]: Ingested LangChain Document objects.

        Raises:
            ValidationError: If directory does not exist or input is invalid.
            DocumentParsingError: If reading or parsing PDF documents fails.
        """
        try:
            path: Path = Path(dirpath)
            if not path.exists():
                raise ValidationError(f"Directory not found: {path}", sys)

            logging.info(f"Ingesting PDF documents from: {path}")
            documents: list[Document] = []
            with console.status("ingesting documents...", spinner="dots"):
                for file in path.rglob("*.pdf"):
                    with pymupdf.open(file) as f:
                        for i, page in enumerate(f):
                            text: str = page.get_text()
                            documents.append(
                                Document(
                                    id=generate_uuid(name=text, as_hex=True),
                                    page_content=text,
                                    metadata={
                                        "source": str(file),
                                        "page": i + 1,
                                        "word_count": len(re.findall(r"\w+", text)),
                                        "character_count": len(text),
                                    },
                                )
                            )

            logging.info(f"Successfully ingested {len(documents)} document pages.")
            return documents

        except MyException:
            raise
        except Exception as e:
            logging.info(f"Failed to ingest documents: {e}")
            raise DocumentParsingError(e, sys) from e

    def _split_documents(
        self,
        documents: list[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[Document]:
        """Split documents into splits using RecursiveCharacterTextSplitter.

        Args:
            documents (list[Document]): List of documents to split.
            chunk_size (int): Maximum chunk character length.
            chunk_overlap (int): Character overlap between consecutive splits.

        Returns:
            list[Document]: Chunked document objects.

        Raises:
            ValidationError: If documents input is empty or invalid.
            DocumentParsingError: If splitting fails.
        """
        try:
            if not documents:
                raise ValidationError("documents cannot be empty", sys)

            logging.info(
                f"Splitting {len(documents)} documents (chunk_size={chunk_size}, overlap={chunk_overlap})..."
            )
            with console.status("splitting documents...", spinner="dots"):
                splitter: RecursiveCharacterTextSplitter = (
                    RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                splits: list[Document] = splitter.split_documents(documents)

                for split in splits:
                    split.page_content = split.page_content.encode(
                        "utf-8", "ignore"
                    ).decode("utf-8", "ignore")

            logging.info(f"Successfully created {len(splits)} document splits.")
            return splits
        except MyException:
            raise
        except Exception as e:
            logging.info(f"Failed to split documents: {e}")
            raise DocumentParsingError(e, sys) from e

    def _embed_documents(
        self,
        documents: list[Document] | None = None,
        force_recreate: bool = False,
    ) -> Chroma:
        """Load existing Chroma vector store or create a new one from documents.

        Args:
            documents (list[Document] | None): Document splits to embed.
            force_recreate (bool): Whether to force indexing even if store exists.

        Returns:
            Chroma: Active Chroma vector store.

        Raises:
            EmbeddingError: If embedding or vector store persistence fails.
        """
        try:
            persist_dir_str: str = str(self.persist_dirpath)
            store_exists: bool = os.path.exists(persist_dir_str) and bool(
                os.listdir(persist_dir_str)
            )

            if self.use_existing and store_exists and not force_recreate:
                logging.info(
                    f"Loading existing vector store from: {persist_dir_str}..."
                )
                with console.status("Loading existing vector store...", spinner="dots"):
                    self._vector_store = Chroma(
                        persist_directory=persist_dir_str,
                        embedding_function=self.embedding_model,
                    )
                logging.info("Vector store loaded successfully from disk.")

            else:
                if documents is None:
                    logging.info(
                        "No documents provided; auto-ingesting from files directory..."
                    )

                    ingested_documents: list[Document] = self._ingest_documents(
                        self.files_dirpath
                    )
                    chunk_size: int = int(SPLIT_CONFIG.get("chunk_size", 1000))
                    chunk_overlap: int = int(SPLIT_CONFIG.get("chunk_overlap", 200))
                    documents = self._split_documents(
                        documents=ingested_documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                logging.info(
                    f"Creating new vector store with {len(documents)} splits..."
                )
                with console.status("Creating new vector store...", spinner="dots"):
                    self._vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embedding_model,
                        persist_directory=persist_dir_str,
                    )
                logging.info("New vector store created and persisted.")

            return self._vector_store

        except Exception as e:
            logging.info(f"Failed to embed documents into vector store: {e}")
            raise EmbeddingError(e, sys) from e

    def run_ingestion_pipeline(self, files_dirpath: Path | str | None = None) -> Chroma:
        """Execute complete pipeline: ingest documents, split documents, and embed into vector store.

        Args:
            files_dirpath (Path | str | None): Directory containing source documents.

        Returns:
            Chroma: Active vector store after ingestion.

        Raises:
            MyException: If any pipeline stage fails.
        """
        try:
            target_dirpath: Path = (
                Path(files_dirpath) if files_dirpath is not None else self.files_dirpath
            )
            logging.info(f"Running ingestion pipeline on: {target_dirpath}...")
            documents: list[Document] = self._ingest_documents(target_dirpath)

            chunk_size: int = int(SPLIT_CONFIG.get("chunk_size", 1000))
            chunk_overlap: int = int(SPLIT_CONFIG.get("chunk_overlap", 200))

            splits: list[Document] = self._split_documents(
                documents=documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            logging.info("Ingestion pipeline ran successfully.")
            return self._embed_documents(documents=splits, force_recreate=True)

        except MyException:
            raise
        except Exception as e:
            logging.info(f"Ingestion pipeline failed: {e}")
            raise MyException(e, sys) from e

    def get_vector_store(self) -> Chroma:
        """Get the active Chroma vector store, initializing if necessary.

        Returns:
            Chroma: The Chroma vector store instance.
        """
        return self._embed_documents(documents=None, force_recreate=False)

    def get_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        """Retrieve documents directly without passing vector_store as an argument.

        Args:
            k (int | None): Number of documents to retrieve. Defaults to configured value.

        Returns:
            VectorStoreRetriever: Configured retriever instance.

        Raises:
            ConfigurationError: If retriever construction fails.
        """
        try:
            k_value: int = k if k is not None else int(RETRIEVER_CONFIG.get("k", 4))
            search_type: str = str(RETRIEVER_CONFIG.get("search_type", "similarity"))
            logging.info(
                f"Configuring retriever (k={k_value}, search_type={search_type})..."
            )

            with console.status("Configuring retriever...", spinner="dots"):
                retriever: VectorStoreRetriever = self.vector_store.as_retriever(
                    search_type=search_type,
                    search_kwargs={"k": k_value},
                )

            logging.info("Retriever configured.")
            return retriever

        except Exception as e:
            logging.info(f"Failed to create retriever: {e}")
            raise ConfigurationError(e, sys) from e
