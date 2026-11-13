"""
rag.py: Vector store management, embeddings, RAG chains, and class-based RAGEngine.
"""

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
from src.utils.display import get_terminal_width
from src.utils.helpers import generate_uuid, load_yaml_file

try:
    load_dotenv()

    WIDTH: int = get_terminal_width()

    CONFIG_DIRPATH: Path = (ROOT / "config").resolve()
    PARAMS_FILEPATH: Path = (CONFIG_DIRPATH / "params.yml").resolve()
    PARAMS: dict[str, Any] = load_yaml_file(PARAMS_FILEPATH)

    PATHS_CONFIG: dict[str, Any] = PARAMS.get("paths", {})
    FILES_DIRNAME: str = str(PATHS_CONFIG.get("files_dir", "files"))
    PERSIST_DIRNAME: str = str(PATHS_CONFIG.get("persist_directory", "chroma_store"))

    DOCUMENTS_DIRPATH: Path = (ROOT / FILES_DIRNAME / "documents").resolve()
    PERSIST_DIRPATH: Path = (ROOT / PERSIST_DIRNAME).resolve()

    SPLIT_CONFIG: dict[str, Any] = PARAMS.get("document_splitting", {})
    CHUNK_SIZE: int = int(SPLIT_CONFIG.get("chunk_size", 1000))
    CHUNK_OVERLAP: int = int(SPLIT_CONFIG.get("chunk_overlap", 200))

    EMBEDDINGS_CONFIG: dict[str, Any] = PARAMS.get("embeddings", {})
    RETRIEVER_CONFIG: dict[str, Any] = PARAMS.get("retriever", {})
    SEARCH_K: int = int(RETRIEVER_CONFIG.get("k", 4))

    EMBEDDING_MODEL: GoogleGenerativeAIEmbeddings = GoogleGenerativeAIEmbeddings(**EMBEDDINGS_CONFIG)

except Exception as e:
    logging.info(f"Failed to initialize rag module dependencies: {e}")
    raise MyException(e, sys) from e


class RAGEngine:
    """Encapsulates vector store lifecycle, embeddings, ingestion, and retriever access."""

    def __init__(
        self,
        files_dirpath: Path | str = DOCUMENTS_DIRPATH,
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

    def get_indexed_sources(self) -> set[str]:
        """Return the set of document filenames already indexed in the vector store.

        Returns:
            set[str]: Lowercase filenames of indexed documents.
        """
        try:
            store_exists: bool = self.persist_dirpath.exists() and any(self.persist_dirpath.iterdir())
            if not store_exists:
                return set()
            if self._vector_store is None:
                self._vector_store = Chroma(
                    persist_directory=str(self.persist_dirpath),
                    embedding_function=self.embedding_model,
                )
            records: Any = self._vector_store._collection.get(include=["metadatas"])
            metadatas: list[dict[str, Any]] = records.get("metadatas", []) or []
            return {Path(str(m.get("source", ""))).name.lower() for m in metadatas if m and m.get("source")}

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            logging.info(f"Failed to check indexed sources: {e}")
            return set()

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

            logging.info(f"Splitting {len(documents)} documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")

            splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            splits: list[Document] = splitter.split_documents(documents)

            for split in splits:
                # Sanitize document text encoding to prevent downstream embedding failures
                split.page_content = split.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
                # Deterministic UUID5 ensures idempotent IDs across re-index runs
                split.id = generate_uuid(name=split.page_content, as_hex=True)

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
            store_exists: bool = self.persist_dirpath.exists() and any(self.persist_dirpath.iterdir())

            if self.use_existing and store_exists and not force_recreate:
                logging.info(f"Loading existing vector store from: {persist_dir_str}...")

                self._vector_store = Chroma(
                    persist_directory=persist_dir_str,
                    embedding_function=self.embedding_model,
                )
                logging.info("Vector store loaded successfully from disk.")

                # Ingest any documents in files_dirpath that are not yet indexed in the vector store
                target_path: Path = Path(self.files_dirpath)
                if target_path.exists():
                    existing_sources: set[str] = self.get_indexed_sources()
                    all_pdf_files: list[Path] = list(target_path.rglob("*.pdf"))
                    missing_pdf_files: list[Path] = [
                        pdf for pdf in all_pdf_files if pdf.name.lower() not in existing_sources
                    ]

                    if missing_pdf_files:
                        logging.info(
                            f"Detected {len(missing_pdf_files)} unindexed document(s): "
                            f"{[f.name for f in missing_pdf_files]}. Ingesting and embedding..."
                        )
                        missing_documents: list[Document] = []

                        for pdf in missing_pdf_files:
                            with pymupdf.open(pdf) as f:
                                for i, page in enumerate(f):
                                    text: str = page.get_text()
                                    missing_documents.append(
                                        Document(
                                            id=generate_uuid(name=text, as_hex=True),
                                            page_content=text,
                                            metadata={
                                                "source": str(pdf),
                                                "page": i + 1,
                                                "word_count": len(re.findall(r"\w+", text)),
                                                "character_count": len(text),
                                            },
                                        )
                                    )

                        chunk_size: int = int(SPLIT_CONFIG.get("chunk_size", CHUNK_SIZE))
                        chunk_overlap: int = int(SPLIT_CONFIG.get("chunk_overlap", CHUNK_OVERLAP))
                        missing_splits: list[Document] = self._split_documents(
                            documents=missing_documents,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )

                        batch_size: int = 100

                        for start_idx in range(0, len(missing_splits), batch_size):
                            batch: list[Document] = missing_splits[start_idx : start_idx + batch_size]
                            batch_ids = [
                                getattr(s, "id", None) or generate_uuid(name=s.page_content, as_hex=True) for s in batch
                            ]
                            self._vector_store.add_documents(documents=batch, ids=batch_ids)

                        logging.info(f"Successfully indexed {len(missing_splits)} new splits into vector store.")

            else:
                if documents is None:
                    logging.info("No documents provided; auto-ingesting from files directory...")

                    ingested_documents: list[Document] = self._ingest_documents(self.files_dirpath)
                    chunk_size = int(SPLIT_CONFIG.get("chunk_size", CHUNK_SIZE))
                    chunk_overlap = int(SPLIT_CONFIG.get("chunk_overlap", CHUNK_OVERLAP))
                    documents = self._split_documents(
                        documents=ingested_documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                logging.info(f"Creating new vector store with {len(documents)} splits...")
                ids: list[str] = [
                    getattr(split, "id", None) or generate_uuid(name=split.page_content, as_hex=True)
                    for split in documents
                ]
                batch_size = 100

                if len(documents) > batch_size:
                    first_batch: list[Document] = documents[:batch_size]
                    first_ids: list[str] = ids[:batch_size]
                    self._vector_store = Chroma.from_documents(
                        documents=first_batch,
                        embedding=self.embedding_model,
                        persist_directory=persist_dir_str,
                        ids=first_ids,
                    )
                    for start_idx in range(batch_size, len(documents), batch_size):
                        batch = documents[start_idx : start_idx + batch_size]
                        batch_ids = ids[start_idx : start_idx + batch_size]
                        self._vector_store.add_documents(documents=batch, ids=batch_ids)
                else:
                    self._vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embedding_model,
                        persist_directory=persist_dir_str,
                        ids=ids,
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
            target_dirpath: Path = Path(files_dirpath) if files_dirpath is not None else self.files_dirpath
            logging.info(f"Running ingestion pipeline on: {target_dirpath}...")
            documents: list[Document] = self._ingest_documents(target_dirpath)

            chunk_size: int = int(SPLIT_CONFIG.get("chunk_size", CHUNK_SIZE))
            chunk_overlap: int = int(SPLIT_CONFIG.get("chunk_overlap", CHUNK_OVERLAP))

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
            k_value: int = k if k is not None else int(RETRIEVER_CONFIG.get("k", SEARCH_K))
            search_type: str = str(RETRIEVER_CONFIG.get("search_type", "similarity"))
            logging.info(f"Configuring retriever (k={k_value}, search_type={search_type})...")

            retriever: VectorStoreRetriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k_value},
            )

            logging.info("Retriever configured.")
            return retriever

        except Exception as e:
            logging.info(f"Failed to create retriever: {e}")
            raise ConfigurationError(e, sys) from e


__all__: list[str] = [
    "RAGEngine",
]
