"""Manage the persistent local Chroma store for document chunks."""

from pathlib import Path
from typing import Any

from chromadb.api import CreateCollectionConfiguration
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import INDEX_CONFIG
from src.embeddings import get_embedding_model


DEFAULT_PERSIST_DIRECTORY = INDEX_CONFIG.persist_directory
COLLECTION_NAME = INDEX_CONFIG.collection_name
DISTANCE_METRIC = INDEX_CONFIG.distance_metric
COLLECTION_CONFIGURATION: CreateCollectionConfiguration = {
    "hnsw": {"space": DISTANCE_METRIC}
}
REQUIRED_CHUNK_METADATA = INDEX_CONFIG.required_chunk_metadata


class VectorStoreError(RuntimeError):
    """Base error for expected local vector-store failures."""


class InvalidChunkError(VectorStoreError, ValueError):
    """Raised when chunks are missing text or required metadata."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when the requested persisted collection does not exist."""


class VectorStoreNotEmptyError(VectorStoreError):
    """Raised when indexing is attempted without an explicit rebuild."""


def validate_chunks(chunks: list[Document]) -> list[str]:
    """Validate chunks and return their deterministic Chroma IDs."""
    if not isinstance(chunks, list) or not chunks:
        raise InvalidChunkError("Provide a non-empty list of chunk Documents.")

    chunk_ids: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, Document):
            raise InvalidChunkError(
                f"Chunk {position} is not a LangChain Document object."
            )
        if not chunk.page_content.strip():
            raise InvalidChunkError(f"Chunk {position} contains no text.")

        missing_fields = [
            field
            for field in REQUIRED_CHUNK_METADATA
            if field not in chunk.metadata or chunk.metadata[field] in (None, "")
        ]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise InvalidChunkError(
                f"Chunk {position} is missing required metadata: {missing}."
            )

        chunk_id = chunk.metadata["chunk_id"]
        if not isinstance(chunk_id, str):
            raise InvalidChunkError(
                f"Chunk {position} has a chunk_id that is not text."
            )
        chunk_ids.append(chunk_id)

    if len(chunk_ids) != len(set(chunk_ids)):
        raise InvalidChunkError("Chunk IDs must be unique before indexing.")
    return chunk_ids


def create_or_open_vector_store(
    embedding_model: Embeddings | None = None,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Create or open the persistent collection with the configured model."""
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = embedding_model or get_embedding_model()

    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=model,
            persist_directory=str(directory),
            collection_configuration=COLLECTION_CONFIGURATION,
            create_collection_if_not_exists=True,
        )
    except Exception as error:
        raise VectorStoreError(
            f"The Chroma collection '{collection_name}' could not be created or "
            f"opened at '{directory}'. Check the directory permissions and try again."
        ) from error


def open_existing_vector_store(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    embedding_model: Embeddings | None = None,
) -> Chroma:
    """Reopen an existing collection without indexing any documents."""
    directory = Path(persist_directory)
    if not directory.exists():
        raise VectorStoreNotFoundError(
            f"No persisted Chroma index exists at '{directory}'. Build it first."
        )

    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(directory),
            collection_configuration=COLLECTION_CONFIGURATION,
            create_collection_if_not_exists=False,
        )
    except Exception as error:
        raise VectorStoreNotFoundError(
            f"The Chroma collection '{collection_name}' was not found at "
            f"'{directory}'. Build the development index first."
        ) from error


def get_stored_document_count(vector_store: Chroma) -> int:
    """Return the stored item count through LangChain Chroma's public API."""
    try:
        result = vector_store.get(include=[])
        return len(result.get("ids", []))
    except Exception as error:
        raise VectorStoreError(
            "The stored Chroma document count could not be read."
        ) from error


def add_chunks(vector_store: Chroma, chunks: list[Document]) -> list[str]:
    """Index validated chunks into an empty collection exactly once."""
    chunk_ids = validate_chunks(chunks)
    existing_count = get_stored_document_count(vector_store)
    if existing_count:
        raise VectorStoreNotEmptyError(
            f"The collection already contains {existing_count} chunks. Open the "
            "existing index or use the explicit rebuild operation instead."
        )

    try:
        stored_ids = vector_store.add_documents(documents=chunks, ids=chunk_ids)
    except Exception as error:
        raise VectorStoreError(
            "The chunks could not be embedded and stored in Chroma. Check the "
            "embedding model, available disk space, and chunk metadata."
        ) from error

    if stored_ids != chunk_ids:
        raise VectorStoreError(
            "Chroma returned document IDs that did not match the deterministic "
            "chunk IDs provided by the project."
        )
    return stored_ids


def rebuild_vector_store(
    chunks: list[Document],
    embedding_model: Embeddings | None = None,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Explicitly replace only the configured development collection."""
    validate_chunks(chunks)
    model = embedding_model or get_embedding_model()
    vector_store = create_or_open_vector_store(
        embedding_model=model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    try:
        vector_store.delete_collection()
    except Exception as error:
        raise VectorStoreError(
            f"The existing Chroma collection '{collection_name}' could not be reset."
        ) from error

    rebuilt_store = create_or_open_vector_store(
        embedding_model=model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    add_chunks(rebuilt_store, chunks)
    return rebuilt_store


def get_stored_record(vector_store: Chroma, chunk_id: str) -> dict[str, Any]:
    """Return one stored chunk's ID, text, and metadata for validation."""
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise InvalidChunkError("A non-empty chunk ID is required.")

    try:
        result = vector_store.get(
            ids=[chunk_id], include=["documents", "metadatas"]
        )
    except Exception as error:
        raise VectorStoreError(
            f"The stored chunk '{chunk_id}' could not be read from Chroma."
        ) from error

    if not result.get("ids"):
        raise VectorStoreNotFoundError(
            f"The chunk '{chunk_id}' does not exist in the Chroma collection."
        )
    return {
        "id": result["ids"][0],
        "page_content": result["documents"][0],
        "metadata": result["metadatas"][0],
    }


def vector_store_exists(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """Check whether the configured persisted collection can be reopened."""
    try:
        open_existing_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except VectorStoreNotFoundError:
        return False
    return True
