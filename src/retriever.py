"""Retrieve relevant document chunks from the persisted Chroma index."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.config import INDEX_CONFIG
from src.embeddings import get_embedding_model
from src.vector_store import (
    COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    VectorStoreError,
    VectorStoreNotFoundError,
    get_stored_document_count,
    open_existing_vector_store,
)


TOP_K = INDEX_CONFIG.top_k
MAX_TOP_K = INDEX_CONFIG.max_top_k
SUPPORTED_CATEGORIES = INDEX_CONFIG.supported_categories
DISTANCE_DESCRIPTION = (
    "Raw cosine distance from Chroma; lower is a better match."
)


class RetrievalError(RuntimeError):
    """Base error for expected retrieval failures."""


class InvalidRetrievalInputError(RetrievalError, ValueError):
    """Raised when a query, k value, or category is invalid."""


class RetrievalIndexUnavailableError(RetrievalError):
    """Raised when the persisted Chroma collection cannot be opened."""


class EmptyRetrievalIndexError(RetrievalError):
    """Raised when the opened collection contains no chunks."""


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked document and its raw Chroma cosine distance."""

    document: Document
    distance: float
    rank: int


def _validate_query(query: str) -> str:
    """Return a cleaned query or raise a clear input error."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidRetrievalInputError(
            "The retrieval query must be a non-empty text string."
        )
    return query.strip()


def _validate_k(k: int) -> int:
    """Restrict result counts to the small development range used by the UI."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_TOP_K:
        raise InvalidRetrievalInputError(
            f"k must be a whole number between 1 and {MAX_TOP_K}."
        )
    return k


def _validate_category(category: str | None) -> str | None:
    """Accept one stored category identifier or None for the complete index."""
    if category is None:
        return None
    if not isinstance(category, str) or category not in SUPPORTED_CATEGORIES:
        accepted = ", ".join(SUPPORTED_CATEGORIES)
        raise InvalidRetrievalInputError(
            f"Invalid retrieval category. Use one of: {accepted}; or None."
        )
    return category


def _open_searchable_store(
    *,
    persist_directory: str | Path,
    collection_name: str,
    embedding_model: Embeddings | None,
) -> Chroma:
    """Open the existing index with the project's one embedding configuration."""
    if not Path(persist_directory).exists():
        raise RetrievalIndexUnavailableError(
            "The persisted Chroma index is unavailable. Build the development "
            "index before running retrieval."
        )
    try:
        model = embedding_model or get_embedding_model()
        vector_store = open_existing_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model=model,
        )
    except VectorStoreNotFoundError as error:
        raise RetrievalIndexUnavailableError(
            "The persisted Chroma index is unavailable. Build the development "
            "index before running retrieval."
        ) from error
    return vector_store


def _require_non_empty_store(vector_store: Chroma) -> None:
    """Reject a collection that has no searchable document chunks."""
    try:
        stored_count = get_stored_document_count(vector_store)
    except VectorStoreError as error:
        raise RetrievalError(
            "The Chroma index could not be inspected before retrieval."
        ) from error
    if stored_count == 0:
        raise EmptyRetrievalIndexError(
            "The Chroma collection is empty. Build the development index first."
        )


def retrieve_with_scores(
    query: str,
    k: int = TOP_K,
    category: str | None = None,
    *,
    vector_store: Chroma | None = None,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    embedding_model: Embeddings | None = None,
) -> list[RetrievalResult]:
    """Return ranked chunks and raw cosine distances without modifying Chroma.

    ``similarity_search_with_score`` in the installed ``langchain-chroma``
    integration returns a distance, not a percentage or normalized relevance
    score. Results are ordered from smallest (best match) to largest distance.
    """
    cleaned_query = _validate_query(query)
    result_count = _validate_k(k)
    selected_category = _validate_category(category)
    store = vector_store or _open_searchable_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    _require_non_empty_store(store)

    metadata_filter = (
        {"category": selected_category} if selected_category is not None else None
    )
    try:
        matches = store.similarity_search_with_score(
            query=cleaned_query,
            k=result_count,
            filter=metadata_filter,
        )
    except Exception as error:
        raise RetrievalError(
            "Chroma could not retrieve document chunks. Check that the index was "
            "built with the configured BGE-M3 model and try again."
        ) from error

    ranked_results: list[RetrievalResult] = []
    for rank, (document, distance) in enumerate(matches, start=1):
        numeric_distance = float(distance)
        if not isfinite(numeric_distance):
            raise RetrievalError("Chroma returned a non-finite retrieval distance.")
        ranked_results.append(
            RetrievalResult(
                document=document,
                distance=numeric_distance,
                rank=rank,
            )
        )
    return ranked_results


def documents_from_results(results: list[RetrievalResult]) -> list[Document]:
    """Extract Documents when score details are not needed by later code."""
    return [result.document for result in results]


def retrieve(
    query: str,
    k: int = TOP_K,
    category: str | None = None,
    **kwargs: object,
) -> list[Document]:
    """Return only relevant Documents using the scored inspection path."""
    return documents_from_results(
        retrieve_with_scores(query=query, k=k, category=category, **kwargs)
    )


def create_langchain_retriever(
    k: int = TOP_K,
    category: str | None = None,
    *,
    vector_store: Chroma | None = None,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    embedding_model: Embeddings | None = None,
) -> BaseRetriever:
    """Create LangChain's standard Documents-only similarity retriever."""
    result_count = _validate_k(k)
    selected_category = _validate_category(category)
    store = vector_store or _open_searchable_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    _require_non_empty_store(store)

    search_kwargs: dict[str, object] = {"k": result_count}
    if selected_category is not None:
        search_kwargs["filter"] = {"category": selected_category}
    return store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
