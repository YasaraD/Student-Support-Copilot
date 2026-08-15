"""Split page-level LangChain documents into retrieval-friendly chunks."""

import re
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import INDEX_CONFIG

DEFAULT_CHUNK_SIZE = INDEX_CONFIG.chunk_size
DEFAULT_CHUNK_OVERLAP = INDEX_CONFIG.chunk_overlap
REQUIRED_METADATA_FIELDS = ("source", "filename", "page", "category")


class TextSplittingError(ValueError):
    """Raised when documents or chunk settings cannot be processed safely."""


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate character-based chunk size and overlap settings."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TextSplittingError("Chunk size must be a whole number.")
    if chunk_size <= 0:
        raise TextSplittingError("Chunk size must be greater than zero.")

    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int):
        raise TextSplittingError("Chunk overlap must be a whole number.")
    if chunk_overlap < 0:
        raise TextSplittingError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise TextSplittingError(
            "Chunk overlap must be smaller than the chunk size."
        )


def _validate_documents(documents: list[Document]) -> None:
    """Ensure source documents and their required metadata are available."""
    if not documents:
        raise TextSplittingError("No documents were provided for text splitting.")

    for document_number, document in enumerate(documents, start=1):
        if not isinstance(document, Document):
            raise TextSplittingError(
                f"Item {document_number} is not a LangChain Document object."
            )

        missing_fields = [
            field
            for field in REQUIRED_METADATA_FIELDS
            if field not in document.metadata
        ]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise TextSplittingError(
                f"Document {document_number} is missing required metadata: {missing}."
            )


def _identifier_part(value: object) -> str:
    """Convert a metadata value into a readable chunk-ID component."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return normalized or "unknown"


def _add_chunk_ids(chunks: list[Document]) -> None:
    """Add deterministic, page-based identifiers to all generated chunks."""
    page_chunk_counters: defaultdict[tuple[str, str, object], int] = defaultdict(int)

    for chunk in chunks:
        category = str(chunk.metadata["category"])
        filename = str(chunk.metadata["filename"])
        page = chunk.metadata["page"]
        counter_key = (category, filename, page)
        chunk_number = page_chunk_counters[counter_key]
        page_chunk_counters[counter_key] += 1

        filename_stem = Path(filename).stem
        chunk.metadata["chunk_id"] = (
            f"{_identifier_part(category)}_"
            f"{_identifier_part(filename_stem)}_"
            f"page_{_identifier_part(page)}_"
            f"chunk_{chunk_number}"
        )


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split page Documents while preserving source metadata.

    ``chunk_size`` and ``chunk_overlap`` are character counts because the
    splitter uses Python's ``len`` function. The defaults are initial experiment
    settings and should be evaluated during later retrieval work.
    """
    _validate_documents(documents)
    _validate_chunk_settings(chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise TextSplittingError(
            "The documents did not contain any text that could be split."
        )

    _add_chunk_ids(chunks)
    return chunks
