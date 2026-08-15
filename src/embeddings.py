"""Create normalized local dense embeddings with BAAI/bge-m3."""

from functools import lru_cache
from math import isfinite

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import INDEX_CONFIG

EMBEDDING_MODEL_NAME = INDEX_CONFIG.embedding_model_name
EXPECTED_EMBEDDING_DIMENSION = INDEX_CONFIG.embedding_dimension
NORMALIZE_EMBEDDINGS = INDEX_CONFIG.normalize_embeddings


class EmbeddingError(RuntimeError):
    """Base error for expected model-loading or embedding failures."""


class InvalidEmbeddingInputError(EmbeddingError, ValueError):
    """Raised when text input is missing or invalid."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when the configured embedding model cannot be loaded."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when vectors cannot be generated or validated."""


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Load and reuse the configured local embedding model.

    The first call may download model files from Hugging Face. The one-item
    process cache stores only the model resource, not input text or vectors.
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
        )
    except Exception as error:
        raise EmbeddingModelLoadError(
            f"The embedding model '{EMBEDDING_MODEL_NAME}' could not be loaded. "
            "On first use, check the internet connection and available disk "
            "space. Later uses can load the model from the local Hugging Face cache."
        ) from error


def _validate_query(text: str) -> str:
    """Return cleaned query text or raise an understandable validation error."""
    if not isinstance(text, str) or not text.strip():
        raise InvalidEmbeddingInputError(
            "The query must be a non-empty text string."
        )
    return text.strip()


def _validate_document_texts(texts: list[str]) -> list[str]:
    """Validate and clean a list of document texts."""
    if not isinstance(texts, list) or not texts:
        raise InvalidEmbeddingInputError(
            "Provide a non-empty list of document text strings."
        )

    cleaned_texts: list[str] = []
    for position, text in enumerate(texts, start=1):
        if not isinstance(text, str) or not text.strip():
            raise InvalidEmbeddingInputError(
                f"Document text {position} must be a non-empty string."
            )
        cleaned_texts.append(text.strip())
    return cleaned_texts


def _validated_vector(vector: list[float], label: str) -> list[float]:
    """Check that an embedding is non-empty, numeric, and finite."""
    if not isinstance(vector, list) or not vector:
        raise EmbeddingGenerationError(f"{label} produced an empty embedding.")

    try:
        numeric_vector = [float(value) for value in vector]
    except (TypeError, ValueError) as error:
        raise EmbeddingGenerationError(
            f"{label} produced an embedding containing non-numeric values."
        ) from error

    if not all(isfinite(value) for value in numeric_vector):
        raise EmbeddingGenerationError(
            f"{label} produced an embedding containing NaN or infinite values."
        )
    return numeric_vector


def embed_query(text: str, model: Embeddings | None = None) -> list[float]:
    """Embed one student query using the configured LangChain interface."""
    cleaned_text = _validate_query(text)
    embedding_model = model or get_embedding_model()

    try:
        vector = embedding_model.embed_query(cleaned_text)
    except Exception as error:
        raise EmbeddingGenerationError(
            "The student query could not be embedded. Check that the local "
            "embedding model loaded correctly and try again."
        ) from error

    return _validated_vector(vector, "The student query")


def embed_documents(
    texts: list[str], model: Embeddings | None = None
) -> list[list[float]]:
    """Embed multiple document texts and verify consistent output vectors."""
    cleaned_texts = _validate_document_texts(texts)
    embedding_model = model or get_embedding_model()

    try:
        vectors = embedding_model.embed_documents(cleaned_texts)
    except Exception as error:
        raise EmbeddingGenerationError(
            "The document texts could not be embedded. Check that the local "
            "embedding model loaded correctly and try again."
        ) from error

    if len(vectors) != len(cleaned_texts):
        raise EmbeddingGenerationError(
            "The embedding model returned a different number of vectors than "
            "the number of document texts provided."
        )

    validated_vectors = [
        _validated_vector(vector, f"Document text {position}")
        for position, vector in enumerate(vectors, start=1)
    ]
    dimensions = {len(vector) for vector in validated_vectors}
    if len(dimensions) != 1:
        raise EmbeddingGenerationError(
            "The embedding model returned vectors with inconsistent dimensions."
        )
    return validated_vectors


def get_embedding_dimension(vector: list[float]) -> int:
    """Validate one vector and return its number of dimensions."""
    return len(_validated_vector(vector, "The supplied text"))
