"""Central configuration for building and querying the local RAG index."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IndexConfiguration:
    """Validated settings that define the contents and behavior of the index."""

    raw_documents_directory: Path = PROJECT_ROOT / "documents" / "raw"
    persist_directory: Path = PROJECT_ROOT / "data" / "chroma_db"
    collection_name: str = "student_support_knowledge"
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    normalize_embeddings: bool = True
    chunk_size: int = 800
    chunk_overlap: int = 120
    distance_metric: str = "cosine"
    top_k: int = 4
    max_top_k: int = 8
    metadata_schema_version: int = 1
    supported_categories: tuple[str, ...] = (
        "examinations",
        "modules",
        "student_services",
        "academic_regulations",
    )
    required_chunk_metadata: tuple[str, ...] = (
        "chunk_id",
        "source",
        "filename",
        "page",
        "category",
    )
    require_document_per_category: bool = True

    def __post_init__(self) -> None:
        """Reject incompatible settings before they can build an index."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "Chunk overlap must be non-negative and smaller than chunk size."
            )
        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")
        if not 1 <= self.top_k <= self.max_top_k:
            raise ValueError("TOP_K must be between 1 and MAX_TOP_K.")
        if self.distance_metric not in {"cosine", "l2", "ip"}:
            raise ValueError("Distance metric must be cosine, l2, or ip.")
        if len(self.supported_categories) != len(set(self.supported_categories)):
            raise ValueError("Supported category identifiers must be unique.")
        if not self.collection_name.strip():
            raise ValueError("The Chroma collection name cannot be empty.")

    @property
    def backup_collection_name(self) -> str:
        """Return the stable name used for the previous working collection."""
        return f"{self.collection_name}__backup"

    @property
    def staging_collection_prefix(self) -> str:
        """Return the prefix used for temporary validated collections."""
        return f"{self.collection_name}__staging_"

    @property
    def manifest_path(self) -> Path:
        """Return the active index-manifest path."""
        return self.persist_directory / "index_manifest.json"

    @property
    def backup_manifest_path(self) -> Path:
        """Return the previous index-manifest path used for rollback."""
        return self.persist_directory / "index_manifest.backup.json"

    def category_directory(self, category: str) -> Path:
        """Return the configured source folder for one category."""
        if category not in self.supported_categories:
            raise ValueError(f"Unsupported category: {category}.")
        return self.raw_documents_directory / category

    def manifest_configuration(self) -> dict[str, object]:
        """Return only settings that determine stored vectors and metadata."""
        return {
            "collection_name": self.collection_name,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "normalize_embeddings": self.normalize_embeddings,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "splitter": "RecursiveCharacterTextSplitter",
            "length_unit": "characters",
            "distance_metric": self.distance_metric,
            "metadata_schema_version": self.metadata_schema_version,
            "required_chunk_metadata": list(self.required_chunk_metadata),
            "supported_categories": list(self.supported_categories),
        }


INDEX_CONFIG = IndexConfiguration()
