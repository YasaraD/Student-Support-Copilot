"""Safely build, promote, inspect, and roll back the local Chroma index."""

import gc
import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import INDEX_CONFIG, IndexConfiguration
from src.index_manifest import (
    DocumentFingerprint,
    IndexChangeReport,
    IndexManifest,
    IndexManifestError,
    compare_index_state,
    configuration_sha256,
    read_index_manifest,
    write_index_manifest,
)


class IndexManagementError(RuntimeError):
    """Base error for safe index-management failures."""


class DocumentDiscoveryError(IndexManagementError):
    """Raised when configured source documents cannot be discovered safely."""


class IndexBuildError(IndexManagementError):
    """Raised when a staging index cannot be prepared or validated."""


class IndexPromotionError(IndexManagementError):
    """Raised when a validated staging index cannot become active safely."""


class IndexRollbackError(IndexManagementError):
    """Raised when the previous collection cannot be restored safely."""


@dataclass(frozen=True)
class SourceDocument:
    """Absolute source path paired with its portable manifest fingerprint."""

    path: Path
    fingerprint: DocumentFingerprint


@dataclass(frozen=True)
class PreparedIndex:
    """Loaded and chunked source material ready for embedding."""

    source_documents: tuple[SourceDocument, ...]
    chunks: tuple[Document, ...]
    page_count: int
    category_chunk_counts: dict[str, int]


@dataclass(frozen=True)
class IndexStatus:
    """Read-only summary of source, manifest, and collection state."""

    source_documents: tuple[SourceDocument, ...]
    manifest: IndexManifest | None
    changes: IndexChangeReport
    active_collection_exists: bool
    active_chunk_count: int
    backup_collection_exists: bool
    backup_chunk_count: int


@dataclass(frozen=True)
class RebuildResult:
    """Outcome of one requested index rebuild."""

    performed: bool
    manifest: IndexManifest | None
    changes: IndexChangeReport
    backup_created: bool


@dataclass(frozen=True)
class RollbackResult:
    """Outcome of swapping the active and backup generations."""

    active_chunk_count: int
    backup_chunk_count: int
    active_manifest: IndexManifest | None


DocumentLoader = Callable[[str | Path, str], list[Document]]
DocumentSplitter = Callable[..., list[Document]]


def _sha256_file(path: Path) -> str:
    """Hash a file in bounded blocks instead of loading it all into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise DocumentDiscoveryError(
            f"The source document '{path}' could not be read for change detection."
        ) from error
    return digest.hexdigest()


def discover_source_documents(
    config: IndexConfiguration = INDEX_CONFIG,
) -> tuple[SourceDocument, ...]:
    """Discover categorized PDFs and create stable content fingerprints."""
    discovered: list[SourceDocument] = []
    for category in config.supported_categories:
        category_directory = config.category_directory(category)
        if not category_directory.exists() or not category_directory.is_dir():
            raise DocumentDiscoveryError(
                f"The configured category folder does not exist: "
                f"'{category_directory}'."
            )
        pdf_paths = sorted(
            (
                path
                for path in category_directory.rglob("*")
                if path.is_file() and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: path.relative_to(
                config.raw_documents_directory
            ).as_posix().casefold(),
        )
        if config.require_document_per_category and not pdf_paths:
            raise DocumentDiscoveryError(
                f"No PDF documents were found for the required category "
                f"'{category}' in '{category_directory}'."
            )
        for path in pdf_paths:
            relative_path = path.relative_to(
                config.raw_documents_directory
            ).as_posix()
            try:
                size_bytes = path.stat().st_size
            except OSError as error:
                raise DocumentDiscoveryError(
                    f"The source document '{path}' could not be inspected."
                ) from error
            discovered.append(
                SourceDocument(
                    path=path,
                    fingerprint=DocumentFingerprint(
                        relative_path=relative_path,
                        category=category,
                        sha256=_sha256_file(path),
                        size_bytes=size_bytes,
                    ),
                )
            )

    if not discovered:
        raise DocumentDiscoveryError(
            f"No PDF documents were found under '{config.raw_documents_directory}'."
        )
    return tuple(discovered)


def _collection_names(client: object) -> set[str]:
    """Return collection names across supported Chroma list return shapes."""
    try:
        collections = client.list_collections()
    except Exception as error:
        raise IndexManagementError("Chroma collections could not be listed.") from error
    return {
        item if isinstance(item, str) else str(item.name)
        for item in collections
    }


def _open_chroma_client(config: IndexConfiguration) -> object:
    """Open the public persistent Chroma client used for collection management."""
    try:
        return chromadb.PersistentClient(path=config.persist_directory)
    except Exception as error:
        raise IndexManagementError(
            f"Chroma could not be opened at '{config.persist_directory}'."
        ) from error


def _collection_count(client: object, name: str) -> int:
    """Return a collection count without loading the embedding model."""
    try:
        return int(client.get_collection(name=name).count())
    except Exception as error:
        raise IndexManagementError(
            f"The Chroma collection '{name}' could not be inspected."
        ) from error


def inspect_index_status(
    config: IndexConfiguration = INDEX_CONFIG,
) -> IndexStatus:
    """Compare current sources/settings with the active index manifest."""
    sources = discover_source_documents(config)
    try:
        manifest = read_index_manifest(config.manifest_path)
    except IndexManifestError as error:
        raise IndexManagementError(str(error)) from error
    changes = compare_index_state(
        tuple(source.fingerprint for source in sources),
        manifest,
        config,
    )

    if not config.persist_directory.exists():
        names: set[str] = set()
        client = None
    else:
        client = _open_chroma_client(config)
        names = _collection_names(client)

    active_exists = config.collection_name in names
    backup_exists = config.backup_collection_name in names
    return IndexStatus(
        source_documents=sources,
        manifest=manifest,
        changes=changes,
        active_collection_exists=active_exists,
        active_chunk_count=(
            _collection_count(client, config.collection_name)
            if client is not None and active_exists
            else 0
        ),
        backup_collection_exists=backup_exists,
        backup_chunk_count=(
            _collection_count(client, config.backup_collection_name)
            if client is not None and backup_exists
            else 0
        ),
    )


def prepare_index_documents(
    source_documents: tuple[SourceDocument, ...],
    config: IndexConfiguration = INDEX_CONFIG,
    *,
    load_fn: DocumentLoader | None = None,
    split_fn: DocumentSplitter | None = None,
) -> PreparedIndex:
    """Load every discovered PDF and create configured LangChain chunks."""
    from src.document_loader import DocumentLoaderError, load_pdf_document
    from src.text_splitter import TextSplittingError, split_documents

    selected_loader = load_fn or load_pdf_document
    selected_splitter = split_fn or split_documents
    all_chunks: list[Document] = []
    page_count = 0
    try:
        for source in source_documents:
            pages = selected_loader(source.path, source.fingerprint.category)
            page_count += len(pages)
            for page in pages:
                page.metadata.update(
                    {
                        "relative_path": source.fingerprint.relative_path,
                        "document_sha256": source.fingerprint.sha256,
                        "metadata_schema_version": config.metadata_schema_version,
                    }
                )
            all_chunks.extend(
                selected_splitter(
                    pages,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
            )
    except (DocumentLoaderError, TextSplittingError) as error:
        raise IndexBuildError(str(error)) from error

    category_counts = Counter(
        str(chunk.metadata.get("category")) for chunk in all_chunks
    )
    missing_categories = [
        category
        for category in config.supported_categories
        if category_counts[category] == 0
    ]
    if missing_categories:
        raise IndexBuildError(
            "No chunks were produced for required categories: "
            + ", ".join(missing_categories)
            + "."
        )
    return PreparedIndex(
        source_documents=source_documents,
        chunks=tuple(all_chunks),
        page_count=page_count,
        category_chunk_counts={
            category: category_counts[category]
            for category in config.supported_categories
        },
    )


def _delete_collection_if_present(client: object, name: str) -> None:
    """Delete exactly one named collection when it exists."""
    if name not in _collection_names(client):
        return
    try:
        client.delete_collection(name=name)
    except Exception as error:
        raise IndexManagementError(
            f"The Chroma collection '{name}' could not be deleted."
        ) from error


def _promote_staging_collection(
    client: object,
    staging_name: str,
    config: IndexConfiguration,
) -> bool:
    """Promote staging and retain the former active collection as backup."""
    names = _collection_names(client)
    if staging_name not in names:
        raise IndexPromotionError("The validated staging collection is missing.")

    active_exists = config.collection_name in names
    try:
        _delete_collection_if_present(client, config.backup_collection_name)
        if active_exists:
            client.get_collection(name=config.collection_name).modify(
                name=config.backup_collection_name
            )
        client.get_collection(name=staging_name).modify(name=config.collection_name)
    except Exception as error:
        try:
            current_names = _collection_names(client)
            if (
                config.collection_name not in current_names
                and config.backup_collection_name in current_names
            ):
                client.get_collection(name=config.backup_collection_name).modify(
                    name=config.collection_name
                )
        except Exception as restore_error:
            raise IndexPromotionError(
                "Staging promotion failed and the previous active collection could "
                "not be restored automatically. Stop the application and inspect "
                "the Chroma collections before continuing."
            ) from restore_error
        raise IndexPromotionError(
            "The staging collection could not be promoted; the previous active "
            "collection was preserved."
        ) from error
    return active_exists


def _restore_collection_after_manifest_failure(
    client: object, config: IndexConfiguration, had_active: bool
) -> None:
    """Restore the previous collection if manifest promotion fails."""
    try:
        _delete_collection_if_present(client, config.collection_name)
        if had_active:
            client.get_collection(name=config.backup_collection_name).modify(
                name=config.collection_name
            )
    except Exception as error:
        raise IndexPromotionError(
            "The manifest update failed and the previous collection could not be "
            "restored automatically."
        ) from error


def _promote_manifest(
    pending_manifest_path: Path,
    config: IndexConfiguration,
) -> None:
    """Promote a prepared manifest while retaining the former one for rollback."""
    try:
        config.backup_manifest_path.unlink(missing_ok=True)
        if config.manifest_path.exists():
            os.replace(config.manifest_path, config.backup_manifest_path)
        os.replace(pending_manifest_path, config.manifest_path)
    except OSError as error:
        try:
            if (
                not config.manifest_path.exists()
                and config.backup_manifest_path.exists()
            ):
                os.replace(config.backup_manifest_path, config.manifest_path)
            pending_manifest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise IndexPromotionError(
            "The new index manifest could not be promoted safely."
        ) from error


def _new_build_id() -> str:
    """Return a readable unique identifier safe for Chroma collection names."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def safe_rebuild_index(
    config: IndexConfiguration = INDEX_CONFIG,
    *,
    embedding_model: Embeddings | None = None,
    force: bool = False,
    load_fn: DocumentLoader | None = None,
    split_fn: DocumentSplitter | None = None,
) -> RebuildResult:
    """Build and validate staging before replacing the active collection."""
    from src.vector_store import (
        VectorStoreError,
        add_chunks,
        create_or_open_vector_store,
        get_stored_document_count,
    )

    status = inspect_index_status(config)
    if not status.changes.rebuild_required and not force:
        return RebuildResult(
            performed=False,
            manifest=status.manifest,
            changes=status.changes,
            backup_created=status.backup_collection_exists,
        )

    prepared = prepare_index_documents(
        status.source_documents,
        config,
        load_fn=load_fn,
        split_fn=split_fn,
    )
    build_id = _new_build_id()
    staging_name = f"{config.staging_collection_prefix}{build_id}"
    pending_manifest_path = config.persist_directory / (
        f".index_manifest.{build_id}.pending.json"
    )
    client = _open_chroma_client(config)

    try:
        staging_store = create_or_open_vector_store(
            embedding_model=embedding_model,
            persist_directory=config.persist_directory,
            collection_name=staging_name,
        )
        stored_ids = add_chunks(staging_store, list(prepared.chunks))
        stored_count = get_stored_document_count(staging_store)
        expected_ids = [
            str(chunk.metadata["chunk_id"]) for chunk in prepared.chunks
        ]
        if stored_count != len(prepared.chunks) or stored_ids != expected_ids:
            raise IndexBuildError(
                "The staging collection failed its stored-count or ID validation."
            )

        manifest = IndexManifest(
            build_id=build_id,
            built_at_utc=datetime.now(timezone.utc).isoformat(),
            configuration=config.manifest_configuration(),
            configuration_sha256=configuration_sha256(
                config.manifest_configuration()
            ),
            documents=tuple(
                source.fingerprint for source in prepared.source_documents
            ),
            page_count=prepared.page_count,
            chunk_count=stored_count,
            category_chunk_counts=prepared.category_chunk_counts,
        )
        write_index_manifest(manifest, pending_manifest_path)

        del staging_store
        gc.collect()
        had_active = _promote_staging_collection(client, staging_name, config)
        try:
            _promote_manifest(pending_manifest_path, config)
        except IndexPromotionError:
            _restore_collection_after_manifest_failure(client, config, had_active)
            raise
    except (VectorStoreError, IndexManifestError, IndexManagementError) as error:
        try:
            pending_manifest_path.unlink(missing_ok=True)
            _delete_collection_if_present(client, staging_name)
        except (OSError, IndexManagementError):
            pass
        if isinstance(error, IndexManagementError):
            raise
        raise IndexBuildError(str(error)) from error
    except Exception as error:
        try:
            pending_manifest_path.unlink(missing_ok=True)
            _delete_collection_if_present(client, staging_name)
        except (OSError, IndexManagementError):
            pass
        raise IndexBuildError(
            "The staging index could not be built. The active collection was not "
            "replaced."
        ) from error

    return RebuildResult(
        performed=True,
        manifest=manifest,
        changes=status.changes,
        backup_created=had_active,
    )


def _swap_collections(client: object, config: IndexConfiguration) -> None:
    """Swap active and backup names using a unique temporary collection name."""
    names = _collection_names(client)
    if config.collection_name not in names:
        raise IndexRollbackError("The active collection is missing.")
    if config.backup_collection_name not in names:
        raise IndexRollbackError("No rollback collection is available.")

    temporary_name = f"{config.collection_name}__rollback_{uuid4().hex[:8]}"
    try:
        client.get_collection(name=config.collection_name).modify(
            name=temporary_name
        )
        client.get_collection(name=config.backup_collection_name).modify(
            name=config.collection_name
        )
        client.get_collection(name=temporary_name).modify(
            name=config.backup_collection_name
        )
    except Exception as error:
        try:
            current_names = _collection_names(client)
            if temporary_name in current_names:
                if config.collection_name in current_names:
                    client.get_collection(name=config.collection_name).modify(
                        name=config.backup_collection_name
                    )
                client.get_collection(name=temporary_name).modify(
                    name=config.collection_name
                )
        except Exception as restore_error:
            raise IndexRollbackError(
                "Collection rollback failed and automatic restoration was not "
                "possible. Inspect Chroma before continuing."
            ) from restore_error
        raise IndexRollbackError(
            "The active and backup collections could not be swapped."
        ) from error


def _replace_bytes(path: Path, content: bytes | None) -> None:
    """Replace one small manifest file atomically, or remove it when absent."""
    if content is None:
        path.unlink(missing_ok=True)
        return
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_path.write_bytes(content)
    os.replace(temporary_path, path)


def _swap_manifest_files(config: IndexConfiguration) -> None:
    """Swap active and backup manifests, allowing a legacy missing manifest."""
    try:
        active_content = (
            config.manifest_path.read_bytes()
            if config.manifest_path.exists()
            else None
        )
        backup_content = (
            config.backup_manifest_path.read_bytes()
            if config.backup_manifest_path.exists()
            else None
        )
        _replace_bytes(config.manifest_path, backup_content)
        _replace_bytes(config.backup_manifest_path, active_content)
    except OSError as error:
        try:
            _replace_bytes(config.manifest_path, active_content)
            _replace_bytes(config.backup_manifest_path, backup_content)
        except OSError:
            pass
        raise IndexRollbackError(
            "The active and backup manifest files could not be swapped."
        ) from error


def rollback_index(
    config: IndexConfiguration = INDEX_CONFIG,
) -> RollbackResult:
    """Restore the previous collection and keep the current one as backup."""
    if not config.persist_directory.exists():
        raise IndexRollbackError("The Chroma persistence directory is missing.")
    client = _open_chroma_client(config)
    _swap_collections(client, config)
    try:
        _swap_manifest_files(config)
    except IndexRollbackError:
        try:
            _swap_collections(client, config)
        except IndexRollbackError as restore_error:
            raise IndexRollbackError(
                "Manifest rollback failed and the collection swap could not be "
                "reversed automatically."
            ) from restore_error
        raise

    try:
        active_manifest = read_index_manifest(config.manifest_path)
    except IndexManifestError as error:
        raise IndexRollbackError(str(error)) from error
    return RollbackResult(
        active_chunk_count=_collection_count(client, config.collection_name),
        backup_chunk_count=_collection_count(
            client, config.backup_collection_name
        ),
        active_manifest=active_manifest,
    )
