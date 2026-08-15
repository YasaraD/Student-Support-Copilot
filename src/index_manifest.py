"""Record index provenance and detect source or configuration changes."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import INDEX_CONFIG, IndexConfiguration


MANIFEST_SCHEMA_VERSION = 1


class IndexManifestError(RuntimeError):
    """Raised when an index manifest cannot be validated or stored safely."""


@dataclass(frozen=True, order=True)
class DocumentFingerprint:
    """Content identity for one categorized source document."""

    relative_path: str
    category: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        return {
            "relative_path": self.relative_path,
            "category": self.category,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentFingerprint":
        """Create a validated fingerprint from manifest data."""
        try:
            fingerprint = cls(
                relative_path=str(data["relative_path"]),
                category=str(data["category"]),
                sha256=str(data["sha256"]),
                size_bytes=int(data["size_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IndexManifestError(
                "The index manifest contains an invalid document fingerprint."
            ) from error
        if len(fingerprint.sha256) != 64 or fingerprint.size_bytes < 0:
            raise IndexManifestError(
                "The index manifest contains an invalid document fingerprint."
            )
        return fingerprint


@dataclass(frozen=True)
class IndexManifest:
    """Provenance record for one successfully validated Chroma collection."""

    build_id: str
    built_at_utc: str
    configuration: dict[str, object]
    configuration_sha256: str
    documents: tuple[DocumentFingerprint, ...]
    page_count: int
    chunk_count: int
    category_chunk_counts: dict[str, int]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-safe manifest."""
        return {
            "schema_version": self.schema_version,
            "build_id": self.build_id,
            "built_at_utc": self.built_at_utc,
            "configuration": self.configuration,
            "configuration_sha256": self.configuration_sha256,
            "documents": [document.to_dict() for document in self.documents],
            "document_count": len(self.documents),
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "category_chunk_counts": self.category_chunk_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexManifest":
        """Create a validated manifest from parsed JSON data."""
        try:
            schema_version = int(data["schema_version"])
            raw_configuration = data["configuration"]
            raw_documents = data["documents"]
            raw_category_counts = data["category_chunk_counts"]
            if not isinstance(raw_configuration, dict):
                raise TypeError
            if not isinstance(raw_documents, list):
                raise TypeError
            if not isinstance(raw_category_counts, dict):
                raise TypeError
            manifest = cls(
                schema_version=schema_version,
                build_id=str(data["build_id"]),
                built_at_utc=str(data["built_at_utc"]),
                configuration=dict(raw_configuration),
                configuration_sha256=str(data["configuration_sha256"]),
                documents=tuple(
                    DocumentFingerprint.from_dict(document)
                    for document in raw_documents
                ),
                page_count=int(data["page_count"]),
                chunk_count=int(data["chunk_count"]),
                category_chunk_counts={
                    str(category): int(count)
                    for category, count in raw_category_counts.items()
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IndexManifestError(
                "The index manifest is missing required or valid fields."
            ) from error

        if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
            raise IndexManifestError(
                "The index manifest uses an unsupported schema version."
            )
        if manifest.page_count < 0 or manifest.chunk_count <= 0:
            raise IndexManifestError(
                "The index manifest contains invalid page or chunk counts."
            )
        if int(data.get("document_count", -1)) != len(manifest.documents):
            raise IndexManifestError(
                "The index manifest document count does not match its document list."
            )
        if configuration_sha256(manifest.configuration) != manifest.configuration_sha256:
            raise IndexManifestError(
                "The index manifest configuration fingerprint is invalid."
            )
        return manifest


@dataclass(frozen=True)
class IndexChangeReport:
    """Differences between current files/settings and the active manifest."""

    manifest_missing: bool
    configuration_changed: bool
    added: tuple[str, ...]
    modified: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def rebuild_required(self) -> bool:
        """Return whether the stored index should be rebuilt."""
        return (
            self.manifest_missing
            or self.configuration_changed
            or bool(self.added)
            or bool(self.modified)
            or bool(self.removed)
        )


def configuration_sha256(configuration: dict[str, object]) -> str:
    """Create a stable fingerprint for index-defining configuration."""
    serialized = json.dumps(
        configuration,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def read_index_manifest(path: Path = INDEX_CONFIG.manifest_path) -> IndexManifest | None:
    """Read the active manifest, or return None when one has not been created."""
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IndexManifestError(
            f"The index manifest at '{path}' could not be read."
        ) from error
    if not isinstance(parsed, dict):
        raise IndexManifestError("The index manifest root must be a JSON object.")
    return IndexManifest.from_dict(parsed)


def write_index_manifest(manifest: IndexManifest, path: Path) -> None:
    """Write a manifest atomically so a partial JSON file is never exposed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except OSError as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise IndexManifestError(
            f"The index manifest could not be written safely to '{path}'."
        ) from error


def compare_index_state(
    current_documents: tuple[DocumentFingerprint, ...],
    manifest: IndexManifest | None,
    config: IndexConfiguration = INDEX_CONFIG,
) -> IndexChangeReport:
    """Compare current document hashes and settings with a previous manifest."""
    current_by_path = {
        document.relative_path: document for document in current_documents
    }
    if manifest is None:
        return IndexChangeReport(
            manifest_missing=True,
            configuration_changed=False,
            added=tuple(sorted(current_by_path)),
            modified=(),
            removed=(),
        )

    previous_by_path = {
        document.relative_path: document for document in manifest.documents
    }
    added = tuple(sorted(current_by_path.keys() - previous_by_path.keys()))
    removed = tuple(sorted(previous_by_path.keys() - current_by_path.keys()))
    modified = tuple(
        sorted(
            path
            for path in current_by_path.keys() & previous_by_path.keys()
            if current_by_path[path] != previous_by_path[path]
        )
    )
    return IndexChangeReport(
        manifest_missing=False,
        configuration_changed=(
            configuration_sha256(config.manifest_configuration())
            != manifest.configuration_sha256
        ),
        added=added,
        modified=modified,
        removed=removed,
    )
