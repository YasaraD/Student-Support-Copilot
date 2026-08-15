"""Tests for staging, promotion, failure safety, and rollback."""

import gc
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import chromadb
from chromadb.api.client import SharedSystemClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import INDEX_CONFIG, IndexConfiguration
from src.index_manager import (
    IndexBuildError,
    discover_source_documents,
    inspect_index_status,
    rollback_index,
    safe_rebuild_index,
)
from src.index_manifest import read_index_manifest
from src.vector_store import add_chunks, create_or_open_vector_store


class FakeEmbeddings(Embeddings):
    """Small deterministic embeddings for local temporary Chroma tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), float(position), 1.0]
            for position, text in enumerate(texts, start=1)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 1.0]


class FailingEmbeddings(FakeEmbeddings):
    """Embedding stub that fails during staging construction."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("controlled embedding failure")


def fake_loader(file_path: str | Path, category: str) -> list[Document]:
    """Return one page without parsing the test placeholder PDF bytes."""
    path = Path(file_path)
    return [
        Document(
            page_content=f"Approved information from {path.name} for {category}.",
            metadata={
                "source": path.name,
                "filename": path.name,
                "page": 1,
                "category": category,
            },
        )
    ]


def old_chunk() -> Document:
    """Create one valid record representing a previous active generation."""
    return Document(
        page_content="Previous active index content.",
        metadata={
            "chunk_id": "examinations_previous_page_1_chunk_0",
            "source": "previous.pdf",
            "filename": "previous.pdf",
            "page": 1,
            "category": "examinations",
        },
    )


class IndexManagerTests(unittest.TestCase):
    """Exercise the complete safe-index lifecycle in a temporary directory."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True
        )
        root = Path(self.temporary_directory.name)
        self.raw_directory = root / "documents" / "raw"
        self.persist_directory = root / "data" / "chroma_db"
        self.config: IndexConfiguration = replace(
            INDEX_CONFIG,
            raw_documents_directory=self.raw_directory,
            persist_directory=self.persist_directory,
            collection_name="student_support_test",
        )
        for category in self.config.supported_categories:
            category_directory = self.raw_directory / category
            category_directory.mkdir(parents=True)
            (category_directory / f"{category}.pdf").write_bytes(
                f"test {category}".encode("utf-8")
            )
        self.embedding_model = FakeEmbeddings()

    def tearDown(self) -> None:
        SharedSystemClient.clear_system_cache()
        gc.collect()
        self.temporary_directory.cleanup()

    def create_legacy_active_collection(self) -> None:
        """Create an active collection that predates manifest support."""
        store = create_or_open_vector_store(
            embedding_model=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.config.collection_name,
        )
        add_chunks(store, [old_chunk()])

    def collection_names(self) -> set[str]:
        client = chromadb.PersistentClient(path=self.persist_directory)
        return {collection.name for collection in client.list_collections()}

    def test_discovery_maps_folders_and_hashes_documents(self) -> None:
        sources = discover_source_documents(self.config)
        self.assertEqual(len(sources), 4)
        self.assertEqual(
            {source.fingerprint.category for source in sources},
            set(self.config.supported_categories),
        )
        self.assertTrue(all(len(source.fingerprint.sha256) == 64 for source in sources))

    def test_safe_rebuild_promotes_staging_and_retains_active_backup(self) -> None:
        self.create_legacy_active_collection()
        result = safe_rebuild_index(
            self.config,
            embedding_model=self.embedding_model,
            load_fn=fake_loader,
        )
        self.assertTrue(result.performed)
        self.assertTrue(result.backup_created)
        self.assertEqual(result.manifest.chunk_count, 4)
        self.assertEqual(
            self.collection_names(),
            {self.config.collection_name, self.config.backup_collection_name},
        )
        client = chromadb.PersistentClient(path=self.persist_directory)
        self.assertEqual(client.get_collection(self.config.collection_name).count(), 4)
        self.assertEqual(
            client.get_collection(self.config.backup_collection_name).count(), 1
        )
        self.assertEqual(read_index_manifest(self.config.manifest_path), result.manifest)

    def test_staging_failure_preserves_active_collection(self) -> None:
        self.create_legacy_active_collection()
        with self.assertRaises(IndexBuildError):
            safe_rebuild_index(
                self.config,
                embedding_model=FailingEmbeddings(),
                load_fn=fake_loader,
            )
        client = chromadb.PersistentClient(path=self.persist_directory)
        self.assertEqual(client.get_collection(self.config.collection_name).count(), 1)
        self.assertNotIn(self.config.backup_collection_name, self.collection_names())
        self.assertFalse(
            any(
                name.startswith(self.config.staging_collection_prefix)
                for name in self.collection_names()
            )
        )

    def test_rollback_restores_previous_generation(self) -> None:
        self.create_legacy_active_collection()
        rebuilt = safe_rebuild_index(
            self.config,
            embedding_model=self.embedding_model,
            load_fn=fake_loader,
        )
        self.assertIsNotNone(rebuilt.manifest)
        rolled_back = rollback_index(self.config)
        self.assertEqual(rolled_back.active_chunk_count, 1)
        self.assertEqual(rolled_back.backup_chunk_count, 4)
        self.assertIsNone(rolled_back.active_manifest)
        self.assertFalse(self.config.manifest_path.exists())
        self.assertTrue(self.config.backup_manifest_path.exists())

    def test_unchanged_manifest_skips_rebuild(self) -> None:
        first = safe_rebuild_index(
            self.config,
            embedding_model=self.embedding_model,
            load_fn=fake_loader,
        )
        second = safe_rebuild_index(
            self.config,
            embedding_model=self.embedding_model,
            load_fn=fake_loader,
        )
        self.assertTrue(first.performed)
        self.assertFalse(second.performed)
        status = inspect_index_status(self.config)
        self.assertFalse(status.changes.rebuild_required)


if __name__ == "__main__":
    unittest.main()
