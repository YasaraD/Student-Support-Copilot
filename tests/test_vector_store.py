"""Tests for the project's persistent Chroma wrapper."""

import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.vector_store import (
    COLLECTION_NAME,
    InvalidChunkError,
    VectorStoreNotEmptyError,
    add_chunks,
    create_or_open_vector_store,
    get_stored_document_count,
    get_stored_record,
    open_existing_vector_store,
    rebuild_vector_store,
    validate_chunks,
)


class FakeEmbeddings(Embeddings):
    """Small deterministic embeddings that avoid loading BGE-M3 in unit tests."""

    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [
            [float(len(text)), float(index), 1.0]
            for index, text in enumerate(texts, start=1)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 1.0]


def make_chunk(
    chunk_id: str = "examinations_mock_policy_page_1_chunk_0",
    *,
    category: str = "examinations",
    page: int = 1,
) -> Document:
    """Create a valid test chunk with realistic source metadata."""
    return Document(
        page_content="Students must follow the fictional examination policy.",
        metadata={
            "chunk_id": chunk_id,
            "source": "mock_policy.pdf",
            "filename": "mock_policy.pdf",
            "page": page,
            "category": category,
            "title": "Mock Policy",
        },
    )


class VectorStoreTests(unittest.TestCase):
    """Verify Chroma configuration, IDs, validation, and persistence."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True
        )
        self.persist_directory = Path(self.temporary_directory.name) / "chroma"
        self.embedding_model = FakeEmbeddings()

    def tearDown(self) -> None:
        SharedSystemClient.clear_system_cache()
        gc.collect()
        self.temporary_directory.cleanup()

    def create_store(self) -> Chroma:
        return create_or_open_vector_store(
            embedding_model=self.embedding_model,
            persist_directory=self.persist_directory,
        )

    def test_valid_vector_store_creation(self) -> None:
        store = self.create_store()
        self.assertIsInstance(store, Chroma)
        self.assertEqual(get_stored_document_count(store), 0)
        self.assertTrue(self.persist_directory.exists())

    def test_required_metadata_is_validated(self) -> None:
        chunk = make_chunk()
        del chunk.metadata["source"]
        with self.assertRaisesRegex(InvalidChunkError, "source"):
            validate_chunks([chunk])

    def test_missing_chunk_id_is_rejected(self) -> None:
        chunk = make_chunk()
        del chunk.metadata["chunk_id"]
        with self.assertRaisesRegex(InvalidChunkError, "chunk_id"):
            validate_chunks([chunk])

    def test_deterministic_ids_are_stored(self) -> None:
        store = self.create_store()
        chunk = make_chunk()
        stored_ids = add_chunks(store, [chunk])
        self.assertEqual(stored_ids, [chunk.metadata["chunk_id"]])
        self.assertEqual(store.get(include=[])["ids"], stored_ids)

    def test_empty_document_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidChunkError, "non-empty"):
            validate_chunks([])

    def test_stored_document_count(self) -> None:
        store = self.create_store()
        chunks = [
            make_chunk(),
            make_chunk("modules_mock_handbook_page_2_chunk_0", category="modules"),
        ]
        add_chunks(store, chunks)
        self.assertEqual(get_stored_document_count(store), 2)

    def test_existing_store_can_be_reopened(self) -> None:
        store = self.create_store()
        add_chunks(store, [make_chunk()])
        reopened = open_existing_vector_store(
            persist_directory=self.persist_directory
        )
        self.assertIsInstance(reopened, Chroma)

    def test_count_persists_across_reopening(self) -> None:
        store = self.create_store()
        add_chunks(store, [make_chunk()])
        reopened = open_existing_vector_store(
            persist_directory=self.persist_directory
        )
        self.assertEqual(get_stored_document_count(reopened), 1)

    def test_metadata_and_text_are_preserved(self) -> None:
        store = self.create_store()
        chunk = make_chunk()
        add_chunks(store, [chunk])
        record = get_stored_record(store, chunk.metadata["chunk_id"])
        self.assertEqual(record["page_content"], chunk.page_content)
        self.assertEqual(record["metadata"], chunk.metadata)

    def test_duplicate_add_requires_explicit_rebuild(self) -> None:
        store = self.create_store()
        chunk = make_chunk()
        add_chunks(store, [chunk])
        with self.assertRaisesRegex(VectorStoreNotEmptyError, "rebuild"):
            add_chunks(store, [chunk])

        rebuilt = rebuild_vector_store(
            [chunk],
            embedding_model=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        self.assertEqual(get_stored_document_count(rebuilt), 1)

    @patch("src.vector_store.get_embedding_model")
    def test_existing_embedding_model_configuration_is_reused(
        self, get_model
    ) -> None:
        get_model.return_value = self.embedding_model
        store = create_or_open_vector_store(
            persist_directory=self.persist_directory
        )
        add_chunks(store, [make_chunk()])
        get_model.assert_called_once_with()
        self.assertEqual(self.embedding_model.document_calls, 1)


if __name__ == "__main__":
    unittest.main()
