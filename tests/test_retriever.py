"""Fast tests for dense retrieval from temporary Chroma collections."""

import gc
import tempfile
import unittest
from pathlib import Path

from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.retriever import (
    EmptyRetrievalIndexError,
    InvalidRetrievalInputError,
    RetrievalIndexUnavailableError,
    RetrievalResult,
    create_langchain_retriever,
    retrieve,
    retrieve_with_scores,
)
from src.vector_store import add_chunks, create_or_open_vector_store


class KeywordEmbeddings(Embeddings):
    """Small deterministic semantic-like vectors for isolated unit tests."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if any(word in lowered for word in ("exam", "illness", "medical")):
            return [1.0, 0.0, 0.0, 0.0]
        if any(word in lowered for word in ("module", "prerequisite", "fail")):
            return [0.0, 1.0, 0.0, 0.0]
        if any(word in lowered for word in ("portal", "library", "support")):
            return [0.0, 0.0, 1.0, 0.0]
        if any(word in lowered for word in ("assignment", "late", "attendance")):
            return [0.0, 0.0, 0.0, 1.0]
        return [0.5, 0.5, 0.5, 0.5]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)


def make_chunk(
    chunk_id: str,
    text: str,
    category: str,
    page: int,
) -> Document:
    """Create a realistic stored chunk for retrieval tests."""
    filename = f"mock_{category}.pdf"
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "source": filename,
            "filename": filename,
            "page": page,
            "category": category,
        },
    )


class RetrieverTests(unittest.TestCase):
    """Verify validation, filtering, ranking, persistence, and read-only use."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True
        )
        self.persist_directory = Path(self.temporary_directory.name) / "chroma"
        self.embedding_model = KeywordEmbeddings()
        self.store = create_or_open_vector_store(
            embedding_model=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        self.chunks = [
            make_chunk(
                "examinations_policy_page_1_chunk_0",
                "Medical evidence is required after an examination illness.",
                "examinations",
                1,
            ),
            make_chunk(
                "modules_handbook_page_2_chunk_0",
                "A failed module may require another module attempt.",
                "modules",
                2,
            ),
            make_chunk(
                "student_services_guide_page_3_chunk_0",
                "IT support helps students who cannot access the portal.",
                "student_services",
                3,
            ),
            make_chunk(
                "academic_regulations_page_4_chunk_0",
                "An assignment submitted late is subject to a mark penalty.",
                "academic_regulations",
                4,
            ),
        ]
        add_chunks(self.store, self.chunks)

    def tearDown(self) -> None:
        SharedSystemClient.clear_system_cache()
        gc.collect()
        self.temporary_directory.cleanup()

    def test_valid_query_retrieval(self) -> None:
        documents = retrieve("missed exam", k=1, vector_store=self.store)
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["category"], "examinations")

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidRetrievalInputError, "non-empty"):
            retrieve("", vector_store=self.store)

    def test_whitespace_only_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidRetrievalInputError, "non-empty"):
            retrieve("   ", vector_store=self.store)

    def test_invalid_k_is_rejected(self) -> None:
        for invalid_k in (0, 9, 1.5, True):
            with self.subTest(k=invalid_k):
                with self.assertRaisesRegex(
                    InvalidRetrievalInputError, "between 1 and 8"
                ):
                    retrieve("exam", k=invalid_k, vector_store=self.store)

    def test_invalid_category_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidRetrievalInputError, "Invalid"):
            retrieve("exam", category="finance", vector_store=self.store)

    def test_optional_category_filtering(self) -> None:
        results = retrieve_with_scores(
            "student support",
            k=4,
            category="modules",
            vector_store=self.store,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document.metadata["category"], "modules")

    def test_result_count_respects_k(self) -> None:
        results = retrieve_with_scores("general question", k=2, vector_store=self.store)
        self.assertEqual(len(results), 2)

    def test_required_metadata_is_preserved(self) -> None:
        result = retrieve_with_scores("late assignment", k=1, vector_store=self.store)[0]
        for field in ("chunk_id", "source", "filename", "page", "category"):
            self.assertIn(field, result.document.metadata)

    def test_missing_vector_store_is_handled(self) -> None:
        missing_path = Path(self.temporary_directory.name) / "missing"
        with self.assertRaises(RetrievalIndexUnavailableError):
            retrieve(
                "exam",
                persist_directory=missing_path,
                embedding_model=self.embedding_model,
            )

    def test_empty_vector_store_is_handled(self) -> None:
        empty_path = Path(self.temporary_directory.name) / "empty"
        empty_store = create_or_open_vector_store(
            embedding_model=self.embedding_model,
            persist_directory=empty_path,
            collection_name="empty_retrieval_test",
        )
        with self.assertRaisesRegex(EmptyRetrievalIndexError, "empty"):
            retrieve("exam", vector_store=empty_store)

    def test_retrieval_does_not_mutate_vector_store(self) -> None:
        before_ids = self.store.get(include=[])["ids"]
        document_embedding_calls = self.embedding_model.document_calls
        retrieve("portal access", k=2, vector_store=self.store)
        after_ids = self.store.get(include=[])["ids"]
        self.assertCountEqual(after_ids, before_ids)
        self.assertEqual(
            self.embedding_model.document_calls,
            document_embedding_calls,
        )
        self.assertEqual(self.embedding_model.query_calls, 1)

    def test_standard_langchain_retriever_creation(self) -> None:
        retriever = create_langchain_retriever(
            k=1,
            category="student_services",
            vector_store=self.store,
        )
        self.assertIsInstance(retriever, BaseRetriever)
        documents = retriever.invoke("portal access")
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["category"], "student_services")

    def test_ranked_result_structure_and_order(self) -> None:
        results = retrieve_with_scores("exam illness", k=4, vector_store=self.store)
        self.assertTrue(all(isinstance(result, RetrievalResult) for result in results))
        self.assertEqual([result.rank for result in results], [1, 2, 3, 4])
        distances = [result.distance for result in results]
        self.assertEqual(distances, sorted(distances))


if __name__ == "__main__":
    unittest.main()
