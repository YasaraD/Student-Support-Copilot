"""Unit tests for the project's LangChain text-splitting logic."""

import unittest

from langchain_core.documents import Document

from src.text_splitter import TextSplittingError, split_documents


def make_document(
    text: str | None = None,
    *,
    page: int = 1,
    category: str = "examinations",
) -> Document:
    """Create a realistic page Document for focused splitter tests."""
    content = text or (("University policy paragraph with useful context. " * 20))
    return Document(
        page_content=content,
        metadata={
            "source": "mock_policy.pdf",
            "filename": "mock_policy.pdf",
            "page": page,
            "category": category,
            "title": "Mock Policy",
        },
    )


class TextSplitterTests(unittest.TestCase):
    """Verify project validation, metadata, and chunk identifiers."""

    def test_documents_are_split_successfully(self) -> None:
        chunks = split_documents([make_document()], chunk_size=180, chunk_overlap=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page_content.strip() for chunk in chunks))

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(TextSplittingError, "No documents"):
            split_documents([])

    def test_invalid_chunk_size_is_rejected(self) -> None:
        for invalid_size in (0, -1, 10.5, True):
            with self.subTest(chunk_size=invalid_size):
                with self.assertRaises(TextSplittingError):
                    split_documents([make_document()], chunk_size=invalid_size)

    def test_invalid_overlap_is_rejected(self) -> None:
        for invalid_overlap in (-1, 10.5, True):
            with self.subTest(chunk_overlap=invalid_overlap):
                with self.assertRaises(TextSplittingError):
                    split_documents(
                        [make_document()],
                        chunk_size=180,
                        chunk_overlap=invalid_overlap,
                    )

    def test_overlap_equal_to_chunk_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(TextSplittingError, "must be smaller"):
            split_documents([make_document()], chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_chunk_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(TextSplittingError, "must be smaller"):
            split_documents([make_document()], chunk_size=100, chunk_overlap=101)

    def test_original_metadata_is_preserved(self) -> None:
        chunks = split_documents([make_document()], chunk_size=180, chunk_overlap=30)
        self.assertTrue(
            all(chunk.metadata["title"] == "Mock Policy" for chunk in chunks)
        )
        self.assertTrue(
            all(chunk.metadata["source"] == "mock_policy.pdf" for chunk in chunks)
        )
        self.assertTrue(
            all(chunk.metadata["filename"] == "mock_policy.pdf" for chunk in chunks)
        )

    def test_category_is_preserved(self) -> None:
        chunks = split_documents(
            [make_document(category="student_services")],
            chunk_size=180,
            chunk_overlap=30,
        )
        self.assertTrue(
            all(chunk.metadata["category"] == "student_services" for chunk in chunks)
        )

    def test_page_number_is_preserved(self) -> None:
        chunks = split_documents(
            [make_document(page=7)], chunk_size=180, chunk_overlap=30
        )
        self.assertTrue(all(chunk.metadata["page"] == 7 for chunk in chunks))

    def test_every_chunk_receives_a_chunk_id(self) -> None:
        chunks = split_documents([make_document()], chunk_size=180, chunk_overlap=30)
        self.assertTrue(all(chunk.metadata.get("chunk_id") for chunk in chunks))
        self.assertTrue(
            all(
                chunk.metadata["chunk_id"].startswith(
                    "examinations_mock_policy_page_1_chunk_"
                )
                for chunk in chunks
            )
        )

    def test_chunk_ids_are_unique(self) -> None:
        documents = [make_document(page=1), make_document(page=2)]
        chunks = split_documents(documents, chunk_size=180, chunk_overlap=30)
        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))


if __name__ == "__main__":
    unittest.main()
