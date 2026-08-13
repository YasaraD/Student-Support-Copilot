"""Tests for deterministic source formatting from Document metadata."""

import unittest

from langchain_core.documents import Document

from src.source_formatter import format_sources


def make_document(
    filename: str | None,
    page: int | None,
    category: str | None,
    chunk_id: str | None,
) -> Document:
    metadata = {}
    if filename is not None:
        metadata["filename"] = filename
    if page is not None:
        metadata["page"] = page
    if category is not None:
        metadata["category"] = category
    if chunk_id is not None:
        metadata["chunk_id"] = chunk_id
    return Document(page_content="Evidence", metadata=metadata)


class SourceFormatterTests(unittest.TestCase):
    """Verify citations use metadata and preserve first-retrieved ordering."""

    def test_filename_page_and_category_are_preserved(self) -> None:
        source = format_sources(
            [make_document("policy.pdf", 2, "examinations", "chunk-1")]
        )[0]
        self.assertEqual(source.filename, "policy.pdf")
        self.assertEqual(source.page, 2)
        self.assertEqual(source.category, "examinations")

    def test_duplicate_filename_page_is_collapsed(self) -> None:
        sources = format_sources(
            [
                make_document("policy.pdf", 2, "examinations", "chunk-1"),
                make_document("policy.pdf", 2, "examinations", "chunk-2"),
            ]
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].chunk_ids, ("chunk-1", "chunk-2"))

    def test_retrieval_order_is_preserved(self) -> None:
        sources = format_sources(
            [
                make_document("second.pdf", 4, "modules", "chunk-2"),
                make_document("first.pdf", 1, "examinations", "chunk-1"),
            ]
        )
        self.assertEqual(
            [source.filename for source in sources],
            ["second.pdf", "first.pdf"],
        )

    def test_missing_optional_metadata_is_handled(self) -> None:
        source = format_sources([make_document(None, None, None, None)])[0]
        self.assertEqual(source.filename, "Unknown source")
        self.assertIsNone(source.page)
        self.assertIsNone(source.category)
        self.assertEqual(source.chunk_ids, ())


if __name__ == "__main__":
    unittest.main()
