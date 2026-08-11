"""Unit tests for page-based PDF document loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from pypdf.errors import FileNotDecryptedError, PdfReadError

from src.document_loader import (
    EncryptedPDFError,
    InvalidCategoryError,
    InvalidPDFFileError,
    NoExtractableTextError,
    PDFFileNotFoundError,
    SUPPORTED_CATEGORIES,
    UnreadablePDFError,
    load_pdf_document,
    validate_category,
)


class CategoryValidationTests(unittest.TestCase):
    """Verify the accepted knowledge-category identifiers."""

    def test_all_supported_categories_are_valid(self) -> None:
        for category in SUPPORTED_CATEGORIES:
            with self.subTest(category=category):
                validate_category(category)

    def test_invalid_category_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidCategoryError, "Invalid category"):
            validate_category("finance")


class DocumentLoadingTests(unittest.TestCase):
    """Verify validation, extraction checks, and metadata normalization."""

    def test_missing_pdf_is_rejected(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "missing-support-document.pdf"
        with self.assertRaisesRegex(PDFFileNotFoundError, "not found"):
            load_pdf_document(missing_path, "examinations")

    def test_non_pdf_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "notes.txt"
            text_path.touch()

            with self.assertRaisesRegex(InvalidPDFFileError, "not a PDF"):
                load_pdf_document(text_path, "modules")

    @patch("src.document_loader.PyPDFLoader")
    def test_metadata_is_normalized_for_each_page(self, loader_class) -> None:
        loader_class.return_value.load.return_value = [
            Document(page_content="First page", metadata={"title": "Guide"}),
            Document(page_content="Second page", metadata={"page": 99}),
        ]

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "temporary-upload.pdf"
            pdf_path.touch()
            documents = load_pdf_document(
                pdf_path,
                "student_services",
                original_filename="Student Guide.pdf",
            )

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].metadata["source"], "Student Guide.pdf")
        self.assertEqual(documents[0].metadata["filename"], "Student Guide.pdf")
        self.assertEqual(documents[0].metadata["page"], 1)
        self.assertEqual(documents[1].metadata["page"], 2)
        self.assertEqual(documents[0].metadata["category"], "student_services")
        self.assertEqual(documents[0].metadata["title"], "Guide")
        loader_class.assert_called_once_with(str(pdf_path), mode="page")

    @patch("src.document_loader.PyPDFLoader")
    def test_pdf_without_extractable_text_is_rejected(self, loader_class) -> None:
        loader_class.return_value.load.return_value = [
            Document(page_content="   ", metadata={"page": 0}),
            Document(page_content="\n", metadata={"page": 1}),
        ]

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "scanned.pdf"
            pdf_path.touch()

            with self.assertRaisesRegex(
                NoExtractableTextError, "No machine-readable text"
            ):
                load_pdf_document(pdf_path, "academic_regulations")

    @patch("src.document_loader.PyPDFLoader")
    def test_encrypted_pdf_has_a_clear_error(self, loader_class) -> None:
        loader_class.return_value.load.side_effect = FileNotDecryptedError(
            "File has not been decrypted"
        )

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "protected.pdf"
            pdf_path.touch()

            with self.assertRaisesRegex(EncryptedPDFError, "password-protected"):
                load_pdf_document(pdf_path, "examinations")

    @patch("src.document_loader.PyPDFLoader")
    def test_damaged_pdf_has_a_clear_error(self, loader_class) -> None:
        loader_class.return_value.load.side_effect = PdfReadError("broken xref")

        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "damaged.pdf"
            pdf_path.touch()

            with self.assertRaisesRegex(UnreadablePDFError, "could not be opened"):
                load_pdf_document(pdf_path, "modules")


if __name__ == "__main__":
    unittest.main()
