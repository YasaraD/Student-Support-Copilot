"""Load text-based PDF files into page-level LangChain documents."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf.errors import EmptyFileError, FileNotDecryptedError, PdfReadError

from src.config import INDEX_CONFIG


SUPPORTED_CATEGORIES = INDEX_CONFIG.supported_categories


class DocumentLoaderError(Exception):
    """Base error for expected document-loading failures."""


class InvalidCategoryError(DocumentLoaderError, ValueError):
    """Raised when a document category is not supported."""


class PDFFileNotFoundError(DocumentLoaderError, FileNotFoundError):
    """Raised when the requested PDF path does not exist."""


class InvalidPDFFileError(DocumentLoaderError, ValueError):
    """Raised when the requested file does not have a PDF extension."""


class EncryptedPDFError(DocumentLoaderError):
    """Raised when a password-protected PDF cannot be read."""


class UnreadablePDFError(DocumentLoaderError):
    """Raised when a PDF is empty, damaged, or unsupported."""


class NoExtractableTextError(DocumentLoaderError):
    """Raised when a PDF contains no machine-readable text."""


def validate_category(category: str) -> None:
    """Check that a category is one of the four supported identifiers."""
    if category not in SUPPORTED_CATEGORIES:
        supported = ", ".join(SUPPORTED_CATEGORIES)
        raise InvalidCategoryError(
            f"Invalid category '{category}'. Choose one of: {supported}."
        )


def _validate_pdf_path(pdf_path: Path) -> None:
    """Check that the source exists and has a PDF extension."""
    if not pdf_path.exists() or not pdf_path.is_file():
        raise PDFFileNotFoundError(
            f"PDF file not found: {pdf_path}. Check the path and try again."
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise InvalidPDFFileError(
            f"The selected file is not a PDF: {pdf_path.name}. Choose a .pdf file."
        )


def _friendly_source_name(pdf_path: Path, original_filename: str | None) -> str:
    """Return a safe filename for source metadata."""
    if original_filename:
        safe_name = Path(original_filename).name
        if safe_name:
            return safe_name
    return pdf_path.name


def _normalize_metadata(
    documents: list[Document], source_name: str, category: str
) -> None:
    """Add consistent, human-readable metadata to each loaded page."""
    for page_number, document in enumerate(documents, start=1):
        document.metadata.update(
            {
                "source": source_name,
                "filename": source_name,
                "page": page_number,
                "category": category,
            }
        )


def load_pdf_document(
    file_path: str | Path,
    category: str,
    *,
    original_filename: str | None = None,
) -> list[Document]:
    """Load a text-based PDF as one LangChain ``Document`` per page.

    Args:
        file_path: Path to a locally readable PDF file.
        category: One of the four supported knowledge-category identifiers.
        original_filename: Optional source filename to preserve when ``file_path``
            points to a temporary upload.

    Returns:
        Page-level LangChain documents with normalized source metadata.

    Raises:
        DocumentLoaderError: If validation or PDF text extraction fails.
    """
    validate_category(category)
    pdf_path = Path(file_path)
    _validate_pdf_path(pdf_path)

    try:
        documents = PyPDFLoader(str(pdf_path), mode="page").load()
    except FileNotDecryptedError as error:
        raise EncryptedPDFError(
            "The PDF is encrypted or password-protected and cannot be opened. "
            "Use an unencrypted text-based PDF."
        ) from error
    except EmptyFileError as error:
        raise UnreadablePDFError(
            "The PDF is empty and cannot be inspected. Choose a non-empty PDF."
        ) from error
    except PdfReadError as error:
        error_text = str(error).lower()
        if (
            "encrypt" in error_text
            or "password" in error_text
            or "decrypt" in error_text
        ):
            raise EncryptedPDFError(
                "The PDF is encrypted or password-protected and cannot be opened. "
                "Use an unencrypted text-based PDF."
            ) from error
        raise UnreadablePDFError(
            "The PDF could not be opened. It may be damaged or use an unsupported "
            "PDF format. Try opening and resaving it with a PDF reader."
        ) from error
    except (OSError, ValueError) as error:
        raise UnreadablePDFError(
            "The PDF could not be opened. Check that it is a valid, readable PDF "
            "and try again."
        ) from error

    if not documents or not any(document.page_content.strip() for document in documents):
        raise NoExtractableTextError(
            "No machine-readable text was found. The PDF may be empty or scanned "
            "as images. OCR is not supported in this milestone."
        )

    source_name = _friendly_source_name(pdf_path, original_filename)
    _normalize_metadata(documents, source_name, category)
    return documents
