"""Create deterministic student-facing sources from retrieved metadata."""

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class SourceCitation:
    """One source/page citation in first-retrieved order."""

    filename: str
    page: int | str | None
    category: str | None
    chunk_ids: tuple[str, ...]


def format_sources(documents: list[Document]) -> list[SourceCitation]:
    """Deduplicate exact filename/page pairs while preserving retrieval order."""
    citations: list[SourceCitation] = []
    positions: dict[tuple[str, int | str | None], int] = {}

    for document in documents:
        metadata = document.metadata
        filename = str(
            metadata.get("filename") or metadata.get("source") or "Unknown source"
        )
        page = metadata.get("page")
        category_value = metadata.get("category")
        category = str(category_value) if category_value is not None else None
        chunk_value = metadata.get("chunk_id")
        chunk_id = str(chunk_value) if chunk_value is not None else None
        key = (filename, page)

        if key not in positions:
            positions[key] = len(citations)
            citations.append(
                SourceCitation(
                    filename=filename,
                    page=page,
                    category=category,
                    chunk_ids=(chunk_id,) if chunk_id else (),
                )
            )
            continue

        position = positions[key]
        existing = citations[position]
        if chunk_id and chunk_id not in existing.chunk_ids:
            citations[position] = SourceCitation(
                filename=existing.filename,
                page=existing.page,
                category=existing.category,
                chunk_ids=existing.chunk_ids + (chunk_id,),
            )
    return citations
