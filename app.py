"""Streamlit interface for the Student Support Copilot."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from src.document_loader import DocumentLoaderError, load_pdf_document
from src.text_splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextSplittingError,
    split_documents,
)


APP_TITLE = "Student Support Copilot"
APP_SUBTITLE = "A RAG-Based University Assistance Chatbot"
CATEGORIES = (
    "Examinations",
    "Modules",
    "Student Services",
    "Academic Regulations",
)
INITIAL_ASSISTANT_MESSAGE = (
    "Hello! This is the initial chatbot interface. You can enter a question to "
    "test the conversation layout."
)
PLACEHOLDER_RESPONSE = (
    "Thanks for your question. The RAG knowledge base has not been connected "
    "yet, so I cannot provide an answer from university documents at this stage."
)
PREVIEW_CHARACTER_LIMIT = 800
CHUNK_PREVIEW_CHARACTER_LIMIT = 320
DEFAULT_SAMPLE_CHUNK_COUNT = 2
MAX_SAMPLE_CHUNK_COUNT = 3
CATEGORY_OPTIONS = {
    "Examinations": "examinations",
    "Modules": "modules",
    "Student Services": "student_services",
    "Academic Regulations": "academic_regulations",
}


def initialize_conversation() -> None:
    """Create the conversation history when a session starts."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": INITIAL_ASSISTANT_MESSAGE}
        ]


def clear_conversation() -> None:
    """Reset the current conversation to its welcome message."""
    st.session_state.messages = [
        {"role": "assistant", "content": INITIAL_ASSISTANT_MESSAGE}
    ]


def display_category_cards() -> None:
    """Show the four university-support categories in a compact row."""
    columns = st.columns(4)
    for column, category in zip(columns, CATEGORIES):
        with column:
            st.info(category)


def display_conversation() -> None:
    """Render all messages stored in the current session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def _create_preview(
    page_content: str, character_limit: int = PREVIEW_CHARACTER_LIMIT
) -> str:
    """Create a compact single-block preview of extracted page text."""
    cleaned_text = " ".join(page_content.split())
    if len(cleaned_text) <= character_limit:
        return cleaned_text
    return f"{cleaned_text[:character_limit].rstrip()}..."


def display_chunk_samples(
    documents: list[Document], sample_count: int
) -> None:
    """Split loaded pages and display a small development summary."""
    try:
        chunks = split_documents(documents)
    except TextSplittingError as error:
        st.error(str(error))
        return

    st.markdown("#### Chunk inspection")
    st.write(f"**Original filename:** {documents[0].metadata['filename']}")
    st.write(f"**Selected category:** {documents[0].metadata['category']}")
    st.write(f"**Page-level Documents:** {len(documents)}")
    st.write(f"**Generated chunks:** {len(chunks)}")
    st.write(f"**Configured chunk size:** {DEFAULT_CHUNK_SIZE} characters")
    st.write(f"**Configured chunk overlap:** {DEFAULT_CHUNK_OVERLAP} characters")

    for sample_number, chunk in enumerate(chunks[:sample_count], start=1):
        st.markdown(f"**Sample chunk {sample_number}**")
        st.write(f"Chunk ID: `{chunk.metadata['chunk_id']}`")
        st.write(f"Page: {chunk.metadata['page']}")
        st.write(f"Category: `{chunk.metadata['category']}`")
        st.write(f"Character count: {len(chunk.page_content)}")
        st.text(
            _create_preview(
                chunk.page_content,
                character_limit=CHUNK_PREVIEW_CHARACTER_LIMIT,
            )
        )


def display_document_inspector() -> None:
    """Provide a development-only interface for inspecting one uploaded PDF."""
    with st.expander("Development tool: inspect one PDF"):
        st.caption(
            "This local development tool temporarily loads one text-based PDF. "
            "The uploaded file is deleted immediately after inspection."
        )
        category_label = st.selectbox(
            "Document category",
            options=tuple(CATEGORY_OPTIONS),
            key="inspection_category",
        )
        uploaded_pdf = st.file_uploader(
            "Upload one text-based PDF",
            type=("pdf",),
            accept_multiple_files=False,
            key="inspection_pdf",
        )
        inspect_chunks = st.checkbox(
            "Also split and inspect sample chunks",
            value=False,
            key="inspect_chunks",
        )
        sample_chunk_count = st.number_input(
            "Number of sample chunks",
            min_value=1,
            max_value=MAX_SAMPLE_CHUNK_COUNT,
            value=DEFAULT_SAMPLE_CHUNK_COUNT,
            step=1,
            disabled=not inspect_chunks,
            key="sample_chunk_count",
        )

        if uploaded_pdf is None or not st.button("Inspect PDF"):
            return

        temporary_path: Path | None = None
        documents = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_pdf.getbuffer())
                temporary_path = Path(temp_file.name)

            documents = load_pdf_document(
                temporary_path,
                CATEGORY_OPTIONS[category_label],
                original_filename=uploaded_pdf.name,
            )
        except DocumentLoaderError as error:
            st.error(str(error))
        except OSError:
            st.error(
                "The temporary PDF copy could not be created. Check available "
                "disk space and file permissions, then try again."
            )
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    os.unlink(temporary_path)
                except OSError:
                    st.warning(
                        "The temporary upload could not be deleted automatically. "
                        "Close the application and remove it from your system's "
                        "temporary folder."
                    )

        if documents is None:
            return

        pages_with_text = sum(
            bool(document.page_content.strip()) for document in documents
        )
        first_text_page = next(
            document for document in documents if document.page_content.strip()
        )

        st.success("The PDF was loaded successfully.")
        st.write(f"**Uploaded filename:** {uploaded_pdf.name}")
        st.write(f"**Selected category:** {category_label}")
        st.write(f"**LangChain Document objects:** {len(documents)}")
        st.write(f"**Pages with extractable text:** {pages_with_text}")
        st.markdown("#### Short text preview")
        st.text(_create_preview(first_text_page.page_content))
        st.markdown("#### Metadata for the first loaded page")
        st.json(documents[0].metadata)

        if inspect_chunks:
            display_chunk_samples(documents, int(sample_chunk_count))


def main() -> None:
    """Render the application and handle chat input."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="centered")
    initialize_conversation()

    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.write(
        "This is the initial interface for the student-support chatbot. "
        "Milestone 3 can load, split, and inspect one text-based PDF, but document "
        "search and AI-generated answers will be added in later milestones."
    )

    st.markdown("#### Supported categories")
    display_category_cards()
    st.divider()

    display_document_inspector()
    st.divider()

    header_column, button_column = st.columns([3, 1])
    with header_column:
        st.markdown("### Conversation")
    with button_column:
        st.button(
            "Clear conversation",
            on_click=clear_conversation,
            use_container_width=True,
        )

    display_conversation()

    user_question = st.chat_input("Ask a university support question")
    if user_question:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.write(user_question)

        st.session_state.messages.append(
            {"role": "assistant", "content": PLACEHOLDER_RESPONSE}
        )
        with st.chat_message("assistant"):
            st.write(PLACEHOLDER_RESPONSE)


if __name__ == "__main__":
    main()
