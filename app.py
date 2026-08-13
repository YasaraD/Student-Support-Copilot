"""Student-facing Streamlit interface for the Student Support Copilot."""

import streamlit as st

from src.embeddings import EmbeddingError
from src.llm import LLMError
from src.prompts import PromptError
from src.rag_pipeline import RAGResponse, InvalidQuestionError, answer_question
from src.retriever import RetrievalError


APP_TITLE = "Student Support Copilot"
APP_SUBTITLE = "A RAG-Based University Assistance Chatbot"
INITIAL_ASSISTANT_MESSAGE = (
    "Hello! I can help you find information about examinations, modules, student "
    "services, and academic regulations. Ask one complete question to get started."
)
CATEGORY_DETAILS = (
    (
        "📝",
        "Examinations",
        "Registration, eligibility, missed exams, appeals, and timetables",
    ),
    (
        "📚",
        "Modules",
        "Credits, prerequisites, assessments, registration, and progression",
    ),
    (
        "🤝",
        "Student Services",
        "Library, IT, careers, counselling, accessibility, and support offices",
    ),
    (
        "📋",
        "Academic Regulations",
        "Attendance, submissions, extensions, appeals, and academic conduct",
    ),
)


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


def format_category(category: object) -> str:
    """Convert a stored category identifier into a readable label."""
    if not isinstance(category, str) or not category.strip():
        return "Category unavailable"
    return category.replace("_", " ").title()


def display_supported_categories() -> None:
    """Display the four supported knowledge areas as student-facing cards."""
    st.markdown("### What I can help with")
    first_row = st.columns(2)
    second_row = st.columns(2)
    for column, (icon, title, description) in zip(
        (*first_row, *second_row), CATEGORY_DETAILS
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {icon} {title}")
                st.caption(description)


def display_answer_sources(sources: list[dict[str, object]]) -> None:
    """Display readable citations derived from retrieved document metadata."""
    if not sources:
        return

    with st.expander(f"View supporting sources ({len(sources)})"):
        for position, source in enumerate(sources, start=1):
            filename = str(source.get("filename") or "Source unavailable")
            page = source.get("page")
            category = format_category(source.get("category"))
            page_text = f"Page {page}" if page is not None else "Page unavailable"
            st.markdown(f"**{position}. {filename}**")
            st.caption(f"{category} · {page_text}")


def display_conversation() -> None:
    """Render all messages stored in the current browser session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("is_error"):
                st.error(message["content"])
            else:
                st.write(message["content"])
            if message["role"] == "assistant":
                display_answer_sources(message.get("sources", []))


def build_source_records(rag_response: RAGResponse) -> list[dict[str, object]]:
    """Convert pipeline source objects into session-safe dictionaries."""
    return [
        {
            "filename": source.filename,
            "page": source.page,
            "category": source.category,
        }
        for source in rag_response.sources
    ]


def display_sidebar() -> None:
    """Show concise guidance without exposing development controls."""
    with st.sidebar:
        st.markdown("## Student Support Copilot")
        st.write(
            "Ask a clear, complete question about one of the supported university "
            "information areas."
        )
        st.markdown("#### Tips for better answers")
        st.markdown(
            "- Include the procedure or rule you need.\n"
            "- Mention the relevant module or examination when applicable.\n"
            "- Ask each question independently; follow-up context is not yet used."
        )
        st.divider()
        st.markdown("#### Important")
        st.caption(
            "This prototype provides informational guidance from its available "
            "documents. It cannot make official university decisions."
        )
        st.caption(
            "Do not enter private student records, medical details, credentials, "
            "or other confidential information."
        )


def handle_question(user_question: str) -> None:
    """Run one independent RAG request and render its result safely."""
    user_message = {"role": "user", "content": user_question}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the knowledge base and preparing an answer..."):
                rag_response = answer_question(user_question)

            sources = build_source_records(rag_response)
            assistant_message = {
                "role": "assistant",
                "content": rag_response.answer,
                "sources": sources,
            }
            st.write(rag_response.answer)
            display_answer_sources(sources)
        except (
            InvalidQuestionError,
            EmbeddingError,
            RetrievalError,
            PromptError,
            LLMError,
        ) as error:
            assistant_message = {
                "role": "assistant",
                "content": str(error),
                "is_error": True,
            }
            st.error(str(error))

    st.session_state.messages.append(assistant_message)


def main() -> None:
    """Render the student-facing application and handle chat input."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    initialize_conversation()
    display_sidebar()

    st.title(APP_TITLE)
    st.markdown(f"**{APP_SUBTITLE}**")
    st.write(
        "Find document-supported guidance for common university questions in one "
        "place."
    )

    display_supported_categories()

    st.info(
        "Prototype notice: the current knowledge base contains sample documents, "
        "not official university policies. Confirm important decisions with the "
        "relevant university department.",
        icon="ℹ️",
    )

    st.divider()
    conversation_column, clear_column = st.columns([4, 1])
    with conversation_column:
        st.markdown("### Ask a question")
    with clear_column:
        st.button(
            "Clear chat",
            on_click=clear_conversation,
            use_container_width=True,
        )

    display_conversation()

    user_question = st.chat_input(
        "Ask about examinations, modules, student services, or regulations"
    )
    if user_question:
        handle_question(user_question)


if __name__ == "__main__":
    main()
