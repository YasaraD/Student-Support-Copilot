"""Streamlit interface for the Student Support Copilot."""

import streamlit as st


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


def main() -> None:
    """Render the application and handle chat input."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="centered")
    initialize_conversation()

    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.write(
        "This is the initial interface for the student-support chatbot. "
        "Document search and AI-generated answers will be added in later milestones."
    )

    st.markdown("#### Supported categories")
    display_category_cards()
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
