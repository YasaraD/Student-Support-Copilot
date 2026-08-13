"""Orchestrate the project's explicit retrieval-then-generation workflow."""

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import BaseMessage

from src.llm import generate_response
from src.prompts import build_rag_messages, format_retrieved_context
from src.retriever import TOP_K, RetrievalResult, retrieve_with_scores
from src.source_formatter import SourceCitation, format_sources


class InvalidQuestionError(ValueError):
    """Raised before retrieval when a student question is invalid."""


@dataclass(frozen=True)
class RAGResponse:
    """Generated answer, deterministic sources, and inspectable retrieval data."""

    answer: str
    sources: list[SourceCitation]
    retrieval_results: list[RetrievalResult]


def answer_question(
    question: str,
    *,
    retrieve_fn: Callable[..., list[RetrievalResult]] = retrieve_with_scores,
    generate_fn: Callable[[list[BaseMessage]], str] = generate_response,
) -> RAGResponse:
    """Run Step 1 retrieval and Step 2 grounded generation for one question."""
    if not isinstance(question, str) or not question.strip():
        raise InvalidQuestionError(
            "The student question must be a non-empty text string."
        )
    cleaned_question = question.strip()

    retrieval_results = retrieve_fn(
        cleaned_question,
        k=TOP_K,
        category=None,
    )
    if not retrieval_results:
        raise InvalidQuestionError(
            "No document evidence was retrieved, so an answer was not generated."
        )

    documents = [result.document for result in retrieval_results]
    context = format_retrieved_context(documents)
    messages = build_rag_messages(cleaned_question, context)
    answer = generate_fn(messages)
    sources = format_sources(documents)
    return RAGResponse(
        answer=answer,
        sources=sources,
        retrieval_results=retrieval_results,
    )
