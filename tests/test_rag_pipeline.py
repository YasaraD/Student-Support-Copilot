"""Tests for explicit retrieval-then-generation orchestration."""

import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from src.rag_pipeline import InvalidQuestionError, answer_question
from src.retriever import TOP_K, RetrievalError, RetrievalResult


def make_result() -> RetrievalResult:
    """Create one realistic scored retrieval result."""
    return RetrievalResult(
        document=Document(
            page_content="A missed examination requires a mitigation request.",
            metadata={
                "filename": "mock_exam_policy.pdf",
                "source": "mock_exam_policy.pdf",
                "page": 2,
                "category": "examinations",
                "chunk_id": "exam_page_2_chunk_0",
            },
        ),
        distance=0.25,
        rank=1,
    )


class RAGPipelineTests(unittest.TestCase):
    """Verify the two phases and deterministic output boundaries."""

    def test_valid_question_returns_answer_and_sources(self) -> None:
        retrieve_fn = Mock(return_value=[make_result()])
        generate_fn = Mock(return_value="Submit a mitigation request.")
        response = answer_question(
            "What should I do?",
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        self.assertEqual(response.answer, "Submit a mitigation request.")
        self.assertEqual(response.sources[0].filename, "mock_exam_policy.pdf")
        self.assertEqual(response.sources[0].page, 2)

    def test_empty_question_is_rejected_before_retrieval(self) -> None:
        retrieve_fn = Mock()
        generate_fn = Mock()
        with self.assertRaises(InvalidQuestionError):
            answer_question("   ", retrieve_fn=retrieve_fn, generate_fn=generate_fn)
        retrieve_fn.assert_not_called()
        generate_fn.assert_not_called()

    def test_retriever_uses_default_top_k_and_all_categories(self) -> None:
        retrieve_fn = Mock(return_value=[make_result()])
        answer_question(
            "Question",
            retrieve_fn=retrieve_fn,
            generate_fn=Mock(return_value="Answer"),
        )
        retrieve_fn.assert_called_once_with("Question", k=TOP_K, category=None)

    def test_retrieved_context_reaches_generation(self) -> None:
        generate_fn = Mock(return_value="Answer")
        result = make_result()
        answer_question(
            "Question",
            retrieve_fn=Mock(return_value=[result]),
            generate_fn=generate_fn,
        )
        messages = generate_fn.call_args.args[0]
        self.assertIn(result.document.page_content, messages[1].content)
        self.assertIn("Question", messages[1].content)

    def test_retrieval_failure_prevents_generation(self) -> None:
        retrieve_fn = Mock(side_effect=RetrievalError("unavailable"))
        generate_fn = Mock()
        with self.assertRaises(RetrievalError):
            answer_question(
                "Question",
                retrieve_fn=retrieve_fn,
                generate_fn=generate_fn,
            )
        generate_fn.assert_not_called()

    def test_empty_retrieval_prevents_generation(self) -> None:
        generate_fn = Mock()
        with self.assertRaises(InvalidQuestionError):
            answer_question(
                "Question",
                retrieve_fn=Mock(return_value=[]),
                generate_fn=generate_fn,
            )
        generate_fn.assert_not_called()

    def test_sources_come_from_metadata_not_generated_answer(self) -> None:
        response = answer_question(
            "Question",
            retrieve_fn=Mock(return_value=[make_result()]),
            generate_fn=Mock(return_value="Invented citation fake.pdf page 99"),
        )
        self.assertEqual(response.sources[0].filename, "mock_exam_policy.pdf")
        self.assertNotEqual(response.sources[0].filename, "fake.pdf")


if __name__ == "__main__":
    unittest.main()
