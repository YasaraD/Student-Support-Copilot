"""Fast unit tests for the project's embedding interface."""

import unittest
from unittest.mock import Mock, patch

from langchain_core.embeddings import Embeddings

from src.embeddings import (
    EMBEDDING_MODEL_NAME,
    EmbeddingGenerationError,
    EmbeddingModelLoadError,
    InvalidEmbeddingInputError,
    embed_documents,
    embed_query,
    get_embedding_dimension,
    get_embedding_model,
)


def make_mock_model() -> Mock:
    """Create a lightweight LangChain embedding-model mock."""
    model = Mock(spec=Embeddings)
    model.embed_query.return_value = [0.1, 0.2, 0.3]
    model.embed_documents.side_effect = lambda texts: [
        [float(index), 0.2, 0.3] for index, _ in enumerate(texts, start=1)
    ]
    return model


class EmbeddingTests(unittest.TestCase):
    """Verify input validation, outputs, dimensions, and error translation."""

    def test_valid_query_embedding(self) -> None:
        model = make_mock_model()
        vector = embed_query("  How do I defer an examination?  ", model=model)
        self.assertEqual(vector, [0.1, 0.2, 0.3])
        model.embed_query.assert_called_once_with("How do I defer an examination?")

    def test_valid_document_embeddings(self) -> None:
        model = make_mock_model()
        vectors = embed_documents(["Exam policy", "Library guide"], model=model)
        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(vector for vector in vectors))

    def test_empty_query_is_rejected(self) -> None:
        for invalid_query in ("", "   ", None):
            with self.subTest(query=invalid_query):
                with self.assertRaises(InvalidEmbeddingInputError):
                    embed_query(invalid_query, model=make_mock_model())

    def test_empty_document_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidEmbeddingInputError, "non-empty list"):
            embed_documents([], model=make_mock_model())

    def test_invalid_document_input_is_rejected(self) -> None:
        for invalid_texts in (("not", "a", "list"), ["valid", ""], [42]):
            with self.subTest(texts=invalid_texts):
                with self.assertRaises(InvalidEmbeddingInputError):
                    embed_documents(invalid_texts, model=make_mock_model())

    def test_embedding_dimensions_are_consistent(self) -> None:
        vectors = embed_documents(["One", "Two", "Three"], model=make_mock_model())
        dimensions = {get_embedding_dimension(vector) for vector in vectors}
        self.assertEqual(dimensions, {3})

    def test_embedding_count_matches_input_count(self) -> None:
        texts = ["One", "Two", "Three"]
        vectors = embed_documents(texts, model=make_mock_model())
        self.assertEqual(len(vectors), len(texts))

    def test_inconsistent_dimensions_are_rejected(self) -> None:
        model = make_mock_model()
        model.embed_documents.side_effect = None
        model.embed_documents.return_value = [[0.1, 0.2], [0.1]]
        with self.assertRaisesRegex(EmbeddingGenerationError, "inconsistent"):
            embed_documents(["One", "Two"], model=model)

    def test_invalid_numeric_values_are_rejected(self) -> None:
        model = make_mock_model()
        model.embed_query.return_value = [0.1, float("nan")]
        with self.assertRaisesRegex(EmbeddingGenerationError, "NaN"):
            embed_query("A valid query", model=model)

    @patch("src.embeddings.HuggingFaceEmbeddings")
    def test_model_is_configured_for_normalized_embeddings(
        self, model_class
    ) -> None:
        get_embedding_model.cache_clear()
        try:
            get_embedding_model()
            model_class.assert_called_once_with(
                model_name=EMBEDDING_MODEL_NAME,
                encode_kwargs={"normalize_embeddings": True},
            )
        finally:
            get_embedding_model.cache_clear()

    @patch("src.embeddings.HuggingFaceEmbeddings", side_effect=OSError("offline"))
    def test_model_loading_failure_has_clear_error(self, _model_class) -> None:
        get_embedding_model.cache_clear()
        try:
            with self.assertRaisesRegex(EmbeddingModelLoadError, "could not be loaded"):
                get_embedding_model()
        finally:
            get_embedding_model.cache_clear()


if __name__ == "__main__":
    unittest.main()
