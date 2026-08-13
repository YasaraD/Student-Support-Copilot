"""Mocked tests for secure Gemini configuration and invocation."""

import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.llm import (
    DEFAULT_LLM_MODEL,
    EmptyModelResponseError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMServiceError,
    MissingAPIKeyError,
    generate_response,
    get_configured_model_name,
    get_google_api_key,
    get_llm,
)


class ProviderFailure(Exception):
    """Provider-like test failure with an HTTP status code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMTests(unittest.TestCase):
    """Verify configuration, output extraction, and safe error messages."""

    def tearDown(self) -> None:
        get_llm.cache_clear()

    @patch.dict(os.environ, {}, clear=True)
    def test_default_configured_model_name(self) -> None:
        self.assertEqual(get_configured_model_name(), DEFAULT_LLM_MODEL)

    @patch.dict(os.environ, {"LLM_MODEL": "custom-model"}, clear=True)
    def test_configured_model_name_can_be_overridden(self) -> None:
        self.assertEqual(get_configured_model_name(), "custom-model")

    @patch("src.llm.load_dotenv")
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_handling(self, _load_dotenv: Mock) -> None:
        with self.assertRaisesRegex(MissingAPIKeyError, "GOOGLE_API_KEY"):
            get_google_api_key()

    def test_valid_mocked_model_response(self) -> None:
        model = Mock()
        model.invoke.return_value = AIMessage(content="Grounded answer")
        answer = generate_response([HumanMessage(content="Question")], model=model)
        self.assertEqual(answer, "Grounded answer")

    def test_content_block_response_is_extracted(self) -> None:
        model = Mock()
        model.invoke.return_value = AIMessage(
            content=[{"type": "text", "text": "Block answer"}]
        )
        self.assertEqual(
            generate_response([HumanMessage(content="Question")], model=model),
            "Block answer",
        )

    def test_empty_model_response_handling(self) -> None:
        model = Mock()
        model.invoke.return_value = AIMessage(content="")
        with self.assertRaises(EmptyModelResponseError):
            generate_response([HumanMessage(content="Question")], model=model)

    def test_authentication_error_is_safe_and_does_not_leak_key(self) -> None:
        secret = "test-secret-that-must-not-leak"
        model = Mock()
        model.invoke.side_effect = ProviderFailure(secret, 401)
        with self.assertRaises(LLMAuthenticationError) as raised:
            generate_response([HumanMessage(content="Question")], model=model)
        self.assertNotIn(secret, str(raised.exception))

    def test_rate_limit_error_is_classified(self) -> None:
        model = Mock()
        model.invoke.side_effect = ProviderFailure("provider details", 429)
        with self.assertRaises(LLMRateLimitError):
            generate_response([HumanMessage(content="Question")], model=model)

    def test_network_or_provider_error_is_safe(self) -> None:
        model = Mock()
        model.invoke.side_effect = OSError("internal network detail")
        with self.assertRaises(LLMServiceError) as raised:
            generate_response([HumanMessage(content="Question")], model=model)
        self.assertNotIn("internal network detail", str(raised.exception))

    @patch("src.llm.ChatGoogleGenerativeAI")
    @patch.dict(
        os.environ,
        {"GOOGLE_API_KEY": "test-key", "LLM_MODEL": "test-model"},
        clear=True,
    )
    def test_model_client_uses_developer_api_without_sampling_overrides(
        self, model_class: Mock
    ) -> None:
        first = get_llm()
        second = get_llm()
        self.assertIs(first, second)
        model_class.assert_called_once_with(
            model="test-model",
            api_key="test-key",
            vertexai=False,
        )


if __name__ == "__main__":
    unittest.main()
