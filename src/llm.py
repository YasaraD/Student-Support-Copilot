"""Configure and invoke Gemini for grounded answer generation only."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
MISSING_API_KEY_MESSAGE = (
    "Gemini API access is not configured. Add GOOGLE_API_KEY to your local .env "
    "file."
)


class LLMError(RuntimeError):
    """Base error for safe, expected answer-generation failures."""


class MissingAPIKeyError(LLMError):
    """Raised when local Gemini credentials are unavailable."""


class LLMAuthenticationError(LLMError):
    """Raised when Gemini rejects the configured credential."""


class LLMRateLimitError(LLMError):
    """Raised when Gemini quota or rate limits prevent generation."""


class LLMServiceError(LLMError):
    """Raised when the external generation service cannot complete a request."""


class EmptyModelResponseError(LLMError):
    """Raised when Gemini returns no usable answer text."""


def _load_local_environment() -> None:
    """Load an optional project-local .env without replacing shell variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_configured_model_name() -> str:
    """Return the configurable Gemini model ID or the project default."""
    _load_local_environment()
    configured = os.getenv("LLM_MODEL", "").strip()
    return configured or DEFAULT_LLM_MODEL


def get_google_api_key() -> str:
    """Return the local API key or raise a safe configuration error."""
    _load_local_environment()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(MISSING_API_KEY_MESSAGE)
    return api_key


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Create and reuse only the Gemini client resource, never responses."""
    return ChatGoogleGenerativeAI(
        model=get_configured_model_name(),
        api_key=get_google_api_key(),
        vertexai=False,
    )


def _extract_text(content: object) -> str:
    """Extract text from current LangChain string or content-block responses."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return ""


def _provider_status_code(error: Exception) -> int | None:
    """Read a provider status code without exposing the provider response."""
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _translate_provider_error(error: Exception) -> LLMError:
    """Convert provider failures into fixed messages that cannot leak secrets."""
    status_code = _provider_status_code(error)
    error_name = type(error).__name__.lower()
    if status_code in (401, 403) or "auth" in error_name or "permission" in error_name:
        return LLMAuthenticationError(
            "Gemini rejected the configured API credential. Check GOOGLE_API_KEY "
            "in the local .env file."
        )
    if status_code == 429 or "quota" in error_name or "ratelimit" in error_name:
        return LLMRateLimitError(
            "The Gemini generation service cannot process this request because a "
            "rate or quota limit was reached. Please try again later."
        )
    return LLMServiceError(
        "The Gemini answer-generation service is temporarily unavailable. Please "
        "check the network connection and try again later."
    )


def generate_response(
    messages: list[BaseMessage],
    model: ChatGoogleGenerativeAI | None = None,
) -> str:
    """Invoke Gemini with prepared messages and return non-empty answer text."""
    if not isinstance(messages, list) or not messages:
        raise LLMError("Prepared LangChain messages are required for generation.")
    chat_model = model or get_llm()
    try:
        response = chat_model.invoke(messages)
    except Exception as error:
        raise _translate_provider_error(error) from error

    answer = _extract_text(response.content)
    if not answer:
        raise EmptyModelResponseError(
            "Gemini returned an empty answer. Please try the question again."
        )
    return answer
