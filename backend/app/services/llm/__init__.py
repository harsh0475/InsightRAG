"""LLM providers package and factory."""
from typing import Optional
from backend.app.core.config import get_settings
from backend.app.services.llm.base import BaseLLMProvider
from backend.app.services.llm.mock_provider import MockLLMProvider
from backend.app.services.llm.openai_provider import OpenAILLMProvider

__all__ = ["BaseLLMProvider", "OpenAILLMProvider", "MockLLMProvider", "get_llm_provider"]


def get_llm_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """Factory function returning the configured LLM provider."""
    settings = get_settings()
    ptype = (provider_type or settings.LLM_PROVIDER).lower()

    if ptype == "openai":
        if settings.LLM_API_KEY:
            return OpenAILLMProvider()
        return MockLLMProvider(model_name="mock-" + settings.LLM_MODEL)
    elif ptype == "mock":
        return MockLLMProvider(model_name="mock-" + settings.LLM_MODEL)
    else:
        raise ValueError(f"Unsupported LLM provider: '{ptype}'. Supported: 'openai', 'mock'")

