"""Unit tests for LLM providers."""
from backend.app.services.llm import BaseLLMProvider, MockLLMProvider, get_llm_provider


def test_mock_llm_grounded_generation():
    """Verify MockLLMProvider generates factual responses with chunk citations."""
    provider = MockLLMProvider()
    prompt = "Context: [ID: doc_oauth_c0001] Access tokens expire in 15 minutes. Question: How long do access tokens last?"
    answer = provider.generate(prompt=prompt)

    assert "OAuth 2.0" in answer or "15-minute" in answer
    assert "[doc_oauth_c0001]" in answer


def test_mock_llm_refusal_on_empty_context():
    """Verify MockLLMProvider refuses when context is empty or absent."""
    provider = MockLLMProvider()
    prompt = "Context: [No relevant context retrieved from knowledge base.] Question: What is the capital of Mars?"
    answer = provider.generate(prompt=prompt)

    assert "I do not have sufficient information" in answer


def test_llm_factory():
    """Verify LLM factory returns BaseLLMProvider."""
    provider = get_llm_provider("mock")
    assert isinstance(provider, BaseLLMProvider)
    assert provider.model_name.startswith("mock-")

