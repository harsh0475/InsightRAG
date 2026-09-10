"""OpenAI-compatible LLM provider implementation."""
import logging
from typing import Optional
from openai import OpenAI

from backend.app.core.config import get_settings
from backend.app.services.llm.base import BaseLLMProvider

logger = logging.getLogger("insightrag.llm.openai")


class OpenAILLMProvider(BaseLLMProvider):
    """Generates completions using OpenAI or any OpenAI-compatible endpoint (Ollama, vLLM, Azure)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        default_temperature: Optional[float] = None,
        default_max_tokens: Optional[int] = None,
    ):
        settings = get_settings()
        self._api_key = api_key or settings.LLM_API_KEY
        self._base_url = base_url or settings.LLM_BASE_URL
        self._model_name = model_name or settings.LLM_MODEL
        self._default_temperature = (
            default_temperature if default_temperature is not None else settings.LLM_TEMPERATURE
        )
        self._default_max_tokens = default_max_tokens or settings.LLM_MAX_TOKENS

        if not self._api_key:
            logger.warning("No LLM_API_KEY configured for OpenAILLMProvider.")

        self.client = OpenAI(
            api_key=self._api_key or "sk-dummy-key-for-local-compatibility",
            base_url=self._base_url,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the chat completions endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else self._default_temperature
        tokens = max_tokens or self._default_max_tokens

        try:
            logger.info(f"Generating LLM completion with model '{self._model_name}' (temp={temp})")
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.error(f"OpenAI LLM API call failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e

