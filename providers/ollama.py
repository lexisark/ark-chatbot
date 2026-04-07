"""Ollama provider via OpenAI-compatible API."""

from __future__ import annotations

from .openai import OpenAIChatProvider
from .registry import registry


class OllamaChatProvider(OpenAIChatProvider):
    """Ollama local models via OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", default_model: str = "llama3.2"):
        super().__init__(api_key="ollama", default_model=default_model, base_url=base_url)


registry.register_chat("ollama", OllamaChatProvider)
