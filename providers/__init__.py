"""Provider package — imports trigger auto-registration."""

from . import anthropic, gemini, ollama, openai, token_counter  # noqa: F401
from .registry import registry  # noqa: F401