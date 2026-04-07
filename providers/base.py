"""Provider protocols and response types for the model harness."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ── Response Types ──────────────────────────────────────


@dataclass
class ChatResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    delta: str
    done: bool = False
    response: ChatResponse | None = None


@dataclass
class EmbeddingResponse:
    embedding: list[float]
    model: str
    tokens: int


# ── Chat Provider ───────────────────────────────────────


@runtime_checkable
class ChatProvider(Protocol):
    """Generates chat completions from an LLM."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse: ...

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]: ...


# ── Embedding Provider ──────────────────────────────────


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generates vector embeddings for text."""

    @property
    def dimensions(self) -> int: ...

    async def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        task_type: str = "retrieval_document",
    ) -> EmbeddingResponse: ...

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        task_type: str = "retrieval_document",
    ) -> list[EmbeddingResponse]: ...


# ── Token Counter ───────────────────────────────────────


@runtime_checkable
class TokenCounter(Protocol):
    """Counts and manages tokens for a specific model family."""

    def count(self, text: str) -> int: ...

    def count_messages(self, messages: list[dict[str, str]]) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...

    def fits(self, text: str, budget: int) -> bool: ...
