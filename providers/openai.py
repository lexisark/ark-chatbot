"""OpenAI provider."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .base import ChatResponse, EmbeddingResponse, StreamChunk
from .registry import registry


class OpenAIChatProvider:
    """OpenAI chat completions (GPT-4o, GPT-4o-mini, o3-mini, etc.)"""

    def __init__(self, api_key: str | None = None, default_model: str = "gpt-4o-mini", base_url: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        model = model or self.default_model
        start_time = time.time()

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = await self.client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        latency_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        model = model or self.default_model
        start_time = time.time()

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        full_text = ""
        stream = await self.client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_text += delta
                yield StreamChunk(delta=delta)

        latency_ms = int((time.time() - start_time) * 1000)
        yield StreamChunk(
            delta="",
            done=True,
            response=ChatResponse(
                content=full_text,
                model=model or self.default_model,
                tokens_in=0,
                tokens_out=0,
                latency_ms=latency_ms,
            ),
        )


class OpenAIEmbeddingProvider:
    """OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)."""

    def __init__(self, api_key: str | None = None, default_model: str = "text-embedding-3-small", dimensions: int = 1536):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        task_type: str = "retrieval_document",
    ) -> EmbeddingResponse:
        model = model or self.default_model
        response = await self.client.embeddings.create(
            model=model, input=text, dimensions=self._dimensions,
        )
        data = response.data[0]
        return EmbeddingResponse(
            embedding=data.embedding,
            model=model,
            tokens=response.usage.total_tokens,
        )

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        task_type: str = "retrieval_document",
    ) -> list[EmbeddingResponse]:
        model = model or self.default_model
        response = await self.client.embeddings.create(
            model=model, input=texts, dimensions=self._dimensions,
        )
        return [
            EmbeddingResponse(
                embedding=d.embedding,
                model=model,
                tokens=response.usage.total_tokens // max(len(texts), 1),
            )
            for d in response.data
        ]


registry.register_chat("openai", OpenAIChatProvider)
registry.register_embedding("openai", OpenAIEmbeddingProvider)
