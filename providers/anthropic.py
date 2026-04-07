"""Anthropic provider."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from .base import ChatResponse, StreamChunk
from .registry import registry


class AnthropicChatProvider:
    """Anthropic chat (Claude Sonnet, Opus, Haiku)."""

    def __init__(self, api_key: str | None = None, default_model: str = "claude-sonnet-4-6"):
        self.client = AsyncAnthropic(api_key=api_key)
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

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)
        latency_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
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

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        full_text = ""
        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                full_text += text
                yield StreamChunk(delta=text)

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


registry.register_chat("anthropic", AnthropicChatProvider)
