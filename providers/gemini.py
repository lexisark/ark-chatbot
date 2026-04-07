"""Google Gemini provider via Vertex AI."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator

from google import genai
from google.genai.types import GenerateContentConfig

from .base import ChatResponse, EmbeddingResponse, StreamChunk
from .registry import registry

logger = logging.getLogger(__name__)


class GeminiChatProvider:
    """Google Gemini chat via Vertex AI (gemini-2.5-flash, gemini-2.5-pro, etc.)"""

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        api_key: str | None = None,
        default_model: str = "gemini-2.5-flash",
    ):
        self.default_model = default_model
        project_id = project_id or os.getenv("GCP_PROJECT_ID", "")
        region = region or os.getenv("GCP_REGION", "us-central1")

        if api_key or os.getenv("GEMINI_API_KEY"):
            # Use API key (simpler, non-Vertex)
            self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        else:
            # Use Vertex AI with application default credentials
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=region,
            )

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

        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        contents = self._format_messages(messages)

        # Retry on 429
        for attempt in range(3):
            try:
                response = await self.client.aio.models.generate_content(
                    model=model, contents=contents, config=config,
                )
                break
            except Exception as e:
                if ("429" in str(e) or "resource exhausted" in str(e).lower()) and attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

        content = response.text or ""
        latency_ms = int((time.time() - start_time) * 1000)

        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0

        return ChatResponse(
            content=content,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
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

        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        contents = self._format_messages(messages)
        full_text = ""

        stream = await self.client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config,
        )
        async for chunk in stream:
            if chunk.text:
                full_text += chunk.text
                yield StreamChunk(delta=chunk.text)

        latency_ms = int((time.time() - start_time) * 1000)
        yield StreamChunk(
            delta="",
            done=True,
            response=ChatResponse(
                content=full_text,
                model=model,
                tokens_in=0,
                tokens_out=0,
                latency_ms=latency_ms,
            ),
        )

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages into a single prompt string for Gemini."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n\n".join(parts)


class GeminiEmbeddingProvider:
    """Google Gemini embeddings (gemini-embedding-001)."""

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        api_key: str | None = None,
        default_model: str = "gemini-embedding-001",
        dimensions: int = 768,
    ):
        self.default_model = default_model
        self._dimensions = dimensions
        project_id = project_id or os.getenv("GCP_PROJECT_ID", "")
        region = region or os.getenv("GCP_REGION", "us-central1")

        if api_key or os.getenv("GEMINI_API_KEY"):
            self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        else:
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=region,
            )

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
        result = await self.client.aio.models.embed_content(
            model=model,
            contents=text,
            config={"task_type": task_type.upper()},
        )
        embedding = result.embeddings[0].values
        return EmbeddingResponse(
            embedding=list(embedding),
            model=model,
            tokens=len(text) // 4,
        )

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        task_type: str = "retrieval_document",
    ) -> list[EmbeddingResponse]:
        return [await self.embed(t, model=model, task_type=task_type) for t in texts]


registry.register_chat("gemini", GeminiChatProvider)
registry.register_embedding("gemini", GeminiEmbeddingProvider)
