"""Embedding service — wraps EmbeddingProvider with normalization and retry."""

from __future__ import annotations

import asyncio
import logging
import math

from providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps an EmbeddingProvider with L2 normalization and retry logic."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self._provider = provider
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions

    async def generate_query_embedding(self, text: str) -> list[float]:
        """Generate embedding for a search query."""
        return await self._embed_with_retry(text, task_type="retrieval_query")

    async def generate_document_embedding(self, text: str) -> list[float]:
        """Generate embedding for a document to be indexed."""
        return await self._embed_with_retry(text, task_type="retrieval_document")

    async def _embed_with_retry(self, text: str, task_type: str) -> list[float]:
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = await self._provider.embed(text, task_type=task_type)
                return self._l2_normalize(response.embedding)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _l2_normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]
