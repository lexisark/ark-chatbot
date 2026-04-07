"""Tests for embedding service."""

import math

import pytest

from context_engine.embedding_service import EmbeddingService
from providers.base import EmbeddingResponse


class MockEmbeddingProvider:
    """Mock that returns deterministic embeddings."""

    def __init__(self, dims: int = 768):
        self._dims = dims
        self.calls: list[dict] = []

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, text, *, model=None, task_type="retrieval_document"):
        self.calls.append({"text": text, "task_type": task_type})
        # Deterministic embedding based on text hash
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        raw = [(h >> (i % 64) & 0xFF) / 255.0 for i in range(self._dims)]
        return EmbeddingResponse(embedding=raw, model="mock", tokens=len(text) // 4)

    async def embed_batch(self, texts, *, model=None, task_type="retrieval_document"):
        return [await self.embed(t, model=model, task_type=task_type) for t in texts]


class FailThenSucceedProvider:
    """Fails N times then succeeds — for retry testing."""

    def __init__(self, fail_count: int = 2, dims: int = 768):
        self._dims = dims
        self._fail_count = fail_count
        self._attempts = 0

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, text, *, model=None, task_type="retrieval_document"):
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise Exception("429 rate limit exceeded")
        return EmbeddingResponse(
            embedding=[0.1] * self._dims, model="mock", tokens=5,
        )

    async def embed_batch(self, texts, **kw):
        return [await self.embed(t, **kw) for t in texts]


@pytest.fixture
def mock_provider():
    return MockEmbeddingProvider(dims=768)


@pytest.fixture
def service(mock_provider):
    return EmbeddingService(mock_provider)


class TestGenerateQueryEmbedding:
    async def test_returns_correct_dimensions(self, service, mock_provider):
        result = await service.generate_query_embedding("hello world")
        assert len(result) == 768

    async def test_uses_retrieval_query_task_type(self, service, mock_provider):
        await service.generate_query_embedding("test query")
        assert mock_provider.calls[-1]["task_type"] == "retrieval_query"

    async def test_returns_list_of_floats(self, service):
        result = await service.generate_query_embedding("test")
        assert all(isinstance(v, float) for v in result)


class TestGenerateDocumentEmbedding:
    async def test_returns_correct_dimensions(self, service):
        result = await service.generate_document_embedding("some document text")
        assert len(result) == 768

    async def test_uses_retrieval_document_task_type(self, service, mock_provider):
        await service.generate_document_embedding("doc text")
        assert mock_provider.calls[-1]["task_type"] == "retrieval_document"


class TestL2Normalization:
    async def test_output_is_normalized(self, service):
        result = await service.generate_query_embedding("test normalization")
        norm = math.sqrt(sum(v * v for v in result))
        assert abs(norm - 1.0) < 0.01  # ~unit length

    async def test_different_inputs_same_norm(self, service):
        r1 = await service.generate_query_embedding("hello")
        r2 = await service.generate_query_embedding("completely different text")
        norm1 = math.sqrt(sum(v * v for v in r1))
        norm2 = math.sqrt(sum(v * v for v in r2))
        assert abs(norm1 - 1.0) < 0.01
        assert abs(norm2 - 1.0) < 0.01


class TestRetryOnRateLimit:
    async def test_retries_on_rate_limit(self):
        provider = FailThenSucceedProvider(fail_count=2, dims=768)
        service = EmbeddingService(provider, max_retries=3, retry_delay=0.01)

        result = await service.generate_query_embedding("test retry")
        assert len(result) == 768  # Eventually succeeds

    async def test_fails_after_max_retries(self):
        provider = FailThenSucceedProvider(fail_count=5, dims=768)
        service = EmbeddingService(provider, max_retries=3, retry_delay=0.01)

        with pytest.raises(Exception, match="rate limit"):
            await service.generate_query_embedding("test failure")


class TestDimensions:
    async def test_dimensions_property(self, service):
        assert service.dimensions == 768

    async def test_custom_dimensions(self):
        provider = MockEmbeddingProvider(dims=1536)
        svc = EmbeddingService(provider)
        assert svc.dimensions == 1536
        result = await svc.generate_query_embedding("test")
        assert len(result) == 1536
