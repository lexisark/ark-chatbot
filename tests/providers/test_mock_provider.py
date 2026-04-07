"""Tests for mock providers — validates protocol conformance."""

import pytest

from providers.base import (
    ChatProvider,
    ChatResponse,
    EmbeddingProvider,
    EmbeddingResponse,
    StreamChunk,
    TokenCounter,
)


class MockChatProvider:
    """Mock chat provider for testing without API calls."""

    def __init__(self, response_text: str = "Mock response", default_model: str = "mock-model"):
        self.response_text = response_text
        self.default_model = default_model
        self.call_history: list[dict] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        self.call_history.append({
            "method": "chat",
            "messages": messages,
            "system_prompt": system_prompt,
            "model": model or self.default_model,
        })
        return ChatResponse(
            content=self.response_text,
            model=model or self.default_model,
            tokens_in=sum(len(m.get("content", "")) // 4 for m in messages),
            tokens_out=len(self.response_text) // 4,
            latency_ms=1,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.call_history.append({
            "method": "chat_stream",
            "messages": messages,
            "system_prompt": system_prompt,
        })
        words = self.response_text.split()
        for word in words:
            yield StreamChunk(delta=word + " ")
        yield StreamChunk(
            delta="",
            done=True,
            response=ChatResponse(
                content=self.response_text,
                model=model or self.default_model,
                tokens_in=10,
                tokens_out=5,
                latency_ms=1,
            ),
        )


class MockEmbeddingProvider:
    """Mock embedding provider for testing without API calls."""

    def __init__(self, dims: int = 768, default_model: str = "mock-embedding"):
        self._dims = dims
        self.default_model = default_model

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        task_type: str = "retrieval_document",
    ) -> EmbeddingResponse:
        # Deterministic fake embedding based on text hash
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        embedding = [(h >> i & 0xFF) / 255.0 for i in range(self._dims)]
        return EmbeddingResponse(
            embedding=embedding,
            model=model or self.default_model,
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


class TestMockChatProvider:
    @pytest.fixture
    def provider(self):
        return MockChatProvider(response_text="Hello from mock!")

    def test_is_chat_provider(self, provider):
        assert isinstance(provider, ChatProvider)

    async def test_chat_returns_chat_response(self, provider):
        messages = [{"role": "user", "content": "hi"}]
        result = await provider.chat(messages)
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello from mock!"
        assert result.model == "mock-model"
        assert result.tokens_in > 0 or result.tokens_in == 0
        assert result.tokens_out >= 0

    async def test_chat_records_history(self, provider):
        await provider.chat([{"role": "user", "content": "test"}], system_prompt="be helpful")
        assert len(provider.call_history) == 1
        assert provider.call_history[0]["system_prompt"] == "be helpful"

    async def test_chat_stream_yields_chunks(self, provider):
        messages = [{"role": "user", "content": "hi"}]
        chunks = []
        async for chunk in provider.chat_stream(messages):
            chunks.append(chunk)
            assert isinstance(chunk, StreamChunk)

        assert len(chunks) >= 2  # at least one word + final
        assert chunks[-1].done is True
        assert chunks[-1].response is not None

    async def test_chat_stream_reconstructs_content(self, provider):
        messages = [{"role": "user", "content": "hi"}]
        text = ""
        async for chunk in provider.chat_stream(messages):
            text += chunk.delta
        assert text.strip() == "Hello from mock!"


class TestMockEmbeddingProvider:
    @pytest.fixture
    def provider(self):
        return MockEmbeddingProvider(dims=768)

    def test_is_embedding_provider(self, provider):
        assert isinstance(provider, EmbeddingProvider)

    def test_dimensions(self, provider):
        assert provider.dimensions == 768

    async def test_embed_returns_response(self, provider):
        result = await provider.embed("hello world")
        assert isinstance(result, EmbeddingResponse)
        assert len(result.embedding) == 768
        assert result.model == "mock-embedding"

    async def test_embed_deterministic(self, provider):
        r1 = await provider.embed("same text")
        r2 = await provider.embed("same text")
        assert r1.embedding == r2.embedding

    async def test_embed_different_texts_differ(self, provider):
        r1 = await provider.embed("text one")
        r2 = await provider.embed("text two")
        assert r1.embedding != r2.embedding

    async def test_embed_batch(self, provider):
        results = await provider.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResponse) for r in results)
        assert all(len(r.embedding) == 768 for r in results)
