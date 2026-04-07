"""Tests for provider base protocols and dataclasses."""

from providers.base import (
    ChatProvider,
    ChatResponse,
    EmbeddingProvider,
    EmbeddingResponse,
    StreamChunk,
    TokenCounter,
)


class TestDataclasses:
    def test_chat_response_construction(self):
        resp = ChatResponse(
            content="Hello!",
            model="gemini-2.5-flash",
            tokens_in=10,
            tokens_out=5,
            latency_ms=120,
        )
        assert resp.content == "Hello!"
        assert resp.model == "gemini-2.5-flash"
        assert resp.tokens_in == 10
        assert resp.tokens_out == 5
        assert resp.latency_ms == 120
        assert resp.cost_usd == 0.0
        assert resp.metadata == {}

    def test_chat_response_with_optional_fields(self):
        resp = ChatResponse(
            content="Hi",
            model="gpt-4o",
            tokens_in=5,
            tokens_out=2,
            latency_ms=50,
            cost_usd=0.001,
            metadata={"provider": "openai"},
        )
        assert resp.cost_usd == 0.001
        assert resp.metadata["provider"] == "openai"

    def test_stream_chunk_defaults(self):
        chunk = StreamChunk(delta="hello")
        assert chunk.delta == "hello"
        assert chunk.done is False
        assert chunk.response is None

    def test_stream_chunk_final(self):
        resp = ChatResponse(content="full", model="m", tokens_in=1, tokens_out=1, latency_ms=1)
        chunk = StreamChunk(delta="", done=True, response=resp)
        assert chunk.done is True
        assert chunk.response.content == "full"

    def test_embedding_response_construction(self):
        resp = EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3],
            model="gemini-embedding-001",
            tokens=5,
        )
        assert len(resp.embedding) == 3
        assert resp.model == "gemini-embedding-001"
        assert resp.tokens == 5


class TestProtocolsAreRuntimeCheckable:
    def test_chat_provider_is_checkable(self):
        assert hasattr(ChatProvider, "__protocol_attrs__") or hasattr(
            ChatProvider, "__abstractmethods__"
        )

    def test_embedding_provider_is_checkable(self):
        assert hasattr(EmbeddingProvider, "__protocol_attrs__") or hasattr(
            EmbeddingProvider, "__abstractmethods__"
        )

    def test_token_counter_is_checkable(self):
        assert hasattr(TokenCounter, "__protocol_attrs__") or hasattr(
            TokenCounter, "__abstractmethods__"
        )

    def test_conforming_class_is_instance(self):
        """A class implementing all ChatProvider methods should pass isinstance check."""

        class FakeChat:
            async def chat(self, messages, system_prompt="", **kwargs):
                return ChatResponse(content="", model="", tokens_in=0, tokens_out=0, latency_ms=0)

            async def chat_stream(self, messages, system_prompt="", **kwargs):
                yield StreamChunk(delta="", done=True)

        assert isinstance(FakeChat(), ChatProvider)

    def test_non_conforming_class_fails(self):
        """A class missing methods should not pass isinstance check."""

        class NotAChat:
            pass

        assert not isinstance(NotAChat(), ChatProvider)
