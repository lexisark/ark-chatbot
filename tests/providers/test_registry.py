"""Tests for provider registry."""

import pytest

from providers.base import ChatResponse, StreamChunk
from providers.registry import ProviderRegistry


class FakeChatProvider:
    def __init__(self, greeting: str = "hi"):
        self.greeting = greeting

    async def chat(self, messages, system_prompt="", **kwargs):
        return ChatResponse(content=self.greeting, model="fake", tokens_in=0, tokens_out=0, latency_ms=0)

    async def chat_stream(self, messages, system_prompt="", **kwargs):
        yield StreamChunk(delta=self.greeting, done=True)


class FakeEmbeddingProvider:
    def __init__(self, dims: int = 3):
        self._dims = dims

    @property
    def dimensions(self):
        return self._dims

    async def embed(self, text, **kwargs):
        pass

    async def embed_batch(self, texts, **kwargs):
        pass


class FakeCounter:
    def __init__(self, multiplier: int = 1):
        self.multiplier = multiplier

    def count(self, text):
        return len(text) * self.multiplier

    def count_messages(self, messages):
        return sum(len(m.get("content", "")) for m in messages) * self.multiplier

    def truncate(self, text, max_tokens):
        return text[:max_tokens]

    def fits(self, text, budget):
        return self.count(text) <= budget


class TestProviderRegistry:
    def test_register_and_create_chat(self):
        reg = ProviderRegistry()
        reg.register_chat("fake", FakeChatProvider)
        provider = reg.create_chat("fake", greeting="hello")
        assert provider.greeting == "hello"

    def test_register_and_create_embedding(self):
        reg = ProviderRegistry()
        reg.register_embedding("fake", FakeEmbeddingProvider)
        provider = reg.create_embedding("fake", dims=768)
        assert provider.dimensions == 768

    def test_register_and_create_counter(self):
        reg = ProviderRegistry()
        reg.register_counter("fake", FakeCounter)
        counter = reg.create_counter("fake", multiplier=2)
        assert counter.count("ab") == 4

    def test_create_unknown_raises_key_error(self):
        reg = ProviderRegistry()
        with pytest.raises(KeyError):
            reg.create_chat("nonexistent")

    def test_create_unknown_embedding_raises_key_error(self):
        reg = ProviderRegistry()
        with pytest.raises(KeyError):
            reg.create_embedding("nonexistent")

    def test_create_unknown_counter_raises_key_error(self):
        reg = ProviderRegistry()
        with pytest.raises(KeyError):
            reg.create_counter("nonexistent")

    def test_duplicate_registration_overwrites(self):
        reg = ProviderRegistry()
        reg.register_chat("fake", FakeChatProvider)

        class AnotherFake:
            def __init__(self):
                self.name = "another"
            async def chat(self, *a, **kw): pass
            async def chat_stream(self, *a, **kw): yield

        reg.register_chat("fake", AnotherFake)
        provider = reg.create_chat("fake")
        assert provider.name == "another"

    def test_list_registered_chat_providers(self):
        reg = ProviderRegistry()
        reg.register_chat("a", FakeChatProvider)
        reg.register_chat("b", FakeChatProvider)
        assert set(reg.list_chat_providers()) == {"a", "b"}

    def test_list_registered_embedding_providers(self):
        reg = ProviderRegistry()
        reg.register_embedding("x", FakeEmbeddingProvider)
        assert reg.list_embedding_providers() == ["x"]
