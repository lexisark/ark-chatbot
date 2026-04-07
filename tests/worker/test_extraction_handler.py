"""Tests for extraction handler."""

import json

import pytest

from db.models import MessageRole
from db.queries import create_chat, create_message
from context_engine.stm_manager import STMManager
from worker.extraction_handler import run_batch_extraction
from providers.base import ChatResponse


class MockExtractionProvider:
    """Returns a canned extraction response."""

    def __init__(self):
        self.calls = []

    async def chat(self, messages, system_prompt="", **kwargs):
        self.calls.append({"messages": messages, "system_prompt": system_prompt})
        return ChatResponse(
            content=json.dumps({
                "entities": [
                    {"type": "pet", "subtype": "dog", "canonical_name": "Max",
                     "attributes": {"breed": "Golden Retriever"}, "overall_confidence": 0.92},
                    {"type": "person", "canonical_name": "Alice",
                     "attributes": {"relation": "sister"}, "overall_confidence": 0.85},
                ],
                "relationships": [
                    {"subject": "user", "predicate": "owns", "object_name": "Max", "confidence": 0.90},
                    {"subject": "user", "predicate": "has_sister", "object_name": "Alice", "confidence": 0.85},
                ],
                "recap_text": "User talked about their dog Max and sister Alice.",
                "keywords": ["Max", "dog", "Alice", "sister"],
            }),
            model="test",
            tokens_in=100,
            tokens_out=200,
            latency_ms=50,
        )

    async def chat_stream(self, *args, **kwargs):
        yield  # not used


class TestRunBatchExtraction:
    async def test_extraction_creates_entities(self, db_session):
        chat = await create_chat(db_session)
        for i in range(5):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"Message {i}")

        provider = MockExtractionProvider()
        stm = STMManager()

        await run_batch_extraction(db_session, provider, stm, chat.id)

        entities = await stm.get_entities(db_session, chat.id, min_confidence=0.0)
        assert len(entities) >= 2

        names = [e.canonical_name for e in entities]
        assert "Max" in names
        assert "Alice" in names

    async def test_extraction_creates_relationships(self, db_session):
        chat = await create_chat(db_session)
        for i in range(5):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"Message {i}")

        provider = MockExtractionProvider()
        stm = STMManager()

        await run_batch_extraction(db_session, provider, stm, chat.id)

        rels = await stm.get_relationships(db_session, chat.id, min_confidence=0.0)
        assert len(rels) >= 1

    async def test_extraction_creates_recap(self, db_session):
        chat = await create_chat(db_session)
        for i in range(5):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"Message {i}")

        provider = MockExtractionProvider()
        stm = STMManager()

        await run_batch_extraction(db_session, provider, stm, chat.id)

        recaps = await stm.get_recaps(db_session, chat.id)
        assert len(recaps) >= 1
        assert "Max" in recaps[0].recap_text

    async def test_extraction_calls_llm(self, db_session):
        chat = await create_chat(db_session)
        await create_message(db_session, chat.id, MessageRole.USER, "Hi I have a dog")
        await create_message(db_session, chat.id, MessageRole.ASSISTANT, "Cool!")

        provider = MockExtractionProvider()
        stm = STMManager()

        await run_batch_extraction(db_session, provider, stm, chat.id)

        assert len(provider.calls) == 1

    async def test_extraction_skips_empty_chat(self, db_session):
        chat = await create_chat(db_session)

        provider = MockExtractionProvider()
        stm = STMManager()

        await run_batch_extraction(db_session, provider, stm, chat.id)

        assert len(provider.calls) == 0
