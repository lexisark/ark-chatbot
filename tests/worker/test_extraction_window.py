"""Tests for extraction message windowing — only extract since last recap."""

import json

import pytest

from db.models import MessageRole
from db.queries import create_chat, create_message
from context_engine.stm_manager import STMManager
from worker.extraction_handler import run_batch_extraction
from providers.base import ChatResponse


class TrackingProvider:
    """Tracks what messages were passed to extraction."""

    def __init__(self):
        self.calls = []

    async def chat(self, messages, **kw):
        self.calls.append(messages[0]["content"] if messages else "")
        return ChatResponse(
            content=json.dumps({
                "entities": [{"type": "pet", "canonical_name": "Max", "overall_confidence": 0.9}],
                "relationships": [],
                "recap_text": "User mentioned Max.",
                "keywords": ["Max"],
            }),
            model="test", tokens_in=100, tokens_out=50, latency_ms=10,
        )

    async def chat_stream(self, *a, **kw):
        yield


class TestExtractionWindow:
    async def test_first_extraction_gets_all_messages(self, db_session):
        """First extraction with no existing recaps should get all messages."""
        chat = await create_chat(db_session)
        for i in range(6):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"Message {i}")

        provider = TrackingProvider()
        stm = STMManager()
        await run_batch_extraction(db_session, provider, stm, chat.id)

        assert len(provider.calls) == 1
        prompt = provider.calls[0]
        # Should contain all messages
        assert "Message 0" in prompt
        assert "Message 5" in prompt

    async def test_second_extraction_gets_only_new_messages(self, db_session):
        """After first extraction creates a recap, second should only get new messages."""
        chat = await create_chat(db_session)

        # First batch of messages
        for i in range(6):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            msg = await create_message(db_session, chat.id, role, f"Old-{i}")

        provider = TrackingProvider()
        stm = STMManager()

        # First extraction — creates a recap with end_msg_id
        await run_batch_extraction(db_session, provider, stm, chat.id)

        # Add new messages
        for i in range(4):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"New-{i}")

        # Second extraction — should only get messages after last recap
        await run_batch_extraction(db_session, provider, stm, chat.id)

        assert len(provider.calls) == 2
        second_prompt = provider.calls[1]
        # Should have new messages
        assert "New-0" in second_prompt
        # Should NOT have old messages (already extracted)
        assert "Old-0" not in second_prompt

    async def test_no_new_messages_skips_extraction(self, db_session):
        """If no new messages since last recap, skip extraction."""
        chat = await create_chat(db_session)
        for i in range(4):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"Msg {i}")

        provider = TrackingProvider()
        stm = STMManager()

        # First extraction
        await run_batch_extraction(db_session, provider, stm, chat.id)

        # No new messages — second extraction should be skipped
        await run_batch_extraction(db_session, provider, stm, chat.id)

        assert len(provider.calls) == 1  # Only the first call
