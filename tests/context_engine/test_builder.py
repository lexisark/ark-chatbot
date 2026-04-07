"""Tests for context builder."""

import uuid

import pytest

from db.models import MessageRole
from db.queries import create_chat, create_message
from context_engine.builder import ContextBuilder
from context_engine.models import ContextAssemblyResult
from providers.token_counter import TiktokenCounter


@pytest.fixture
def builder():
    return ContextBuilder(token_counter=TiktokenCounter())


class TestBudgetAllocation:
    async def test_system_prompt_gets_10_percent(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="You are helpful. " * 50)
        await create_message(db_session, chat.id, MessageRole.USER, "hi")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi")

        assert result.system_prompt_tokens <= 100  # 10% of 1000
        assert result.system_prompt_tokens > 0

    async def test_total_tokens_within_budget(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="Be helpful.")
        for i in range(20):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"Message number {i} with some content here")

        result = await builder.build_context(db_session, chat.id, budget_tokens=500, current_message="test")

        assert result.total_tokens <= 500

    async def test_budget_split_ratios(self, builder, db_session):
        """System prompt ~10%, messages ~50%, RAG ~40% (RAG placeholder = 0 in Phase 2)."""
        chat = await create_chat(db_session, system_prompt="You are a helpful assistant.")
        for i in range(10):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"This is message {i}")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="test")

        # System prompt should be within persona budget
        assert result.system_prompt_tokens <= 100  # 10% of 1000
        # Recent messages should use their allocation
        assert result.recent_tokens > 0
        # Memories placeholder is 0 in Phase 2
        assert result.memories_tokens == 0


class TestSystemPromptTruncation:
    async def test_long_system_prompt_truncated(self, builder, db_session):
        # Create a very long system prompt
        long_prompt = "You must follow these rules. " * 100  # ~700 tokens
        chat = await create_chat(db_session, system_prompt=long_prompt)
        await create_message(db_session, chat.id, MessageRole.USER, "hi")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi")

        # Should be truncated to ~10% of 1000 = 100 tokens
        assert result.system_prompt_tokens <= 100

    async def test_short_system_prompt_unchanged(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="Be helpful.")
        await create_message(db_session, chat.id, MessageRole.USER, "hi")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi")

        assert "Be helpful." in result.system_prompt

    async def test_no_system_prompt(self, builder, db_session):
        chat = await create_chat(db_session)
        await create_message(db_session, chat.id, MessageRole.USER, "hi")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi")

        assert result.system_prompt == ""
        assert result.system_prompt_tokens == 0


class TestMessageWindowing:
    async def test_recent_messages_newest_kept(self, builder, db_session):
        chat = await create_chat(db_session)
        for i in range(20):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            await create_message(db_session, chat.id, role, f"msg-{i}")

        result = await builder.build_context(db_session, chat.id, budget_tokens=200, current_message="test")

        # The most recent message should be included
        assert any("msg-19" in m["content"] for m in result.recent_messages)

    async def test_empty_chat(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="Hello")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi")

        assert result.recent_messages == []
        assert result.recent_tokens == 0

    async def test_messages_in_chronological_order(self, builder, db_session):
        chat = await create_chat(db_session)
        await create_message(db_session, chat.id, MessageRole.USER, "first")
        await create_message(db_session, chat.id, MessageRole.ASSISTANT, "second")
        await create_message(db_session, chat.id, MessageRole.USER, "third")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="test")

        contents = [m["content"] for m in result.recent_messages]
        assert contents == ["first", "second", "third"]


class TestFormatForLLM:
    async def test_format_produces_system_instruction(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="You are an expert.")
        await create_message(db_session, chat.id, MessageRole.USER, "hi")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi")
        system_instruction, messages = builder.format_for_llm(result)

        assert "You are an expert." in system_instruction
        assert isinstance(messages, list)

    async def test_format_messages_are_conversation_only(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="System stuff")
        await create_message(db_session, chat.id, MessageRole.USER, "hello")
        await create_message(db_session, chat.id, MessageRole.ASSISTANT, "hi there")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="test")
        system_instruction, messages = builder.format_for_llm(result)

        # Messages should not contain system prompt
        for msg in messages:
            assert msg["role"] in ("user", "assistant")
            assert "System stuff" not in msg["content"]

    async def test_format_includes_memory_block_placeholder(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="Be helpful.")
        await create_message(db_session, chat.id, MessageRole.USER, "hi")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi")
        # In Phase 2, no memories — system instruction should just have the prompt
        system_instruction, _ = builder.format_for_llm(result)
        assert "Be helpful." in system_instruction

    async def test_format_with_rag_disabled(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="Test")
        await create_message(db_session, chat.id, MessageRole.USER, "hi")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi", enable_rag=False)

        assert result.memories_text == ""
        assert result.memories_tokens == 0


class TestAssemblyResult:
    async def test_result_has_timing(self, builder, db_session):
        chat = await create_chat(db_session)
        await create_message(db_session, chat.id, MessageRole.USER, "hi")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="hi")

        assert isinstance(result, ContextAssemblyResult)
        assert result.assembly_time_ms >= 0

    async def test_result_has_token_breakdown(self, builder, db_session):
        chat = await create_chat(db_session, system_prompt="Hello")
        await create_message(db_session, chat.id, MessageRole.USER, "test")

        result = await builder.build_context(db_session, chat.id, budget_tokens=1000, current_message="test")

        assert result.system_prompt_tokens >= 0
        assert result.recent_tokens >= 0
        assert result.memories_tokens >= 0
        assert result.total_tokens == result.system_prompt_tokens + result.recent_tokens + result.memories_tokens
