"""Tests for database query functions."""

import uuid

import pytest

from db.models import MessageRole
from db.queries import (
    count_user_messages,
    create_chat,
    create_message,
    delete_chat,
    get_chat,
    get_chat_messages,
    list_chats,
    update_chat,
)


class TestChatQueries:
    async def test_create_chat(self, db_session):
        chat = await create_chat(
            db_session,
            title="Test Chat",
            system_prompt="Be helpful.",
            scope_id="user-1",
        )
        assert chat.id is not None
        assert chat.title == "Test Chat"
        assert chat.system_prompt == "Be helpful."
        assert chat.scope_id == "user-1"

    async def test_create_chat_minimal(self, db_session):
        chat = await create_chat(db_session)
        assert chat.id is not None
        assert chat.title is None

    async def test_get_chat(self, db_session):
        chat = await create_chat(db_session, title="Find me")
        result = await get_chat(db_session, chat.id)
        assert result is not None
        assert result.title == "Find me"

    async def test_get_chat_not_found(self, db_session):
        result = await get_chat(db_session, uuid.uuid4())
        assert result is None

    async def test_list_chats(self, db_session):
        await create_chat(db_session, title="Chat 1")
        await create_chat(db_session, title="Chat 2")
        await create_chat(db_session, title="Chat 3")

        chats = await list_chats(db_session, limit=10, offset=0)
        assert len(chats) >= 3

    async def test_list_chats_pagination(self, db_session):
        for i in range(5):
            await create_chat(db_session, title=f"Chat {i}")

        page1 = await list_chats(db_session, limit=2, offset=0)
        page2 = await list_chats(db_session, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    async def test_list_chats_filter_by_scope(self, db_session):
        await create_chat(db_session, title="A", scope_id="scope-1")
        await create_chat(db_session, title="B", scope_id="scope-2")
        await create_chat(db_session, title="C", scope_id="scope-1")

        results = await list_chats(db_session, scope_id="scope-1")
        assert len(results) == 2
        assert all(c.scope_id == "scope-1" for c in results)

    async def test_update_chat(self, db_session):
        chat = await create_chat(db_session, title="Old Title")
        updated = await update_chat(db_session, chat.id, title="New Title", system_prompt="New prompt")
        assert updated is not None
        assert updated.title == "New Title"
        assert updated.system_prompt == "New prompt"

    async def test_update_chat_not_found(self, db_session):
        result = await update_chat(db_session, uuid.uuid4(), title="Nope")
        assert result is None

    async def test_delete_chat(self, db_session):
        chat = await create_chat(db_session, title="Delete me")
        deleted = await delete_chat(db_session, chat.id)
        assert deleted is True

        result = await get_chat(db_session, chat.id)
        assert result is None

    async def test_delete_chat_not_found(self, db_session):
        deleted = await delete_chat(db_session, uuid.uuid4())
        assert deleted is False


class TestMessageQueries:
    async def test_create_message(self, db_session):
        chat = await create_chat(db_session)
        msg = await create_message(
            db_session, chat_id=chat.id, role=MessageRole.USER, content="Hello!"
        )
        assert msg.id is not None
        assert msg.chat_id == chat.id
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"

    async def test_get_chat_messages_chronological(self, db_session):
        chat = await create_chat(db_session)
        await create_message(db_session, chat.id, MessageRole.USER, "first")
        await create_message(db_session, chat.id, MessageRole.ASSISTANT, "second")
        await create_message(db_session, chat.id, MessageRole.USER, "third")

        messages = await get_chat_messages(db_session, chat.id)
        assert len(messages) == 3
        assert messages[0].content == "first"
        assert messages[1].content == "second"
        assert messages[2].content == "third"

    async def test_get_chat_messages_with_limit(self, db_session):
        chat = await create_chat(db_session)
        for i in range(10):
            await create_message(db_session, chat.id, MessageRole.USER, f"msg {i}")

        messages = await get_chat_messages(db_session, chat.id, limit=5)
        assert len(messages) == 5

    async def test_count_user_messages(self, db_session):
        chat = await create_chat(db_session)
        await create_message(db_session, chat.id, MessageRole.USER, "u1")
        await create_message(db_session, chat.id, MessageRole.ASSISTANT, "a1")
        await create_message(db_session, chat.id, MessageRole.USER, "u2")
        await create_message(db_session, chat.id, MessageRole.ASSISTANT, "a2")
        await create_message(db_session, chat.id, MessageRole.USER, "u3")

        count = await count_user_messages(db_session, chat.id)
        assert count == 3
