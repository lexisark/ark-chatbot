"""Tests for database models."""

import uuid
from datetime import datetime, timezone

import pytest

from db.models import (
    Base,
    Chat,
    LTMEntity,
    LTMEpisode,
    LTMRelationship,
    Message,
    MessageRole,
    STMEntity,
    STMRecap,
    STMRelationship,
)


class TestChatModel:
    async def test_create_chat(self, db_session):
        chat = Chat(
            title="Test Chat",
            system_prompt="You are a helpful assistant.",
            scope_id="user-123",
        )
        db_session.add(chat)
        await db_session.flush()

        assert chat.id is not None
        assert isinstance(chat.id, uuid.UUID)
        assert chat.title == "Test Chat"
        assert chat.system_prompt == "You are a helpful assistant."
        assert chat.scope_id == "user-123"
        assert chat.created_at is not None

    async def test_chat_defaults(self, db_session):
        chat = Chat()
        db_session.add(chat)
        await db_session.flush()

        assert chat.id is not None
        assert chat.title is None
        assert chat.system_prompt is None
        assert chat.scope_id is None
        assert chat.metadata_ == {}

    async def test_chat_with_metadata(self, db_session):
        chat = Chat(metadata_={"key": "value", "nested": {"a": 1}})
        db_session.add(chat)
        await db_session.flush()

        assert chat.metadata_["key"] == "value"
        assert chat.metadata_["nested"]["a"] == 1


class TestMessageModel:
    async def test_create_message(self, db_session):
        chat = Chat(title="Test")
        db_session.add(chat)
        await db_session.flush()

        msg = Message(
            chat_id=chat.id,
            role=MessageRole.USER,
            content="Hello!",
            token_count=5,
        )
        db_session.add(msg)
        await db_session.flush()

        assert msg.id is not None
        assert msg.chat_id == chat.id
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello!"
        assert msg.token_count == 5

    async def test_message_roles(self, db_session):
        chat = Chat()
        db_session.add(chat)
        await db_session.flush()

        for role in MessageRole:
            msg = Message(chat_id=chat.id, role=role, content=f"{role.value} msg")
            db_session.add(msg)

        await db_session.flush()


class TestSTMEntityModel:
    async def test_create_entity(self, db_session):
        chat = Chat()
        db_session.add(chat)
        await db_session.flush()

        entity = STMEntity(
            chat_id=chat.id,
            entity_type="person",
            entity_subtype="family",
            canonical_name="Alice",
            attributes={"role": "sister", "age": "25"},
            overall_confidence=0.85,
            mention_count=3,
        )
        db_session.add(entity)
        await db_session.flush()

        assert entity.id is not None
        assert entity.entity_type == "person"
        assert entity.canonical_name == "Alice"
        assert entity.attributes["role"] == "sister"
        assert entity.overall_confidence == 0.85
        assert entity.mention_count == 3


class TestSTMRelationshipModel:
    async def test_create_relationship(self, db_session):
        chat = Chat()
        db_session.add(chat)
        await db_session.flush()

        subj = STMEntity(chat_id=chat.id, entity_type="person", canonical_name="User")
        obj = STMEntity(chat_id=chat.id, entity_type="pet", canonical_name="Max")
        db_session.add_all([subj, obj])
        await db_session.flush()

        rel = STMRelationship(
            chat_id=chat.id,
            subject_entity_id=subj.id,
            predicate="owns",
            object_entity_id=obj.id,
            confidence=0.9,
            source_msg_ids=[],
        )
        db_session.add(rel)
        await db_session.flush()

        assert rel.id is not None
        assert rel.predicate == "owns"
        assert rel.confidence == 0.9


class TestSTMRecapModel:
    async def test_create_recap(self, db_session):
        chat = Chat()
        db_session.add(chat)
        await db_session.flush()

        recap = STMRecap(
            chat_id=chat.id,
            recap_text="User talked about their dog Max.",
            keywords=["dog", "Max", "pet"],
            confidence=0.8,
        )
        db_session.add(recap)
        await db_session.flush()

        assert recap.id is not None
        assert recap.recap_text == "User talked about their dog Max."
        assert "Max" in recap.keywords


class TestLTMEpisodeModel:
    async def test_create_episode(self, db_session):
        episode = LTMEpisode(
            scope_id="user-123",
            episode_summary="User discussed their golden retriever Max.",
            keywords=["Max", "golden retriever", "dog"],
            importance_score=0.8,
            is_final=True,
            emotional_tone="positive",
            source_chat_id=uuid.uuid4(),
        )
        db_session.add(episode)
        await db_session.flush()

        assert episode.id is not None
        assert episode.scope_id == "user-123"
        assert episode.is_final is True
        assert episode.importance_score == 0.8


class TestLTMEntityModel:
    async def test_create_ltm_entity(self, db_session):
        entity = LTMEntity(
            scope_id="user-123",
            entity_type="pet",
            entity_subtype="dog",
            canonical_name="Max",
            attributes={"breed": "Golden Retriever"},
            overall_confidence=0.92,
            mention_count=5,
        )
        db_session.add(entity)
        await db_session.flush()

        assert entity.id is not None
        assert entity.scope_id == "user-123"


class TestLTMRelationshipModel:
    async def test_create_ltm_relationship(self, db_session):
        subj = LTMEntity(scope_id="user-123", entity_type="person", canonical_name="User")
        obj = LTMEntity(scope_id="user-123", entity_type="pet", canonical_name="Max")
        db_session.add_all([subj, obj])
        await db_session.flush()

        rel = LTMRelationship(
            scope_id="user-123",
            subject_entity_id=subj.id,
            predicate="owns",
            object_entity_id=obj.id,
            confidence=0.95,
        )
        db_session.add(rel)
        await db_session.flush()

        assert rel.id is not None


class TestCascadeDelete:
    async def test_delete_chat_cascades_messages(self, db_session):
        chat = Chat(title="To delete")
        db_session.add(chat)
        await db_session.flush()

        msg = Message(chat_id=chat.id, role=MessageRole.USER, content="hi")
        db_session.add(msg)
        await db_session.flush()
        msg_id = msg.id

        await db_session.delete(chat)
        await db_session.flush()
        db_session.expire_all()

        result = await db_session.get(Message, msg_id)
        assert result is None

    async def test_delete_chat_cascades_stm(self, db_session):
        chat = Chat()
        db_session.add(chat)
        await db_session.flush()

        entity = STMEntity(chat_id=chat.id, entity_type="person", canonical_name="Test")
        recap = STMRecap(chat_id=chat.id, recap_text="test", keywords=[], confidence=0.5)
        db_session.add_all([entity, recap])
        await db_session.flush()
        entity_id = entity.id
        recap_id = recap.id

        await db_session.delete(chat)
        await db_session.flush()
        db_session.expire_all()

        assert await db_session.get(STMEntity, entity_id) is None
        assert await db_session.get(STMRecap, recap_id) is None
