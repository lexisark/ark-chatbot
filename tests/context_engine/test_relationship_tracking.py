"""Tests for relationship mention tracking."""

import pytest

from context_engine.stm_manager import STMManager
from db.queries import create_chat


@pytest.fixture
def stm():
    return STMManager()


class TestRelationshipMentionTracking:
    async def test_new_relationship_starts_at_1(self, stm, db_session):
        chat = await create_chat(db_session)
        subj = await stm.upsert_entity(db_session, chat.id, "person", "User")
        obj = await stm.upsert_entity(db_session, chat.id, "pet", "Max")

        rel = await stm.upsert_relationship(
            db_session, chat.id, subj.id, "owns", obj.id, confidence=0.9,
        )
        assert rel.mention_count == 1
        assert rel.last_mentioned is not None

    async def test_upsert_increments_mention_count(self, stm, db_session):
        chat = await create_chat(db_session)
        subj = await stm.upsert_entity(db_session, chat.id, "person", "User")
        obj = await stm.upsert_entity(db_session, chat.id, "pet", "Max")

        r1 = await stm.upsert_relationship(db_session, chat.id, subj.id, "owns", obj.id, confidence=0.7)
        r2 = await stm.upsert_relationship(db_session, chat.id, subj.id, "owns", obj.id, confidence=0.8)

        assert r1.id == r2.id
        assert r2.mention_count == 2

    async def test_upsert_updates_last_mentioned(self, stm, db_session):
        chat = await create_chat(db_session)
        subj = await stm.upsert_entity(db_session, chat.id, "person", "User")
        obj = await stm.upsert_entity(db_session, chat.id, "pet", "Max")

        r1 = await stm.upsert_relationship(db_session, chat.id, subj.id, "owns", obj.id, confidence=0.7)
        first_mentioned = r1.last_mentioned

        r2 = await stm.upsert_relationship(db_session, chat.id, subj.id, "owns", obj.id, confidence=0.8)

        assert r2.last_mentioned >= first_mentioned
