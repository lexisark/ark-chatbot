"""Tests for STM (Short-Term Memory) manager."""

import uuid

import pytest

from db.models import STMEntity, STMRecap, STMRelationship
from db.queries import create_chat
from context_engine.stm_manager import STMManager


@pytest.fixture
def stm():
    return STMManager()


class TestUpsertEntity:
    async def test_insert_new_entity(self, stm, db_session):
        chat = await create_chat(db_session)

        entity = await stm.upsert_entity(
            db_session, chat.id,
            entity_type="person",
            canonical_name="Alice",
            attributes={"role": "sister"},
            confidence=0.7,
        )

        assert entity.id is not None
        assert entity.canonical_name == "Alice"
        assert entity.entity_type == "person"
        assert entity.attributes["role"] == "sister"
        assert entity.mention_count == 1

    async def test_insert_new_entity_confidence_floor(self, stm, db_session):
        chat = await create_chat(db_session)

        entity = await stm.upsert_entity(
            db_session, chat.id,
            entity_type="person",
            canonical_name="Bob",
            confidence=0.1,  # Below floor
        )

        assert entity.overall_confidence == 0.30  # Floor

    async def test_upsert_existing_reinforces_confidence(self, stm, db_session):
        chat = await create_chat(db_session)

        e1 = await stm.upsert_entity(
            db_session, chat.id,
            entity_type="pet", canonical_name="Max", confidence=0.5,
        )
        original_confidence = e1.overall_confidence

        e2 = await stm.upsert_entity(
            db_session, chat.id,
            entity_type="pet", canonical_name="Max", confidence=0.6,
        )

        assert e2.id == e1.id
        assert e2.overall_confidence == original_confidence + 0.20  # boost
        assert e2.mention_count == 2

    async def test_upsert_existing_merges_attributes(self, stm, db_session):
        chat = await create_chat(db_session)

        await stm.upsert_entity(
            db_session, chat.id,
            entity_type="pet", canonical_name="Max",
            attributes={"breed": "Golden Retriever"},
        )
        e2 = await stm.upsert_entity(
            db_session, chat.id,
            entity_type="pet", canonical_name="Max",
            attributes={"age": "3 years"},
        )

        assert e2.attributes["breed"] == "Golden Retriever"
        assert e2.attributes["age"] == "3 years"

    async def test_upsert_confidence_capped_at_095(self, stm, db_session):
        chat = await create_chat(db_session)

        await stm.upsert_entity(
            db_session, chat.id,
            entity_type="person", canonical_name="Alice", confidence=0.9,
        )
        # Boost again — should cap at 0.95
        e = await stm.upsert_entity(
            db_session, chat.id,
            entity_type="person", canonical_name="Alice", confidence=0.9,
        )

        assert e.overall_confidence <= 0.95

    async def test_upsert_case_insensitive_dedup(self, stm, db_session):
        chat = await create_chat(db_session)

        e1 = await stm.upsert_entity(
            db_session, chat.id,
            entity_type="person", canonical_name="alice",
        )
        e2 = await stm.upsert_entity(
            db_session, chat.id,
            entity_type="person", canonical_name="Alice",
        )

        assert e1.id == e2.id
        assert e2.mention_count == 2


class TestUpsertRelationship:
    async def test_insert_new_relationship(self, stm, db_session):
        chat = await create_chat(db_session)

        subj = await stm.upsert_entity(db_session, chat.id, "person", "User")
        obj = await stm.upsert_entity(db_session, chat.id, "pet", "Max")

        rel = await stm.upsert_relationship(
            db_session, chat.id,
            subject_entity_id=subj.id,
            predicate="owns",
            object_entity_id=obj.id,
            confidence=0.9,
        )

        assert rel.id is not None
        assert rel.predicate == "owns"
        assert rel.confidence == 0.9

    async def test_upsert_existing_relationship(self, stm, db_session):
        chat = await create_chat(db_session)

        subj = await stm.upsert_entity(db_session, chat.id, "person", "User")
        obj = await stm.upsert_entity(db_session, chat.id, "pet", "Max")

        r1 = await stm.upsert_relationship(
            db_session, chat.id, subj.id, "owns", obj.id, confidence=0.5,
        )
        r2 = await stm.upsert_relationship(
            db_session, chat.id, subj.id, "owns", obj.id, confidence=0.8,
        )

        assert r1.id == r2.id
        assert r2.confidence == 0.8


class TestInsertRecap:
    async def test_insert_recap(self, stm, db_session):
        chat = await create_chat(db_session)

        recap = await stm.insert_recap(
            db_session, chat.id,
            recap_text="User talked about their dog Max.",
            keywords=["dog", "Max"],
            confidence=0.8,
        )

        assert recap.id is not None
        assert recap.recap_text == "User talked about their dog Max."
        assert "Max" in recap.keywords


class TestGetEntities:
    async def test_get_entities_above_threshold(self, stm, db_session):
        chat = await create_chat(db_session)

        await stm.upsert_entity(db_session, chat.id, "person", "Alice", confidence=0.8)
        await stm.upsert_entity(db_session, chat.id, "person", "Bob", confidence=0.2)

        entities = await stm.get_entities(db_session, chat.id, min_confidence=0.5)

        names = [e.canonical_name for e in entities]
        assert "Alice" in names
        assert "Bob" not in names

    async def test_get_entities_default_threshold(self, stm, db_session):
        chat = await create_chat(db_session)

        await stm.upsert_entity(db_session, chat.id, "person", "Alice", confidence=0.5)
        await stm.upsert_entity(db_session, chat.id, "person", "Bob", confidence=0.1)

        entities = await stm.get_entities(db_session, chat.id)
        names = [e.canonical_name for e in entities]
        assert "Alice" in names


class TestGetRelationships:
    async def test_get_relationships(self, stm, db_session):
        chat = await create_chat(db_session)

        subj = await stm.upsert_entity(db_session, chat.id, "person", "User")
        obj = await stm.upsert_entity(db_session, chat.id, "pet", "Max")
        await stm.upsert_relationship(db_session, chat.id, subj.id, "owns", obj.id, confidence=0.9)

        rels = await stm.get_relationships(db_session, chat.id)
        assert len(rels) == 1
        assert rels[0].predicate == "owns"


class TestGetRecaps:
    async def test_get_recaps_limited(self, stm, db_session):
        chat = await create_chat(db_session)

        for i in range(5):
            await stm.insert_recap(db_session, chat.id, f"Recap {i}", [f"kw{i}"], confidence=0.8)

        recaps = await stm.get_recaps(db_session, chat.id, limit=3)
        assert len(recaps) == 3


class TestEntityCap:
    async def test_entity_count_capped(self, stm, db_session):
        chat = await create_chat(db_session)

        # Insert more than default cap (we'll use a small cap for testing)
        stm._max_entities = 5
        for i in range(10):
            await stm.upsert_entity(db_session, chat.id, "thing", f"Entity{i}", confidence=0.5)

        entities = await stm.get_entities(db_session, chat.id, min_confidence=0.0)
        assert len(entities) <= 5
