"""Tests for hybrid search schema — embedding columns on STM + LTM."""

import uuid

from db.models import LTMEntity, LTMEpisode, STMEntity, STMRecap
from db.queries import create_chat


class TestSTMEntityEmbedding:
    async def test_entity_stores_embedding(self, db_session):
        chat = await create_chat(db_session)
        embedding = [0.1] * 768

        entity = STMEntity(
            chat_id=chat.id,
            entity_type="pet",
            canonical_name="Max",
            embedding=embedding,
            overall_confidence=0.9,
        )
        db_session.add(entity)
        await db_session.flush()

        assert entity.embedding is not None
        assert len(entity.embedding) == 768

    async def test_entity_embedding_nullable(self, db_session):
        chat = await create_chat(db_session)
        entity = STMEntity(
            chat_id=chat.id,
            entity_type="person",
            canonical_name="Alice",
        )
        db_session.add(entity)
        await db_session.flush()

        assert entity.embedding is None


class TestSTMRecapEmbedding:
    async def test_recap_stores_embedding(self, db_session):
        chat = await create_chat(db_session)
        embedding = [0.2] * 768

        recap = STMRecap(
            chat_id=chat.id,
            recap_text="User talked about their dog.",
            keywords=["dog"],
            embedding=embedding,
            confidence=0.8,
        )
        db_session.add(recap)
        await db_session.flush()

        assert recap.embedding is not None
        assert len(recap.embedding) == 768

    async def test_recap_embedding_nullable(self, db_session):
        chat = await create_chat(db_session)
        recap = STMRecap(
            chat_id=chat.id,
            recap_text="Some recap.",
            keywords=[],
            confidence=0.5,
        )
        db_session.add(recap)
        await db_session.flush()

        assert recap.embedding is None


class TestLTMEntityEmbedding:
    async def test_ltm_entity_stores_embedding(self, db_session):
        embedding = [0.3] * 768

        entity = LTMEntity(
            scope_id="scope-1",
            entity_type="pet",
            canonical_name="Max",
            embedding=embedding,
            overall_confidence=0.9,
        )
        db_session.add(entity)
        await db_session.flush()

        assert entity.embedding is not None
        assert len(entity.embedding) == 768


class TestLTMEpisodeEmbeddingStillWorks:
    async def test_episode_embedding_unchanged(self, db_session):
        embedding = [0.4] * 768

        episode = LTMEpisode(
            scope_id="scope-1",
            episode_summary="User discussed their dog Max.",
            keywords=["Max", "dog"],
            embedding=embedding,
            importance_score=0.8,
            is_final=True,
            source_chat_id=uuid.uuid4(),
        )
        db_session.add(episode)
        await db_session.flush()

        assert episode.embedding is not None
        assert len(episode.embedding) == 768
