"""Tests for LTM (Long-Term Memory) manager."""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from db.models import LTMEntity, LTMEpisode, LTMRelationship
from db.queries import create_chat
from context_engine.stm_manager import STMManager
from context_engine.ltm_manager import LTMManager
from providers.base import ChatResponse, EmbeddingResponse


class MockLTMChatProvider:
    async def chat(self, messages, system_prompt="", **kwargs):
        return ChatResponse(
            content=json.dumps({
                "episode_summary": "User discussed their golden retriever Max and sister Alice visiting.",
                "keywords": ["Max", "golden retriever", "Alice", "sister", "visit"],
                "importance_score": 0.8,
                "emotional_tone": "positive",
            }),
            model="mock", tokens_in=100, tokens_out=50, latency_ms=10,
        )

    async def chat_stream(self, *a, **kw):
        yield


class MockLTMEmbeddingProvider:
    @property
    def dimensions(self):
        return 768

    async def embed(self, text, **kw):
        return EmbeddingResponse(embedding=[0.1] * 768, model="mock", tokens=10)

    async def embed_batch(self, texts, **kw):
        return [await self.embed(t, **kw) for t in texts]


@pytest.fixture
def stm():
    return STMManager()


@pytest.fixture
def ltm():
    return LTMManager()


class TestGenerateEpisode:
    async def test_creates_episode(self, ltm, stm, db_session):
        chat = await create_chat(db_session, scope_id="scope-1")
        await stm.upsert_entity(db_session, chat.id, "pet", "Max", confidence=0.9)
        await stm.insert_recap(db_session, chat.id, "User talked about Max.", ["Max"], confidence=0.8)

        provider = MockLTMChatProvider()
        embedding = MockLTMEmbeddingProvider()

        episode = await ltm.generate_episode(
            db_session, provider, embedding, chat.id, "scope-1",
        )

        assert episode is not None
        assert episode.scope_id == "scope-1"
        assert "Max" in episode.episode_summary
        assert episode.importance_score == 0.8
        assert episode.is_final is True
        assert episode.embedding is not None
        assert len(episode.embedding) == 768

    async def test_episode_has_keywords(self, ltm, stm, db_session):
        chat = await create_chat(db_session, scope_id="scope-2")
        await stm.upsert_entity(db_session, chat.id, "pet", "Max", confidence=0.9)

        provider = MockLTMChatProvider()
        embedding = MockLTMEmbeddingProvider()

        episode = await ltm.generate_episode(db_session, provider, embedding, chat.id, "scope-2")

        assert "Max" in episode.keywords

    async def test_episode_skips_empty_chat(self, ltm, db_session):
        chat = await create_chat(db_session, scope_id="scope-3")

        provider = MockLTMChatProvider()
        embedding = MockLTMEmbeddingProvider()

        episode = await ltm.generate_episode(db_session, provider, embedding, chat.id, "scope-3")

        assert episode is None


class TestPromoteEntities:
    async def test_promotes_stm_to_ltm(self, ltm, stm, db_session):
        chat = await create_chat(db_session, scope_id="scope-p")
        await stm.upsert_entity(db_session, chat.id, "pet", "Max",
                                attributes={"breed": "Golden"}, confidence=0.9)
        await stm.upsert_entity(db_session, chat.id, "person", "Alice", confidence=0.8)

        await ltm.promote_entities(db_session, chat.id, "scope-p")

        from sqlalchemy import select
        stmt = select(LTMEntity).where(LTMEntity.scope_id == "scope-p")
        result = await db_session.execute(stmt)
        ltm_entities = list(result.scalars().all())

        assert len(ltm_entities) == 2
        names = {e.canonical_name for e in ltm_entities}
        assert "Max" in names
        assert "Alice" in names

    async def test_promote_deduplicates(self, ltm, stm, db_session):
        chat1 = await create_chat(db_session, scope_id="scope-d")
        await stm.upsert_entity(db_session, chat1.id, "pet", "Max", confidence=0.9)
        await ltm.promote_entities(db_session, chat1.id, "scope-d")

        # Promote again from different chat with same entity
        chat2 = await create_chat(db_session, scope_id="scope-d")
        await stm.upsert_entity(db_session, chat2.id, "pet", "Max", confidence=0.8)
        await ltm.promote_entities(db_session, chat2.id, "scope-d")

        from sqlalchemy import select, func
        stmt = select(func.count()).select_from(LTMEntity).where(
            LTMEntity.scope_id == "scope-d",
            func.lower(LTMEntity.canonical_name) == "max",
        )
        result = await db_session.execute(stmt)
        assert result.scalar_one() == 1  # Not duplicated


class TestPromoteRelationships:
    async def test_promotes_relationships(self, ltm, stm, db_session):
        chat = await create_chat(db_session, scope_id="scope-r")
        subj = await stm.upsert_entity(db_session, chat.id, "person", "User", confidence=0.9)
        obj = await stm.upsert_entity(db_session, chat.id, "pet", "Max", confidence=0.9)
        await stm.upsert_relationship(db_session, chat.id, subj.id, "owns", obj.id, confidence=0.9)

        await ltm.promote_entities(db_session, chat.id, "scope-r")
        await ltm.promote_relationships(db_session, chat.id, "scope-r")

        from sqlalchemy import select
        stmt = select(LTMRelationship).where(LTMRelationship.scope_id == "scope-r")
        result = await db_session.execute(stmt)
        ltm_rels = list(result.scalars().all())
        assert len(ltm_rels) == 1
        assert ltm_rels[0].predicate == "owns"


class TestImportanceDecay:
    async def test_decay_reduces_old_episodes(self, ltm, db_session):
        old_date = datetime.now(timezone.utc) - timedelta(days=14)

        episode = LTMEpisode(
            scope_id="scope-decay",
            episode_summary="Old episode",
            keywords=["old"],
            importance_score=0.9,
            is_final=True,
            episode_date=old_date,
            source_chat_id=uuid.uuid4(),
        )
        db_session.add(episode)
        await db_session.flush()
        original_score = episode.importance_score

        await ltm.apply_importance_decay(db_session, "scope-decay")
        await db_session.refresh(episode)

        assert episode.importance_score < original_score

    async def test_recent_episodes_barely_decay(self, ltm, db_session):
        recent_date = datetime.now(timezone.utc) - timedelta(hours=1)

        episode = LTMEpisode(
            scope_id="scope-recent",
            episode_summary="Recent episode",
            keywords=["recent"],
            importance_score=0.9,
            is_final=True,
            episode_date=recent_date,
            source_chat_id=uuid.uuid4(),
        )
        db_session.add(episode)
        await db_session.flush()

        await ltm.apply_importance_decay(db_session, "scope-recent")
        await db_session.refresh(episode)

        # Should barely change
        assert episode.importance_score > 0.85
