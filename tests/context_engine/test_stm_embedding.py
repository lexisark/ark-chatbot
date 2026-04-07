"""Tests for STM Manager embedding generation on write."""

import pytest

from context_engine.embedding_service import EmbeddingService
from context_engine.stm_manager import STMManager
from db.queries import create_chat
from providers.base import EmbeddingResponse


class MockEmbProvider:
    @property
    def dimensions(self):
        return 768

    async def embed(self, text, **kw):
        # Deterministic: just use length-based values
        return EmbeddingResponse(embedding=[len(text) / 100.0] * 768, model="mock", tokens=5)

    async def embed_batch(self, texts, **kw):
        return [await self.embed(t, **kw) for t in texts]


@pytest.fixture
def embedding_service():
    return EmbeddingService(MockEmbProvider())


@pytest.fixture
def stm(embedding_service):
    return STMManager(embedding_service=embedding_service)


class TestEntityEmbeddingOnWrite:
    async def test_upsert_entity_generates_embedding(self, stm, db_session):
        chat = await create_chat(db_session)
        entity = await stm.upsert_entity(
            db_session, chat.id, "pet", "Max",
            attributes={"breed": "Golden Retriever"}, confidence=0.9,
        )
        assert entity.embedding is not None
        assert len(entity.embedding) == 768

    async def test_upsert_entity_updates_embedding_on_new_attributes(self, stm, db_session):
        chat = await create_chat(db_session)

        e1 = await stm.upsert_entity(db_session, chat.id, "pet", "Max", attributes={"breed": "Golden"})
        emb1 = list(e1.embedding)

        e2 = await stm.upsert_entity(db_session, chat.id, "pet", "Max", attributes={"age": "3 years"})
        emb2 = list(e2.embedding)

        # Embedding should change because attributes changed (text for embedding is different)
        assert emb1 != emb2

    async def test_no_embedding_service_skips_embedding(self, db_session):
        """STMManager without embedding_service should still work, just no embedding."""
        stm_no_emb = STMManager()
        chat = await create_chat(db_session)
        entity = await stm_no_emb.upsert_entity(db_session, chat.id, "pet", "Max")
        assert entity.embedding is None


class TestRecapEmbeddingOnWrite:
    async def test_insert_recap_generates_embedding(self, stm, db_session):
        chat = await create_chat(db_session)
        recap = await stm.insert_recap(
            db_session, chat.id, "User discussed their dog Max.", ["Max", "dog"],
            confidence=0.8,
        )
        assert recap.embedding is not None
        assert len(recap.embedding) == 768

    async def test_no_embedding_service_skips_embedding(self, db_session):
        stm_no_emb = STMManager()
        chat = await create_chat(db_session)
        recap = await stm_no_emb.insert_recap(
            db_session, chat.id, "Some recap.", ["test"], confidence=0.5,
        )
        assert recap.embedding is None


class TestLTMPromotionCopiesEmbedding:
    async def test_promote_entities_copies_embedding(self, stm, db_session):
        from context_engine.ltm_manager import LTMManager
        from sqlalchemy import select
        from db.models import LTMEntity

        chat = await create_chat(db_session, scope_id="emb-promo")
        await stm.upsert_entity(db_session, chat.id, "pet", "Max", attributes={"breed": "Golden"}, confidence=0.9)

        ltm = LTMManager()
        await ltm.promote_entities(db_session, chat.id, "emb-promo")

        result = await db_session.execute(select(LTMEntity).where(LTMEntity.scope_id == "emb-promo"))
        ltm_entity = result.scalar_one()

        assert ltm_entity.embedding is not None
        assert len(ltm_entity.embedding) == 768
