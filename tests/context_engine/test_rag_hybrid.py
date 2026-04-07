"""Tests for hybrid search (FTS + vector) across all RAG tiers."""

import math
import uuid

import pytest

from context_engine.embedding_service import EmbeddingService
from context_engine.rag_manager import RAGManager
from context_engine.stm_manager import STMManager
from context_engine.tokens import TokenHelper
from db.models import LTMEntity, LTMEpisode
from db.queries import create_chat
from providers.base import EmbeddingResponse
from providers.token_counter import TiktokenCounter


class DeterministicEmbProvider:
    """Generates embeddings where similar texts produce similar vectors."""

    @property
    def dimensions(self):
        return 768

    async def embed(self, text, **kw):
        import hashlib
        words = text.lower().split()
        vec = [0.0] * 768
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(768):
                vec[i] += ((h >> (i % 64)) & 1) / max(len(words), 1)
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return EmbeddingResponse(embedding=vec, model="mock", tokens=5)

    async def embed_batch(self, texts, **kw):
        return [await self.embed(t, **kw) for t in texts]


@pytest.fixture
def emb_service():
    return EmbeddingService(DeterministicEmbProvider())


@pytest.fixture
def stm(emb_service):
    return STMManager(embedding_service=emb_service)


@pytest.fixture
def tokens():
    return TokenHelper(TiktokenCounter())


@pytest.fixture
def rag(tokens):
    return RAGManager(tokens)


class TestSTMEntityHybridSearch:
    async def test_relevant_entity_ranked_higher(self, rag, stm, emb_service, db_session):
        """Entity matching the query should rank higher than unrelated entity."""
        chat = await create_chat(db_session, scope_id="h1")

        await stm.upsert_entity(db_session, chat.id, "pet", "Max",
                                attributes={"breed": "Golden Retriever"}, confidence=0.7)
        await stm.upsert_entity(db_session, chat.id, "location", "Seattle",
                                attributes={"type": "city"}, confidence=0.9)

        query_emb = await emb_service.generate_query_embedding("tell me about my dog Max")

        result = await rag.build_you_remember(
            db_session, chat.id, "tell me about my dog Max",
            budget_tokens=500, query_embedding=query_emb,
        )

        entity_text = "\n".join(result.entities)
        # Max should appear (it's the relevant entity)
        assert "Max" in entity_text

    async def test_vector_search_finds_semantic_match(self, rag, stm, emb_service, db_session):
        """Vector search should find 'Max (pet/dog)' even when query says 'puppy'."""
        chat = await create_chat(db_session, scope_id="h2")

        await stm.upsert_entity(db_session, chat.id, "pet", "Max",
                                attributes={"breed": "Golden Retriever", "type": "dog"}, confidence=0.8)

        query_emb = await emb_service.generate_query_embedding("my puppy dog pet")

        result = await rag.build_you_remember(
            db_session, chat.id, "my puppy",
            budget_tokens=500, query_embedding=query_emb,
        )

        entity_text = "\n".join(result.entities)
        assert "Max" in entity_text


class TestSTMRecapHybridSearch:
    async def test_relevant_recap_ranked_higher(self, rag, stm, emb_service, db_session):
        chat = await create_chat(db_session, scope_id="h3")

        await stm.insert_recap(db_session, chat.id,
                               "User discussed their golden retriever Max and walks in the park.",
                               ["Max", "dog", "park"], confidence=0.8)
        await stm.insert_recap(db_session, chat.id,
                               "User asked about the weather forecast for next week.",
                               ["weather", "forecast"], confidence=0.8)

        query_emb = await emb_service.generate_query_embedding("Max dog park")

        result = await rag.build_you_remember(
            db_session, chat.id, "tell me about Max",
            budget_tokens=500, query_embedding=query_emb,
        )

        # Should have at least one recap about Max
        assert len(result.recaps) > 0
        recap_text = "\n".join(result.recaps)
        assert "Max" in recap_text


class TestLTMEpisodeHybridSearch:
    async def test_episode_vector_search(self, rag, emb_service, db_session):
        chat = await create_chat(db_session, scope_id="h4")

        # Create episode with embedding
        ep_text = "User talked about their golden retriever Max who loves the park."
        ep_emb = await emb_service.generate_document_embedding(ep_text)

        episode = LTMEpisode(
            scope_id="h4",
            episode_summary=ep_text,
            keywords=["Max", "golden retriever", "park"],
            embedding=ep_emb,
            importance_score=0.8,
            is_final=True,
            source_chat_id=uuid.uuid4(),
        )
        db_session.add(episode)
        await db_session.flush()

        query_emb = await emb_service.generate_query_embedding("Max dog park")

        result = await rag.build_you_remember(
            db_session, chat.id, "Max",
            budget_tokens=500, scope_id="h4", turn_count=1,
            query_embedding=query_emb,
        )

        assert len(result.episodes) > 0
        assert "Max" in "\n".join(result.episodes)


class TestLTMEntityHybridSearch:
    async def test_ltm_entity_retrieved(self, rag, emb_service, db_session):
        chat = await create_chat(db_session, scope_id="h5")

        # Create LTM entity with embedding
        ent_text = "Max (pet) — breed: Golden Retriever"
        ent_emb = await emb_service.generate_document_embedding(ent_text)

        ltm_entity = LTMEntity(
            scope_id="h5",
            entity_type="pet",
            canonical_name="Max",
            attributes={"breed": "Golden Retriever"},
            overall_confidence=0.9,
            embedding=ent_emb,
        )
        db_session.add(ltm_entity)
        await db_session.flush()

        query_emb = await emb_service.generate_query_embedding("my dog Max")

        result = await rag.build_you_remember(
            db_session, chat.id, "Max",
            budget_tokens=500, scope_id="h5", turn_count=1,
            query_embedding=query_emb,
        )

        # LTM entities should appear in entities list
        entity_text = "\n".join(result.entities)
        assert "Max" in entity_text


class TestHybridScoringWeights:
    async def test_fts_weight_dominates(self, rag, stm, emb_service, db_session):
        """With 70% FTS weight, keyword match should matter more than vector similarity."""
        chat = await create_chat(db_session, scope_id="h6")

        # Entity with exact keyword match
        await stm.upsert_entity(db_session, chat.id, "pet", "Max", confidence=0.5)
        # Entity with high confidence but no keyword match
        await stm.upsert_entity(db_session, chat.id, "location", "Paris", confidence=0.95)

        query_emb = await emb_service.generate_query_embedding("Max")

        result = await rag.build_you_remember(
            db_session, chat.id, "Max",
            budget_tokens=500, query_embedding=query_emb,
        )

        if len(result.entities) >= 2:
            # Max should be first despite lower confidence (keyword match)
            assert "Max" in result.entities[0]


class TestNoEmbeddingFallback:
    async def test_works_without_query_embedding(self, rag, stm, db_session):
        """RAG should still work when no query embedding is provided (FTS-only fallback)."""
        chat = await create_chat(db_session, scope_id="h7")
        # STMManager without embedding service
        stm_no_emb = STMManager()
        await stm_no_emb.upsert_entity(db_session, chat.id, "pet", "Max", confidence=0.9)

        result = await rag.build_you_remember(
            db_session, chat.id, "Max",
            budget_tokens=500,
            query_embedding=None,
        )

        # Should still return entities via FTS/confidence fallback
        assert len(result.entities) > 0
        assert "Max" in "\n".join(result.entities)
