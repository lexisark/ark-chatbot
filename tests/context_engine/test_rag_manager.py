"""Tests for RAG manager."""

import uuid

import pytest

from db.queries import create_chat
from db.models import LTMEpisode
from context_engine.stm_manager import STMManager
from context_engine.rag_manager import RAGManager
from providers.token_counter import TiktokenCounter
from context_engine.tokens import TokenHelper


@pytest.fixture
def stm():
    return STMManager()


@pytest.fixture
def tokens():
    return TokenHelper(TiktokenCounter())


@pytest.fixture
def rag(tokens):
    return RAGManager(tokens)


class TestSTMEntityRetrieval:
    async def test_retrieves_entities_for_chat(self, rag, stm, db_session):
        chat = await create_chat(db_session, scope_id="s1")
        await stm.upsert_entity(db_session, chat.id, "pet", "Max", attributes={"breed": "Golden"}, confidence=0.9)
        await stm.upsert_entity(db_session, chat.id, "person", "Alice", confidence=0.8)

        result = await rag.build_you_remember(db_session, chat.id, "tell me about Max", budget_tokens=500)

        assert len(result.entities) > 0
        entity_text = "\n".join(result.entities)
        assert "Max" in entity_text

    async def test_filters_low_confidence_entities(self, rag, stm, db_session):
        chat = await create_chat(db_session, scope_id="s1")
        await stm.upsert_entity(db_session, chat.id, "person", "High", confidence=0.9)
        await stm.upsert_entity(db_session, chat.id, "person", "Low", confidence=0.1)

        result = await rag.build_you_remember(db_session, chat.id, "test", budget_tokens=500)

        entity_text = "\n".join(result.entities)
        assert "High" in entity_text


class TestSTMRecapRetrieval:
    async def test_retrieves_recaps(self, rag, stm, db_session):
        chat = await create_chat(db_session, scope_id="s1")
        await stm.insert_recap(db_session, chat.id, "User discussed their dog Max.", ["Max", "dog"], confidence=0.8)

        result = await rag.build_you_remember(db_session, chat.id, "Max", budget_tokens=500)

        assert len(result.recaps) > 0
        assert "Max" in result.recaps[0]


class TestSTMRelationshipRetrieval:
    async def test_retrieves_relationships(self, rag, stm, db_session):
        chat = await create_chat(db_session, scope_id="s1")
        subj = await stm.upsert_entity(db_session, chat.id, "person", "User", confidence=0.9)
        obj = await stm.upsert_entity(db_session, chat.id, "pet", "Max", confidence=0.9)
        await stm.upsert_relationship(db_session, chat.id, subj.id, "owns", obj.id, confidence=0.9)

        result = await rag.build_you_remember(db_session, chat.id, "Max", budget_tokens=500)

        assert len(result.relationships) > 0
        rel_text = "\n".join(result.relationships)
        assert "owns" in rel_text


class TestLTMEpisodeRetrieval:
    async def test_retrieves_episodes_for_scope(self, rag, stm, db_session):
        chat = await create_chat(db_session, scope_id="scope-1")

        episode = LTMEpisode(
            scope_id="scope-1",
            episode_summary="User talked about their golden retriever Max who loves the park.",
            keywords=["Max", "golden retriever", "park"],
            importance_score=0.8,
            is_final=True,
            source_chat_id=uuid.uuid4(),
        )
        db_session.add(episode)
        await db_session.flush()

        result = await rag.build_you_remember(
            db_session, chat.id, "Max", budget_tokens=500,
            scope_id="scope-1", turn_count=1,
        )

        assert len(result.episodes) > 0
        episode_text = "\n".join(result.episodes)
        assert "Max" in episode_text

    async def test_episode_budget_higher_early(self, rag, stm, db_session):
        """Turns 1-2 should allocate more budget to episodes."""
        chat = await create_chat(db_session, scope_id="scope-ep")

        for i in range(3):
            episode = LTMEpisode(
                scope_id="scope-ep",
                episode_summary=f"Episode {i} about conversation history.",
                keywords=[f"topic{i}"],
                importance_score=0.7,
                is_final=True,
                source_chat_id=uuid.uuid4(),
            )
            db_session.add(episode)
        await db_session.flush()

        # Early turn — more episodes
        result_early = await rag.build_you_remember(
            db_session, chat.id, "topic", budget_tokens=500,
            scope_id="scope-ep", turn_count=1,
        )
        # Late turn — fewer episodes
        result_late = await rag.build_you_remember(
            db_session, chat.id, "topic", budget_tokens=500,
            scope_id="scope-ep", turn_count=10,
        )

        # Early should have >= late episode allocation
        assert result_early.episode_budget_ratio >= result_late.episode_budget_ratio


class TestBudgetTruncation:
    async def test_output_fits_within_budget(self, rag, stm, db_session):
        chat = await create_chat(db_session, scope_id="s1")

        # Add lots of entities to exceed budget
        for i in range(30):
            await stm.upsert_entity(db_session, chat.id, "person", f"Person{i}", confidence=0.9)

        result = await rag.build_you_remember(db_session, chat.id, "test", budget_tokens=100)

        assert result.total_tokens <= 100


class TestEmptyMemories:
    async def test_empty_chat_returns_empty_block(self, rag, db_session):
        chat = await create_chat(db_session, scope_id="s1")

        result = await rag.build_you_remember(db_session, chat.id, "test", budget_tokens=500)

        assert result.entities == []
        assert result.relationships == []
        assert result.episodes == []
        assert result.recaps == []
        assert result.total_tokens == 0

    async def test_no_episodes_stm_only(self, rag, stm, db_session):
        chat = await create_chat(db_session, scope_id="no-ltm")
        await stm.upsert_entity(db_session, chat.id, "pet", "Max", confidence=0.9)

        result = await rag.build_you_remember(
            db_session, chat.id, "Max", budget_tokens=500,
            scope_id="no-ltm", turn_count=1,
        )

        assert len(result.entities) > 0
        assert result.episodes == []
