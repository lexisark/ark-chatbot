"""RAG manager — hybrid retrieval of STM + LTM memories for context assembly."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from context_engine.config import RAG_CONFIG
from context_engine.models import YouRememberBlock
from context_engine.tokens import TokenHelper
from db.models import LTMEpisode, STMEntity, STMRecap, STMRelationship

logger = logging.getLogger(__name__)


class RAGManager:
    """Retrieves and formats memories for the context window."""

    def __init__(self, tokens: TokenHelper):
        self.tokens = tokens

    async def build_you_remember(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        query: str,
        budget_tokens: int,
        *,
        scope_id: str | None = None,
        turn_count: int = 1,
        query_embedding: list[float] | None = None,
    ) -> YouRememberBlock:
        """Build the "You remember" block from STM + LTM.

        Budget split:
        - Episode budget varies by turn count (25% early → 10% late)
        - Remaining goes to STM (entities, relationships, recaps)
        """
        block = YouRememberBlock()

        # Determine episode budget ratio
        if turn_count <= 2:
            block.episode_budget_ratio = RAG_CONFIG["episode_budget_turn_1_2"]
        elif turn_count <= 5:
            block.episode_budget_ratio = RAG_CONFIG["episode_budget_turn_3_5"]
        elif turn_count <= 19:
            block.episode_budget_ratio = RAG_CONFIG["episode_budget_turn_6_19"]
        else:
            block.episode_budget_ratio = RAG_CONFIG["episode_budget_turn_20_plus"]

        episode_budget = int(budget_tokens * block.episode_budget_ratio)
        stm_budget = budget_tokens - episode_budget

        # ── STM: entities ───────────────────────────────
        entities = await self._get_stm_entities(db, chat_id)
        for e in entities:
            line = f"- {e.canonical_name} ({e.entity_type})"
            if e.attributes:
                attrs = ", ".join(f"{k}: {v}" for k, v in e.attributes.items())
                line += f" — {attrs}"
            block.entities.append(line)

        # ── STM: relationships ──────────────────────────
        relationships = await self._get_stm_relationships(db, chat_id)
        for r in relationships:
            # Resolve entity names
            subj = await db.get(STMEntity, r.subject_entity_id)
            obj = await db.get(STMEntity, r.object_entity_id)
            subj_name = subj.canonical_name if subj else "?"
            obj_name = obj.canonical_name if obj else "?"
            block.relationships.append(f"- {subj_name} {r.predicate} {obj_name}")

        # ── STM: recaps ─────────────────────────────────
        recaps = await self._get_stm_recaps(db, chat_id)
        for rc in recaps:
            block.recaps.append(f"- {rc.recap_text}")

        # ── LTM: episodes ──────────────────────────────
        if scope_id and episode_budget > 0:
            episodes = await self._get_ltm_episodes(db, scope_id, chat_id)
            for ep in episodes:
                block.episodes.append(f"- {ep.episode_summary}")

        # ── Truncate to budget ──────────────────────────
        self._truncate_block(block, stm_budget, episode_budget)

        return block

    async def _get_stm_entities(
        self, db: AsyncSession, chat_id: uuid.UUID,
    ) -> list[STMEntity]:
        stmt = (
            select(STMEntity)
            .where(
                STMEntity.chat_id == chat_id,
                STMEntity.overall_confidence >= 0.30,
            )
            .order_by(STMEntity.overall_confidence.desc())
            .limit(50)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def _get_stm_relationships(
        self, db: AsyncSession, chat_id: uuid.UUID,
    ) -> list[STMRelationship]:
        stmt = (
            select(STMRelationship)
            .where(
                STMRelationship.chat_id == chat_id,
                STMRelationship.confidence >= 0.30,
            )
            .order_by(STMRelationship.confidence.desc())
            .limit(30)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def _get_stm_recaps(
        self, db: AsyncSession, chat_id: uuid.UUID,
    ) -> list[STMRecap]:
        stmt = (
            select(STMRecap)
            .where(STMRecap.chat_id == chat_id)
            .order_by(STMRecap.created_at.desc())
            .limit(5)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def _get_ltm_episodes(
        self, db: AsyncSession, scope_id: str, exclude_chat_id: uuid.UUID,
    ) -> list[LTMEpisode]:
        stmt = (
            select(LTMEpisode)
            .where(
                LTMEpisode.scope_id == scope_id,
                LTMEpisode.source_chat_id != exclude_chat_id,
            )
            .order_by(
                LTMEpisode.is_final.desc(),
                LTMEpisode.importance_score.desc(),
                LTMEpisode.episode_date.desc(),
            )
            .limit(RAG_CONFIG.get("episode_retrieval_limit", 5))
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    def _truncate_block(
        self, block: YouRememberBlock, stm_budget: int, episode_budget: int,
    ) -> None:
        """Truncate block sections to fit within token budgets."""

        # Truncate STM sections
        stm_used = 0
        block.entities = self._fit_lines(block.entities, stm_budget, stm_used)
        stm_used += self._count_lines(block.entities)

        block.relationships = self._fit_lines(block.relationships, stm_budget, stm_used)
        stm_used += self._count_lines(block.relationships)

        block.recaps = self._fit_lines(block.recaps, stm_budget, stm_used)
        stm_used += self._count_lines(block.recaps)

        # Truncate episodes
        block.episodes = self._fit_lines(block.episodes, episode_budget, 0)
        episode_used = self._count_lines(block.episodes)

        block.total_tokens = stm_used + episode_used

    def _fit_lines(self, lines: list[str], budget: int, used: int) -> list[str]:
        """Keep lines that fit within remaining budget."""
        fitted = []
        current = used
        for line in lines:
            line_tokens = self.tokens.count(line)
            if current + line_tokens > budget:
                break
            fitted.append(line)
            current += line_tokens
        return fitted

    def _count_lines(self, lines: list[str]) -> int:
        return sum(self.tokens.count(line) for line in lines)
