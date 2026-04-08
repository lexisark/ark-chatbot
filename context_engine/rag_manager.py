"""RAG manager — hybrid FTS + vector retrieval across STM + LTM tiers."""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime

from sqlalchemy import select, text, func
from sqlalchemy.ext.asyncio import AsyncSession

from context_engine.config import RAG_CONFIG, STM_CONFIG
from context_engine.models import YouRememberBlock
from context_engine.tokens import TokenHelper
from db.models import LTMEntity, LTMEpisode, STMEntity, STMRecap, STMRelationship

logger = logging.getLogger(__name__)

# Stop words to filter from query keywords
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could i me my we our you your he she it they them "
    "his her its their what which who whom this that these those am in on at to for "
    "of and but or not with from by as if then so than too very just about up out "
    "how when where why all each every both few more most some any no nor".split()
)


def _extract_keywords(query: str, max_keywords: int | None = None) -> list[str]:
    if max_keywords is None:
        max_keywords = RAG_CONFIG.get("max_keywords", 10)
    """Extract keywords from a query string, filtering stop words."""
    words = query.lower().split()
    keywords = [w.strip(".,!?;:'\"()[]{}") for w in words]
    keywords = [w for w in keywords if w and w not in _STOP_WORDS and len(w) > 1]
    return keywords[:max_keywords]


def _keyword_match_score(text_to_match: str, keywords: list[str]) -> float:
    """Score how well text matches the query keywords (0.0 to 1.0)."""
    if not keywords:
        return 0.0
    text_lower = text_to_match.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches / len(keywords)


def _cosine_distance_to_score(distance: float) -> float:
    """Convert cosine distance (0=identical, 2=opposite) to score (1=identical, 0=far)."""
    return max(0.0, 1.0 - distance)


# ── Recency scoring ────────────────────────────────────


def recency_score(
    last_mentioned: datetime,
    mention_count: int = 1,
    confidence: float = 0.5,
) -> float:
    """Adaptive recency score with mention and confidence factors.

    All parameters configurable via RECENCY_CONFIG (from .env).
    """
    from context_engine.config import RECENCY_CONFIG
    from datetime import timezone as tz

    now = datetime.now(tz.utc)
    age_hours = max(0.0, (now - last_mentioned).total_seconds() / 3600)

    base = RECENCY_CONFIG["base_half_life_hours"]
    m_factor = RECENCY_CONFIG["mention_factor"]
    m_cap = RECENCY_CONFIG["mention_cap"]
    c_factor = RECENCY_CONFIG["confidence_factor"]

    mention_mult = 1.0 + m_factor * min(mention_count, m_cap)
    confidence_mult = 1.0 + c_factor * confidence
    effective_half_life = base * mention_mult * confidence_mult

    return math.pow(0.5, age_hours / effective_half_life)


class RAGManager:
    """Retrieves and formats memories using hybrid FTS + vector search."""

    def __init__(self, tokens: TokenHelper):
        self.tokens = tokens
        self._fts_weight = RAG_CONFIG["hybrid_fts_weight"]
        self._vector_weight = RAG_CONFIG["hybrid_vector_weight"]
        self._vector_threshold = RAG_CONFIG["vector_distance_threshold"]
        self._score_hybrid = RAG_CONFIG["score_hybrid_weight"]
        self._score_confidence = RAG_CONFIG["score_confidence_weight"]
        self._score_recency = RAG_CONFIG["score_recency_weight"]

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
        context_window_start: datetime | None = None,
    ) -> YouRememberBlock:
        block = YouRememberBlock()
        keywords = _extract_keywords(query)

        # Determine episode budget ratio by turn count
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

        # ── STM: entities (hybrid scored, context window dedup) ───
        stm_entities = await self._search_stm_entities(
            db, chat_id, keywords, query_embedding, context_window_start,
        )
        for e in stm_entities:
            line = f"- {e['entity'].canonical_name} ({e['entity'].entity_type})"
            if e['entity'].attributes:
                attrs = ", ".join(f"{k}: {v}" for k, v in e['entity'].attributes.items())
                line += f" — {attrs}"
            block.entities.append(line)

        # ── STM: relationships ──────────────────────────
        relationships = await self._get_stm_relationships(db, chat_id)
        for r in relationships:
            subj = await db.get(STMEntity, r.subject_entity_id)
            obj = await db.get(STMEntity, r.object_entity_id)
            subj_name = subj.canonical_name if subj else "?"
            obj_name = obj.canonical_name if obj else "?"
            block.relationships.append(f"- {subj_name} {r.predicate} {obj_name}")

        # ── STM: recaps (hybrid scored, context window dedup) ───
        stm_recaps = await self._search_stm_recaps(
            db, chat_id, keywords, query_embedding, context_window_start,
        )
        for r in stm_recaps:
            block.recaps.append(f"- {r['recap'].recap_text}")

        # ── LTM: episodes (hybrid scored) ───────────────
        if scope_id and episode_budget > 0:
            ltm_episodes = await self._search_ltm_episodes(db, scope_id, chat_id, keywords, query_embedding)
            for ep in ltm_episodes:
                block.episodes.append(f"- {ep['episode'].episode_summary}")

        # ── LTM: entities (hybrid scored) ───────────────
        if scope_id:
            ltm_entities = await self._search_ltm_entities(db, scope_id, keywords, query_embedding)
            for e in ltm_entities:
                # Avoid duplicating STM entities already in block
                name = e['entity'].canonical_name
                if not any(name in line for line in block.entities):
                    line = f"- {name} ({e['entity'].entity_type})"
                    if e['entity'].attributes:
                        attrs = ", ".join(f"{k}: {v}" for k, v in e['entity'].attributes.items())
                        line += f" — {attrs}"
                    block.entities.append(line)

        # ── Truncate to budget ──────────────────────────
        self._truncate_block(block, stm_budget, episode_budget)

        return block

    # ── Search Methods ──────────────────────────────────

    async def _search_stm_entities(
        self, db: AsyncSession, chat_id: uuid.UUID,
        keywords: list[str], query_embedding: list[float] | None,
        context_window_start: datetime | None = None,
    ) -> list[dict]:
        """Hybrid search STM entities: FTS keyword match + vector similarity.
        Excludes entities first_mentioned within the context window."""
        stmt = (
            select(STMEntity)
            .where(STMEntity.chat_id == chat_id, STMEntity.overall_confidence >= STM_CONFIG["min_confidence_threshold"])
            .limit(RAG_CONFIG["stm_entity_limit"])
        )
        if context_window_start:
            stmt = stmt.where(STMEntity.first_mentioned < context_window_start)
        result = await db.execute(stmt)
        entities = list(result.scalars().all())

        scored = []
        for e in entities:
            # FTS score: keyword match against name + type + attributes
            match_text = self._entity_match_text(e)
            fts_score = _keyword_match_score(match_text, keywords)

            # Vector score: cosine similarity if both embeddings exist
            vector_score = 0.0
            if query_embedding and e.embedding is not None:
                distance = self._cosine_distance(query_embedding, list(e.embedding))
                if distance <= self._vector_threshold:
                    vector_score = _cosine_distance_to_score(distance)

            # Hybrid score
            hybrid = self._fts_weight * fts_score + self._vector_weight * vector_score

            # Recency score
            rec_score = recency_score(e.last_mentioned, e.mention_count, e.overall_confidence)

            final_score = self._score_hybrid * hybrid + self._score_confidence * e.overall_confidence + self._score_recency * rec_score

            scored.append({"entity": e, "score": final_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _search_stm_recaps(
        self, db: AsyncSession, chat_id: uuid.UUID,
        keywords: list[str], query_embedding: list[float] | None,
        context_window_start: datetime | None = None,
    ) -> list[dict]:
        """Hybrid search STM recaps. Excludes recaps within the context window."""
        stmt = (
            select(STMRecap)
            .where(STMRecap.chat_id == chat_id)
            .order_by(STMRecap.created_at.desc())
            .limit(RAG_CONFIG["stm_recap_limit"])
        )
        if context_window_start:
            stmt = stmt.where(STMRecap.created_at < context_window_start)
        result = await db.execute(stmt)
        recaps = list(result.scalars().all())

        scored = []
        for r in recaps:
            # FTS: match against recap text + keywords
            match_text = r.recap_text + " " + " ".join(r.keywords or [])
            fts_score = _keyword_match_score(match_text, keywords)

            # Vector
            vector_score = 0.0
            if query_embedding and r.embedding is not None:
                distance = self._cosine_distance(query_embedding, list(r.embedding))
                if distance <= self._vector_threshold:
                    vector_score = _cosine_distance_to_score(distance)

            hybrid = self._fts_weight * fts_score + self._vector_weight * vector_score

            # Recency for recaps (use created_at, mention_count=1)
            rec_score = recency_score(r.created_at, mention_count=1, confidence=r.confidence)

            final_score = self._score_hybrid * hybrid + self._score_confidence * r.confidence + self._score_recency * rec_score

            scored.append({"recap": r, "score": final_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _search_ltm_episodes(
        self, db: AsyncSession, scope_id: str, exclude_chat_id: uuid.UUID,
        keywords: list[str], query_embedding: list[float] | None,
    ) -> list[dict]:
        """Hybrid search LTM episodes: FTS on summary + vector on embedding."""
        stmt = (
            select(LTMEpisode)
            .where(
                LTMEpisode.scope_id == scope_id,
                LTMEpisode.source_chat_id != exclude_chat_id,
            )
            .order_by(LTMEpisode.importance_score.desc())
            .limit(RAG_CONFIG["ltm_episode_limit"])
        )
        result = await db.execute(stmt)
        episodes = list(result.scalars().all())

        scored = []
        for ep in episodes:
            # FTS: match against summary + keywords
            match_text = ep.episode_summary + " " + " ".join(ep.keywords or [])
            fts_score = _keyword_match_score(match_text, keywords)

            # Vector
            vector_score = 0.0
            if query_embedding and ep.embedding is not None:
                distance = self._cosine_distance(query_embedding, list(ep.embedding))
                if distance <= self._vector_threshold:
                    vector_score = _cosine_distance_to_score(distance)

            hybrid = self._fts_weight * fts_score + self._vector_weight * vector_score

            # Weight by importance
            final_score = 0.5 * hybrid + 0.5 * ep.importance_score

            scored.append({"episode": ep, "score": final_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _search_ltm_entities(
        self, db: AsyncSession, scope_id: str,
        keywords: list[str], query_embedding: list[float] | None,
    ) -> list[dict]:
        """Hybrid search LTM entities."""
        stmt = (
            select(LTMEntity)
            .where(LTMEntity.scope_id == scope_id, LTMEntity.overall_confidence >= STM_CONFIG["min_confidence_threshold"])
            .limit(RAG_CONFIG["ltm_entity_limit"])
        )
        result = await db.execute(stmt)
        entities = list(result.scalars().all())

        scored = []
        for e in entities:
            match_text = self._entity_match_text_ltm(e)
            fts_score = _keyword_match_score(match_text, keywords)

            vector_score = 0.0
            if query_embedding and e.embedding is not None:
                distance = self._cosine_distance(query_embedding, list(e.embedding))
                if distance <= self._vector_threshold:
                    vector_score = _cosine_distance_to_score(distance)

            hybrid = self._fts_weight * fts_score + self._vector_weight * vector_score
            final_score = 0.5 * hybrid + 0.5 * e.overall_confidence

            scored.append({"entity": e, "score": final_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _get_stm_relationships(
        self, db: AsyncSession, chat_id: uuid.UUID,
    ) -> list[STMRelationship]:
        stmt = (
            select(STMRelationship)
            .where(STMRelationship.chat_id == chat_id, STMRelationship.confidence >= STM_CONFIG["min_confidence_threshold"])
            .order_by(STMRelationship.confidence.desc())
            .limit(RAG_CONFIG["stm_relationship_limit"])
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    # ── Scoring Helpers ─────────────────────────────────

    @staticmethod
    def _entity_match_text(e: STMEntity) -> str:
        parts = [e.canonical_name, e.entity_type]
        if e.entity_subtype:
            parts.append(e.entity_subtype)
        if e.attributes:
            parts.extend(str(v) for v in e.attributes.values())
        return " ".join(parts)

    @staticmethod
    def _entity_match_text_ltm(e: LTMEntity) -> str:
        parts = [e.canonical_name, e.entity_type]
        if e.entity_subtype:
            parts.append(e.entity_subtype)
        if e.attributes:
            parts.extend(str(v) for v in e.attributes.values())
        return " ".join(parts)

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """Compute cosine distance (0=identical, 2=opposite)."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity  # distance

    # ── Budget Truncation ───────────────────────────────

    def _truncate_block(
        self, block: YouRememberBlock, stm_budget: int, episode_budget: int,
    ) -> None:
        stm_used = 0
        block.entities = self._fit_lines(block.entities, stm_budget, stm_used)
        stm_used += self._count_lines(block.entities)

        block.relationships = self._fit_lines(block.relationships, stm_budget, stm_used)
        stm_used += self._count_lines(block.relationships)

        block.recaps = self._fit_lines(block.recaps, stm_budget, stm_used)
        stm_used += self._count_lines(block.recaps)

        block.episodes = self._fit_lines(block.episodes, episode_budget, 0)
        episode_used = self._count_lines(block.episodes)

        block.total_tokens = stm_used + episode_used

    def _fit_lines(self, lines: list[str], budget: int, used: int) -> list[str]:
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
