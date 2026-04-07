"""Long-Term Memory manager — episode generation, entity promotion, importance decay."""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from context_engine.embedding_service import EmbeddingService
from context_engine.stm_manager import STMManager
from db.models import (
    LTMEntity,
    LTMEpisode,
    LTMRelationship,
    STMEntity,
    STMRelationship,
)
from providers.base import ChatProvider, EmbeddingProvider

logger = logging.getLogger(__name__)

EPISODE_DECAY_HALF_LIFE_DAYS = 7


class LTMManager:

    async def generate_episode(
        self,
        db: AsyncSession,
        chat_provider: ChatProvider,
        embedding_provider: EmbeddingProvider,
        chat_id: uuid.UUID,
        scope_id: str,
    ) -> LTMEpisode | None:
        """Generate an LTM episode from a chat's STM data + recent messages."""
        from db.queries import get_chat_messages

        stm = STMManager()
        entities = await stm.get_entities(db, chat_id, min_confidence=0.0)
        relationships = await stm.get_relationships(db, chat_id, min_confidence=0.0)
        recaps = await stm.get_recaps(db, chat_id, limit=100)  # all recaps
        messages = await get_chat_messages(db, chat_id, limit=10)  # last 10 messages

        if not entities and not recaps and not messages:
            return None

        # Build context for episode generation
        context_parts = []

        if messages:
            context_parts.append("Recent conversation:")
            for m in messages:
                context_parts.append(f"  {m.role.value.upper()}: {m.content}")

        if entities:
            context_parts.append("\nEntities from this conversation:")
            for e in entities[:50]:
                line = f"- {e.canonical_name} ({e.entity_type})"
                if e.attributes:
                    attrs = ", ".join(f"{k}: {v}" for k, v in e.attributes.items())
                    line += f" — {attrs}"
                context_parts.append(line)

        if relationships:
            context_parts.append("\nRelationships:")
            for r in relationships[:30]:
                subj = await db.get(STMEntity, r.subject_entity_id)
                obj = await db.get(STMEntity, r.object_entity_id)
                subj_name = subj.canonical_name if subj else "?"
                obj_name = obj.canonical_name if obj else "?"
                context_parts.append(f"- {subj_name} {r.predicate} {obj_name}")

        if recaps:
            context_parts.append("\nConversation summaries:")
            for r in recaps:
                context_parts.append(f"- {r.recap_text}")

        context = "\n".join(context_parts)

        # Call LLM to generate episode narrative
        prompt = f"""Summarize this conversation into a rich narrative episode (200-300 tokens).
Include key entities, relationships, what was discussed, and the emotional tone.
This episode will be used to recall this conversation in the future.

{context}

Respond with ONLY valid JSON:
{{
    "episode_summary": "Rich narrative summary of the conversation...",
    "keywords": ["keyword1", "keyword2"],
    "importance_score": 0.8,
    "emotional_tone": "positive|negative|neutral|mixed"
}}"""

        response = await chat_provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )

        # Parse response — reuse extraction's robust parser
        import re
        cleaned = response.content.strip()
        # Try closed code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            # Try truncated code block
            match = re.search(r"```(?:json)?\s*\n?(.*)", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            from context_engine.extraction import _try_repair_json
            data = _try_repair_json(cleaned)
            if data is None:
                logger.warning(f"Failed to parse episode response: {response.content[:200]}")
                # Fall back to using the raw text as the summary
                data = {
                    "episode_summary": response.content[:500],
                    "keywords": [],
                    "importance_score": 0.5,
                    "emotional_tone": "neutral",
                }

        # Generate embedding
        embedding_service = EmbeddingService(embedding_provider)
        embedding = await embedding_service.generate_document_embedding(data["episode_summary"])

        # Store episode
        episode = LTMEpisode(
            scope_id=scope_id,
            episode_summary=data["episode_summary"],
            keywords=data.get("keywords", []),
            embedding=embedding,
            importance_score=data.get("importance_score", 0.5),
            is_final=True,
            emotional_tone=data.get("emotional_tone", "neutral"),
            source_chat_id=chat_id,
        )
        db.add(episode)
        await db.flush()

        logger.info(f"Generated LTM episode for scope {scope_id}, importance={episode.importance_score}")
        return episode

    async def promote_entities(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        scope_id: str,
    ) -> int:
        """Promote STM entities to LTM with deduplication."""
        stm_entities = await db.execute(
            select(STMEntity).where(STMEntity.chat_id == chat_id)
        )
        promoted = 0

        for stm_e in stm_entities.scalars().all():
            # Check for existing LTM entity (case-insensitive)
            existing = await db.execute(
                select(LTMEntity).where(
                    LTMEntity.scope_id == scope_id,
                    LTMEntity.entity_type == stm_e.entity_type,
                    func.lower(LTMEntity.canonical_name) == stm_e.canonical_name.lower(),
                )
            )
            ltm_e = existing.scalar_one_or_none()

            if ltm_e:
                # Merge: update confidence, attributes, mention count, embedding
                ltm_e.overall_confidence = max(ltm_e.overall_confidence, stm_e.overall_confidence)
                ltm_e.mention_count += stm_e.mention_count
                if stm_e.attributes:
                    ltm_e.attributes = {**ltm_e.attributes, **stm_e.attributes}
                if stm_e.embedding is not None:
                    ltm_e.embedding = stm_e.embedding
            else:
                ltm_e = LTMEntity(
                    scope_id=scope_id,
                    entity_type=stm_e.entity_type,
                    entity_subtype=stm_e.entity_subtype,
                    canonical_name=stm_e.canonical_name,
                    attributes=stm_e.attributes,
                    overall_confidence=stm_e.overall_confidence,
                    mention_count=stm_e.mention_count,
                    embedding=stm_e.embedding,
                )
                db.add(ltm_e)
                promoted += 1

        await db.flush()
        return promoted

    async def promote_relationships(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        scope_id: str,
    ) -> int:
        """Promote STM relationships to LTM with deduplication."""
        stm_rels = await db.execute(
            select(STMRelationship).where(STMRelationship.chat_id == chat_id)
        )
        promoted = 0

        for stm_r in stm_rels.scalars().all():
            # Resolve STM entity IDs to LTM entity IDs via canonical name
            stm_subj = await db.get(STMEntity, stm_r.subject_entity_id)
            stm_obj = await db.get(STMEntity, stm_r.object_entity_id)
            if not stm_subj or not stm_obj:
                continue

            # Find corresponding LTM entities
            ltm_subj = await self._find_ltm_entity(db, scope_id, stm_subj)
            ltm_obj = await self._find_ltm_entity(db, scope_id, stm_obj)
            if not ltm_subj or not ltm_obj:
                continue

            # Check for existing relationship
            existing = await db.execute(
                select(LTMRelationship).where(
                    LTMRelationship.scope_id == scope_id,
                    LTMRelationship.subject_entity_id == ltm_subj.id,
                    LTMRelationship.predicate == stm_r.predicate,
                    LTMRelationship.object_entity_id == ltm_obj.id,
                )
            )
            if existing.scalar_one_or_none():
                continue

            ltm_rel = LTMRelationship(
                scope_id=scope_id,
                subject_entity_id=ltm_subj.id,
                predicate=stm_r.predicate,
                object_entity_id=ltm_obj.id,
                confidence=stm_r.confidence,
            )
            db.add(ltm_rel)
            promoted += 1

        await db.flush()
        return promoted

    async def apply_importance_decay(
        self,
        db: AsyncSession,
        scope_id: str,
    ) -> None:
        """Apply importance decay to episodes based on age (7-day half-life)."""
        episodes = await db.execute(
            select(LTMEpisode).where(LTMEpisode.scope_id == scope_id)
        )

        now = datetime.now(timezone.utc)
        for ep in episodes.scalars().all():
            age_days = (now - ep.episode_date).total_seconds() / 86400
            decay = math.pow(0.5, age_days / EPISODE_DECAY_HALF_LIFE_DAYS)
            # Apply decay to the original score, not compound
            ep.importance_score = ep.importance_score * decay

        await db.flush()

    async def _find_ltm_entity(
        self, db: AsyncSession, scope_id: str, stm_entity: STMEntity,
    ) -> LTMEntity | None:
        result = await db.execute(
            select(LTMEntity).where(
                LTMEntity.scope_id == scope_id,
                LTMEntity.entity_type == stm_entity.entity_type,
                func.lower(LTMEntity.canonical_name) == stm_entity.canonical_name.lower(),
            )
        )
        return result.scalar_one_or_none()
