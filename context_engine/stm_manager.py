"""Short-Term Memory (STM) manager — entity/relationship upsert with dedup."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from context_engine.config import STM_CONFIG
from db.models import STMEntity, STMRecap, STMRelationship

logger = logging.getLogger(__name__)

CONFIDENCE_FLOOR = 0.30
CONFIDENCE_CAP = 0.95


class STMManager:
    def __init__(self, embedding_service=None):
        self._max_entities = STM_CONFIG["max_entities_per_chat"]
        self._boost = STM_CONFIG["fact_reinforcement_boost"]
        self._min_confidence = STM_CONFIG["min_confidence_threshold"]
        self._embedding_service = embedding_service

    def _entity_text(self, canonical_name: str, entity_type: str, attributes: dict | None) -> str:
        """Build text representation for embedding."""
        text = f"{canonical_name} ({entity_type})"
        if attributes:
            attrs = ", ".join(f"{k}: {v}" for k, v in attributes.items())
            text += f" — {attrs}"
        return text

    async def _generate_embedding(self, text: str) -> list[float] | None:
        if self._embedding_service is None:
            return None
        try:
            return await self._embedding_service.generate_document_embedding(text)
        except Exception:
            logger.warning(f"Failed to generate embedding for: {text[:50]}", exc_info=True)
            return None

    async def upsert_entity(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        entity_type: str,
        canonical_name: str,
        *,
        entity_subtype: str | None = None,
        attributes: dict | None = None,
        confidence: float = 0.5,
    ) -> STMEntity:
        """Insert or update an entity. Deduplicates by (chat_id, entity_type, canonical_name) case-insensitive."""

        # Find existing
        stmt = (
            select(STMEntity)
            .where(
                STMEntity.chat_id == chat_id,
                STMEntity.entity_type == entity_type,
                func.lower(STMEntity.canonical_name) == canonical_name.lower(),
            )
        )
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        now = datetime.now(timezone.utc)

        if existing:
            # Reinforce confidence
            existing.overall_confidence = min(
                CONFIDENCE_CAP,
                existing.overall_confidence + self._boost,
            )
            existing.mention_count += 1
            existing.last_mentioned = now
            existing.updated_at = now

            # Merge attributes
            attrs_changed = False
            if attributes:
                merged = {**existing.attributes, **attributes}
                if merged != existing.attributes:
                    attrs_changed = True
                existing.attributes = merged

            if entity_subtype and not existing.entity_subtype:
                existing.entity_subtype = entity_subtype

            # Regenerate embedding if attributes changed
            if attrs_changed:
                existing.embedding = await self._generate_embedding(
                    self._entity_text(existing.canonical_name, existing.entity_type, existing.attributes)
                )

            await db.flush()
            return existing

        # Check entity cap
        count_stmt = (
            select(func.count())
            .select_from(STMEntity)
            .where(STMEntity.chat_id == chat_id)
        )
        count_result = await db.execute(count_stmt)
        current_count = count_result.scalar_one()

        if current_count >= self._max_entities:
            # Evict lowest-confidence entity
            evict_stmt = (
                select(STMEntity)
                .where(STMEntity.chat_id == chat_id)
                .order_by(STMEntity.overall_confidence.asc())
                .limit(1)
            )
            evict_result = await db.execute(evict_stmt)
            evict_entity = evict_result.scalar_one_or_none()
            if evict_entity:
                await db.delete(evict_entity)
                await db.flush()

        # Insert new
        clamped_confidence = max(CONFIDENCE_FLOOR, min(CONFIDENCE_CAP, confidence))

        entity_attrs = attributes or {}
        embedding = await self._generate_embedding(
            self._entity_text(canonical_name, entity_type, entity_attrs)
        )

        entity = STMEntity(
            chat_id=chat_id,
            entity_type=entity_type,
            entity_subtype=entity_subtype,
            canonical_name=canonical_name,
            attributes=entity_attrs,
            overall_confidence=clamped_confidence,
            mention_count=1,
            embedding=embedding,
            first_mentioned=now,
            last_mentioned=now,
        )
        db.add(entity)
        await db.flush()
        return entity

    async def upsert_relationship(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        subject_entity_id: uuid.UUID,
        predicate: str,
        object_entity_id: uuid.UUID,
        *,
        confidence: float = 0.5,
        source_msg_ids: list[uuid.UUID] | None = None,
    ) -> STMRelationship:
        """Insert or update a relationship. Dedup by (chat_id, subject, predicate, object)."""

        stmt = (
            select(STMRelationship)
            .where(
                STMRelationship.chat_id == chat_id,
                STMRelationship.subject_entity_id == subject_entity_id,
                STMRelationship.predicate == predicate,
                STMRelationship.object_entity_id == object_entity_id,
            )
        )
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.confidence = confidence
            if source_msg_ids:
                existing.source_msg_ids = list(set(existing.source_msg_ids or []) | set(source_msg_ids))
            await db.flush()
            return existing

        rel = STMRelationship(
            chat_id=chat_id,
            subject_entity_id=subject_entity_id,
            predicate=predicate,
            object_entity_id=object_entity_id,
            confidence=confidence,
            source_msg_ids=source_msg_ids or [],
        )
        db.add(rel)
        await db.flush()
        return rel

    async def insert_recap(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        recap_text: str,
        keywords: list[str],
        *,
        confidence: float = 0.5,
        entity_ids: list[uuid.UUID] | None = None,
        relationship_ids: list[uuid.UUID] | None = None,
        start_msg_id: uuid.UUID | None = None,
        end_msg_id: uuid.UUID | None = None,
    ) -> STMRecap:
        embedding = await self._generate_embedding(recap_text)

        recap = STMRecap(
            chat_id=chat_id,
            recap_text=recap_text,
            keywords=keywords,
            embedding=embedding,
            confidence=confidence,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            start_msg_id=start_msg_id,
            end_msg_id=end_msg_id,
        )
        db.add(recap)
        await db.flush()
        return recap

    async def get_entities(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        min_confidence: float | None = None,
    ) -> list[STMEntity]:
        threshold = min_confidence if min_confidence is not None else self._min_confidence
        stmt = (
            select(STMEntity)
            .where(
                STMEntity.chat_id == chat_id,
                STMEntity.overall_confidence >= threshold,
            )
            .order_by(STMEntity.overall_confidence.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_relationships(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        min_confidence: float | None = None,
    ) -> list[STMRelationship]:
        threshold = min_confidence if min_confidence is not None else self._min_confidence
        stmt = (
            select(STMRelationship)
            .where(
                STMRelationship.chat_id == chat_id,
                STMRelationship.confidence >= threshold,
            )
            .order_by(STMRelationship.confidence.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_recaps(
        self,
        db: AsyncSession,
        chat_id: uuid.UUID,
        limit: int = 10,
    ) -> list[STMRecap]:
        stmt = (
            select(STMRecap)
            .where(STMRecap.chat_id == chat_id)
            .order_by(STMRecap.created_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())
