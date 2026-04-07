"""Batch extraction handler — extracts entities, relationships, and recaps.

Loads existing STM memory before calling the LLM so the extraction prompt
can deduplicate against known entities and focus on NEW information.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from context_engine.extraction import build_extraction_prompt, parse_extraction_response
from context_engine.stm_manager import STMManager
from db.queries import get_chat_messages
from providers.base import ChatProvider

logger = logging.getLogger(__name__)


async def run_batch_extraction(
    db: AsyncSession,
    chat_provider: ChatProvider,
    stm: STMManager,
    chat_id: uuid.UUID,
) -> None:
    """Run batch extraction on recent messages with existing memory context.

    1. Load existing STM entities/relationships for dedup context
    2. Load recent messages
    3. Call LLM with extraction prompt (includes existing memory)
    4. Parse structured output
    5. Upsert entities/relationships via STMManager
    6. Insert recap
    """
    messages = await get_chat_messages(db, chat_id, limit=100)
    if not messages:
        return

    llm_messages = [{"role": m.role.value, "content": m.content} for m in messages]

    # Load existing STM for dedup context
    existing_entities = await _load_existing_entities(stm, db, chat_id)
    existing_relationships = await _load_existing_relationships(stm, db, chat_id)

    # Build extraction prompt with memory context
    prompt = build_extraction_prompt(
        llm_messages,
        existing_entities=existing_entities,
        existing_relationships=existing_relationships,
    )

    # Call LLM
    response = await chat_provider.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )

    # Parse response
    extracted = parse_extraction_response(response.content)

    if not extracted.entities and not extracted.relationships and not extracted.recap:
        logger.debug(f"No data extracted for chat {chat_id}")
        return

    # Deduplicate extracted entities before upserting
    from context_engine.dedup import deduplicate_entities
    deduped_entity_dicts = deduplicate_entities([
        {
            "type": e.entity_type, "subtype": e.entity_subtype,
            "canonical_name": e.canonical_name, "attributes": e.attributes,
            "attribute_confidence": e.attribute_confidence,
            "overall_confidence": e.confidence, "tags": e.tags,
        }
        for e in extracted.entities
    ])
    # Rebuild entity list from deduped dicts
    from context_engine.extraction import ExtractedEntity
    extracted.entities = [
        ExtractedEntity(
            entity_type=d["type"], canonical_name=d["canonical_name"],
            entity_subtype=d.get("subtype"), attributes=d.get("attributes", {}),
            attribute_confidence=d.get("attribute_confidence", {}),
            confidence=d.get("overall_confidence", 0.5), tags=d.get("tags", []),
        )
        for d in deduped_entity_dicts
    ]

    # Upsert entities — build name→id map for relationship linking
    entity_map: dict[str, uuid.UUID] = {}

    for e in extracted.entities:
        entity = await stm.upsert_entity(
            db, chat_id,
            entity_type=e.entity_type,
            canonical_name=e.canonical_name,
            entity_subtype=e.entity_subtype,
            attributes=e.attributes,
            confidence=e.confidence,
        )
        entity_map[e.canonical_name.lower()] = entity.id

    # Upsert relationships — resolve entity names to IDs
    for r in extracted.relationships:
        subject_id = entity_map.get(r.subject.lower())
        object_id = entity_map.get(r.object_name.lower())

        if not subject_id:
            subj_entity = await stm.upsert_entity(
                db, chat_id, entity_type="person", canonical_name=r.subject, confidence=0.5,
            )
            subject_id = subj_entity.id
            entity_map[r.subject.lower()] = subject_id

        if not object_id:
            obj_entity = await stm.upsert_entity(
                db, chat_id, entity_type="other", canonical_name=r.object_name, confidence=0.5,
            )
            object_id = obj_entity.id
            entity_map[r.object_name.lower()] = object_id

        await stm.upsert_relationship(
            db, chat_id,
            subject_entity_id=subject_id,
            predicate=r.predicate,
            object_entity_id=object_id,
            confidence=r.confidence,
        )

    # Insert recap
    if extracted.recap:
        entity_ids = list(entity_map.values())
        await stm.insert_recap(
            db, chat_id,
            recap_text=extracted.recap.recap_text,
            keywords=extracted.recap.keywords,
            confidence=extracted.recap.confidence,
            entity_ids=entity_ids,
        )

    await db.flush()
    logger.info(
        f"Extraction complete for chat {chat_id}: "
        f"{len(extracted.entities)} entities, {len(extracted.relationships)} relationships"
    )


async def _load_existing_entities(
    stm: STMManager, db: AsyncSession, chat_id: uuid.UUID,
) -> list[dict]:
    """Load existing STM entities formatted for the extraction prompt."""
    entities = await stm.get_entities(db, chat_id, min_confidence=0.0)
    return [
        {
            "canonical_name": e.canonical_name,
            "entity_type": e.entity_type,
            "entity_subtype": e.entity_subtype,
            "attributes": e.attributes,
            "confidence": e.overall_confidence,
        }
        for e in entities
    ]


async def _load_existing_relationships(
    stm: STMManager, db: AsyncSession, chat_id: uuid.UUID,
) -> list[dict]:
    """Load existing STM relationships formatted for the extraction prompt."""
    from db.models import STMEntity

    relationships = await stm.get_relationships(db, chat_id, min_confidence=0.0)
    result = []
    for r in relationships:
        subj = await db.get(STMEntity, r.subject_entity_id)
        obj = await db.get(STMEntity, r.object_entity_id)
        if subj and obj:
            result.append({
                "subject": subj.canonical_name,
                "predicate": r.predicate,
                "object_name": obj.canonical_name,
                "confidence": r.confidence,
            })
    return result
