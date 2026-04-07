"""Batch extraction handler — extracts entities, relationships, and recaps from messages."""

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
    """Run batch extraction on recent messages in a chat.

    1. Load recent messages
    2. Call LLM with extraction prompt (low temp)
    3. Parse structured output
    4. Upsert entities/relationships via STMManager
    5. Insert recap
    """
    messages = await get_chat_messages(db, chat_id, limit=100)
    if not messages:
        return

    llm_messages = [{"role": m.role.value, "content": m.content} for m in messages]

    # Build extraction prompt
    prompt = build_extraction_prompt(llm_messages)

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

    # Upsert entities — build a name→id map for relationship linking
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

        if not subject_id or not object_id:
            # Create placeholder entities if not found
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
