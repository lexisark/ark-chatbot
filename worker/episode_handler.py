"""Episode generation handler — orchestrates LTM episode creation."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from context_engine.ltm_manager import LTMManager
from providers.base import ChatProvider, EmbeddingProvider

logger = logging.getLogger(__name__)


async def run_episode_generation(
    db: AsyncSession,
    chat_provider: ChatProvider,
    embedding_provider: EmbeddingProvider,
    chat_id: uuid.UUID,
    scope_id: str,
) -> None:
    """Generate LTM episode and promote entities/relationships."""
    ltm = LTMManager()

    # Generate episode
    episode = await ltm.generate_episode(db, chat_provider, embedding_provider, chat_id, scope_id)
    if episode is None:
        logger.debug(f"No episode generated for chat {chat_id}")
        return

    # Promote STM → LTM
    entities_promoted = await ltm.promote_entities(db, chat_id, scope_id)
    rels_promoted = await ltm.promote_relationships(db, chat_id, scope_id)

    # Apply decay to old episodes
    await ltm.apply_importance_decay(db, scope_id)

    await db.flush()
    logger.info(
        f"Episode generation complete for chat {chat_id}: "
        f"promoted {entities_promoted} entities, {rels_promoted} relationships"
    )
