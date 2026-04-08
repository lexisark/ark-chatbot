"""Chat CRUD endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas import ChatCreate, ChatResponse, ChatUpdate
from db import queries

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chats", tags=["chats"])


@router.post("", status_code=201, response_model=ChatResponse)
async def create_chat(body: ChatCreate, request: Request, db: AsyncSession = Depends(get_db)):
    # Before creating new chat, trigger episode for previous chat in same scope
    if body.scope_id:
        await _trigger_episode_for_previous_chat(request, db, body.scope_id)

    chat = await queries.create_chat(
        db,
        title=body.title,
        system_prompt=body.system_prompt,
        scope_id=body.scope_id,
        metadata=body.metadata,
    )
    await db.commit()
    return _to_response(chat)


@router.get("", response_model=list[ChatResponse])
async def list_chats(
    limit: int = 50,
    offset: int = 0,
    scope_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    chats = await queries.list_chats(db, limit=limit, offset=offset, scope_id=scope_id)
    return [_to_response(c) for c in chats]


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    chat = await queries.get_chat(db, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return _to_response(chat)


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(chat_id: uuid.UUID, body: ChatUpdate, db: AsyncSession = Depends(get_db)):
    chat = await queries.update_chat(
        db, chat_id, title=body.title, system_prompt=body.system_prompt,
    )
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    await db.commit()
    return _to_response(chat)


@router.delete("/{chat_id}", status_code=204)
async def delete_chat(chat_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    deleted = await queries.delete_chat(db, chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    await db.commit()


@router.post("/{chat_id}/complete")
async def complete_chat(
    chat_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Mark a chat as complete and generate an LTM episode."""
    chat = await queries.get_chat(db, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    from app.config import settings

    # Check minimum message count
    user_msg_count = await queries.count_user_messages(db, chat_id)
    if user_msg_count < settings.final_episode_min_messages:
        return {"episode_generated": False, "reason": f"Need at least {settings.final_episode_min_messages} user messages"}

    # Generate episode
    chat_provider = getattr(request.app.state, "chat_provider", None)
    embedding_service = getattr(request.app.state, "embedding_service", None)
    generated = await _generate_episode_for_chat(chat_provider, embedding_service, db, chat)
    return {"episode_generated": generated}


async def _generate_episode_for_chat(chat_provider, embedding_service, db: AsyncSession, chat) -> bool:
    """Generate an LTM episode for a chat."""
    if not chat_provider or not embedding_service or not chat.scope_id:
        return False

    from context_engine.ltm_manager import LTMManager

    ltm = LTMManager()
    episode = await ltm.generate_episode(
        db, chat_provider, embedding_service._provider, chat.id, chat.scope_id,
    )
    if episode:
        await ltm.promote_entities(db, chat.id, chat.scope_id)
        await ltm.promote_relationships(db, chat.id, chat.scope_id)
        # Note: importance decay is computed at query time (RAG manager),
        # not persisted, to avoid compounding on repeated calls.
        await db.commit()
        logger.info(f"Episode generated for chat {chat.id} in scope {chat.scope_id}")
        return True
    return False


async def _trigger_episode_for_previous_chat(
    request: Request, db: AsyncSession, scope_id: str,
) -> None:
    """When creating a new chat, generate episode for the previous chat in the same scope."""
    from app.config import settings

    # Find the most recent chat in this scope
    previous_chats = await queries.list_chats(db, limit=1, scope_id=scope_id)
    if not previous_chats:
        return

    prev_chat = previous_chats[0]

    # Check minimum messages
    user_msg_count = await queries.count_user_messages(db, prev_chat.id)
    if user_msg_count < settings.final_episode_min_messages:
        logger.debug(f"Previous chat {prev_chat.id} has {user_msg_count} user msgs, need {settings.final_episode_min_messages}")
        return

    # Check if episode already exists for this chat
    from sqlalchemy import select
    from db.models import LTMEpisode
    existing = await db.execute(
        select(LTMEpisode).where(
            LTMEpisode.source_chat_id == prev_chat.id,
            LTMEpisode.is_final == True,
        ).limit(1)
    )
    if existing.scalar_one_or_none():
        return  # Already has a final episode

    # Capture providers from app state before async task
    chat_provider = getattr(request.app.state, "chat_provider", None)
    embedding_service = getattr(request.app.state, "embedding_service", None)

    if not chat_provider or not embedding_service:
        logger.warning("Cannot generate episode: missing chat_provider or embedding_service on app state")
        return

    prev_chat_id = prev_chat.id

    # Generate episode async
    from worker.in_process import InProcessQueue

    queue = InProcessQueue()

    async def _gen():
        from db.session import async_session_factory
        async with async_session_factory() as episode_db:
            prev = await queries.get_chat(episode_db, prev_chat_id)
            if prev:
                await _generate_episode_for_chat(chat_provider, embedding_service, episode_db, prev)

    await queue.enqueue("episode_generation", _gen)
    logger.info(f"Queued episode generation for previous chat {prev_chat_id} in scope {scope_id}")


def _to_response(chat) -> ChatResponse:
    return ChatResponse(
        id=chat.id,
        title=chat.title,
        system_prompt=chat.system_prompt,
        scope_id=chat.scope_id,
        metadata=chat.metadata_,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )
