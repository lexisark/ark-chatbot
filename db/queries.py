"""Database query functions."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Chat, Message, MessageRole


# ── Chat Queries ────────────────────────────────────────


async def create_chat(
    db: AsyncSession,
    *,
    title: str | None = None,
    system_prompt: str | None = None,
    scope_id: str | None = None,
    metadata: dict | None = None,
) -> Chat:
    chat = Chat(
        title=title,
        system_prompt=system_prompt,
        scope_id=scope_id,
        metadata_=metadata or {},
    )
    db.add(chat)
    await db.flush()
    return chat


async def get_chat(db: AsyncSession, chat_id: uuid.UUID) -> Chat | None:
    return await db.get(Chat, chat_id)


async def list_chats(
    db: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    scope_id: str | None = None,
) -> list[Chat]:
    stmt = select(Chat).order_by(Chat.created_at.desc()).limit(limit).offset(offset)
    if scope_id is not None:
        stmt = stmt.where(Chat.scope_id == scope_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_chat(
    db: AsyncSession,
    chat_id: uuid.UUID,
    *,
    title: str | None = None,
    system_prompt: str | None = None,
) -> Chat | None:
    chat = await db.get(Chat, chat_id)
    if chat is None:
        return None
    if title is not None:
        chat.title = title
    if system_prompt is not None:
        chat.system_prompt = system_prompt
    await db.flush()
    return chat


async def delete_chat(db: AsyncSession, chat_id: uuid.UUID) -> bool:
    chat = await db.get(Chat, chat_id)
    if chat is None:
        return False
    await db.delete(chat)
    await db.flush()
    return True


# ── Message Queries ─────────────────────────────────────


async def create_message(
    db: AsyncSession,
    chat_id: uuid.UUID,
    role: MessageRole,
    content: str,
    token_count: int | None = None,
) -> Message:
    msg = Message(
        chat_id=chat_id,
        role=role,
        content=content,
        token_count=token_count,
    )
    db.add(msg)
    await db.flush()
    return msg


async def get_chat_messages(
    db: AsyncSession,
    chat_id: uuid.UUID,
    *,
    limit: int = 100,
) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def count_user_messages(db: AsyncSession, chat_id: uuid.UUID) -> int:
    stmt = (
        select(func.count())
        .select_from(Message)
        .where(Message.chat_id == chat_id, Message.role == MessageRole.USER)
    )
    result = await db.execute(stmt)
    return result.scalar_one()
