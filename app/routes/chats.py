"""Chat CRUD endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas import ChatCreate, ChatResponse, ChatUpdate
from db import queries

router = APIRouter(prefix="/api/chats", tags=["chats"])


@router.post("", status_code=201, response_model=ChatResponse)
async def create_chat(body: ChatCreate, db: AsyncSession = Depends(get_db)):
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
