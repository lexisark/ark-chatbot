"""Pydantic request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── Chat ────────────────────────────────────────────────


class ChatCreate(BaseModel):
    title: str | None = None
    system_prompt: str | None = None
    scope_id: str | None = None
    metadata: dict | None = None


class ChatUpdate(BaseModel):
    title: str | None = None
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    id: uuid.UUID
    title: str | None
    system_prompt: str | None
    scope_id: str | None
    metadata: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Message ─────────────────────────────────────────────


class MessageSend(BaseModel):
    content: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    role: str
    content: str
    token_count: int | None
    created_at: datetime

    model_config = {"from_attributes": True}
