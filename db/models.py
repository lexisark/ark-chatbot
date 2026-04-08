"""SQLAlchemy 2.0 models for the context engine."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector

from app.config import settings

_EMBEDDING_DIM = settings.embedding_dimensions
from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ── Chat & Messages ─────────────────────────────────────


class Chat(Base):
    __tablename__ = "chat"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    messages: Mapped[list[Message]] = relationship("Message", back_populates="chat", cascade="all, delete-orphan", passive_deletes=True)
    stm_entities: Mapped[list[STMEntity]] = relationship("STMEntity", back_populates="chat", cascade="all, delete-orphan", passive_deletes=True)
    stm_relationships: Mapped[list[STMRelationship]] = relationship("STMRelationship", back_populates="chat", cascade="all, delete-orphan", passive_deletes=True)
    stm_recaps: Mapped[list[STMRecap]] = relationship("STMRecap", back_populates="chat", cascade="all, delete-orphan", passive_deletes=True)


class Message(Base):
    __tablename__ = "message"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, name="message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chat: Mapped[Chat] = relationship("Chat", back_populates="messages")


# ── Short-Term Memory (Chat-Scoped) ────────────────────


class STMEntity(Base):
    __tablename__ = "stm_entity"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_subtype: Mapped[str | None] = mapped_column(String(50), nullable=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    embedding = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    first_mentioned: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_mentioned: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    chat: Mapped[Chat] = relationship("Chat", back_populates="stm_entities")


class STMRelationship(Base):
    __tablename__ = "stm_relationship"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id", ondelete="CASCADE"), nullable=False, index=True)
    subject_entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("stm_entity.id", ondelete="CASCADE"), nullable=False)
    predicate: Mapped[str] = mapped_column(String(100), nullable=False)
    object_entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("stm_entity.id", ondelete="CASCADE"), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    last_mentioned: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    source_msg_ids: Mapped[list] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chat: Mapped[Chat] = relationship("Chat", back_populates="stm_relationships")


class STMRecap(Base):
    __tablename__ = "stm_recap"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat.id", ondelete="CASCADE"), nullable=False, index=True)
    recap_text: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[list] = mapped_column(ARRAY(String), default=list)
    entity_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=True)
    relationship_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=True)
    start_msg_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("message.id"), nullable=True)
    end_msg_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("message.id"), nullable=True)
    embedding = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chat: Mapped[Chat] = relationship("Chat", back_populates="stm_recaps")


# ── Long-Term Memory (Cross-Chat, scoped by scope_id) ──


class LTMEpisode(Base):
    __tablename__ = "ltm_episode"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    episode_summary: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[list] = mapped_column(ARRAY(String), default=list)
    embedding = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    importance_score: Mapped[float] = mapped_column(Float, default=0.5)
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    emotional_tone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    episode_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    source_chat_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class LTMEntity(Base):
    __tablename__ = "ltm_entity"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_subtype: Mapped[str | None] = mapped_column(String(50), nullable=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict, server_default="{}")
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    embedding = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class LTMRelationship(Base):
    __tablename__ = "ltm_relationship"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    subject_entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ltm_entity.id", ondelete="CASCADE"), nullable=False)
    predicate: Mapped[str] = mapped_column(String(100), nullable=False)
    object_entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ltm_entity.id", ondelete="CASCADE"), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
