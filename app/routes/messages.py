"""Message endpoints."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.schemas import MessageResponse, MessageSend
from db import queries
from db.models import MessageRole

router = APIRouter(prefix="/api/chats/{chat_id}", tags=["messages"])


@router.post("/messages", response_model=MessageResponse)
async def send_message(
    chat_id: uuid.UUID,
    body: MessageSend,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    # Verify chat exists
    chat = await queries.get_chat(db, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save user message (commit immediately so it's durable even if LLM fails)
    await queries.create_message(db, chat_id, MessageRole.USER, body.content)
    await db.commit()

    # Build context with token budgeting
    from context_engine.builder import ContextBuilder
    from app.config import settings as app_settings

    token_counter = getattr(request.app.state, "token_counter", None)
    if token_counter is None:
        from providers.token_counter import TiktokenCounter
        token_counter = TiktokenCounter()
    embedding_service = getattr(request.app.state, "embedding_service", None)
    builder = ContextBuilder(token_counter=token_counter, embedding_service=embedding_service)
    context = await builder.build_context(db, chat_id, budget_tokens=app_settings.builder_total_budget, current_message=body.content)
    system_instruction, llm_messages = builder.format_for_llm(context)

    # Call LLM via provider on app state
    chat_provider = request.app.state.chat_provider
    response = await chat_provider.chat(
        messages=llm_messages,
        system_prompt=system_instruction,
    )

    # Save assistant message
    assistant_msg = await queries.create_message(
        db, chat_id, MessageRole.ASSISTANT, response.content, token_count=response.tokens_out,
    )
    await db.commit()

    # Trigger extraction if interval reached
    from app.config import settings
    user_msg_count = await queries.count_user_messages(db, chat_id)
    if user_msg_count > 0 and user_msg_count % settings.recap_interval_messages == 0:
        from context_engine.stm_manager import STMManager
        from worker.extraction_handler import run_batch_extraction
        from worker.in_process import InProcessQueue

        queue = InProcessQueue()
        stm = STMManager()

        async def _extract():
            from db.session import async_session_factory
            async with async_session_factory() as extract_db:
                await run_batch_extraction(extract_db, chat_provider, stm, chat_id)
                await extract_db.commit()

        await queue.enqueue("batch_extraction", _extract)

    return MessageResponse(
        id=assistant_msg.id,
        chat_id=assistant_msg.chat_id,
        role=assistant_msg.role.value,
        content=assistant_msg.content,
        token_count=assistant_msg.token_count,
        created_at=assistant_msg.created_at,
    )


@router.post("/messages/stream")
async def send_message_stream(
    chat_id: uuid.UUID,
    body: MessageSend,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Send a message and stream the response via SSE."""
    chat = await queries.get_chat(db, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Save user message (commit immediately so it's durable even if LLM fails)
    await queries.create_message(db, chat_id, MessageRole.USER, body.content)
    await db.commit()

    # Build context
    from context_engine.builder import ContextBuilder
    from providers.token_counter import TiktokenCounter

    embedding_service = getattr(request.app.state, "embedding_service", None)
    builder = ContextBuilder(token_counter=TiktokenCounter(), embedding_service=embedding_service)
    context = await builder.build_context(db, chat_id, current_message=body.content)
    system_instruction, llm_messages = builder.format_for_llm(context)

    chat_provider = request.app.state.chat_provider

    async def event_stream():
        full_text = ""
        try:
            async for chunk in chat_provider.chat_stream(
                messages=llm_messages,
                system_prompt=system_instruction,
            ):
                if chunk.done:
                    break
                full_text += chunk.delta
                yield f"data: {json.dumps({'delta': chunk.delta})}\n\n"

            # Save assistant message
            assistant_msg = await queries.create_message(
                db, chat_id, MessageRole.ASSISTANT, full_text,
            )
            await db.commit()

            yield f"data: {json.dumps({'done': True, 'message_id': str(assistant_msg.id)})}\n\n"

            # Trigger extraction if interval reached
            from app.config import settings
            user_msg_count = await queries.count_user_messages(db, chat_id)
            if user_msg_count > 0 and user_msg_count % settings.recap_interval_messages == 0:
                from context_engine.stm_manager import STMManager
                from worker.extraction_handler import run_batch_extraction
                from worker.in_process import InProcessQueue

                queue = InProcessQueue()
                stm = STMManager()

                async def _extract():
                    from db.session import async_session_factory
                    async with async_session_factory() as extract_db:
                        await run_batch_extraction(extract_db, chat_provider, stm, chat_id)
                        await extract_db.commit()

                await queue.enqueue("batch_extraction", _extract)

        except Exception as e:
            # Save partial response on error
            if full_text:
                await queries.create_message(db, chat_id, MessageRole.ASSISTANT, full_text)
                await db.commit()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/messages", response_model=list[MessageResponse])
async def get_messages(
    chat_id: uuid.UUID,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    messages = await queries.get_chat_messages(db, chat_id, limit=limit)
    return [
        MessageResponse(
            id=m.id,
            chat_id=m.chat_id,
            role=m.role.value,
            content=m.content,
            token_count=m.token_count,
            created_at=m.created_at,
        )
        for m in messages
    ]
