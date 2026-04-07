"""Context builder — assembles token-budgeted context for LLM inference."""

from __future__ import annotations

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from context_engine.config import BUILDER_CONFIG
from context_engine.models import ContextAssemblyResult
from context_engine.tokens import TokenHelper
from db.queries import get_chat, get_chat_messages
from providers.base import TokenCounter

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Assembles conversation context with token budgeting.

    Budget allocation:
        - System prompt: 10% (configurable)
        - Recent messages: 50% (configurable)
        - RAG memories: 40% (configurable, placeholder in Phase 2)
    """

    def __init__(self, token_counter: TokenCounter, embedding_service=None):
        self.tokens = TokenHelper(token_counter)
        self._embedding_service = embedding_service

    async def build_context(
        self,
        db: AsyncSession,
        chat_id,
        budget_tokens: int = 4000,
        current_message: str = "",
        enable_rag: bool | None = None,
    ) -> ContextAssemblyResult:
        start_time = time.time()
        result = ContextAssemblyResult()

        chat = await get_chat(db, chat_id)
        if chat is None:
            result.assembly_time_ms = int((time.time() - start_time) * 1000)
            return result

        # Step 1: System prompt (10% of budget, hard cap at persona_max_tokens)
        prompt_budget = min(
            BUILDER_CONFIG["persona_max_tokens"],
            int(budget_tokens * BUILDER_CONFIG["persona_budget_ratio"]),
        )

        if chat.system_prompt:
            result.system_prompt = self.tokens.truncate(chat.system_prompt, prompt_budget)
            result.system_prompt_tokens = self.tokens.count(result.system_prompt)
        else:
            result.system_prompt = ""
            result.system_prompt_tokens = 0

        # Step 2: Recent messages (50% of budget)
        recent_budget = int(budget_tokens * BUILDER_CONFIG["recent_messages_budget_ratio"])
        messages = await get_chat_messages(db, chat_id, limit=100)

        llm_messages = [{"role": m.role.value, "content": m.content} for m in messages]
        result.recent_messages = self.tokens.fit_messages_to_budget(llm_messages, recent_budget)
        result.recent_tokens = self.tokens.count_messages(result.recent_messages) if result.recent_messages else 0

        # Step 3: RAG memories (40% of budget)
        use_rag = enable_rag if enable_rag is not None else True
        if use_rag and current_message:
            from context_engine.rag_manager import RAGManager
            from db.queries import count_user_messages

            rag_budget = int(budget_tokens * BUILDER_CONFIG["rag_budget_ratio"])
            rag = RAGManager(self.tokens)

            # Generate query embedding once for all RAG search tiers
            query_embedding = None
            if self._embedding_service:
                try:
                    query_embedding = await self._embedding_service.generate_query_embedding(current_message)
                except Exception:
                    logger.warning("Failed to generate query embedding", exc_info=True)

            turn_count = await count_user_messages(db, chat_id)
            you_remember = await rag.build_you_remember(
                db, chat_id, current_message, rag_budget,
                scope_id=chat.scope_id,
                turn_count=turn_count,
                query_embedding=query_embedding,
            )

            # Format into text block
            parts = []
            if you_remember.entities:
                parts.append("Entities you know about:")
                parts.extend(you_remember.entities)
            if you_remember.relationships:
                parts.append("Relationships:")
                parts.extend(you_remember.relationships)
            if you_remember.episodes:
                parts.append("Past conversations:")
                parts.extend(you_remember.episodes)
            if you_remember.recaps:
                parts.append("Recent context:")
                parts.extend(you_remember.recaps)

            result.memories_text = "\n".join(parts)
            result.memories_tokens = you_remember.total_tokens
        else:
            result.memories_text = ""
            result.memories_tokens = 0

        # Totals
        result.total_tokens = result.system_prompt_tokens + result.recent_tokens + result.memories_tokens
        result.assembly_time_ms = int((time.time() - start_time) * 1000)

        logger.debug(
            f"Context assembly: {result.total_tokens}/{budget_tokens} tokens "
            f"(prompt={result.system_prompt_tokens}, msgs={result.recent_tokens}, mem={result.memories_tokens}) "
            f"in {result.assembly_time_ms}ms"
        )

        return result

    def format_for_llm(self, result: ContextAssemblyResult) -> tuple[str, list[dict[str, str]]]:
        """Format assembled context into system_instruction + messages.

        Returns:
            (system_instruction, messages) — system_instruction is passed to the LLM
            separately from the conversation messages.
        """
        parts = []

        if result.system_prompt:
            parts.append(result.system_prompt)

        if result.memories_text:
            parts.append(
                "MEMORIES:\n"
                "What you remember about this user from past conversations:\n\n"
                + result.memories_text
            )

        system_instruction = "\n\n".join(parts)
        result.system_instruction = system_instruction

        return system_instruction, list(result.recent_messages)
