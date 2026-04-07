# Ark Chatbot Context Engine

**Open-source conversational memory engine extracted from [Arkadia](https://arkadia.chat)**

A production-tested context engine that gives any LLM-powered chatbot persistent memory across conversations. It manages short-term memory (within a chat), long-term memory (across chats), and intelligent context assembly — so your chatbot remembers what matters without blowing through token budgets.

---

## What This Is

A standalone Python backend that sits between your frontend and any LLM provider. You send messages, define system prompts, and the context engine handles:

- **Token-budgeted context assembly** — dynamically allocates tokens across system prompt, conversation history, and memories
- **Short-term memory (STM)** — extracts entities, relationships, and recaps within a chat session
- **Long-term memory (LTM)** — promotes important memories across chat sessions via episode generation
- **Hybrid RAG retrieval** — combines full-text search (70%) and vector similarity (30%) to surface relevant memories
- **Async memory processing** — extraction and episode generation happen in background workers, keeping chat responses fast

**What this is NOT:** a full chat application. No auth, no user management, no frontend, no character system. This is the engine — bring your own UI and auth layer.

---

## Origin

This context engine powers [Arkadia](https://arkadia.chat), where AI animal characters maintain persistent memory across thousands of conversations. The system achieves:

- 91-94% memory recall accuracy
- 99.4% entity extraction accuracy
- ~89ms P95 context assembly time
- 94% cross-chat continuity

We stripped out the character/persona system, auth, and Arkadia-specific features to release the core memory engine as a reusable building block.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Your Frontend                       │
└────────────────────────┬────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI Server                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Chat CRUD   │  │   Message    │  │   System     │  │
│  │  Endpoints   │  │   Streaming  │  │   Prompts    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│  ┌──────▼─────────────────▼──────────────────▼───────┐  │
│  │              Context Engine                       │  │
│  │                                                   │  │
│  │  ┌───────────┐ ┌───────────┐ ┌─────────────────┐  │  │
│  │  │  Builder  │ │    RAG    │ │  Token Budgeter │  │  │
│  │  │ (assembly)│ │ (retrieval)│ │  (allocation)   │  │  │
│  │  └───────────┘ └───────────┘ └─────────────────┘  │  │
│  │                                                   │  │
│  │  ┌───────────┐ ┌───────────┐ ┌─────────────────┐  │  │
│  │  │    STM    │ │    LTM    │ │   Embedding     │  │  │
│  │  │ (in-chat) │ │(cross-chat)│ │   Service       │  │  │
│  │  └───────────┘ └───────────┘ └─────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Background Worker                      │   │
│  │  (batch extraction + episode generation)          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   PostgreSQL + pgvector         │
        │   (FTS + vector embeddings)     │
        └─────────────────────────────────┘
```

---

## Core Concepts

### Three Processing Flows

| Flow | Timing | What Happens |
|------|--------|--------------|
| **Chat Response** | Synchronous (~1.8s E2E) | User message → context assembly → LLM → streamed response |
| **Batch Extraction** | Async (~3-5s) | Every N messages: extract entities + relationships + recap |
| **Episode Generation** | Async (~8-15s) | On session return or chat end: create LTM episode narrative |

### Token Budget Allocation

The context engine splits a configurable token budget (default 4000) across three areas:

```
┌─────────────────────────────────────────────────┐
│              Total Budget (4000 tokens)          │
├──────────┬────────────────────┬─────────────────┤
│ System   │  Recent Messages   │  RAG/Memories   │
│ Prompt   │                    │                 │
│  10%     │       50%          │      40%        │
│ (400)    │     (2000)         │    (1600)       │
└──────────┴────────────────────┴─────────────────┘
```

Budget ratios are fully configurable via environment variables.

### Memory Tiers

#### Short-Term Memory (STM) — Chat-Scoped

Extracted every N user messages (default: 5) via a background worker:

- **Entities** — structured objects: people, places, things, events
  - Tracked with: `entity_type`, `canonical_name`, `attributes` (JSONB), `confidence`, `mention_count`
  - Confidence reinforcement: +0.20 boost on re-mention, capped at 0.95
- **Relationships** — subject-predicate-object triples between entities
  - Example: `User → owns → Max (dog)`
- **Recaps** — 2-4 sentence summaries of conversation chunks with keywords for FTS

#### Long-Term Memory (LTM) — Cross-Chat

Generated on chat completion or session return (30+ min gap):

- **Episodes** — rich narrative summaries (~200-300 tokens) with:
  - Vector embeddings (768-dim) for semantic search
  - Importance scores (LLM-rated 0-1, decays over time)
  - Keywords for FTS recall
- **Promoted Entities/Relationships** — STM entities that persist across chats, deduplicated

### Hybrid RAG Retrieval

When assembling context, the RAG manager searches memories using:

- **Full-Text Search (70% weight)** — keyword matching on entity names, recap keywords, episode summaries
- **Vector Similarity (30% weight)** — semantic matching via embeddings, cosine distance threshold <0.6
- **Dynamic episode budget** — allocates more cross-chat memory early in conversation, fades as in-chat context grows:

| Turn Count | Episode Budget | STM Budget |
|------------|---------------|------------|
| 1-2        | 25%           | 75%        |
| 3-5        | 15%           | 85%        |
| 6-19       | 10%           | 90%        |
| 20+        | 10%           | 90%        |

---

## Data Model

### Core Tables

```
┌───────────────┐       ┌───────────────┐
│     chat      │──────<│    message     │
│               │       │               │
│ id (PK)       │       │ id (PK)       │
│ system_prompt │       │ chat_id (FK)  │
│ title         │       │ role          │
│ created_at    │       │ content       │
│ updated_at    │       │ token_count   │
│ metadata      │       │ created_at    │
└───────┬───────┘       └───────────────┘
        │
        │ chat_id (FK)
        │
┌───────▼───────┐  ┌─────────────────┐  ┌───────────────┐
│  stm_entity   │──│ stm_relationship│  │   stm_recap   │
│               │  │                 │  │               │
│ id (PK)       │  │ id (PK)        │  │ id (PK)       │
│ chat_id (FK)  │  │ chat_id (FK)   │  │ chat_id (FK)  │
│ entity_type   │  │ subject_id (FK)│  │ recap_text    │
│ entity_subtype│  │ predicate      │  │ keywords[]    │
│ canonical_name│  │ object_id (FK) │  │ start_msg_id  │
│ attributes    │  │ confidence     │  │ end_msg_id    │
│ confidence    │  │ source_msg_ids │  │ confidence    │
│ mention_count │  │ created_at     │  │ created_at    │
│ created_at    │  └─────────────────┘  └───────────────┘
│ updated_at    │
└───────────────┘

        Cross-chat scope (keyed by chat.metadata or user-defined scope)

┌───────────────┐  ┌─────────────────┐  ┌───────────────────┐
│  ltm_entity   │──│ ltm_relationship│  │   ltm_episode     │
│               │  │                 │  │                   │
│ id (PK)       │  │ id (PK)        │  │ id (PK)           │
│ scope_id      │  │ scope_id       │  │ scope_id          │
│ entity_type   │  │ subject_id (FK)│  │ episode_summary   │
│ canonical_name│  │ predicate      │  │ keywords[]        │
│ attributes    │  │ object_id (FK) │  │ embedding (768d)  │
│ confidence    │  │ confidence     │  │ importance_score   │
│ mention_count │  │ created_at     │  │ is_final          │
│ created_at    │  └─────────────────┘  │ emotional_tone    │
│ updated_at    │                       │ episode_date      │
└───────────────┘                       │ created_at        │
                                        └───────────────────┘
```

**Key design note:** In Arkadia, LTM is scoped by `(user_id, character_id)`. In the open-source version, we use a generic `scope_id` — you decide what defines a memory scope (per user, per assistant, per project, etc.).

---

## API Design

### Chats

```
POST   /api/chats                    Create a new chat (with optional system prompt)
GET    /api/chats                    List chats (with pagination)
GET    /api/chats/{chat_id}          Get chat details
PATCH  /api/chats/{chat_id}          Update chat (title, system prompt)
DELETE /api/chats/{chat_id}          Delete chat (cascades STM cleanup)
```

### Messages

```
POST   /api/chats/{chat_id}/messages          Send message, get response
POST   /api/chats/{chat_id}/messages/stream   Send message, stream response (SSE)
GET    /api/chats/{chat_id}/messages          Get message history (paginated)
```

### System Prompts

System prompts are stored per-chat. When creating or updating a chat, provide the system prompt:

```json
POST /api/chats
{
  "system_prompt": "You are a helpful coding assistant who explains concepts clearly.",
  "title": "Coding Help",
  "scope_id": "user-123",
  "metadata": {}
}
```

The context engine treats the system prompt as the "persona" budget slice (10% of token budget, max 400 tokens by default).

### Context Debug (optional, useful for development)

```
GET  /api/chats/{chat_id}/context    Inspect assembled context (token breakdown, memories included)
```

---

## Context Assembly Flow

What happens when a user sends a message:

```
1. Receive user message
   │
2. Save user message to database
   │
3. CONTEXT ASSEMBLY (synchronous, target <120ms)
   │
   ├─ 3a. Load system prompt (10% budget, cached 5 min)
   │      └─ Truncate to budget if needed
   │
   ├─ 3b. Load recent messages (50% budget)
   │      └─ Last 100 messages → budget-truncate from oldest
   │      └─ Track oldest message timestamp for STM dedup
   │
   └─ 3c. RAG memory retrieval (40% budget)
          ├─ STM entities + relationships (FTS match)
          ├─ STM recaps (FTS + keyword match)
          └─ LTM episodes (hybrid: 70% FTS + 30% vector)
   │
4. Format for LLM
   │  ├─ System instruction = system_prompt + memory block
   │  └─ Messages = recent conversation history
   │
5. Call LLM provider → stream response
   │
6. Save assistant message to database
   │
7. Queue async jobs (non-blocking)
   └─ If message_count % recap_interval == 0:
      └─ Queue batch extraction job
```

---

## Memory Extraction Pipeline

### Batch Extraction (every N user messages)

```
Worker receives job
  │
  ├─ Load last N message pairs from chat
  │
  ├─ Call LLM with extraction prompt (low temp: 0.1)
  │
  ├─ Parse structured output:
  │   {
  │     "entities": [
  │       {"type": "person", "canonical_name": "Alice", "attributes": {"role": "sister"}, "confidence": 0.92}
  │     ],
  │     "relationships": [
  │       {"subject": "user", "predicate": "has_sister", "object_name": "Alice", "confidence": 0.90}
  │     ],
  │     "recap_text": "User mentioned their sister Alice is visiting next week...",
  │     "keywords": ["Alice", "sister", "visit"]
  │   }
  │
  ├─ Upsert entities (deduplicate by type + canonical_name)
  │   └─ If exists: reinforce confidence (+0.20), merge attributes, bump mention_count
  │   └─ If new: insert with confidence floor (0.30)
  │
  ├─ Upsert relationships
  │
  └─ Insert recap with keywords for FTS
```

### Episode Generation (on chat completion / session return)

```
Trigger: chat marked complete OR user returns after 30+ min gap
  │
  ├─ Collect all STM entities + relationships + recaps from chat
  │
  ├─ Call LLM to generate narrative summary (~200-300 tokens)
  │
  ├─ Generate vector embedding (768-dim)
  │
  ├─ Store LTM episode with:
  │   └─ summary, embedding, importance_score, keywords, is_final flag
  │
  └─ Promote STM entities/relationships → LTM (deduplicate across chats)
```

---

## Configuration

All configuration is via environment variables with sensible defaults:

### Token Budget

| Variable | Default | Description |
|----------|---------|-------------|
| `BUILDER_PERSONA_RATIO` | `0.10` | System prompt budget (% of total) |
| `BUILDER_PERSONA_MAX` | `400` | System prompt hard cap (tokens) |
| `BUILDER_RECENT_RATIO` | `0.50` | Recent messages budget (% of total) |
| `BUILDER_RAG_RATIO` | `0.40` | Memory/RAG budget (% of total) |

### STM Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `RECAP_INTERVAL_MESSAGES` | `5` | Extract memory every N user messages |
| `STM_MAX_ENTITIES` | `150` | Max entities per chat |
| `STM_MAX_RELATIONSHIPS` | `200` | Max relationships per chat |
| `STM_MIN_CONFIDENCE` | `0.30` | Minimum confidence to store |
| `STM_FACT_BOOST` | `0.20` | Confidence boost on re-mention |
| `EXTRACTION_MODEL` | `gemini-2.5-flash` | Model for extraction (defaults to Gemini) |
| `EXTRACTION_TEMPERATURE` | `0.1` | Low temp for consistent extraction |

### RAG Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_SEARCH_TIMEOUT` | `1.0` | Search timeout (seconds) |
| `RAG_HYBRID_FTS_WEIGHT` | `0.7` | FTS weight in hybrid search |
| `RAG_HYBRID_VECTOR_WEIGHT` | `0.3` | Vector weight in hybrid search |
| `RAG_EPISODE_BUDGET_EARLY` | `0.25` | Episode budget for turns 1-2 |
| `RAG_EPISODE_BUDGET_MID` | `0.15` | Episode budget for turns 3-5 |
| `RAG_EPISODE_BUDGET_LATE` | `0.10` | Episode budget for turns 6-19 |
| `RAG_VECTOR_DISTANCE` | `0.6` | Max cosine distance threshold |

### LTM Episodes

| Variable | Default | Description |
|----------|---------|-------------|
| `EPISODE_INTERVAL_MESSAGES` | `20` | Chunk episode every N user messages |
| `FINAL_EPISODE_MIN_MESSAGES` | `3` | Min messages for final episode |
| `EPISODE_RETRIEVAL_LIMIT` | `5` | Top episodes to retrieve per search |

---

## Model Harness Design

The model harness is the layer between the context engine and LLM providers. The core design principle: **chat, embeddings, and extraction are separate concerns that can use different providers.**

You might want Anthropic for chat, OpenAI for embeddings, and a cheap Gemini model for background memory extraction. The harness makes this natural.

### Design Principles

1. **Protocol-based** — define interfaces, implement per provider. No base classes, no inheritance trees.
2. **Three separate protocols** — `ChatProvider`, `EmbeddingProvider`, `TokenCounter`. Not one god-class.
3. **Registry pattern** — register providers by name, resolve at runtime from config.
4. **Use real model names** — no internal key mapping (`"chat-fast"` → `"gpt-4o-mini"`). Just pass the model name your provider expects.
5. **No coupling** — the context engine never imports a specific provider. It depends only on the protocols.
6. **Circuit breaker is opt-in** — simple use cases don't need fallback chains. Add it when you need it.

### Three Protocols

```python
# providers/base.py

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable


# ── Response Types ──────────────────────────────────────

@dataclass
class ChatResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    delta: str            # text fragment
    done: bool = False    # True on final chunk
    response: ChatResponse | None = None  # populated on final chunk only


@dataclass
class EmbeddingResponse:
    embedding: list[float]
    model: str
    tokens: int


# ── Chat Provider ───────────────────────────────────────

@runtime_checkable
class ChatProvider(Protocol):
    """Generates chat completions from an LLM."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse: ...

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str = "",
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]: ...


# ── Embedding Provider ──────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generates vector embeddings for text."""

    @property
    def dimensions(self) -> int:
        """Embedding vector dimensions (e.g. 1536 for OpenAI, 768 for Gemini)."""
        ...

    async def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        task_type: str = "retrieval_document",
    ) -> EmbeddingResponse: ...

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        task_type: str = "retrieval_document",
    ) -> list[EmbeddingResponse]: ...


# ── Token Counter ───────────────────────────────────────

@runtime_checkable
class TokenCounter(Protocol):
    """Counts and manages tokens for a specific model family."""

    def count(self, text: str) -> int: ...

    def count_messages(self, messages: list[dict[str, str]]) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...

    def fits(self, text: str, budget: int) -> bool: ...
```

### Provider Registry

The registry maps provider names to factory functions. No magic — just a dict.

```python
# providers/registry.py

class ProviderRegistry:
    """Registry for chat, embedding, and token counter providers."""

    def __init__(self):
        self._chat_factories: dict[str, type] = {}
        self._embedding_factories: dict[str, type] = {}
        self._counter_factories: dict[str, type] = {}

    def register_chat(self, name: str, factory: type) -> None: ...
    def register_embedding(self, name: str, factory: type) -> None: ...
    def register_counter(self, name: str, factory: type) -> None: ...

    def create_chat(self, name: str, **kwargs) -> ChatProvider: ...
    def create_embedding(self, name: str, **kwargs) -> EmbeddingProvider: ...
    def create_counter(self, name: str, **kwargs) -> TokenCounter: ...


# Global registry — providers register themselves on import
registry = ProviderRegistry()
```

### Provider Implementations

Each provider is a single file that implements one or more protocols:

```python
# providers/openai.py

from openai import AsyncOpenAI
from .base import ChatProvider, ChatResponse, StreamChunk, EmbeddingProvider, EmbeddingResponse
from .registry import registry


class OpenAIChatProvider:
    """OpenAI chat completions (GPT-4o, GPT-4o-mini, o3-mini, etc.)"""

    def __init__(self, api_key: str | None = None, default_model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    async def chat(self, messages, system_prompt="", *, model=None, temperature=0.7, max_tokens=1024) -> ChatResponse:
        model = model or self.default_model
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = await self.client.chat.completions.create(
            model=model, messages=full_messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        return ChatResponse(
            content=choice.message.content,
            model=response.model,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            latency_ms=0,  # calculated by caller or middleware
        )

    async def chat_stream(self, messages, system_prompt="", *, model=None, temperature=0.7, max_tokens=1024):
        model = model or self.default_model
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        stream = await self.client.chat.completions.create(
            model=model, messages=full_messages,
            temperature=temperature, max_tokens=max_tokens,
            stream=True, stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(delta=chunk.choices[0].delta.content)
        yield StreamChunk(delta="", done=True)


class OpenAIEmbeddingProvider:
    """OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)"""

    def __init__(self, api_key: str | None = None, default_model: str = "text-embedding-3-small"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model
        self._dimensions = 1536  # text-embedding-3-small default

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text, *, model=None, task_type="retrieval_document") -> EmbeddingResponse:
        model = model or self.default_model
        response = await self.client.embeddings.create(model=model, input=text)
        data = response.data[0]
        return EmbeddingResponse(
            embedding=data.embedding, model=model, tokens=response.usage.total_tokens,
        )

    async def embed_batch(self, texts, *, model=None, task_type="retrieval_document") -> list[EmbeddingResponse]:
        model = model or self.default_model
        response = await self.client.embeddings.create(model=model, input=texts)
        return [
            EmbeddingResponse(embedding=d.embedding, model=model, tokens=response.usage.total_tokens // len(texts))
            for d in response.data
        ]


# Auto-register on import
registry.register_chat("openai", OpenAIChatProvider)
registry.register_embedding("openai", OpenAIEmbeddingProvider)
```

```python
# providers/anthropic.py

from anthropic import AsyncAnthropic
from .base import ChatProvider, ChatResponse, StreamChunk
from .registry import registry


class AnthropicChatProvider:
    """Anthropic chat (Claude Sonnet, Opus, Haiku)"""

    def __init__(self, api_key: str | None = None, default_model: str = "claude-sonnet-4-6"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.default_model = default_model

    async def chat(self, messages, system_prompt="", *, model=None, temperature=0.7, max_tokens=1024) -> ChatResponse:
        model = model or self.default_model
        kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)
        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            latency_ms=0,
        )

    async def chat_stream(self, messages, system_prompt="", *, model=None, temperature=0.7, max_tokens=1024):
        model = model or self.default_model
        kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if system_prompt:
            kwargs["system"] = system_prompt

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text)
        yield StreamChunk(delta="", done=True)


registry.register_chat("anthropic", AnthropicChatProvider)
```

```python
# providers/gemini.py

from google import genai
from .base import ChatProvider, ChatResponse, StreamChunk, EmbeddingProvider, EmbeddingResponse
from .registry import registry


class GeminiChatProvider:
    """Google Gemini chat (gemini-2.5-flash, gemini-2.5-pro)"""

    def __init__(self, api_key: str | None = None, default_model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model
    # ... implements chat() and chat_stream()


class GeminiEmbeddingProvider:
    """Google Gemini embeddings (gemini-embedding-001)"""

    def __init__(self, api_key: str | None = None, default_model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model
        self._dimensions = 768
    # ... implements embed() and embed_batch()


registry.register_chat("gemini", GeminiChatProvider)
registry.register_embedding("gemini", GeminiEmbeddingProvider)
```

```python
# providers/ollama.py
# Uses OpenAI-compatible API, so it's just a thin wrapper

from .openai import OpenAIChatProvider, OpenAIEmbeddingProvider
from .registry import registry


class OllamaChatProvider(OpenAIChatProvider):
    """Ollama local models via OpenAI-compatible API"""

    def __init__(self, base_url: str = "http://localhost:11434/v1", default_model: str = "llama3.2"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        self.default_model = default_model


registry.register_chat("ollama", OllamaChatProvider)
```

### Token Counter Implementations

```python
# providers/token_counter.py

import tiktoken
from functools import lru_cache
from .base import TokenCounter
from .registry import registry


class TiktokenCounter:
    """Token counter using tiktoken (works for OpenAI + approximate for others)."""

    def __init__(self, encoding: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += 4  # message overhead
            total += self.count(msg.get("content", ""))
            total += 1  # role token
        total += 2  # conversation overhead
        return total

    def truncate(self, text: str, max_tokens: int) -> str:
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._enc.decode(tokens[:max_tokens])

    def fits(self, text: str, budget: int) -> bool:
        return self.count(text) <= budget


class CharacterEstimateCounter:
    """Rough token counter (1 token ≈ 4 chars). No dependencies. Good for local models."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return sum(self.count(m.get("content", "")) + 5 for m in messages)

    def truncate(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text

    def fits(self, text: str, budget: int) -> bool:
        return self.count(text) <= budget


registry.register_counter("tiktoken", TiktokenCounter)
registry.register_counter("character_estimate", CharacterEstimateCounter)
```

### Configuration

The harness is configured entirely via environment variables. Three separate provider slots:

```env
# ── Chat Provider (generates responses) ──────────────
CHAT_PROVIDER=gemini                   # gemini | openai | anthropic | ollama
CHAT_MODEL=gemini-2.5-flash            # model name your provider expects
GCP_PROJECT_ID=your-gcp-project        # required for Gemini via Vertex AI
GCP_REGION=us-central1                 # Vertex AI region

# ── Embedding Provider (vector search) ────────────────
EMBEDDING_PROVIDER=gemini              # gemini | openai | ollama
EMBEDDING_MODEL=gemini-embedding-001   # 768-dim vectors

# ── Extraction Provider (background memory extraction) ─
# Uses CHAT_PROVIDER by default, but you can override for cost savings
EXTRACTION_PROVIDER=                   # defaults to CHAT_PROVIDER
EXTRACTION_MODEL=gemini-2.5-flash      # cheap model for structured extraction
EXTRACTION_TEMPERATURE=0.1

# ── Token Counter ─────────────────────────────────────
TOKEN_COUNTER=tiktoken                 # tiktoken | character_estimate

# ── Alternative: OpenAI ───────────────────────────────
# CHAT_PROVIDER=openai
# CHAT_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...

# ── Alternative: Anthropic ────────────────────────────
# CHAT_PROVIDER=anthropic
# CHAT_MODEL=claude-sonnet-4-6
# ANTHROPIC_API_KEY=sk-ant-...

# ── Alternative: Ollama (local) ───────────────────────
# CHAT_PROVIDER=ollama
# CHAT_MODEL=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434/v1
```

### How the Context Engine Uses It

The context engine depends only on the protocols — never on specific providers:

```python
# context_engine/builder.py

class ContextBuilder:
    def __init__(
        self,
        chat_provider: ChatProvider,
        embedding_provider: EmbeddingProvider,
        token_counter: TokenCounter,
    ):
        self.chat = chat_provider
        self.embedding = embedding_provider
        self.tokens = token_counter

    async def build_context(self, chat_id, current_message, budget_tokens=4000):
        result = ContextAssemblyResult()

        # 1. System prompt (10% budget)
        prompt_budget = int(budget_tokens * 0.10)
        system_prompt = await self._load_system_prompt(chat_id)
        result.system_prompt = self.tokens.truncate(system_prompt, prompt_budget)
        result.system_prompt_tokens = self.tokens.count(result.system_prompt)

        # 2. Recent messages (50% budget)
        recent_budget = int(budget_tokens * 0.50)
        messages = await self._load_recent_messages(chat_id)
        result.recent_messages = self._fit_messages_to_budget(messages, recent_budget)

        # 3. RAG memories (40% budget)
        rag_budget = int(budget_tokens * 0.40)
        query_embedding = await self.embedding.embed(current_message, task_type="retrieval_query")
        result.memories = await self.rag.search(chat_id, current_message, query_embedding, rag_budget)

        return result
```

### Adding a New Provider

Adding a provider = one file, implements the protocol, auto-registers:

```python
# providers/my_custom_provider.py

from .base import ChatProvider, ChatResponse, StreamChunk
from .registry import registry


class MyCustomChatProvider:
    def __init__(self, base_url: str, api_key: str, default_model: str = "my-model"):
        self.base_url = base_url
        self.api_key = api_key
        self.default_model = default_model

    async def chat(self, messages, system_prompt="", **kwargs) -> ChatResponse:
        # Your implementation here
        ...

    async def chat_stream(self, messages, system_prompt="", **kwargs):
        # Your implementation here
        ...


registry.register_chat("my_custom", MyCustomChatProvider)
```

Then set `CHAT_PROVIDER=my_custom` in `.env`. Done.

### What We Skip from Arkadia (for now)

| Arkadia Feature | Open-Source v1 | Why |
|----------------|---------------|-----|
| Circuit breaker | Skip | Adds complexity; add when needed |
| Provider chain (fallback) | Skip | Single provider is simpler to debug |
| Model key mapping | Skip | Use real model names directly |
| Combined LLM/Vision/TTS class | Skip | Only need chat + embeddings |
| Cost calculation per model | Include (simple) | Useful for monitoring |
| Thinking budget control | Skip | Gemini-specific, not universal |

Circuit breaker and fallback chains can be added later as a `ResilientChatProvider` wrapper that takes two `ChatProvider` instances — composable, not baked in.

---

## Background Worker Strategy

Arkadia uses Google Cloud Pub/Sub for job queuing. For the open-source version, we'll support multiple backends:

| Backend | Best For | Notes |
|---------|----------|-------|
| **In-process** (default) | Development, single-instance | `asyncio.create_task()` — no external deps |
| **Redis + ARQ** | Production, multi-instance | Lightweight, Python-native |
| **PostgreSQL LISTEN/NOTIFY** | Simple production | No extra infra, uses existing DB |

The default in-process worker means you can run the entire system with just PostgreSQL — no Redis, no Pub/Sub, no message broker.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **API Framework** | FastAPI (async) |
| **ORM** | SQLAlchemy 2.0 (async) |
| **Database** | PostgreSQL 15+ with pgvector |
| **Validation** | Pydantic v2 |
| **Migrations** | Alembic |
| **Token Counting** | tiktoken (OpenAI) / model-specific |
| **Testing** | pytest + pytest-asyncio |
| **Background Jobs** | In-process (default) / Redis+ARQ / PG NOTIFY |

---

## Project Structure

```
ark-chatbot/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # App configuration
│   ├── routes/
│   │   ├── chats.py               # Chat CRUD endpoints
│   │   └── messages.py            # Message + streaming endpoints
│   └── dependencies.py            # DB session, provider injection
│
├── context_engine/
│   ├── builder.py                 # Context assembly (the core)
│   ├── rag_manager.py             # Hybrid RAG retrieval
│   ├── stm_manager.py             # Short-term memory operations
│   ├── ltm_manager.py             # Long-term memory operations
│   ├── embedding_service.py       # Vector embedding generation
│   ├── extraction.py              # Entity/relationship extraction prompts
│   ├── tokens.py                  # Token counting + truncation
│   ├── config.py                  # Engine configuration
│   └── models.py                  # Data structures
│
├── providers/
│   ├── base.py                    # LLMProvider protocol
│   ├── openai_provider.py         # OpenAI implementation
│   ├── anthropic_provider.py      # Anthropic implementation
│   ├── gemini_provider.py         # Google Gemini implementation
│   └── embedding/
│       ├── base.py                # EmbeddingProvider protocol
│       ├── openai_embedding.py    # OpenAI embeddings
│       └── local_embedding.py     # Sentence transformers (optional)
│
├── db/
│   ├── models.py                  # SQLAlchemy models
│   ├── session.py                 # Async session factory
│   └── migrations/                # Alembic migrations
│
├── worker/
│   ├── base.py                    # Worker protocol
│   ├── in_process.py              # Default: asyncio tasks
│   ├── extraction_handler.py      # Batch extraction logic
│   └── episode_handler.py         # Episode generation logic
│
├── tests/
│   ├── test_builder.py
│   ├── test_rag_manager.py
│   ├── test_stm_manager.py
│   ├── test_ltm_manager.py
│   └── test_api.py
│
├── docker-compose.yml             # PostgreSQL + pgvector + app
├── Dockerfile
├── pyproject.toml
├── alembic.ini
├── .env.example
└── README.md
```

---

## What We Strip from Arkadia

| Arkadia Feature | Open-Source Equivalent |
|----------------|----------------------|
| Character personas + traits | User-defined system prompt per chat |
| `(user_id, character_id)` LTM scope | Generic `scope_id` (you define) |
| Breed knowledge RAG | Removed (no domain-specific knowledge) |
| Firebase Auth | None — bring your own auth |
| Vertex AI primary + OpenAI fallback | Pluggable provider (you choose) |
| Google Cloud Pub/Sub workers | In-process default, optional Redis/PG |
| Sentry + custom observability | Standard Python logging |
| Roleplay instructions | Removed (user controls system prompt) |
| Paw file data / voice style | Removed |
| Cloud Run deployment | Docker Compose (self-host anywhere) |

---

## Implementation Phases

### Phase 1: Foundation
- [ ] Project scaffolding (FastAPI, SQLAlchemy, Alembic)
- [ ] Database models (chat, message, STM tables, LTM tables)
- [ ] Alembic migrations with pgvector setup
- [ ] Basic chat CRUD API (create chat, list chats, send message, get history)
- [ ] LLM provider abstraction + OpenAI implementation
- [ ] Token counting and truncation utilities
- [ ] Docker Compose for local development

### Phase 2: Context Engine Core
- [ ] Context builder (token-budgeted assembly)
- [ ] System prompt integration (replaces persona)
- [ ] Recent message windowing with budget truncation
- [ ] STM manager (entity + relationship upsert with dedup)
- [ ] Extraction worker (in-process, batch entity/relationship/recap extraction)
- [ ] Extraction prompt engineering

### Phase 3: RAG + Long-Term Memory
- [ ] Embedding service abstraction + OpenAI implementation
- [ ] RAG manager with hybrid search (FTS + vector)
- [ ] Dynamic episode budget allocation by turn count
- [ ] LTM episode generation worker
- [ ] STM → LTM entity/relationship promotion
- [ ] Episode importance decay over time

### Phase 4: Production Readiness
- [ ] SSE streaming endpoint
- [ ] Anthropic + Gemini provider implementations
- [ ] Redis/ARQ worker backend (optional)
- [ ] Comprehensive test suite
- [ ] Configuration validation
- [ ] Context debug endpoint
- [ ] Documentation + examples
- [ ] Docker production image

---

## Quick Start (Target Developer Experience)

```bash
# Clone and start
git clone https://github.com/ArkadiaChat/ark-chatbot-context-engine.git
cd ark-chatbot-context-engine
cp .env.example .env  # Set GCP_PROJECT_ID (or OPENAI_API_KEY for OpenAI)

# Start PostgreSQL + app
docker compose up -d

# Create a chat with a system prompt
curl -X POST http://localhost:8000/api/chats \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "You are a helpful assistant who remembers everything about the user.",
    "title": "My Chat"
  }'

# Send a message
curl -X POST http://localhost:8000/api/chats/{chat_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Hi! My name is Alice and I have a golden retriever named Max."}'

# Later, in the same or different chat (same scope_id):
# The engine remembers Alice and Max automatically
```

---

## License

MIT
