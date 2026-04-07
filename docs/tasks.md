# Ark Chatbot Context Engine — Tasks

Reference: [ark-chatbot-context-engine.md](./ark-chatbot-context-engine.md)

**Approach: TDD** — For each section: plan the interface → write failing tests → implement to green → refactor.

---

## Phase 1: Foundation

### 1.1 Project Scaffolding

No tests — just setup.

- [ ] `pyproject.toml` with dependencies (fastapi, uvicorn, sqlalchemy[asyncio], asyncpg, alembic, pydantic, tiktoken, pgvector, openai, anthropic, google-genai, httpx, pytest, pytest-asyncio, httpx for test client)
- [ ] `app/main.py` — FastAPI app with lifespan (DB init, provider init), CORS, health check
- [ ] `app/config.py` — Pydantic `Settings` class loading from env: DB URL, provider config (defaults to Gemini + Arkadia's GCP dev project), token budgets, STM/RAG/LTM config
- [ ] `app/dependencies.py` — FastAPI dependency injection: async DB session, chat provider, embedding provider, token counter
- [ ] `.env.example` with all env vars documented
- [ ] `Dockerfile` (Python 3.12, multi-stage)
- [ ] `docker-compose.yml` — PostgreSQL 16 with pgvector extension + app service
- [ ] `tests/conftest.py` — pytest config, async fixtures, test DB setup (separate test database with auto-rollback)

### 1.2 Model Harness

#### Plan

Three protocols (`ChatProvider`, `EmbeddingProvider`, `TokenCounter`), registry pattern, Gemini-first.

#### Tests first

- [ ] `tests/providers/test_base.py`:
  - Test `ChatResponse`, `StreamChunk`, `EmbeddingResponse` dataclass construction
  - Test that protocol classes are runtime-checkable (`isinstance` works)
- [ ] `tests/providers/test_registry.py`:
  - Test `register_chat()` + `create_chat()` round-trip
  - Test `register_embedding()` + `create_embedding()` round-trip
  - Test `register_counter()` + `create_counter()` round-trip
  - Test `create_chat("unknown")` raises `KeyError`
  - Test duplicate registration overwrites cleanly
- [ ] `tests/providers/test_token_counter.py`:
  - Test `TiktokenCounter.count("hello world")` returns reasonable int
  - Test `TiktokenCounter.count_messages([...])` includes overhead
  - Test `TiktokenCounter.truncate("long text", 5)` produces ≤5 tokens
  - Test `TiktokenCounter.fits("short", 100)` → True
  - Test `TiktokenCounter.fits("short", 1)` → False
  - Test `CharacterEstimateCounter` same interface, rough correctness
- [ ] `tests/providers/test_mock_provider.py`:
  - Create `MockChatProvider` + `MockEmbeddingProvider` implementing protocols
  - Test `chat()` returns `ChatResponse`
  - Test `chat_stream()` yields `StreamChunk` sequence ending with `done=True`
  - Test `embed()` returns `EmbeddingResponse` with correct dimensions
  - Test `embed_batch()` returns list of correct length

#### Implement

- [ ] `providers/base.py` — protocols + dataclasses
- [ ] `providers/registry.py` — `ProviderRegistry` + global `registry`
- [ ] `providers/token_counter.py` — `TiktokenCounter`, `CharacterEstimateCounter`
- [ ] `providers/gemini.py` — `GeminiChatProvider`, `GeminiEmbeddingProvider` (default provider, Vertex AI via `google-genai`)
- [ ] `providers/openai.py` — `OpenAIChatProvider`, `OpenAIEmbeddingProvider`
- [ ] `providers/anthropic.py` — `AnthropicChatProvider`
- [ ] `providers/ollama.py` — `OllamaChatProvider` (wraps OpenAI-compatible API)
- [ ] `providers/__init__.py` — imports all modules to trigger auto-registration
- [ ] `tests/providers/test_gemini_live.py` — live integration test (marked `@pytest.mark.live`, skipped in CI): send a real message via Gemini, assert response structure

### 1.3 Database Layer

#### Plan

8 tables: Chat, Message, STMEntity, STMRelationship, STMRecap, LTMEpisode, LTMEntity, LTMRelationship. Async SQLAlchemy 2.0, pgvector for embeddings, Alembic migrations.

#### Tests first

- [ ] `tests/db/test_models.py`:
  - Test `Chat` model creation with all fields
  - Test `Message` model with FK to Chat
  - Test `STMEntity` model with JSONB attributes
  - Test `STMRelationship` model with entity FKs
  - Test `STMRecap` model with array fields
  - Test `LTMEpisode` model with vector embedding field
  - Test `LTMEntity` and `LTMRelationship` with scope_id
  - Test cascade: deleting Chat cascades Messages, STM* rows
- [ ] `tests/db/test_queries.py`:
  - Test `create_chat()` → returns Chat with UUID
  - Test `get_chat()` → returns Chat or None
  - Test `list_chats()` → pagination works (offset/limit)
  - Test `list_chats(scope_id=...)` → filters correctly
  - Test `update_chat()` → updates title, system_prompt
  - Test `delete_chat()` → removes chat + cascaded rows
  - Test `create_message()` → returns Message with FK
  - Test `get_chat_messages()` → returns in chronological order, respects limit
  - Test `count_user_messages()` → counts only role=user

#### Implement

- [ ] `db/session.py` — async engine + `AsyncSessionLocal`
- [ ] `db/models.py` — all 8 SQLAlchemy models
- [ ] `db/queries.py` — all query functions
- [ ] `alembic.ini` + `db/migrations/env.py` — async Alembic config
- [ ] `db/migrations/versions/001_initial.py` — initial migration with pgvector extension + all tables + FTS indexes

### 1.4 Chat API

#### Plan

REST endpoints for chat CRUD and messaging. No context engine yet — just system prompt + recent messages + direct LLM call.

#### Tests first

- [ ] `tests/api/test_chats.py` (using `httpx.AsyncClient` + `TestClient`):
  - `POST /api/chats` → 201, returns chat with id, title, system_prompt
  - `POST /api/chats` with missing fields → uses defaults
  - `GET /api/chats` → 200, returns list with pagination
  - `GET /api/chats?scope_id=x` → filters by scope
  - `GET /api/chats/{id}` → 200, returns chat
  - `GET /api/chats/{bad_id}` → 404
  - `PATCH /api/chats/{id}` → 200, updates fields
  - `DELETE /api/chats/{id}` → 204
  - `DELETE /api/chats/{bad_id}` → 404
- [ ] `tests/api/test_messages.py` (using mock chat provider):
  - `POST /api/chats/{id}/messages` with `{"content": "hello"}` → 200, returns assistant message
  - `POST /api/chats/{id}/messages` stores both user + assistant messages
  - `GET /api/chats/{id}/messages` → returns message history in order
  - `GET /api/chats/{id}/messages?limit=5` → pagination works
  - `POST /api/chats/{bad_id}/messages` → 404
- [ ] `tests/api/test_schemas.py`:
  - Test request/response model validation
  - Test required vs optional fields

#### Implement

- [ ] `app/schemas.py` — Pydantic request/response models
- [ ] `app/routes/chats.py` — CRUD endpoints
- [ ] `app/routes/messages.py` — messaging endpoints (naive: system_prompt + last N messages → LLM)
- [ ] Wire routes into `app/main.py`

### 1.5 End-to-End Smoke Test

- [ ] `tests/test_e2e.py`:
  - Full flow with mock provider: create chat → send message → get response → list messages → delete chat
  - Verify message persistence across requests
- [ ] Manual: `docker compose up` → curl create chat → curl send message → verify response

---

## Phase 2: Context Engine Core

### 2.1 Token Utilities

#### Tests first

- [ ] `tests/context_engine/test_tokens.py`:
  - Test `count_tokens("hello world")` returns positive int
  - Test `count_message_tokens([{"role": "user", "content": "hi"}])` includes overhead
  - Test `truncate_to_budget("long text...", 5)` → output is ≤5 tokens
  - Test `truncate_to_budget("short", 100)` → returns original text
  - Test `fits_in_budget("text", 100)` → True/False correctly
  - Test with empty string → 0 tokens

#### Implement

- [ ] `context_engine/tokens.py` — wraps injected `TokenCounter` protocol

### 2.2 Context Builder

#### Tests first

- [ ] `tests/context_engine/test_builder.py`:
  - Test budget allocation: system_prompt gets ≤10%, recent ≤50%, RAG ≤40%
  - Test system_prompt truncation when it exceeds budget
  - Test recent messages loaded newest-first, budget-truncated from oldest
  - Test empty chat (no messages) → system_prompt only
  - Test `format_for_llm()` → system_instruction contains prompt + memory block, messages are just conversation
  - Test total_tokens ≤ budget_tokens
  - Test assembly with RAG disabled → no memories_text

#### Implement

- [ ] `context_engine/config.py` — `BUILDER_CONFIG`, `STM_CONFIG`, `RAG_CONFIG`, `LTM_EPISODE_CONFIG` with env loading + validation
- [ ] `context_engine/models.py` — `ContextAssemblyResult` dataclass
- [ ] `context_engine/builder.py` — `ContextBuilder` class (RAG placeholder returns empty in Phase 2)
- [ ] Integrate into `POST /api/chats/{chat_id}/messages` — replace naive prompt with budgeted context

### 2.3 STM Manager

#### Tests first

- [ ] `tests/context_engine/test_stm_manager.py`:
  - Test `upsert_entity()` new → inserts with confidence 0.30 floor
  - Test `upsert_entity()` existing → confidence +0.20, attributes merged, mention_count bumped
  - Test `upsert_entity()` confidence capped at 0.95
  - Test `upsert_entity()` case-insensitive dedup on canonical_name
  - Test `upsert_relationship()` new → inserts
  - Test `upsert_relationship()` existing → updates confidence
  - Test `insert_recap()` → stores with keywords and msg range
  - Test `get_entities(min_confidence=0.5)` → filters below threshold
  - Test `get_relationships()` → returns relationships for chat
  - Test `get_recaps(limit=3)` → returns most recent N
  - Test entity count cap at `max_entities_per_chat` (150)

#### Implement

- [ ] `context_engine/stm_manager.py` — `STMManager` class with all methods

### 2.4 Extraction Worker

#### Tests first

- [ ] `tests/context_engine/test_extraction.py`:
  - Test `build_extraction_prompt()` → produces valid prompt string with message content
  - Test `parse_extraction_response()` with valid JSON → returns `ExtractedData` with entities, relationships, recap
  - Test `parse_extraction_response()` with malformed JSON → handles gracefully (returns empty)
  - Test `parse_extraction_response()` with partial data → extracts what's available
- [ ] `tests/worker/test_extraction_handler.py`:
  - Test `run_batch_extraction()` with mock provider → calls LLM, upserts entities via STMManager
  - Test extraction skipped if no new messages since last extraction
  - Test extraction uses correct message window (last N pairs)
- [ ] `tests/worker/test_in_process.py`:
  - Test `InProcessQueue.enqueue()` → job runs asynchronously
  - Test job failure doesn't crash the queue
  - Test `JobQueue` protocol compliance

#### Implement

- [ ] `context_engine/extraction.py` — prompt templates + response parsing
- [ ] `context_engine/models.py` — add `ExtractedEntity`, `ExtractedRelationship`, `ExtractedRecap`, `ExtractedData`
- [ ] `worker/base.py` — `JobQueue` protocol
- [ ] `worker/in_process.py` — `InProcessQueue`
- [ ] `worker/extraction_handler.py` — `run_batch_extraction()`
- [ ] Integrate trigger into message route: `user_msg_count % recap_interval == 0` → enqueue

---

## Phase 3: RAG + Long-Term Memory

### 3.1 Embedding Integration

#### Tests first

- [ ] `tests/context_engine/test_embedding_service.py`:
  - Test `generate_query_embedding()` → returns list[float] of correct dimensions
  - Test `generate_document_embedding()` → returns list[float] of correct dimensions
  - Test L2 normalization: output vector norm ≈ 1.0
  - Test with mock provider → correct task_type passed through
  - Test retry on rate limit (mock 429 → success on retry)

#### Implement

- [ ] `context_engine/embedding_service.py` — `EmbeddingService` wrapping `EmbeddingProvider` protocol

### 3.2 RAG Manager

#### Tests first

- [ ] `tests/context_engine/test_rag_manager.py`:
  - Test STM entity retrieval via FTS keyword match
  - Test STM recap retrieval via keyword match
  - Test LTM episode retrieval via hybrid search (FTS + vector)
  - Test hybrid scoring: 70% FTS + 30% vector weight
  - Test dynamic episode budget: turns 1-2 get 25%, turns 6+ get 10%
  - Test dedup: entities already in context window excluded
  - Test output fits within budget_tokens
  - Test `YouRememberBlock` formatting: entities, relationships, episodes, recaps sections
  - Test empty memories → returns empty block (not None)
  - Test with no LTM episodes → STM-only results

#### Implement

- [ ] `context_engine/models.py` — add `YouRememberBlock`, `MemoryCandidate`
- [ ] `context_engine/rag_manager.py` — `RAGManager` class
- [ ] Wire RAG into `ContextBuilder.build_context()` — replace placeholder

### 3.3 LTM Episode Generation

#### Tests first

- [ ] `tests/context_engine/test_ltm_manager.py`:
  - Test `generate_episode()` → creates LTMEpisode with summary, embedding, importance_score
  - Test episode includes keywords extracted from summary
  - Test `promote_entities()` → STM entities copied to LTM with dedup (same canonical_name merges)
  - Test `promote_relationships()` → STM relationships copied to LTM with dedup
  - Test `apply_importance_decay()` → old episodes get reduced importance (7-day half-life)
  - Test episode with `is_final=True` for chat-end episodes
- [ ] `tests/worker/test_episode_handler.py`:
  - Test `run_episode_generation()` orchestrates LTMManager correctly
  - Test triggered on chat completion

#### Implement

- [ ] `context_engine/ltm_manager.py` — `LTMManager` class
- [ ] `worker/episode_handler.py` — `run_episode_generation()`
- [ ] Integrate episode trigger into chat lifecycle

### 3.4 FTS + Vector Indexes

No unit tests — migration correctness verified by RAG manager integration tests.

- [ ] `db/migrations/versions/002_search_indexes.py`:
  - GIN index on `stm_entity.canonical_name` (tsvector)
  - GIN index on `stm_recap.keywords` (array ops)
  - GIN index on `ltm_episode.keywords` (array ops)
  - GiST/IVFFlat index on `ltm_episode.embedding` (pgvector)
  - FTS tsvector column + trigger on `ltm_episode.episode_summary`

---

## Phase 4: Production Readiness

### 4.1 Streaming

#### Tests first

- [ ] `tests/api/test_streaming.py`:
  - Test `POST /api/chats/{id}/messages/stream` → returns SSE event stream
  - Test stream contains multiple chunks ending with `done` event
  - Test stream saves complete assistant message to DB after completion
  - Test stream interruption → partial response still saved

#### Implement

- [ ] SSE streaming in `POST /api/chats/{chat_id}/messages/stream` using `StreamingResponse` + `chat_stream()`

### 4.2 Additional Provider Testing

- [ ] `tests/providers/test_openai_live.py` — live test (marked `@pytest.mark.live`)
- [ ] `tests/providers/test_anthropic_live.py` — live test
- [ ] `tests/providers/test_ollama_live.py` — live test
- [ ] Provider-specific error mapping to common exceptions

### 4.3 Context Debug

#### Tests first

- [ ] `tests/api/test_context_debug.py`:
  - Test `GET /api/chats/{id}/context` → returns token breakdown
  - Test response includes system_prompt_tokens, recent_messages_tokens, memories_tokens
  - Test includes list of entities/episodes included

#### Implement

- [ ] `GET /api/chats/{chat_id}/context` endpoint

### 4.4 Configuration Validation

- [ ] Test budget ratios sum to ~1.0 validation
- [ ] Test missing API key for chosen provider → clear error on startup
- [ ] Test invalid thresholds rejected

### 4.5 Documentation

- [ ] `README.md` — quick start, configuration reference, architecture overview
- [ ] `.env.example` — comprehensive, commented
- [ ] Provider setup guides (Gemini, OpenAI, Anthropic, Ollama)

---

## Build Order

Within each section: **tests → implement → green → next section.**

```text
Phase 1 ──────────────────────────────────────────────
  1.1 Scaffolding (no tests)
  1.2 Model Harness
      tests: base, registry, token_counter, mock_provider
      impl:  base.py, registry.py, token_counter.py, gemini.py, openai.py, anthropic.py, ollama.py
  1.3 Database Layer
      tests: models, queries
      impl:  session.py, models.py, queries.py, alembic, migration
  1.4 Chat API
      tests: chats, messages, schemas
      impl:  schemas.py, routes/chats.py, routes/messages.py
  1.5 Smoke Test (e2e)

Phase 2 ──────────────────────────────────────────────
  2.1 Token Utilities
      tests: tokens
      impl:  context_engine/tokens.py
  2.2 Context Builder
      tests: builder
      impl:  config.py, models.py, builder.py, route integration
  2.3 STM Manager
      tests: stm_manager
      impl:  stm_manager.py
  2.4 Extraction Worker
      tests: extraction, extraction_handler, in_process
      impl:  extraction.py, models.py, worker/base.py, worker/in_process.py, extraction_handler.py

Phase 3 ──────────────────────────────────────────────
  3.1 Embedding Integration
      tests: embedding_service
      impl:  embedding_service.py
  3.2 RAG Manager
      tests: rag_manager
      impl:  models.py, rag_manager.py, builder integration
  3.3 LTM Episode Generation
      tests: ltm_manager, episode_handler
      impl:  ltm_manager.py, episode_handler.py
  3.4 Search Indexes (migration only)

Phase 4 ──────────────────────────────────────────────
  4.1 Streaming        tests → impl
  4.2 Provider Testing  live tests
  4.3 Context Debug     tests → impl
  4.4 Config Validation tests → impl
  4.5 Documentation
```
