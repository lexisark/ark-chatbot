# Context Architecture for Conversational Chatbots

A technical deep-dive into how the Ark Chatbot Context Engine manages memory, retrieval, and context assembly for LLM-powered chatbots.

---

## The Problem

LLMs are stateless. Every API call starts from zero. The standard workaround — stuffing the last N messages into the prompt — breaks down quickly:

- **Token limits**: Long conversations overflow the context window
- **No cross-session memory**: Start a new chat and everything is forgotten
- **Flat history**: The LLM sees raw messages but doesn't understand what's important
- **No structure**: "The user mentioned their dog Max three times" is invisible information

Building a chatbot that genuinely remembers users requires a memory system that extracts, stores, retrieves, and assembles context — all within a token budget, all fast enough to not slow down responses.

---

## Architecture Overview

The context engine operates three concurrent processing flows:

```
                                 ┌──────────────────┐
                                 │   User Message    │
                                 └────────┬─────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
              SYNCHRONOUS           ASYNCHRONOUS          ASYNCHRONOUS
              (per message)      (every N messages)    (on new chat start)
                    │                     │                     │
            ┌───────▼───────┐    ┌───────▼───────┐    ┌───────▼───────┐
            │    Context    │    │   Batch       │    │   Episode     │
            │   Assembly    │    │  Extraction   │    │  Generation   │
            │   Fast Path   │    │  Background   │    │  Background   │
            └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                    │                     │                     │
            ┌───────▼───────┐    ┌───────▼───────┐    ┌───────▼───────┐
            │  LLM Call +   │    │ STM Entities  │    │ LTM Episode   │
            │  Stream Back  │    │ Relationships │    │ Entity Promo  │
            │               │    │ Recaps        │    │ Decay         │
            └───────────────┘    └───────────────┘    └───────────────┘
```

**Flow 1: Context Assembly** runs on every message. It must be fast — the user is waiting. It reads from memory but never writes.

**Flow 2: Batch Extraction** runs asynchronously after every N user messages (default: 5). It writes to short-term memory. The user never waits for it.

**Flow 3: Episode Generation** runs asynchronously when a user starts a new chat. It promotes short-term memory to long-term memory. Again, non-blocking.

This separation is critical. The chat response path stays fast because memory writes happen in the background.

---

## Memory Model

### Two-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                SHORT-TERM MEMORY (STM)                      │
│                    Scoped per chat                           │
│                                                             │
│  Entities          Relationships        Recaps              │
│  ┌──────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ Max      │     │ user → owns  │     │ "User talked │    │
│  │ (pet/dog)│     │   → Max      │     │  about Max"  │    │
│  │ conf:0.92│     │ conf: 0.90   │     │ kw: [Max,dog]│    │
│  │ embed:768│     │ mentions: 3  │     │ embed: 768   │    │
│  └──────────┘     └──────────────┘     └──────────────┘    │
│                                                             │
│  Extracted every 5 user messages                            │
│  Reinforced on re-mention (+0.20 confidence)                │
│  Deduped by canonical name (case-insensitive, fuzzy)        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    On new chat start:
                    promote + generate episode
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                LONG-TERM MEMORY (LTM)                       │
│                   Scoped per scope_id                       │
│                                                             │
│  Episodes              Entities           Relationships     │
│  ┌──────────────┐     ┌──────────┐       ┌─────────────┐   │
│  │ "User has a  │     │ Max      │       │ user → owns │   │
│  │  dog named   │     │ (pet/dog)│       │   → Max     │   │
│  │  Max who..." │     │ merged   │       │ merged      │   │
│  │ embed: 768   │     │ across   │       │ across      │   │
│  │ importance:  │     │ chats    │       │ chats       │   │
│  │  0.8 (decays)│     └──────────┘       └─────────────┘   │
│  └──────────────┘                                           │
│                                                             │
│  Persists across chat sessions                              │
│  Episode importance decays (7-day half-life)                │
│  Entities merged with dedup on promotion                    │
└─────────────────────────────────────────────────────────────┘
```

### Why Two Tiers?

**STM is fast and detailed.** It stores granular entities and relationships extracted from the current conversation. It's optimized for the current chat — "what did the user just tell me?"

**LTM is durable and compressed.** It stores narrative episodes summarizing entire conversations. It's optimized for cross-chat recall — "what do I know about this user from past conversations?"

The separation matters because:
- STM can grow without limit during a chat, then gets pruned on promotion
- LTM episodes are compact (~200 tokens each) with vector embeddings for semantic search
- Old LTM episodes decay in importance, naturally prioritizing recent conversations
- Entity dedup on promotion prevents the same fact from inflating across chats

---

## Context Assembly

When a user sends a message, the context builder assembles everything the LLM needs to respond — within a fixed token budget.

### Token Budget Allocation

```
Total Budget: 4000 tokens (configurable)

┌──────────────────────────────────────────────────────┐
│ System Prompt │    Recent Messages    │ RAG Memories  │
│     10%       │        50%            │     40%       │
│   (400 max)   │      (2000)           │   (1600)      │
└──────────────────────────────────────────────────────┘
```

Each section is independently truncated to its budget:

1. **System Prompt (10%)** — the chat's system prompt, truncated if too long
2. **Recent Messages (50%)** — most recent messages that fit, walked backwards from newest
3. **RAG Memories (40%)** — hybrid-searched entities, relationships, recaps, and episodes

### RAG Budget: Dynamic Episode Allocation

The RAG budget is further split between STM (current chat) and LTM (past chats), dynamically based on conversation depth:

| Turn Count | Episode Budget | STM Budget | Rationale |
|------------|---------------|------------|-----------|
| 1-2 | 25% | 75% | Early: user may reference past chats |
| 3-5 | 15% | 85% | Shifting to current conversation |
| 6-19 | 10% | 90% | In-chat context is king |
| 20+ | 10% | 90% | Stable long conversations |

This prevents the LLM from being overwhelmed by historical context when there's plenty of in-chat context to work with.

### Assembly Flow

```
1. Load system prompt → truncate to 10% budget
2. Load recent messages → walk backwards, fit to 50% budget
   → Track context_window_start (oldest included message)
3. Generate query embedding from current user message (once)
4. Search all memory tiers with hybrid scoring:
   a. STM entities (exclude those in context window)
   b. STM relationships
   c. STM recaps (exclude those in context window)
   d. LTM episodes (for scope, exclude current chat)
   e. LTM entities (for scope, dedup against STM)
5. Truncate each section to budget
6. Format: system_instruction = prompt + memory block
           messages = conversation history
```

---

## Hybrid Search

Every memory retrieval uses the same hybrid scoring formula across all tiers:

```
hybrid_score = fts_weight × keyword_match + vector_weight × (1 - cosine_distance)
final_score  = score_hybrid × hybrid + score_confidence × confidence + score_recency × recency
```

Default weights (configurable):
- Vector weight: 0.7, FTS weight: 0.3
- Final: 0.4 hybrid + 0.3 confidence + 0.3 recency

### Why Hybrid?

**FTS alone** fails on semantic similarity. User says "my puppy" but the entity is stored as "Max (pet/dog)". No keyword overlap.

**Vector alone** is noisy. Cosine similarity returns vaguely related results without considering exact matches.

**Hybrid** gets the best of both: exact keyword matches rank highest, semantic matches fill in the gaps.

### Embeddings on Write

Embeddings are generated at write time, not search time:

- **STM Entity**: embedded from `"{name} ({type}) — {attributes}"`
- **STM Recap**: embedded from recap text
- **LTM Episode**: embedded from episode summary
- **LTM Entity**: copied from STM on promotion

This means search is a simple cosine distance calculation — no LLM calls during context assembly.

### Adaptive Recency Scoring

Memories decay over time, but frequently mentioned memories decay slower:

```
effective_half_life = base_hours × (1 + mention_factor × min(mentions, cap))
                                 × (1 + confidence_factor × confidence)

recency_score = 0.5 ^ (age_hours / effective_half_life)
```

With defaults (72h base, 0.2 mention factor, cap 5):
- Entity mentioned once, 3 days ago: score ~0.63
- Entity mentioned 5 times, 3 days ago: score ~0.76
- Entity mentioned once, 7 days ago: score ~0.42

This naturally surfaces important, frequently-discussed topics while letting one-off mentions fade.

### Context Window Dedup

The builder tracks the oldest message in the recent messages window. RAG excludes any STM entity or recap whose timestamp falls within that window — the LLM can already see those in the conversation. This prevents wasting tokens on redundant memory.

---

## Extraction Pipeline

Every 5 user messages (configurable), a batch extraction runs asynchronously:

### What Gets Extracted

The LLM receives:
- **New messages** since last extraction (max 10)
- **Existing entities** for dedup context (top 50 by confidence)
- **Existing relationships** for dedup context (top 30)
- **Previous recaps** for continuity (last 7)

And produces:
```json
{
  "entities": [
    {"type": "person", "canonical_name": "Sarah", "attributes": {"role": "manager"},
     "attribute_confidence": {"role": 0.90}, "overall_confidence": 0.88}
  ],
  "relationships": [
    {"subject": "user", "predicate": "works_with", "object_name": "Sarah", "confidence": 0.88}
  ],
  "recap_text": "User mentioned their manager Sarah in engineering.",
  "keywords": ["Sarah", "manager", "engineering"]
}
```

### Post-Extraction Validation

Before upserting, the engine:
1. **Rejects pronouns** — "he", "she", "it" are not entity names
2. **Rejects short names** — must be at least 2 characters
3. **Rejects assistant entities** — only extract user-side information
4. **Clamps confidence** — floor 0.30, cap 0.95 (never 1.0)
5. **Fuzzy deduplicates** — "Max" and "Maxi" merge as the same entity
6. **Generates embeddings** — each entity and recap gets a 768-dim vector

### Entity Upsert Logic

When an extracted entity matches an existing one (case-insensitive canonical name + type):
- **Confidence reinforced**: +0.20, capped at 0.95
- **Attributes merged**: new attributes override, existing preserved
- **Mention count incremented**
- **Embedding regenerated** if attributes changed
- **Timestamp updated**

When it's new:
- Inserted with confidence floor (0.30 minimum)
- Embedding generated from entity text
- If entity cap reached (150), lowest-confidence entity evicted

### Extraction Window Tracking

Each recap stores `start_msg_id` and `end_msg_id`. The next extraction only loads messages after the last recap's `end_msg_id`. This prevents reprocessing — each message batch is extracted exactly once.

---

## Episode Generation

When a user starts a new chat, the engine generates a long-term memory episode for the previous chat in the same scope.

### Trigger

```
User creates new chat with scope_id="user-123"
  → Find previous chat in scope "user-123"
  → Check: >= 2 user messages? No existing final episode?
  → Generate episode asynchronously (non-blocking)
```

### What Goes In

The LLM receives:
- Last 10 messages from the chat
- Top 50 entities by confidence
- Top 30 relationships
- All recaps (the full summary chain)

### What Comes Out

A narrative episode (~200 tokens) focused on what the user shared:

```json
{
  "episode_summary": "The user works as a software engineer at Google in Seattle.
    They have a golden retriever named Max and are planning a trip to Japan
    next summer with their partner Sarah.",
  "keywords": ["software engineer", "Google", "Seattle", "Max", "Japan", "Sarah"],
  "importance_score": 0.85,
  "emotional_tone": "positive"
}
```

The episode is stored with a configurable vector embedding for semantic search.

### Promotion

After episode generation:
1. **STM entities → LTM entities**: merged by canonical name, highest confidence wins
2. **STM relationships → LTM relationships**: deduped by subject/predicate/object
3. **Importance decay**: episode importance is computed from age at retrieval time using a configurable half-life

This keeps LTM clean: recent conversations are prominent, old ones fade unless reinforced by re-mention.

---

## Token Efficiency

The engine is designed to run well on cost-efficient models. Key efficiency decisions:

**Context assembly has zero LLM calls.** It reads from the database and does Python-side scoring. Embeddings are pre-computed. The only LLM call is the actual chat response.

**Extraction batches, not per-message.** Instead of calling the LLM on every message, extraction runs every 5 user messages. One LLM call processes 10 messages at once.

**Episodes are one-shot.** One LLM call + one embedding call per chat session, not per message.

**Token budget is enforced at every level.** System prompt, messages, and RAG each have hard caps. The total context sent to the LLM is always within budget regardless of how much memory exists.

**Embeddings are generated once, searched many times.** Write-time embedding means search is pure math — no API calls.

### Cost Profile (illustrative)

| Operation | Frequency | Token Cost |
|-----------|-----------|------------|
| Chat response | Every message | ~4000 in + ~500 out |
| Batch extraction | Every 5 messages | ~3000 in + ~1000 out |
| Episode generation | Per chat session | ~2000 in + ~500 out |
| Embedding | Per entity/recap/episode | ~100 tokens |

For a 20-message conversation: ~6 chat calls + ~4 extraction calls + 1 episode = ~11 LLM calls total. Exact token cost depends on your provider, model, extraction cadence, and embedding settings.

---

## Extending the Engine

### Adding RAG Sources

The context builder's RAG step is where custom sources plug in. The 40% RAG budget can be shared between conversation memory and your sources:

```python
# In context_engine/builder.py
rag_budget = int(budget_tokens * 0.40)

# Split: 70% conversation memory, 30% your docs
memory_budget = int(rag_budget * 0.70)
doc_budget = int(rag_budget * 0.30)

you_remember = await rag.build_you_remember(db, chat_id, query, memory_budget, ...)
doc_results = await your_search(query, budget=doc_budget)
```

Since everything uses the same `EmbeddingProvider` protocol, your document embeddings live in the same vector space — you can even search across conversations and documents in a single query.

### Custom Scoring

Override `RAGManager._search_stm_entities()` or change the scoring weights via `.env`:

```env
RAG_SCORE_HYBRID_WEIGHT=0.5    # Increase match relevance
RAG_SCORE_CONFIDENCE_WEIGHT=0.2  # Decrease confidence bias
RAG_SCORE_RECENCY_WEIGHT=0.3   # Keep recency the same
```

### Custom Entity Types

The extraction prompt defines entity types. Add your own by editing `context_engine/extraction.py`:

```
ENTITY TYPES (use ONLY these):
  person, location, object, event, pet, user, other
  your_custom_type: description of when to use it
```

No schema changes needed — entities are stored with a string `entity_type` field.

---

## Database Schema

```sql
-- Core
chat (id, title, system_prompt, scope_id, metadata, created_at, updated_at)
message (id, chat_id, role, content, token_count, created_at)

-- Short-Term Memory (per chat)
stm_entity (id, chat_id, entity_type, entity_subtype, canonical_name,
            attributes JSONB, overall_confidence, mention_count,
            embedding vector(768), first_mentioned, last_mentioned)
stm_relationship (id, chat_id, subject_entity_id, predicate, object_entity_id,
                   confidence, mention_count, last_mentioned, source_msg_ids)
stm_recap (id, chat_id, recap_text, keywords[], embedding vector(768),
           entity_ids[], start_msg_id, end_msg_id, confidence)

-- Long-Term Memory (per scope)
ltm_episode (id, scope_id, episode_summary, keywords[], embedding vector(768),
             importance_score, is_final, emotional_tone, source_chat_id)
ltm_entity (id, scope_id, entity_type, canonical_name, attributes JSONB,
            overall_confidence, mention_count, embedding vector(768))
ltm_relationship (id, scope_id, subject_entity_id, predicate,
                   object_entity_id, confidence)
```

Indexes: HNSW on all embedding columns, GIN on keyword arrays, composite indexes on (chat_id, confidence) and (scope_id, importance_score).
