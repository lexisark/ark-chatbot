# Context Engine Gaps — ark-chatbot vs Arkadia

Audit date: 2026-04-07

This document catalogs the gaps between ark-chatbot's context engine and Arkadia's production implementation, with a remediation plan for each.

---

## Gap 1: RAG Manager — No Vector/Hybrid Search (STM + LTM)

**Severity: Critical**

### Current State

RAG manager (`context_engine/rag_manager.py`) does simple `SELECT ... ORDER BY confidence DESC` queries against both STM and LTM tables. The `query_embedding` parameter is accepted by `build_you_remember()` but never used anywhere. No semantic similarity search exists at any tier.

### What Arkadia Does

Arkadia implements hybrid search for LTM episodes only (FTS + vector). STM is FTS-only.

### What We Want (goes beyond Arkadia)

**Hybrid search across BOTH STM and LTM** — not just LTM episodes. This is an improvement over Arkadia's approach:

#### STM Hybrid Search
- **STM Entities**: Add an `embedding` column (vector). When entities are upserted, generate an embedding from `"{canonical_name} ({entity_type}) — {attributes}"`. Search with hybrid: FTS on `canonical_name`/`attributes` + vector cosine similarity on `embedding`. This enables fuzzy matches like user says "my puppy" → finds entity "Max (pet/dog)".
- **STM Recaps**: Add an `embedding` column (vector). When recaps are inserted, generate an embedding from `recap_text`. Search with hybrid: FTS on `recap_text`/`keywords` + vector cosine on `embedding`.

#### LTM Hybrid Search
- **LTM Episodes**: Already have `embedding` column. Enable hybrid: FTS on `summary_tsv` + vector cosine on `embedding`. Combined scoring with configurable weights.
- **LTM Entities**: Add an `embedding` column. Same hybrid approach as STM entities.

#### Unified Scoring
All tiers use the same hybrid formula:
```
hybrid_score = fts_weight * fts_rank + vector_weight * (1 - cosine_distance)
```
Default weights: 70% FTS / 30% vector (configurable via `RAG_CONFIG`).

### Remediation Plan

#### Schema Changes
1. Add `embedding` column (Vector(768), nullable) to `STMEntity`
2. Add `embedding` column (Vector(768), nullable) to `STMRecap`
3. Add `embedding` column (Vector(768), nullable) to `LTMEntity`
4. Add tsvector columns for FTS:
   - `STMEntity`: generated from `canonical_name || ' ' || entity_type || ' ' || attributes_text`
   - `STMRecap`: generated from `recap_text`
   - `LTMEpisode`: `summary_tsv` already exists in migration
5. Migration: `002_hybrid_search.sql`

#### Embedding Generation
6. In `STMManager.upsert_entity()`: generate embedding from entity text (async, via EmbeddingService)
7. In `STMManager.insert_recap()`: generate embedding from recap_text
8. In `LTMManager.promote_entities()`: copy STM entity embedding to LTM entity, or regenerate

#### RAG Hybrid Search
9. Rewrite `_get_stm_entities()`:
   - FTS: `ts_rank(entity_tsv, plainto_tsquery(:query))`
   - Vector: `embedding <=> :query_embedding` where distance < 0.6
   - Combined: `fts_weight * fts_rank + vector_weight * (1 - distance)`
   - Final: `combined * confidence_weight * confidence`
10. Rewrite `_get_stm_recaps()`: same hybrid approach
11. Rewrite `_get_ltm_episodes()`: same hybrid approach (already has embedding column)
12. Add `_get_ltm_entities()` with hybrid search (new — not in current RAG)
13. `build_you_remember()` generates query embedding once, passes to all search methods

### Files to Change

- `db/models.py` — add embedding columns to STMEntity, STMRecap, LTMEntity
- `db/migrations/versions/002_hybrid_search.sql` — new migration
- `context_engine/stm_manager.py` — accept EmbeddingService, generate embeddings on upsert/insert
- `context_engine/ltm_manager.py` — copy/generate embeddings on promotion
- `context_engine/rag_manager.py` — rewrite all search methods with hybrid scoring
- `context_engine/builder.py` — generate query embedding once, pass through
- `tests/context_engine/test_rag_manager.py` — test hybrid search, vector scoring

---

## Gap 2: Extraction — No Existing Memory Context

**Severity: High**

### Current State

`worker/extraction_handler.py` calls `build_extraction_prompt(messages)` with only the recent messages. The extraction prompt in `context_engine/extraction.py` has no knowledge of what entities/relationships already exist in STM.

### What Arkadia Does

Arkadia's batch extractor loads existing STM entities and relationships before calling the LLM, and includes them in the prompt as an "existing memory" block:

```
EXISTING MEMORY (do not re-extract these unless new information is found):
- Max (pet/dog) — breed: Golden Retriever, age: 3 years [confidence: 0.92]
- Alice (person/family) — relation: sister [confidence: 0.85]
- User owns Max [confidence: 0.90]
```

This prevents the LLM from re-extracting the same entities every batch, and instead focuses on NEW or UPDATED information.

### Impact Without Fix

- Duplicate extraction: "Max" gets re-extracted every 5 messages with slightly different attributes
- Wasted LLM tokens on redundant extraction
- Confidence scores get artificially inflated (reinforced on every batch even without new mentions)

### Remediation Plan

1. Update `build_extraction_prompt()` to accept `existing_entities` and `existing_relationships` parameters
2. Format existing memory as a dedup block in the prompt with instruction to only extract NEW or CHANGED information
3. In `run_batch_extraction()`, load existing STM entities/relationships before calling LLM
4. Pass them to the prompt builder
5. Add instruction: "If an entity already exists and has NEW attributes, include it with only the new/changed attributes. Do not re-emit unchanged entities."

### Files to Change

- `context_engine/extraction.py` — update `build_extraction_prompt()` signature and template
- `worker/extraction_handler.py` — load existing STM before calling LLM
- `tests/context_engine/test_extraction.py` — test prompt includes existing memory block
- `tests/worker/test_extraction_handler.py` — test existing entities passed to prompt

---

## Gap 3: RAG Manager — No FTS Scoring for STM

**Severity: Medium**

### Current State

STM entity and recap retrieval uses `ORDER BY confidence DESC` — the current user message is not factored into which entities/recaps are returned. A user asking about "Max" gets the same entity list as a user asking about "weather".

### What Arkadia Does

Arkadia scores STM entities by relevance to the current query:
- Extracts keywords from the user message (stop-word filtered, max 10)
- Scores entities by: `50% confidence + 25% recency + 10% keyword_match + 15% mention_boost`
- Scores recaps by keyword array overlap with query keywords
- Returns highest-scoring candidates within budget

### Remediation Plan

1. Extract keywords from `query` parameter (simple: split + lowercase + filter stop words)
2. Score entities by keyword match against `canonical_name` + `attributes` values
3. Score recaps by keyword overlap with `keywords` array
4. Combine match score with confidence and recency for final ranking
5. Use `ts_rank` where tsvector indexes exist, fall back to simple string matching

### Files to Change

- `context_engine/rag_manager.py` — add keyword extraction, scoring functions, update retrieval methods
- `tests/context_engine/test_rag_manager.py` — test keyword-relevant entities ranked higher

---

## Gap 4: RAG Manager — No Context Window Dedup

**Severity: Medium**

### Current State

The "You remember" block may include entities that are already visible in the recent messages window. For example, if the user just said "Max is at the park" and Max is an STM entity, the memory block will still include "Max (pet/dog)" — redundant with what the LLM can already see.

### What Arkadia Does

Arkadia tracks the `context_window_start` timestamp (the oldest message in the recent messages budget) and excludes STM entities whose `first_mentioned` timestamp falls within the context window. This prevents memory–message overlap.

### Remediation Plan

1. In `ContextBuilder.build_context()`, record the `created_at` of the oldest message included in recent_messages
2. Pass `context_window_start` to `RAGManager.build_you_remember()`
3. In STM entity retrieval, add `WHERE first_mentioned < :context_window_start` to exclude entities the LLM can already see in the conversation
4. Optionally do the same for recaps (exclude recaps whose message range overlaps the context window)

### Files to Change

- `context_engine/builder.py` — track and pass `context_window_start`
- `context_engine/rag_manager.py` — filter by `context_window_start`
- `tests/context_engine/test_rag_manager.py` — test dedup excludes recent entities

---

## Gap 5: RAG Manager — Adaptive Recency Scoring

**Severity: Low**

### Current State

No recency decay applied to entity or recap scores. A 3-day-old entity with 1 mention ranks the same as a 1-hour-old entity with 10 mentions (assuming equal confidence).

### What Arkadia Does

Arkadia applies adaptive recency decay:
- Base half-life: 72 hours for STM entities
- Adjusted by mention count: more mentions → slower decay
- Adjusted by confidence: higher confidence → slower decay
- Formula: `decay = 0.5 ^ (age_hours / (base_half_life * (1 + mention_factor * min(mentions, cap)) * (1 + confidence_factor * confidence)))`

### Remediation Plan

1. Add recency scoring function that factors in `last_mentioned` timestamp, `mention_count`, and `confidence`
2. Apply as a multiplier during entity/recap ranking in `_get_stm_entities()` and `_get_stm_recaps()`
3. Use config values from `RAG_CONFIG` for decay parameters

### Files to Change

- `context_engine/rag_manager.py` — add `_recency_score()`, apply in retrieval
- `tests/context_engine/test_rag_manager.py` — test old entities decay, frequently mentioned entities decay slower

---

## Gap 6: Extraction Prompt — Missing Fields

**Severity: Low**

### Current State

Extraction prompt lacks several fields that Arkadia extracts:
- No `tags` field (semantic categorization like "pet_care", "introduction")
- No per-attribute `attribute_confidence` (e.g., breed: 0.95, age: 0.70)
- No extended thinking token request
- No image message handling guidance

### Remediation Plan

1. Add `tags` to extraction JSON schema and `ExtractedRecap` dataclass
2. Add `attribute_confidence` to extraction JSON schema (store in entity attributes alongside values)
3. Add thinking token guidance in prompt ("Think step by step before extracting")
4. Add note about image messages format
5. Update `parse_extraction_response()` to handle new fields
6. Add `tags` column to `STMRecap` model (or store in existing `keywords` array)

### Files to Change

- `context_engine/extraction.py` — update prompt and parser
- `context_engine/models.py` — update `ExtractedRecap` to include tags  
- `db/models.py` — consider adding tags to STMRecap (optional, can reuse keywords)
- `tests/context_engine/test_extraction.py` — test tags/attribute_confidence parsing

---

## Gap 7: STM Relationship — Missing Mention Tracking

**Severity: Low**

### Current State

`STMRelationship` doesn't track `mention_count` or `last_mentioned`. When the same relationship is upserted, only `confidence` is updated.

### What Arkadia Does

Arkadia tracks `mention_count` and `last_mentioned` on relationships, same as entities. This feeds into recency scoring.

### Remediation Plan

1. Add `mention_count` (Integer, default 1) and `last_mentioned` (DateTime) to `STMRelationship` model
2. In `STMManager.upsert_relationship()`, increment `mention_count` and update `last_mentioned` on existing match
3. Migration to add columns

### Files to Change

- `db/models.py` — add fields to `STMRelationship`
- `context_engine/stm_manager.py` — update upsert logic
- `tests/context_engine/test_stm_manager.py` — test mention tracking

---

## Remediation Priority Order

| Priority | Gap | Severity | Effort | Impact |
|----------|-----|----------|--------|--------|
| **P1** | Gap 1: Vector/Hybrid Search (STM + LTM) | Critical | High | Core RAG quality — semantic search across ALL tiers, fuzzy entity matching |
| **P2** | Gap 2: Extraction Memory Dedup | High | Medium | Prevents duplicate entities, reduces wasted LLM tokens |
| **P3** | Gap 3: FTS Scoring for STM | Medium | Medium | Absorbed into P1 — FTS is half of hybrid search |
| **P4** | Gap 4: Context Window Dedup | Medium | Low | Cleaner memory blocks, no redundancy with visible messages |
| **P5** | Gap 5: Adaptive Recency Scoring | Low | Low | Better ranking of frequently mentioned entities |
| **P6** | Gap 6: Extraction Prompt Fields | Low | Low | Richer extraction (tags, attribute confidence) |
| **P7** | Gap 7: Relationship Mention Tracking | Low | Low | Feeds into recency scoring |

### Suggested Build Order

**Phase A — Hybrid Search Across All Tiers (P1 + P3)**

P3 is absorbed into P1 since FTS is the first half of hybrid search. This is the largest change:

1. **Schema**: Add `embedding` (Vector) to STMEntity, STMRecap, LTMEntity. Add tsvector columns for FTS.
2. **Embedding on write**: STMManager generates embeddings on entity upsert and recap insert. LTMManager copies/generates on promotion.
3. **Hybrid search**: All RAG retrieval methods use `fts_weight * ts_rank + vector_weight * (1 - cosine_distance)` scoring.
4. **Query embedding**: Builder generates one query embedding, passes to all RAG search methods.

**Phase B — Extraction Quality (P2 + P6)**

Address together since both touch the extraction prompt and handler. Add existing memory context and richer output fields.

**Phase C — Scoring & Dedup (P4 + P5 + P7)**

Address together since all are incremental improvements to RAG scoring. Context window dedup, adaptive recency, and relationship mention tracking.
