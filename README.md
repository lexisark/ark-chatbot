# Ark Chatbot Context Engine

Persistent memory for chatbots.

Ark Chatbot Context Engine is a memory layer for conversational applications. If you already have a chatbot, this gives it cross-session memory without forcing you to build a custom memory stack from scratch.

It extracts entities, relationships, and summaries from conversations, stores them as short-term and long-term memory, and retrieves what fits inside the current token budget.

Use it when you need:
- Cross-session memory, not just chat history
- Structured recall of people, pets, places, events, and preferences
- Token-budgeted retrieval that stays practical in real prompts
- A self-hosted base you can adapt to your product

This project is for teams building support, coaching, companion, sales, education, or other personalized chatbots. It is not an agent framework; it is the memory layer behind one.

## Why Use It

- **Built for real chatbot flows**: chat/session model, message APIs, background extraction, episodic long-term memory, and a demo UI are included.
- **Memory that stays usable**: retrieval is hybrid-scored and token-budgeted so memory does not drown the live conversation.
- **Easy to tune**: extraction cadence, scoring weights, decay rates, and context budgets are all configurable via `.env`.
- **Easy to extend**: plug in new providers or additional RAG sources without rewriting the pipeline.

## Tech Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn
- **Database**: PostgreSQL 17, pgvector
- **ORM / Validation**: SQLAlchemy asyncio, Pydantic
- **LLM Providers**: Gemini, OpenAI, Anthropic, Ollama
- **Embeddings**: Gemini or OpenAI
- **Frontend Demo**: vanilla HTML, CSS, and JavaScript
- **Testing**: pytest, pytest-asyncio

## What It Does

```
User: "I have a dog named River and a cat named Goli"
  → Extracts: River (pet/dog), Goli (pet/cat), user owns River, user owns Goli

User (10 messages later): "How's my puppy doing?"
  → Hybrid search finds River via semantic similarity ("puppy" → "dog")
  → LLM responds with context about River

User (new chat, days later): "Do you remember my pets?"
  → Episode from previous chat recalled via vector search
  → LLM knows about River and Goli from long-term memory
```

## How It Works

The engine manages three processing flows:

```
MESSAGE FLOW (synchronous, per message)
User message → Save → Build Context → Call LLM → Stream Response → Save
                           |
                    Token-budgeted assembly:
                    10% system prompt
                    50% recent messages
                    40% RAG memories (hybrid FTS + vector)

EXTRACTION FLOW (async, every 5 user messages)
New messages → LLM extracts → Entities + Relationships + Recap
                                    |
                              Upserted to STM with:
                              - Case-insensitive dedup
                              - Confidence reinforcement
                              - Attribute merging
                              - Embedding generation

EPISODE FLOW (async, on new chat creation)
Previous chat's STM → LLM generates narrative → Vector embedding → LTM Episode
                           |
                    Entities & relationships promoted to LTM
                    Old episode importance decayed (7-day half-life)
```

## Memory Architecture

```
SHORT-TERM MEMORY (STM) — per chat
├── Entities: people, pets, places, objects, events (with embeddings)
├── Relationships: user owns Max, Alice lives_in Seattle
└── Recaps: "User discussed their dog River and upcoming trip" (with embeddings)

LONG-TERM MEMORY (LTM) — per scope (cross-chat)
├── Episodes: narrative summaries with vector embeddings
├── Promoted entities: merged across chats with dedup
└── Promoted relationships: carried across conversations
```

**Hybrid Search** — every RAG query searches across all tiers using:
- Vector cosine similarity (70% weight)
- FTS keyword matching (30% weight)
- Adaptive recency scoring (mention frequency + confidence decay)
- Context window dedup (excludes memories already visible in recent messages)

> **Note on search strategy**: This implementation uses hybrid search (FTS + vector) across all tiers — STM and LTM. For many use cases, **lexical search (FTS) alone is fast and accurate enough**, especially for STM where entity names are exact and queries are short. Hybrid is enabled everywhere here to handle fuzzy matches like "my puppy" → "Max (pet/dog)", but you can disable vector search per tier by setting `RAG_HYBRID_VECTOR_WEIGHT=0` and skipping embedding generation. Pure FTS is faster, cheaper (no embedding API calls on writes), and works without pgvector.

## Quick Start

```bash
git clone <your-repo-url>
cd ark-chatbot-context-engine

# Copy and edit config
cp .env.example .env
# Set GCP_PROJECT_ID for Gemini, or OPENAI_API_KEY for OpenAI, etc.

# Start PostgreSQL + app
docker compose up -d db
pip install -e ".[dev]"
./start.sh

# Open http://localhost:8000
```

### Or with Docker Compose

```bash
cp .env.example .env
# Edit .env with your provider credentials
docker compose up -d
# Open http://localhost:8000
```

## Supported Providers

Configure via `CHAT_PROVIDER` and `EMBEDDING_PROVIDER` in `.env`:

| Provider | Chat | Embeddings | Config |
|----------|------|------------|--------|
| **Gemini** (default) | gemini-2.5-flash | gemini-embedding-001 | `GCP_PROJECT_ID` |
| **OpenAI** | gpt-4o-mini | text-embedding-3-small | `OPENAI_API_KEY` |
| **Anthropic** | claude-sonnet-4-6 | — | `ANTHROPIC_API_KEY` |
| **Ollama** (local) | llama3.2 | — | `OLLAMA_BASE_URL` |

Mix and match — use Anthropic for chat, OpenAI for embeddings:
```env
CHAT_PROVIDER=anthropic
CHAT_MODEL=claude-sonnet-4-6
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

If you use a non-default embedding dimension, make sure your database vector column size matches `EMBEDDING_DIMENSIONS` before applying migrations or writing embeddings.

## API

### Chats

```
POST   /api/chats                    Create chat (with system prompt, scope_id)
GET    /api/chats                    List chats (filter by scope_id)
GET    /api/chats/{id}               Get chat
PATCH  /api/chats/{id}               Update title, system prompt
DELETE /api/chats/{id}               Delete chat (cascades STM)
POST   /api/chats/{id}/complete      Generate LTM episode for this chat
```

### Messages

```
POST   /api/chats/{id}/messages          Send message, get response
POST   /api/chats/{id}/messages/stream   Send message, stream response (SSE)
GET    /api/chats/{id}/messages          Get message history
```

### Example

```bash
# Create a chat
curl -X POST http://localhost:8000/api/chats \
  -H "Content-Type: application/json" \
  -d '{"title": "My Chat", "system_prompt": "You are a helpful assistant.", "scope_id": "user-123"}'

# Send a message
curl -X POST http://localhost:8000/api/chats/{id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Hi, my name is Alex and I live in Seattle."}'

# Stream a response (SSE)
curl -N -X POST http://localhost:8000/api/chats/{id}/messages/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "What do you remember about me?"}'
```

## Configuration

All engine behavior is configurable via `.env` — no code changes needed. See [.env.example](.env.example) for the full list.

### Key Settings

**Context Budget** — how token budget is split:
```env
BUILDER_PERSONA_RATIO=0.10     # 10% for system prompt
BUILDER_RECENT_RATIO=0.50      # 50% for recent messages
BUILDER_RAG_RATIO=0.40         # 40% for RAG memories
BUILDER_TOTAL_BUDGET=4000      # Total tokens
```

**Extraction** — when and how memories are extracted:
```env
RECAP_INTERVAL_MESSAGES=5      # Extract every 5 user messages
EXTRACTION_MESSAGE_LIMIT=10    # Messages per batch (5 user + 5 assistant)
STM_CONFIDENCE_BOOST=0.20      # Confidence boost on re-mention
STM_MAX_CONFIDENCE=0.95        # Never reach 1.0
```

**RAG Scoring** — how memories are ranked:
```env
RAG_HYBRID_FTS_WEIGHT=0.3      # Keyword match weight
RAG_HYBRID_VECTOR_WEIGHT=0.7   # Semantic similarity weight
RAG_SCORE_HYBRID_WEIGHT=0.4    # Match score in final ranking
RAG_SCORE_CONFIDENCE_WEIGHT=0.3  # Confidence in final ranking
RAG_SCORE_RECENCY_WEIGHT=0.3   # Recency in final ranking
```

**Recency Decay** — how fast memories fade:
```env
RECENCY_BASE_HALF_LIFE_HOURS=72  # 3-day base half-life
RECENCY_MENTION_FACTOR=0.2      # More mentions = slower decay
RECENCY_MENTION_CAP=5           # Cap mention effect at 5
```

**Episodes** — long-term memory generation:
```env
FINAL_EPISODE_MIN_MESSAGES=2   # Min messages to generate episode
EPISODE_DECAY_HALF_LIFE_DAYS=7 # Old episodes fade over 7 days
```

## Project Structure

```
ark-chatbot/
├── app/
│   ├── main.py              # FastAPI app, lifespan, provider init
│   ├── config.py            # All settings (env-configurable)
│   ├── routes/
│   │   ├── chats.py         # Chat CRUD + episode triggers
│   │   └── messages.py      # Send/stream messages
│   └── schemas.py           # Pydantic models
│
├── context_engine/
│   ├── builder.py           # Token-budgeted context assembly
│   ├── rag_manager.py       # Hybrid FTS + vector search
│   ├── stm_manager.py       # Entity/relationship upsert with dedup
│   ├── ltm_manager.py       # Episode generation, entity promotion
│   ├── embedding_service.py # L2 normalization, retry logic
│   ├── extraction.py        # Extraction prompt, JSON parsing, validation
│   ├── dedup.py             # Fuzzy name matching, entity dedup
│   ├── tokens.py            # Token counting, budget fitting
│   ├── config.py            # Engine config (from app settings)
│   └── models.py            # Data structures
│
├── providers/
│   ├── base.py              # ChatProvider, EmbeddingProvider, TokenCounter protocols
│   ├── registry.py          # Provider registry (register + create by name)
│   ├── gemini.py            # Google Gemini (default)
│   ├── openai.py            # OpenAI
│   ├── anthropic.py         # Anthropic
│   ├── ollama.py            # Ollama (local)
│   └── token_counter.py     # tiktoken + character estimate
│
├── db/
│   ├── models.py            # 8 SQLAlchemy tables
│   ├── queries.py           # Async query functions
│   └── migrations/          # SQL migrations
│
├── worker/
│   ├── extraction_handler.py  # Batch extraction pipeline
│   ├── episode_handler.py     # Episode generation pipeline
│   └── in_process.py          # Async job queue (no external deps)
│
├── static/                  # Chat frontend (vanilla HTML/CSS/JS)
├── tests/                   # Test suite
├── docker-compose.yml       # PostgreSQL 17 + pgvector
└── .env.example             # All configuration documented
```

## Database Schema

8 tables with pgvector embeddings and FTS indexes:

```
Chat ──< Message
  |
  ├──< STMEntity (with embedding)
  ├──< STMRelationship
  └──< STMRecap (with embedding)

LTMEpisode (with embedding, per scope)
LTMEntity (with embedding, per scope)
LTMRelationship (per scope)
```

`scope_id` groups chats for shared long-term memory. One user talking to one assistant = one scope. Multiple users = multiple scopes.

## Extending with Custom RAG Sources

The context engine assembles context from multiple sources into a single token budget. You can add your own sources alongside the built-in conversation memory.

The RAG budget (40% of total by default) is where additional sources plug in. For example, to add document search:

```python
# In context_engine/builder.py — add your source in the RAG step

# Built-in: conversation memory (entities, recaps, episodes)
you_remember = await rag.build_you_remember(...)

# Your addition: document search
doc_results = await your_doc_search(current_message, budget=200)

# Combine into the memory block
result.memories_text = you_remember_text + "\n\nRelevant documents:\n" + doc_results
```

Since everything goes through the same `EmbeddingProvider` protocol, your document embeddings use the same vector space as conversation memory — they're searchable together.

Use cases:
- **Knowledge base RAG** — product docs, FAQs, policies
- **User-uploaded documents** — PDFs, notes, files
- **External APIs** — CRM data, calendar events, task lists
- **Domain knowledge** — medical, legal, technical references

## Adding a Provider

Implement the protocol, register it:

```python
# providers/my_provider.py
from providers.base import ChatProvider, ChatResponse, StreamChunk
from providers.registry import registry

class MyChatProvider:
    def __init__(self, api_key: str, default_model: str = "my-model"):
        self.api_key = api_key
        self.default_model = default_model

    async def chat(self, messages, system_prompt="", **kwargs) -> ChatResponse:
        # Your implementation
        ...

    async def chat_stream(self, messages, system_prompt="", **kwargs):
        # Yield StreamChunk objects
        ...

registry.register_chat("my_provider", MyChatProvider)
```

Then set `CHAT_PROVIDER=my_provider` in `.env`.

## Known Limitations

- Background extraction and episode generation run in-process by default. For heavier workloads or multi-instance deployments, replace the in-process queue with a durable worker system.
- The database vector column size must match `EMBEDDING_DIMENSIONS`. If you change embedding dimensions from the default, update your schema and migrations accordingly.
- The included API and demo UI are intentionally minimal. Authentication, authorization, multi-tenancy, observability, and hosted-service concerns are left to the integrating application.
- Provider support is focused on the main chat and embedding paths. If you use non-default provider combinations, validate the exact configuration you plan to deploy.

## Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Skip live API tests
python -m pytest tests/ -m "not live"
```

## Origin

This project was extracted from a production conversational memory system and generalized into a standalone engine for building chatbot applications with persistent memory.

## License

MIT
