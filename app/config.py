from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── Database ──────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ark_chatbot"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/ark_chatbot"

    # ── Chat Provider ─────────────────────────────────────
    chat_provider: str = "gemini"
    chat_model: str = "gemini-2.5-flash"
    chat_api_key: str = ""
    chat_temperature: float = 0.7
    chat_max_tokens: int = 1024

    # ── Embedding Provider ────────────────────────────────
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    # ── Extraction Provider (defaults to chat provider) ───
    extraction_provider: str = ""
    extraction_model: str = "gemini-2.5-flash"
    extraction_temperature: float = 0.1
    extraction_max_tokens: int = 4096
    extraction_message_limit: int = 10      # Max messages per extraction batch (5 user + 5 assistant)
    extraction_entity_context: int = 50     # Existing entities passed for dedup
    extraction_relationship_context: int = 30  # Existing relationships passed for dedup
    extraction_recap_context: int = 7       # Existing recaps passed for dedup

    # ── GCP (for Gemini via Vertex AI) ────────────────────
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"

    # ── OpenAI / Anthropic (alternative providers) ────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Ollama ────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434/v1"

    # ── Token Counter ─────────────────────────────────────
    token_counter: str = "tiktoken"

    # ── Context Builder Budget ────────────────────────────
    builder_persona_ratio: float = 0.10     # System prompt budget (% of total)
    builder_persona_max: int = 400          # Hard cap on system prompt tokens
    builder_recent_ratio: float = 0.50      # Recent messages budget (% of total)
    builder_rag_ratio: float = 0.40         # RAG/memories budget (% of total)
    builder_total_budget: int = 4000        # Total token budget for context assembly
    builder_recent_messages_limit: int = 100  # Max messages loaded from DB

    # ── STM (Short-Term Memory) ───────────────────────────
    recap_interval_messages: int = 5        # Extract memory every N user messages
    stm_max_entities: int = 150             # Max entities per chat
    stm_max_relationships: int = 200        # Max relationships per chat
    stm_min_confidence: float = 0.30        # Floor confidence for new entities
    stm_max_confidence: float = 0.95        # Cap confidence (never 1.0)
    stm_confidence_boost: float = 0.20      # Confidence reinforcement on re-mention

    # ── RAG Retrieval ─────────────────────────────────────
    rag_search_timeout: float = 1.0         # Search timeout (seconds)
    rag_hybrid_fts_weight: float = 0.3      # FTS weight in hybrid scoring
    rag_hybrid_vector_weight: float = 0.7   # Vector weight in hybrid scoring
    rag_vector_distance_threshold: float = 0.6  # Max cosine distance for vector matches
    rag_max_keywords: int = 10              # Max keywords extracted from query

    # ── RAG Scoring Weights ───────────────────────────────
    rag_score_hybrid_weight: float = 0.4    # Weight for hybrid match score
    rag_score_confidence_weight: float = 0.3  # Weight for entity confidence
    rag_score_recency_weight: float = 0.3   # Weight for recency score

    # ── RAG Retrieval Limits ──────────────────────────────
    rag_stm_entity_limit: int = 50          # Max STM entities searched
    rag_stm_recap_limit: int = 10           # Max STM recaps searched
    rag_stm_relationship_limit: int = 30    # Max STM relationships returned
    rag_ltm_entity_limit: int = 50          # Max LTM entities searched
    rag_ltm_episode_limit: int = 20         # Max LTM episodes searched

    # ── RAG Episode Budget by Turn Count ──────────────────
    rag_episode_budget_turn_1_2: float = 0.25   # 25% for first 2 turns
    rag_episode_budget_turn_3_5: float = 0.15   # 15% for turns 3-5
    rag_episode_budget_turn_6_19: float = 0.10  # 10% for turns 6-19
    rag_episode_budget_turn_20_plus: float = 0.10  # 10% for 20+ turns

    # ── Recency Scoring ───────────────────────────────────
    recency_base_half_life_hours: float = 72    # 3-day base half-life
    recency_mention_factor: float = 0.2         # Per-mention multiplier for half-life
    recency_mention_cap: int = 5                # Max mentions factored into decay
    recency_confidence_factor: float = 0.5      # Confidence contribution to half-life

    # ── LTM (Long-Term Memory) Episodes ───────────────────
    episode_interval_messages: int = 20     # Chunk episode every N user messages (unused)
    final_episode_min_messages: int = 2     # Min user messages to generate final episode
    episode_retrieval_limit: int = 5        # Top episodes to retrieve per RAG search
    episode_decay_half_life_days: float = 7  # Importance decay half-life
    episode_max_tokens: int = 1000          # Max tokens for episode generation LLM call
    episode_messages_context: int = 10      # Recent messages fed to episode generation
    episode_entities_context: int = 50      # Max entities fed to episode generation
    episode_relationships_context: int = 30  # Max relationships fed to episode generation

    # ── Server ────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @property
    def effective_extraction_provider(self) -> str:
        return self.extraction_provider or self.chat_provider


settings = Settings()
