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

    # ── Embedding Provider ────────────────────────────────
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"

    # ── Extraction Provider (defaults to chat provider) ───
    extraction_provider: str = ""
    extraction_model: str = "gemini-2.5-flash"
    extraction_temperature: float = 0.1

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
    builder_persona_ratio: float = 0.10
    builder_persona_max: int = 400
    builder_recent_ratio: float = 0.50
    builder_rag_ratio: float = 0.40
    builder_total_budget: int = 4000

    # ── STM Extraction ────────────────────────────────────
    recap_interval_messages: int = 5
    stm_max_entities: int = 150
    stm_max_relationships: int = 200
    stm_min_confidence: float = 0.30
    stm_fact_boost: float = 0.20

    # ── RAG Retrieval ─────────────────────────────────────
    rag_search_timeout: float = 1.0
    rag_hybrid_fts_weight: float = 0.7
    rag_hybrid_vector_weight: float = 0.3
    rag_episode_budget_early: float = 0.25
    rag_episode_budget_mid: float = 0.15
    rag_episode_budget_late: float = 0.10
    rag_vector_distance: float = 0.6

    # ── LTM Episodes ─────────────────────────────────────
    episode_interval_messages: int = 20
    final_episode_min_messages: int = 2
    episode_retrieval_limit: int = 5

    # ── Server ────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    @property
    def effective_extraction_provider(self) -> str:
        return self.extraction_provider or self.chat_provider


settings = Settings()
