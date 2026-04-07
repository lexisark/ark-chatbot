"""Context engine configuration loaded from app settings."""

from app.config import settings

BUILDER_CONFIG = {
    "persona_budget_ratio": settings.builder_persona_ratio,
    "persona_max_tokens": settings.builder_persona_max,
    "recent_messages_budget_ratio": settings.builder_recent_ratio,
    "rag_budget_ratio": settings.builder_rag_ratio,
}

STM_CONFIG = {
    "recap_interval_messages": settings.recap_interval_messages,
    "max_entities_per_chat": settings.stm_max_entities,
    "max_relationships_per_chat": settings.stm_max_relationships,
    "min_confidence_threshold": settings.stm_min_confidence,
    "fact_reinforcement_boost": settings.stm_fact_boost,
    "enable_rag": True,
}

RAG_CONFIG = {
    "search_timeout_seconds": settings.rag_search_timeout,
    "hybrid_fts_weight": settings.rag_hybrid_fts_weight,
    "hybrid_vector_weight": settings.rag_hybrid_vector_weight,
    "episode_budget_turn_1_2": settings.rag_episode_budget_early,
    "episode_budget_turn_3_5": settings.rag_episode_budget_mid,
    "episode_budget_turn_6_19": settings.rag_episode_budget_late,
    "episode_budget_turn_20_plus": settings.rag_episode_budget_late,
    "episode_vector_distance_threshold": settings.rag_vector_distance,
}

LTM_EPISODE_CONFIG = {
    "episode_interval_messages": settings.episode_interval_messages,
    "final_episode_min_messages": settings.final_episode_min_messages,
    "episode_retrieval_limit": settings.episode_retrieval_limit,
}
