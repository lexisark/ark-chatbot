"""Context engine configuration loaded from app settings."""

from app.config import settings

BUILDER_CONFIG = {
    "persona_budget_ratio": settings.builder_persona_ratio,
    "persona_max_tokens": settings.builder_persona_max,
    "recent_messages_budget_ratio": settings.builder_recent_ratio,
    "rag_budget_ratio": settings.builder_rag_ratio,
    "recent_messages_limit": settings.builder_recent_messages_limit,
}

STM_CONFIG = {
    "recap_interval_messages": settings.recap_interval_messages,
    "max_entities_per_chat": settings.stm_max_entities,
    "max_relationships_per_chat": settings.stm_max_relationships,
    "min_confidence_threshold": settings.stm_min_confidence,
    "max_confidence": settings.stm_max_confidence,
    "fact_reinforcement_boost": settings.stm_confidence_boost,
    "enable_rag": True,
}

RAG_CONFIG = {
    "search_timeout_seconds": settings.rag_search_timeout,
    "hybrid_fts_weight": settings.rag_hybrid_fts_weight,
    "hybrid_vector_weight": settings.rag_hybrid_vector_weight,
    "vector_distance_threshold": settings.rag_vector_distance_threshold,
    "max_keywords": settings.rag_max_keywords,
    # Scoring weights
    "score_hybrid_weight": settings.rag_score_hybrid_weight,
    "score_confidence_weight": settings.rag_score_confidence_weight,
    "score_recency_weight": settings.rag_score_recency_weight,
    # Retrieval limits
    "stm_entity_limit": settings.rag_stm_entity_limit,
    "stm_recap_limit": settings.rag_stm_recap_limit,
    "stm_relationship_limit": settings.rag_stm_relationship_limit,
    "ltm_entity_limit": settings.rag_ltm_entity_limit,
    "ltm_episode_limit": settings.rag_ltm_episode_limit,
    # Episode budget by turn count
    "episode_budget_turn_1_2": settings.rag_episode_budget_turn_1_2,
    "episode_budget_turn_3_5": settings.rag_episode_budget_turn_3_5,
    "episode_budget_turn_6_19": settings.rag_episode_budget_turn_6_19,
    "episode_budget_turn_20_plus": settings.rag_episode_budget_turn_20_plus,
    "episode_retrieval_limit": settings.episode_retrieval_limit,
}

RECENCY_CONFIG = {
    "base_half_life_hours": settings.recency_base_half_life_hours,
    "mention_factor": settings.recency_mention_factor,
    "mention_cap": settings.recency_mention_cap,
    "confidence_factor": settings.recency_confidence_factor,
}

LTM_EPISODE_CONFIG = {
    "episode_interval_messages": settings.episode_interval_messages,
    "final_episode_min_messages": settings.final_episode_min_messages,
    "episode_retrieval_limit": settings.episode_retrieval_limit,
    "decay_half_life_days": settings.episode_decay_half_life_days,
    "max_tokens": settings.episode_max_tokens,
    "messages_context": settings.episode_messages_context,
    "entities_context": settings.episode_entities_context,
    "relationships_context": settings.episode_relationships_context,
}

EXTRACTION_CONFIG = {
    "temperature": settings.extraction_temperature,
    "max_tokens": settings.extraction_max_tokens,
    "message_limit": settings.extraction_message_limit,
    "entity_context": settings.extraction_entity_context,
    "relationship_context": settings.extraction_relationship_context,
    "recap_context": settings.extraction_recap_context,
}
