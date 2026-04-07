-- Initial migration: pgvector extension + search indexes
-- Run after SQLAlchemy create_all or as a standalone migration

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- FTS index on stm_entity canonical_name
CREATE INDEX IF NOT EXISTS idx_stm_entity_name_trgm
ON stm_entity USING gin (lower(canonical_name) gin_trgm_ops);

-- GIN index on stm_recap keywords array
CREATE INDEX IF NOT EXISTS idx_stm_recap_keywords
ON stm_recap USING gin (keywords);

-- GIN index on ltm_episode keywords array
CREATE INDEX IF NOT EXISTS idx_ltm_episode_keywords
ON ltm_episode USING gin (keywords);

-- Full-text search on ltm_episode summary
ALTER TABLE ltm_episode ADD COLUMN IF NOT EXISTS summary_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', episode_summary)) STORED;

CREATE INDEX IF NOT EXISTS idx_ltm_episode_summary_fts
ON ltm_episode USING gin (summary_tsv);

-- Vector similarity index on ltm_episode embedding (IVFFlat)
-- Note: IVFFlat requires data to exist first for training.
-- For small datasets, use exact search (no index). Add IVFFlat when > 1000 rows:
-- CREATE INDEX idx_ltm_episode_embedding ON ltm_episode USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_stm_entity_chat_confidence
ON stm_entity (chat_id, overall_confidence DESC);

CREATE INDEX IF NOT EXISTS idx_stm_relationship_chat
ON stm_relationship (chat_id, confidence DESC);

CREATE INDEX IF NOT EXISTS idx_ltm_episode_scope_importance
ON ltm_episode (scope_id, importance_score DESC, episode_date DESC);

CREATE INDEX IF NOT EXISTS idx_ltm_entity_scope_name
ON ltm_entity (scope_id, entity_type, lower(canonical_name));
