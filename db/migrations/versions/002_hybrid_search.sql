-- Migration 002: Add embedding columns + tsvector for hybrid search across all tiers
-- NOTE: Default dimension is 768 (Gemini). If using OpenAI (1536) or other dimensions,
-- change the vector size below to match EMBEDDING_DIMENSIONS in your .env.

-- STM Entity: embedding for semantic search
ALTER TABLE stm_entity ADD COLUMN IF NOT EXISTS embedding vector(768);

-- STM Recap: embedding for semantic search
ALTER TABLE stm_recap ADD COLUMN IF NOT EXISTS embedding vector(768);

-- LTM Entity: embedding for semantic search
ALTER TABLE ltm_entity ADD COLUMN IF NOT EXISTS embedding vector(768);

-- Vector indexes (cosine similarity)
CREATE INDEX IF NOT EXISTS idx_stm_entity_embedding
ON stm_entity USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_stm_recap_embedding
ON stm_recap USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_ltm_entity_embedding
ON ltm_entity USING hnsw (embedding vector_cosine_ops);

-- LTM Episode embedding index (column already exists from 001)
CREATE INDEX IF NOT EXISTS idx_ltm_episode_embedding
ON ltm_episode USING hnsw (embedding vector_cosine_ops);
