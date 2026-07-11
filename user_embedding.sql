CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS user_embeddings (
    user_id BIGINT PRIMARY KEY,
    embedding VECTOR(512) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE user_embeddings IS
'Stores user interest profile embeddings owned by Recommendation';
