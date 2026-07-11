CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE user_embeddings (
    user_id BIGINT PRIMARY KEY,
    embedding VECTOR(512) NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE user_embeddings IS
'Stores user interest profile embeddings';

COMMENT ON COLUMN user_embeddings.user_id IS
'User identifier';

COMMENT ON COLUMN user_embeddings.embedding IS
'Vector representation of user interests';
