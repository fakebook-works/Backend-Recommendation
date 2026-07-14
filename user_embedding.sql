CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE SCHEMA IF NOT EXISTS recommendation;

CREATE TABLE IF NOT EXISTS recommendation.user_embeddings (
    user_id BIGINT PRIMARY KEY,
    embedding public.vector(512) NOT NULL
);

COMMENT ON TABLE recommendation.user_embeddings IS
'Stores user interest profile embeddings owned by Recommendation';
