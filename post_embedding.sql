CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE SCHEMA IF NOT EXISTS recommendation;

CREATE TABLE IF NOT EXISTS recommendation.post_embeddings (
    post_id BIGINT PRIMARY KEY,
    embedding public.vector(512) NOT NULL
);

COMMENT ON TABLE recommendation.post_embeddings IS
'Stores multimodal post embeddings owned by Recommendation';
