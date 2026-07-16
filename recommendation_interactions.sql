CREATE SCHEMA IF NOT EXISTS recommendation;

CREATE TABLE IF NOT EXISTS recommendation.recommendation_interactions (
    idempotency_key VARCHAR(128) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    target_id BIGINT NOT NULL,
    action VARCHAR(16) NOT NULL,
    weight REAL NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_recommendation_interactions_user_created
    ON recommendation.recommendation_interactions (user_id, created_at DESC);

COMMENT ON TABLE recommendation.recommendation_interactions IS
'Idempotent interaction feedback used to personalize each user embedding';

