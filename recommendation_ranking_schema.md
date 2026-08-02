# Recommendation Persistence Schema

This document describes the schema used by the current Recommendation runtime. Candidate sets and ranked lists are computed per request and are not persisted in this version.

## Extension

The authoritative root embedding schema files enable pgvector in `public`:

```sql
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
```

## `user_embeddings`

| Column | Type | Meaning |
| --- | --- | --- |
| `user_id` | `BIGINT PRIMARY KEY` | Canonical SocialGraph Snowflake ID |
| `embedding` | `VECTOR(512) NOT NULL` | Current normalized preference vector |

Registration uses an idempotent insert: an existing vector is retained on retry.

## `post_embeddings`

| Column | Type | Meaning |
| --- | --- | --- |
| `post_id` | `BIGINT PRIMARY KEY` | Canonical SocialGraph post ID |
| `embedding` | `VECTOR(512) NOT NULL` | Multimodal content vector |

Post creation and update use an upsert keyed by `post_id`.

## `recommendation_interactions`

An idempotent feedback ledger backing `record_recommendation_interaction`. Each row keys a SocialGraph outbox event so at-least-once interaction delivery logs and applies each interaction to the user vector only once; replayed keys are ignored.

| Column | Type |
| --- | --- |
| `idempotency_key` | `VARCHAR(128) PRIMARY KEY` |
| `user_id` | `BIGINT NOT NULL` |
| `target_id` | `BIGINT NOT NULL` |
| `action` | `VARCHAR(16) NOT NULL` |
| `weight` | `REAL NOT NULL` |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP` |

The required non-unique btree index has keys `(user_id, created_at DESC)`.

## Runtime Data Flow

1. Candidate IDs come from SocialGraph's authenticated ID-only `post-candidate-ids` REST API.
2. Recommendation selects only embeddings whose `post_id` appears in that candidate set.
3. Ranking is calculated in memory.
4. The GraphQL response returns ranked IDs only, without persisting a ranked-list table or exposing internal scores.
5. Gateway Fusion hydrates each ranked item through SocialGraph; hydration is not persisted by Recommendation.

The old `fb.rec_candidate_set`, `fb.rec_candidate`, `fb.rec_ranked_list`, and `fb.rec_ranked_item` design is not part of the current runtime and must not be treated as a deployed contract.

## Applying the Schema

```powershell
psql -d fakebook -f .\user_embedding.sql
psql -d fakebook -f .\post_embedding.sql
psql -d fakebook -f .\recommendation_interactions.sql
```

All three scripts are idempotent and the default-on startup migrator runs them under a
PostgreSQL session advisory lock. Applied versions and SHA-256 checksums are stored in
`recommendation.schema_migrations`. Configure
`RECOMMENDATION_MIGRATION_DATABASE_URL` for the migration owner because creating the
`vector` extension can require elevated privileges; startup fails if pgvector or a
migration is unavailable. `RECOMMENDATION_DB_MIGRATIONS_ENABLED=false` is the explicit
opt-out for deployments that migrate in a separate release step. The root-level SQL
files are authoritative. Every migration target is checked against its exact catalog
shape before a ledger row is inserted, recorded rows are rechecked on every startup,
and a final full validation rejects drift, partial legacy tables, or an invalid
interaction index.
