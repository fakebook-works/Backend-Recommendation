# Recommendation Persistence Schema

This document describes the schema used by the current Recommendation runtime. Candidate sets and ranked lists are computed per request and are not persisted in this version.

## Extension

Both schema scripts enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## `user_embeddings`

| Column | Type | Meaning |
| --- | --- | --- |
| `user_id` | `BIGINT PRIMARY KEY` | Canonical SocialGraph Snowflake ID |
| `embedding` | `VECTOR(512) NOT NULL` | Current normalized preference vector |
| `updated_at` | `TIMESTAMPTZ NOT NULL` | Last vector update time |

Registration uses an idempotent insert: an existing vector is retained on retry.

## `post_embeddings`

| Column | Type | Meaning |
| --- | --- | --- |
| `post_id` | `BIGINT PRIMARY KEY` | Canonical SocialGraph post ID |
| `embedding` | `VECTOR(512) NOT NULL` | Multimodal content vector |
| `created_at` | `TIMESTAMPTZ NOT NULL` | First creation time |
| `updated_at` | `TIMESTAMPTZ NOT NULL` | Last upsert time |

Post creation and update use an upsert keyed by `post_id`.

## Runtime Data Flow

1. Candidate IDs and social source labels come from SocialGraph's authenticated internal REST API.
2. Recommendation selects only embeddings whose `post_id` appears in that candidate set.
3. Ranking is calculated in memory.
4. The GraphQL response returns ranked IDs and scores without persisting a ranked-list table.

The old `fb.rec_candidate_set`, `fb.rec_candidate`, `fb.rec_ranked_list`, and `fb.rec_ranked_item` design is not part of the current runtime and must not be treated as a deployed contract.

## Applying the Schema

```powershell
psql -d fakebook -f .\user_embedding.sql
psql -d fakebook -f .\post_embedding.sql
```

Both scripts are idempotent and can be run repeatedly.
