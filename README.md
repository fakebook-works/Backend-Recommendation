# Fakebook Backend Recommendation

Backend Recommendation owns user and post embeddings and ranks a personalized feed. SocialGraph remains the source of truth for users, posts, relationships, privacy, and candidate generation.

## Service Boundaries

- SocialGraph creates the canonical Snowflake `userId`.
- After Authentication accepts that ID, SocialGraph calls Recommendation to create the initial user embedding.
- SocialGraph sends post content and media URLs when a searchable post is created.
- Recommendation fetches candidate IDs from SocialGraph over an authenticated internal API. It does not query SocialGraph tables directly.
- Internal REST owns embedding writes. Public GraphQL only exposes recommendation reads.

All IDs must be positive signed 64-bit integers. GraphQL uses `ID`, so clients should send Snowflake IDs as strings.

## Project Layout

```text
ForFakebook/
  EmbeddingModel.py          FastAPI and Strawberry entry point
  operations.py              Application operations and validation
  database.py                SQLAlchemy session and pgvector persistence
  embedding_service.py       Lazy-loaded text/image/video embedding models
  recommendation_service.py Candidate-ID retrieval and semantic ranking
tests/                       Automated contract and ranking tests
user_embedding.sql           Idempotent user embedding schema
post_embedding.sql           Idempotent post embedding schema
```

## Requirements

- Python 3.9 or later
- PostgreSQL with the `vector` extension
- The two schema files in this repository

Create and activate a virtual environment, then install production dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Apply the idempotent schemas:

```powershell
psql -d fakebook -f .\user_embedding.sql
psql -d fakebook -f .\post_embedding.sql
```

Both tables use `vector(512)` and a `BIGINT` primary key.

## Configuration

Set configuration through environment variables. Do not commit production secrets.

```powershell
$env:DATABASE_URL="postgresql://postgres:postgres@localhost:5432/fakebook"
$env:INTERNAL_SHARED_SECRET="replace-with-at-least-32-bytes"
$env:SOCIAL_GRAPH_BASE_URL="http://localhost:5223"
```

`INTERNAL_SHARED_SECRET` must match Gateway and SocialGraph. Every `/internal/*` request requires:

```http
X-Gateway-Secret: <shared secret>
X-Correlation-ID: <optional trace id>
```

The service returns `403` for a missing or invalid secret and fails closed with `503` when the server secret is shorter than 32 bytes.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn ForFakebook.EmbeddingModel:app --host 0.0.0.0 --port 8000
```

Health check:

```http
GET /health
```

## Internal REST Contracts

### User Embedding

```http
PUT /internal/recommendation/users/{userId}/embedding
DELETE /internal/recommendation/users/{userId}/embedding
```

`PUT` is idempotent. It creates a deterministic normalized initial vector when the user has no embedding and reports `created: false` on a retry. `DELETE` is idempotent and returns `204`.

### Post Embedding

```http
PUT /internal/recommendation/posts/{postId}/embedding
Content-Type: application/json

{
  "content": "Post text",
  "mediaUrls": ["https://example.com/photo.jpg"]
}
```

```http
DELETE /internal/recommendation/posts/{postId}/embedding
```

The model is loaded lazily on the first post embedding request. `PUT` upserts the row; `DELETE` is idempotent and returns `204`.

## GraphQL

Endpoint: `POST /graphql`

`recommendFeed` is viewer-specific. Gateway must send headers generated from the validated session; frontend must never provide these trusted headers directly:

```http
X-Gateway-Secret: <shared secret at least 32 bytes>
X-User-Id: <authenticated user ID>
X-Correlation-ID: <trace ID>
```

The `userId` argument must match `X-User-Id`.

```graphql
query RecommendedFeed($userId: ID!, $skip: Int!, $take: Int!) {
  recommendFeed(userId: $userId, skip: $skip, take: $take) {
    postId
  }
}
```

Example variables:

```json
{
  "userId": "9000000000000001",
  "skip": 0,
  "take": 20
}
```

The public result deliberately contains only ranked `postId` values. Ranking diagnostics are internal implementation details, not a frontend contract. In the composed Gateway schema, each item also has SocialGraph's nullable `post` field, including group metadata for `GroupPostDetail`. There is no GraphQL mutation for embedding writes.

## Candidate and Ranking Flow

1. Recommendation calls `GET {SOCIAL_GRAPH_BASE_URL}/internal/recommendation/post-candidate-ids?userId=...&limit=...` with the shared secret and correlation ID.
2. SocialGraph returns a deduplicated JSON array of positive signed 64-bit post IDs after privacy and block filtering.
3. Recommendation loads the user's vector and available candidate post vectors from its own database.
4. It ranks by semantic dot product. Candidates without a stored embedding receive score `0`; Python's stable sort preserves SocialGraph order for ties.
5. It applies bounded offset pagination (`skip >= 0`, `take` clamped to `1..100`) and returns only IDs.
6. Fusion uses SocialGraph's internal `RecommendationItem` lookup to batch-hydrate `post`; deleted or newly unauthorized posts become `post: null`.

## Registration Integration

The normal registration sequence is:

```text
Gateway createUser
  -> SocialGraph creates canonical userId
  -> Authentication POST /internal/users (required)
  -> Search index PUT and Recommendation user embedding PUT (concurrent, best-effort)
  -> SocialGraph returns userId
```

Recommendation receives exactly the canonical ID generated by SocialGraph. Retrying the `PUT` does not create a duplicate embedding.

## Tests

Install the lightweight test dependencies and run pytest:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The suite covers internal authentication, fail-closed configuration, correlation propagation, Snowflake IDs, idempotent user creation, canonical post/delete contracts, trusted GraphQL viewer enforcement, ID-only schema output, SocialGraph candidate request validation/deduplication, and semantic ranking order.
