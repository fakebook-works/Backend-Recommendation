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
recommendation_interactions.sql Additive personalization feedback ledger
```

## Requirements

- Python 3.11
- PostgreSQL with the `vector` extension
- The three schema files in this repository

Create and activate a virtual environment, then install production dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Apply the idempotent schemas with the database migration owner before starting the
runtime service:

```powershell
psql -d fakebook -f .\user_embedding.sql
psql -d fakebook -f .\post_embedding.sql
psql -d fakebook -f .\recommendation_interactions.sql
```

The runtime `fakebook_recommendation` role intentionally has no schema `CREATE`
permission and never executes DDL. `/health/ready` returns `503` when these owner-run
migrations are missing or unreadable instead of attempting to elevate runtime access.
The two embedding tables use `vector(512)` and a `BIGINT` primary key; the interaction
table is the idempotent personalization ledger.

## Configuration

Set configuration through environment variables. Do not commit production secrets.

```powershell
$env:DATABASE_URL="postgresql://fakebook:<password>@<host>:5432/fakebook?options=-csearch_path%3Drecommendation%2Cpublic"
$env:INTERNAL_SHARED_SECRET="replace-with-at-least-32-bytes"
$env:RECOMMENDATION_INTERNAL_SECRET="replace-with-a-different-secret-at-least-32-bytes"
$env:SOCIAL_GRAPH_SERVICE_SECRET="replace-with-a-different-secret-at-least-32-bytes"
$env:SOCIAL_GRAPH_BASE_URL="http://localhost:1002"
$env:INTERNAL_AUTH_REQUIRE_SIGNATURE="true"
$env:INTERNAL_AUTH_SEND_LEGACY_SECRET="false"
$env:RECOMMENDATION_MEDIA_ALLOWED_HOSTS="trusted-media-host.example"
$env:RECOMMENDATION_MEDIA_BASE_URL="https://trusted-media-host.example"
$env:RECOMMENDATION_MEDIA_REQUIRE_ALLOWLIST="true"
```

Recommendation tables live in the `recommendation` PostgreSQL schema. Keep the
password in environment configuration and never commit it to the repository.

`INTERNAL_SHARED_SECRET` authenticates Gateway requests to public GraphQL.
`RECOMMENDATION_INTERNAL_SECRET` signs SocialGraph calls to Recommendation's
embedding REST API. Managed environments require:

```http
X-Internal-Timestamp: <unix seconds>
X-Internal-Nonce: <single-use 32-character hex nonce>
X-Internal-Signature: <HMAC-SHA256>
X-Correlation-ID: <optional trace id>
```

The service returns `403` for missing, invalid, stale or replayed signatures and fails
closed with `503` when the configured signing key is shorter than 32 bytes. Bare service
runs retain an explicit legacy migration mode; Compose and `start-local.ps1` disable it.

`SOCIAL_GRAPH_SERVICE_SECRET` is the independent outbound signing key used when
Recommendation calls SocialGraph candidate endpoints. Managed environments send only
the HMAC headers; they never reuse `X-Gateway-Secret` or transmit the raw key.

Media URLs are user-controlled input. Production therefore requires an exact hostname
allowlist, rejects loopback/link-local/reserved/private destinations unless that private
hostname is explicitly allowlisted, disables redirects, and caps both declared and
actually streamed bytes. Videos are downloaded to a bounded temporary file before
OpenCV reads them; OpenCV never receives a remote URL.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn ForFakebook.EmbeddingModel:app --host 0.0.0.0 --port 1003
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
  "content": "Post text or an empty string for media-only posts",
  "mediaUrls": ["https://example.com/photo.jpg"]
}
```

```http
DELETE /internal/recommendation/posts/{postId}/embedding
```

At least one non-blank `content` value or media URL is required. Media-only posts
use the processed media embedding without an empty-text contribution. The model
is loaded lazily on the first post embedding request. `PUT` upserts the row;
`DELETE` is idempotent and returns `204`.

### Personalization Feedback

```http
POST /internal/recommendation/users/{userId}/interactions
X-Internal-Timestamp: <unix seconds>
X-Internal-Nonce: <single-use nonce>
X-Internal-Signature: <HMAC-SHA256>
Idempotency-Key: <stable SocialGraph outbox event id>
Content-Type: application/json

{
  "targetId": 9000000000000002,
  "action": "SAVE"
}
```

Supported actions are `LIKE`, `UNLIKE`, `SAVE`, `UNSAVE`, `WATCH`, `SHARE`, and
`COMMENT`. Each first-seen event applies a signed normalized EMA step to the user
vector. The ledger insert and vector update commit atomically; concurrent feedback for
one user is serialized. Replayed idempotency keys return `applied: false`. A target whose
post embedding is not ready returns HTTP 425 so the SocialGraph outbox retries it.
Deleting a user or post embedding also removes its owned feedback-ledger rows.

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

The public result deliberately contains only ranked `postId` values. The field name is retained for backward compatibility, but the value is a SocialGraph content-object ID and Home candidates may be `FeedPost`, `GroupPost`, or `Reel`. Ranking diagnostics are internal implementation details, not a frontend contract. In the composed Gateway schema, each item also has SocialGraph's nullable `post` field, hydrated as `FeedPostDetail`, `GroupPostDetail`, or `ReelDetail`. There is no GraphQL mutation for embedding writes.

Reel ranking uses the same trusted-viewer boundary and embedding store. `FOR_YOU`
uses all SocialGraph-approved candidates; `FOLLOWING` keeps candidates authored by
accepted friends or users the viewer actively follows (SocialGraph has already
applied privacy, block and membership policy):

```graphql
query RecommendedReels($userId: ID!, $mode: ReelRecommendationMode!) {
  recommendReels(userId: $userId, mode: $mode, skip: 0, take: 20) {
    reelId
  }
}
```

Gateway authentication uses `INTERNAL_SHARED_SECRET`. Calls into Recommendation's
internal embedding endpoints use `RECOMMENDATION_INTERNAL_SECRET`. Recommendation
calls SocialGraph with the independent `SOCIAL_GRAPH_SERVICE_SECRET`.

## Candidate and Ranking Flow

1. Recommendation calls `GET {SOCIAL_GRAPH_BASE_URL}/internal/recommendation/post-candidate-ids?userId=...&limit=...` with a timestamped HMAC derived from `SOCIAL_GRAPH_SERVICE_SECRET` and the correlation ID.
2. SocialGraph returns a deduplicated JSON array of positive signed 64-bit content IDs for eligible `FeedPost`, `GroupPost`, and `Reel` objects after privacy and block filtering. Recommendation treats these IDs as opaque and does not infer object type or visibility from an embedding.
3. Recommendation loads the interaction-trained user vector and available candidate post vectors from its own database.
4. It ranks by semantic dot product. Candidates without a stored embedding receive score `0`; Python's stable sort preserves SocialGraph order for ties.
5. It applies bounded offset pagination (`skip >= 0`, `take` clamped to `1..100`) and returns only IDs.
6. Fusion uses SocialGraph's internal `RecommendationItem` lookup to batch-hydrate `post`; SocialGraph checks visibility again and returns `FeedPostDetail`, `GroupPostDetail`, or `ReelDetail`. Deleted or newly unauthorized content becomes `post: null`.

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

The suite covers internal authentication, fail-closed configuration, correlation propagation, Snowflake IDs, idempotent user creation, canonical post/delete contracts, idempotent interaction learning, normalized vector updates, trusted GraphQL viewer enforcement, ID-only schema output, SocialGraph candidate request validation/deduplication, and semantic ranking order.
