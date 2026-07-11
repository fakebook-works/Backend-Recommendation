# Recommendation Design

## Ownership

Recommendation owns only derived vector data and ranking logic:

- `user_embeddings`: one 512-dimensional preference vector per canonical user ID.
- `post_embeddings`: one 512-dimensional multimodal vector per canonical post ID.

SocialGraph owns users, posts, relationships, blocks, privacy, and candidate generation. Recommendation must not read or write SocialGraph tables directly.

## Candidate Generation

For each feed request, Recommendation calls SocialGraph:

```http
GET /internal/recommendation/post-candidates?userId={userId}&limit={limit}
X-Gateway-Secret: <shared secret>
X-Correlation-ID: <trace id>
```

Candidate records contain `id`, `authorId`, `source`, and `createdAt`. Current source labels and base scores are:

| Source | Social score |
| --- | ---: |
| `friend` | 1.0 |
| `followed` | 0.9 |
| `group` | 0.7 |
| `recent_public` | 0.3 |
| unknown | 0.2 |

SocialGraph is responsible for candidate deduplication and block filtering before returning this pool.

## Embeddings

Post embeddings combine text and any downloadable image/video media. Models are loaded lazily so health checks and user provisioning do not download model weights.

New users receive a normalized deterministic initial vector derived from their canonical ID. This makes retry behavior stable while no interaction-history initializer exists. The endpoint uses idempotent `PUT`, so an existing user vector is not replaced during registration retries.

## Ranking

For every candidate with a stored post embedding:

```text
semanticScore = dot(normalizedUserVector, normalizedPostVector)
socialScore   = scoreByCandidateSource
score         = 0.6 * semanticScore + 0.4 * socialScore
```

Candidates without a post embedding receive `semanticScore = 0` and can still rank by social source. Results are sorted by final score and paginated after ranking.

## Failure Model

- Authentication remains the required registration dependency.
- User embedding provisioning occurs only after Authentication succeeds.
- Recommendation provisioning is best-effort from SocialGraph and safe to retry.
- Recommendation feed errors should not change SocialGraph source data.
- Internal endpoints fail closed if shared-secret authentication is not configured.

Durable retry/outbox handling is a later step; current failed best-effort calls are logged with a correlation ID.
