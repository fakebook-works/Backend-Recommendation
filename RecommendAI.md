# Recommendation Design

## Ownership

Recommendation owns only derived vector data and ranking logic:

- `user_embeddings`: one 512-dimensional preference vector per canonical user ID.
- `post_embeddings`: one 512-dimensional multimodal vector per canonical post ID.

SocialGraph owns users, posts, relationships, blocks, privacy, and candidate generation. Recommendation must not read or write SocialGraph tables directly.

## Candidate Generation

For each feed request, Recommendation calls SocialGraph:

```http
GET /internal/recommendation/post-candidate-ids?userId={userId}&limit={limit}
X-Gateway-Secret: <shared secret>
X-Correlation-ID: <trace id>
```

Response la JSON array ID-only, vi du `[789, 788, 777]`. SocialGraph chiu trach nhiem social source selection, deduplication, privacy va block filtering truoc khi tra pool. Recommendation validate moi ID la positive signed 64-bit va khong phu thuoc vao source label.

## Embeddings

Post embeddings combine text and any downloadable image/video media. Models are loaded lazily so health checks and user provisioning do not download model weights.

New users receive a normalized deterministic initial vector derived from their canonical ID. This makes retry behavior stable while no interaction-history initializer exists. The endpoint uses idempotent `PUT`, so an existing user vector is not replaced during registration retries.

## Ranking

For every candidate with a stored post embedding:

```text
semanticScore = dot(normalizedUserVector, normalizedPostVector)
```

Candidates without a post embedding receive `semanticScore = 0`. Results are stable-sorted by semantic score, paginated after ranking, and GraphQL returns only `postId`; score fields are not a public contract.

Gateway Fusion uses SocialGraph's internal `RecommendationItem(postId)` lookup to add nullable `post: HomePost` to each item. SocialGraph performs a batched authorization-aware read and returns group metadata for group posts.

## Failure Model

- Authentication remains the required registration dependency.
- User embedding provisioning occurs only after Authentication succeeds.
- Recommendation provisioning is best-effort from SocialGraph and safe to retry.
- Recommendation feed errors should not change SocialGraph source data.
- Internal endpoints fail closed if shared-secret authentication is not configured.
- `recommendFeed` requires matching trusted `X-Gateway-Secret`, `X-User-Id`, and `userId` argument.

Durable retry/outbox handling is a later step; current failed best-effort calls are logged with a correlation ID.
