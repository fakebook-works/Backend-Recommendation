# Recommendation Module Notes

## Module Responsibilities

- `EmbeddingModel.py` defines FastAPI middleware, internal REST routes, and the read-only Strawberry schema.
- `operations.py` validates signed 64-bit IDs and coordinates database/model operations.
- `database.py` owns SQLAlchemy setup and pgvector serialization.
- `embedding_service.py` lazily loads the text/image models and creates 512-dimensional vectors.
- `recommendation_service.py` fetches SocialGraph candidates and applies hybrid ranking.

Use package imports and start the service with:

```powershell
python -m uvicorn ForFakebook.EmbeddingModel:app --port 8000
```

Starting `EmbeddingModel.py` as a top-level file is unsupported because the application uses package-relative imports.

## Internal Write Boundary

Embedding writes are service-to-service operations:

```text
PUT    /internal/recommendation/users/{userId}/embedding
DELETE /internal/recommendation/users/{userId}/embedding
PUT    /internal/recommendation/posts/{postId}/embedding
DELETE /internal/recommendation/posts/{postId}/embedding
```

All four routes require `X-Gateway-Secret`. User and post IDs are path parameters; post upsert is the only write route with a JSON body.

The GraphQL schema deliberately has no mutation type. This prevents frontend callers from creating or deleting derived vectors directly.

## Model Loading

Model imports and weights are deferred until a post embedding is requested. Therefore:

- `/health` does not require model weights.
- User registration does not load CLIP or SentenceTransformer.
- Unit tests can validate HTTP and orchestration contracts without downloading production models.

## Database Isolation

This service queries only `user_embeddings` and `post_embeddings`. Candidate retrieval goes through SocialGraph REST with the same shared secret and correlation ID used by the incoming feed request.

## Test Boundary

The automated suite replaces database/model operations at the FastAPI dependency boundary and separately tests candidate request construction and ranking. A deployment smoke test should additionally verify PostgreSQL/pgvector connectivity and one real model inference using deployment-specific model caches.
