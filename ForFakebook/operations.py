from __future__ import annotations

import os

import numpy as np
from sqlalchemy import text

from .database import SessionLocal, create_user_embedding_if_missing, save_post_embedding
from .embedding_service import generate_multimodal_embedding
from .recommendation_service import recommend_feed_logic


class RecommendationUnavailableError(RuntimeError):
    pass


class RecommendationOperations:
    def __init__(self, session_factory=SessionLocal, embedding_generator=generate_multimodal_embedding):
        self._session_factory = session_factory
        self._embedding_generator = embedding_generator

    def ensure_user_embedding(self, user_id: int) -> bool:
        self._validate_id(user_id, "user_id")
        random = np.random.default_rng(user_id)
        embedding = random.normal(size=512)
        embedding = embedding / np.linalg.norm(embedding)
        with self._session() as db:
            return create_user_embedding_if_missing(db, user_id, embedding.tolist())

    def delete_user_embedding(self, user_id: int) -> None:
        self._validate_id(user_id, "user_id")
        with self._session() as db:
            db.execute(
                text("DELETE FROM user_embeddings WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
            db.commit()

    def upsert_post_embedding(self, post_id: int, content: str, media_urls: list[str]) -> None:
        self._validate_id(post_id, "post_id")
        if not content.strip():
            raise ValueError("content is required")

        embedding = self._embedding_generator(content, media_urls)
        with self._session() as db:
            save_post_embedding(db, post_id, embedding)

    def delete_post_embedding(self, post_id: int) -> None:
        self._validate_id(post_id, "post_id")
        with self._session() as db:
            db.execute(
                text("DELETE FROM post_embeddings WHERE post_id = :post_id"),
                {"post_id": post_id},
            )
            db.commit()

    def recommend_feed(
        self,
        user_id: int,
        skip: int,
        take: int,
        correlation_id: str | None = None,
    ) -> list[dict]:
        self._validate_id(user_id, "user_id")
        social_graph_url = os.getenv("SOCIAL_GRAPH_BASE_URL", "http://localhost:5223")
        shared_secret = os.getenv("INTERNAL_SHARED_SECRET", "")
        if len(shared_secret.encode("utf-8")) < 32:
            raise RecommendationUnavailableError("INTERNAL_SHARED_SECRET is not configured.")

        with self._session() as db:
            return recommend_feed_logic(
                db,
                user_id,
                social_graph_url,
                shared_secret,
                skip,
                take,
                correlation_id,
            )

    def _session(self):
        if self._session_factory is None:
            raise RecommendationUnavailableError("Database connection is not configured.")
        return self._session_factory()

    @staticmethod
    def _validate_id(value: int, name: str) -> None:
        if value <= 0 or value > 9_223_372_036_854_775_807:
            raise ValueError(f"{name} must be a positive signed 64-bit integer")


_operations = RecommendationOperations()


def get_operations() -> RecommendationOperations:
    return _operations
