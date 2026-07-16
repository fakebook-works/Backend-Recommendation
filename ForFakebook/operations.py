from __future__ import annotations

import os

import numpy as np
from sqlalchemy import text

from .database import (
    SessionLocal,
    create_user_embedding_if_missing,
    ensure_interaction_schema,
    parse_vector,
    save_post_embedding,
    vector_literal,
)
from .embedding_service import generate_multimodal_embedding
from .recommendation_service import recommend_feed_logic, recommend_reels_logic


INTERACTION_WEIGHTS = {
    "LIKE": 0.25,
    "UNLIKE": -0.15,
    "SAVE": 0.35,
    "UNSAVE": -0.20,
    "WATCH": 0.08,
    "SHARE": 0.40,
    "COMMENT": 0.20,
}


class RecommendationUnavailableError(RuntimeError):
    pass


class InteractionTargetUnavailableError(RecommendationUnavailableError):
    """The interaction arrived before its target embedding was materialized."""


def deterministic_user_embedding(user_id: int) -> np.ndarray:
    random = np.random.default_rng(user_id)
    embedding = random.normal(size=512)
    return embedding / np.linalg.norm(embedding)


def apply_interaction_to_embedding(
    current_embedding: np.ndarray,
    target_embedding: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Apply a signed, normalized EMA step without changing vector magnitude."""
    current = np.asarray(current_embedding, dtype=float)
    target = np.asarray(target_embedding, dtype=float)
    if current.shape != target.shape or current.ndim != 1:
        raise ValueError("current and target embeddings must be one-dimensional and equal in size")

    current_norm = np.linalg.norm(current)
    target_norm = np.linalg.norm(target)
    if current_norm == 0 or target_norm == 0:
        raise ValueError("interaction embeddings must have non-zero magnitude")

    current = current / current_norm
    target = target / target_norm
    candidate = (1.0 - abs(weight)) * current + weight * target
    candidate_norm = np.linalg.norm(candidate)
    if candidate_norm == 0:
        return current
    return candidate / candidate_norm


class RecommendationOperations:
    def __init__(self, session_factory=SessionLocal, embedding_generator=generate_multimodal_embedding):
        self._session_factory = session_factory
        self._embedding_generator = embedding_generator

    def ensure_user_embedding(self, user_id: int) -> bool:
        self._validate_id(user_id, "user_id")
        embedding = deterministic_user_embedding(user_id)
        with self._session() as db:
            return create_user_embedding_if_missing(db, user_id, embedding.tolist())

    def record_interaction(
        self,
        user_id: int,
        target_id: int,
        action: str,
        idempotency_key: str,
    ) -> bool:
        self._validate_id(user_id, "user_id")
        self._validate_id(target_id, "target_id")
        normalized_action = action.strip().upper()
        if normalized_action not in INTERACTION_WEIGHTS:
            raise ValueError(f"unsupported recommendation interaction action: {action}")
        normalized_key = idempotency_key.strip()
        if not normalized_key or len(normalized_key) > 128:
            raise ValueError("Idempotency-Key must contain between 1 and 128 characters")

        weight = INTERACTION_WEIGHTS[normalized_action]
        with self._session() as db:
            try:
                ensure_interaction_schema(db)
                claimed = db.execute(
                    text(
                        """
                        INSERT INTO recommendation_interactions
                            (idempotency_key, user_id, target_id, action, weight)
                        VALUES
                            (:idempotency_key, :user_id, :target_id, :action, :weight)
                        ON CONFLICT (idempotency_key) DO NOTHING
                        """
                    ),
                    {
                        "idempotency_key": normalized_key,
                        "user_id": user_id,
                        "target_id": target_id,
                        "action": normalized_action,
                        "weight": weight,
                    },
                )
                if claimed.rowcount != 1:
                    db.commit()
                    return False

                # Serialize all feedback for one user, including the first event before a
                # user_embeddings row exists, so concurrent events cannot lose updates.
                db.execute(
                    text("SELECT pg_advisory_xact_lock(:user_id)"),
                    {"user_id": user_id},
                )
                target_row = db.execute(
                    text("SELECT embedding FROM post_embeddings WHERE post_id = :target_id"),
                    {"target_id": target_id},
                ).first()
                if target_row is None:
                    raise InteractionTargetUnavailableError(
                        f"Embedding for recommendation target {target_id} is not available yet."
                    )

                target_embedding = parse_vector(target_row[0])
                user_row = db.execute(
                    text(
                        "SELECT embedding FROM user_embeddings "
                        "WHERE user_id = :user_id FOR UPDATE"
                    ),
                    {"user_id": user_id},
                ).first()
                if user_row is None:
                    current_embedding = (
                        target_embedding
                        if weight > 0
                        else deterministic_user_embedding(user_id)
                    )
                else:
                    current_embedding = parse_vector(user_row[0])

                updated_embedding = apply_interaction_to_embedding(
                    current_embedding,
                    target_embedding,
                    weight,
                )
                db.execute(
                    text(
                        """
                        INSERT INTO user_embeddings (user_id, embedding)
                        VALUES (:user_id, CAST(:embedding AS vector))
                        ON CONFLICT (user_id) DO UPDATE
                        SET embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "user_id": user_id,
                        "embedding": vector_literal(updated_embedding),
                    },
                )
                db.commit()
                return True
            except Exception:
                db.rollback()
                raise

    def delete_user_embedding(self, user_id: int) -> None:
        self._validate_id(user_id, "user_id")
        with self._session() as db:
            try:
                ensure_interaction_schema(db)
                db.execute(
                    text("DELETE FROM recommendation_interactions WHERE user_id = :user_id"),
                    {"user_id": user_id},
                )
                db.execute(
                    text("DELETE FROM user_embeddings WHERE user_id = :user_id"),
                    {"user_id": user_id},
                )
                db.commit()
            except Exception:
                db.rollback()
                raise

    def upsert_post_embedding(self, post_id: int, content: str, media_urls: list[str]) -> None:
        self._validate_id(post_id, "post_id")
        if not content.strip() and not any(url.strip() for url in media_urls):
            raise ValueError("content or at least one media URL is required")

        embedding = self._embedding_generator(content, media_urls)
        with self._session() as db:
            save_post_embedding(db, post_id, embedding)

    def delete_post_embedding(self, post_id: int) -> None:
        self._validate_id(post_id, "post_id")
        with self._session() as db:
            try:
                ensure_interaction_schema(db)
                db.execute(
                    text("DELETE FROM recommendation_interactions WHERE target_id = :post_id"),
                    {"post_id": post_id},
                )
                db.execute(
                    text("DELETE FROM post_embeddings WHERE post_id = :post_id"),
                    {"post_id": post_id},
                )
                db.commit()
            except Exception:
                db.rollback()
                raise

    def recommend_feed(
        self,
        user_id: int,
        skip: int,
        take: int,
        correlation_id: str | None = None,
    ) -> list[dict]:
        self._validate_id(user_id, "user_id")
        social_graph_url = os.getenv("SOCIAL_GRAPH_BASE_URL", "http://localhost:1002")
        shared_secret = os.getenv("SOCIAL_GRAPH_SERVICE_SECRET", "")
        if len(shared_secret.encode("utf-8")) < 32:
            raise RecommendationUnavailableError("SOCIAL_GRAPH_SERVICE_SECRET is not configured.")

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

    def recommend_reels(
        self,
        user_id: int,
        mode: str,
        skip: int,
        take: int,
        correlation_id: str | None = None,
    ) -> list[dict]:
        self._validate_id(user_id, "user_id")
        social_graph_url = os.getenv("SOCIAL_GRAPH_BASE_URL", "http://localhost:1002")
        shared_secret = os.getenv("SOCIAL_GRAPH_SERVICE_SECRET", "")
        if len(shared_secret.encode("utf-8")) < 32:
            raise RecommendationUnavailableError("SOCIAL_GRAPH_SERVICE_SECRET is not configured.")

        with self._session() as db:
            return recommend_reels_logic(
                db,
                user_id,
                social_graph_url,
                shared_secret,
                mode,
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
