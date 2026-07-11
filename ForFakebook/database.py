from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres@localhost:5432/fakebook",
)

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception:
    engine = None
    SessionLocal = None


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def save_post_embedding(db, post_id: int, embedding: Sequence[float]) -> None:
    db.execute(
        text(
            """
            INSERT INTO post_embeddings (post_id, embedding)
            VALUES (:post_id, CAST(:embedding AS vector))
            ON CONFLICT (post_id) DO UPDATE
            SET embedding = EXCLUDED.embedding, updated_at = NOW()
            """
        ),
        {"post_id": post_id, "embedding": vector_literal(embedding)},
    )
    db.commit()


def create_user_embedding_if_missing(db, user_id: int, embedding: Sequence[float]) -> bool:
    result = db.execute(
        text(
            """
            INSERT INTO user_embeddings (user_id, embedding, updated_at)
            VALUES (:user_id, CAST(:embedding AS vector), NOW())
            ON CONFLICT (user_id) DO NOTHING
            """
        ),
        {"user_id": user_id, "embedding": vector_literal(embedding)},
    )
    db.commit()
    return result.rowcount == 1


def parse_vector(value) -> np.ndarray:
    if isinstance(value, str):
        clean_value = value.strip("[]")
        if not clean_value:
            return np.zeros(512)
        return np.array([float(item) for item in clean_value.split(",")])
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    return np.asarray(value, dtype=float)
