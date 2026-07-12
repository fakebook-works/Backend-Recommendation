from __future__ import annotations

from collections.abc import Callable

import numpy as np
import requests
from sqlalchemy import text

from .database import parse_vector


MAX_SIGNED_64_BIT_ID = 9_223_372_036_854_775_807


def fetch_post_candidate_ids(
    user_id: int,
    limit: int,
    social_graph_base_url: str,
    shared_secret: str,
    correlation_id: str | None = None,
    http_get: Callable = requests.get,
) -> list[int]:
    headers = {"X-Gateway-Secret": shared_secret}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    response = http_get(
        f"{social_graph_base_url.rstrip('/')}/internal/recommendation/post-candidate-ids",
        params={"userId": user_id, "limit": limit},
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("SocialGraph candidate response must be a JSON array of IDs.")

    candidate_ids: list[int] = []
    seen: set[int] = set()
    for value in payload:
        if type(value) is not int:
            raise ValueError("SocialGraph returned an invalid post candidate ID.")

        post_id = value
        if post_id <= 0 or post_id > MAX_SIGNED_64_BIT_ID:
            raise ValueError("SocialGraph returned an invalid post candidate ID.")
        if post_id not in seen:
            seen.add(post_id)
            candidate_ids.append(post_id)

    return candidate_ids


def recommend_feed_logic(
    db,
    user_id: int,
    social_graph_base_url: str,
    shared_secret: str,
    skip: int = 0,
    take: int = 20,
    correlation_id: str | None = None,
) -> list[dict]:
    normalized_skip = max(0, skip)
    normalized_take = min(max(1, take), 100)
    candidate_ids = fetch_post_candidate_ids(
        user_id,
        min(500, normalized_skip + normalized_take + 200),
        social_graph_base_url,
        shared_secret,
        correlation_id,
    )
    if not candidate_ids:
        return []

    user_row = db.execute(
        text("SELECT embedding FROM user_embeddings WHERE user_id = :user_id"),
        {"user_id": user_id},
    ).fetchone()
    user_embedding = parse_vector(user_row[0]) if user_row else np.zeros(512)

    post_rows = db.execute(
        text(
            """
            SELECT post_id, embedding
            FROM post_embeddings
            WHERE post_id = ANY(:candidate_ids)
            """
        ),
        {"candidate_ids": candidate_ids},
    ).fetchall()
    post_embeddings = {int(row[0]): parse_vector(row[1]) for row in post_rows}

    user_norm = np.linalg.norm(user_embedding)
    ranked: list[tuple[int, float]] = []
    for post_id in candidate_ids:
        post_embedding = post_embeddings.get(post_id)
        semantic_score = 0.0
        if post_embedding is not None and user_norm > 0:
            semantic_score = float(np.dot(user_embedding, post_embedding))
        ranked.append((post_id, semantic_score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [
        {"postId": post_id}
        for post_id, _ in ranked[normalized_skip : normalized_skip + normalized_take]
    ]
