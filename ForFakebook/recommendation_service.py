from __future__ import annotations

from collections.abc import Callable

import numpy as np
import requests
from sqlalchemy import text

from .database import parse_vector


SOURCE_SCORES = {
    "friend": 1.0,
    "followed": 0.9,
    "group": 0.7,
    "recent_public": 0.3,
}


def fetch_post_candidates(
    user_id: int,
    limit: int,
    social_graph_base_url: str,
    shared_secret: str,
    correlation_id: str | None = None,
    http_get: Callable = requests.get,
) -> list[dict]:
    headers = {"X-Gateway-Secret": shared_secret}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    response = http_get(
        f"{social_graph_base_url.rstrip('/')}/internal/recommendation/post-candidates",
        params={"userId": user_id, "limit": limit},
        headers=headers,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


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
    candidates = fetch_post_candidates(
        user_id,
        min(500, normalized_skip + normalized_take + 200),
        social_graph_base_url,
        shared_secret,
        correlation_id,
    )
    candidate_ids = [int(candidate["id"]) for candidate in candidates]
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

    ranked: list[dict] = []
    for candidate in candidates:
        post_id = int(candidate["id"])
        post_embedding = post_embeddings.get(post_id)
        semantic_score = 0.0
        if post_embedding is not None and np.linalg.norm(user_embedding) > 0:
            semantic_score = float(np.dot(user_embedding, post_embedding))

        social_score = SOURCE_SCORES.get(str(candidate.get("source", "")), 0.2)
        final_score = 0.6 * semantic_score + 0.4 * social_score
        ranked.append(
            {
                "postId": post_id,
                "score": final_score,
                "semanticScore": semantic_score,
                "socialScore": social_score,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[normalized_skip : normalized_skip + normalized_take]
