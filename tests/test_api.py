import os

import pytest
from fastapi.testclient import TestClient

from ForFakebook.EmbeddingModel import (
    CORRELATION_HEADER,
    GATEWAY_SECRET_HEADER,
    RECOMMENDATION_INTERNAL_SECRET_HEADER,
    USER_ID_HEADER,
    app,
)
from ForFakebook.operations import InteractionTargetUnavailableError, get_operations


GATEWAY_SHARED_SECRET = "gateway-test-shared-secret-at-least-32-bytes"
SOCIAL_GRAPH_SHARED_SECRET = "social-graph-test-shared-secret-at-least-32-bytes"
SNOWFLAKE_ID = 9_000_000_000_000_001


class FakeOperations:
    def __init__(self):
        self.calls = []
        self.user_exists = False

    def ensure_user_embedding(self, user_id):
        self.calls.append(("ensure_user", user_id))
        created = not self.user_exists
        self.user_exists = True
        return created

    def delete_user_embedding(self, user_id):
        self.calls.append(("delete_user", user_id))

    def upsert_post_embedding(self, post_id, content, media_urls):
        self.calls.append(("upsert_post", post_id, content, media_urls))

    def delete_post_embedding(self, post_id):
        self.calls.append(("delete_post", post_id))

    def record_interaction(self, user_id, target_id, action, idempotency_key):
        self.calls.append(
            ("record_interaction", user_id, target_id, action, idempotency_key)
        )
        if target_id == SNOWFLAKE_ID + 99:
            raise InteractionTargetUnavailableError("Target embedding is not available yet.")
        return not idempotency_key.endswith("duplicate")

    def recommend_feed(self, user_id, skip, take, correlation_id=None):
        self.calls.append(("recommend", user_id, skip, take, correlation_id))
        return [
            {
                "postId": SNOWFLAKE_ID + 1,
            }
        ]

    def recommend_reels(self, user_id, mode, skip, take, correlation_id=None):
        self.calls.append(("recommend_reels", user_id, mode, skip, take, correlation_id))
        return [{"reelId": SNOWFLAKE_ID + 2}]


@pytest.fixture()
def api(monkeypatch):
    fake = FakeOperations()
    monkeypatch.setenv("INTERNAL_SHARED_SECRET", GATEWAY_SHARED_SECRET)
    monkeypatch.setenv("RECOMMENDATION_INTERNAL_SECRET", SOCIAL_GRAPH_SHARED_SECRET)
    monkeypatch.setenv("SOCIAL_GRAPH_SERVICE_SECRET", SOCIAL_GRAPH_SHARED_SECRET)
    app.dependency_overrides[get_operations] = lambda: fake
    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()


def internal_headers(correlation_id=None):
    headers = {RECOMMENDATION_INTERNAL_SECRET_HEADER: SOCIAL_GRAPH_SHARED_SECRET}
    if correlation_id:
        headers[CORRELATION_HEADER] = correlation_id
    return headers


def test_internal_routes_reject_missing_and_invalid_secret(api):
    client, _ = api
    path = f"/internal/recommendation/users/{SNOWFLAKE_ID}/embedding"

    assert client.put(path).status_code == 403
    assert client.put(path, headers={RECOMMENDATION_INTERNAL_SECRET_HEADER: "wrong"}).status_code == 403
    assert client.put(path, headers={RECOMMENDATION_INTERNAL_SECRET_HEADER: b"\xffinvalid"}).status_code == 403
    assert client.put(
        path,
        headers={"X-Internal-SocialGraphService-Secret": SOCIAL_GRAPH_SHARED_SECRET},
    ).status_code == 403
    assert client.put(path, headers={GATEWAY_SECRET_HEADER: GATEWAY_SHARED_SECRET}).status_code == 403


def test_internal_auth_matches_path_segment_only(api):
    client, _ = api

    response = client.get("/internalevil")

    assert response.status_code == 404
    assert response.headers[CORRELATION_HEADER]


def test_internal_routes_fail_closed_when_secret_is_not_configured(api, monkeypatch):
    client, _ = api
    monkeypatch.delenv("RECOMMENDATION_INTERNAL_SECRET", raising=False)

    response = client.put(
        f"/internal/recommendation/users/{SNOWFLAKE_ID}/embedding",
        headers={RECOMMENDATION_INTERNAL_SECRET_HEADER: SOCIAL_GRAPH_SHARED_SECRET},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "RECOMMENDATION_AUTH_NOT_CONFIGURED"


def test_user_embedding_upsert_supports_snowflake_and_is_idempotent(api):
    client, fake = api
    path = f"/internal/recommendation/users/{SNOWFLAKE_ID}/embedding"

    first = client.put(path, headers=internal_headers("register-correlation"))
    second = client.put(path, headers=internal_headers())

    assert first.status_code == 200
    assert first.json() == {
        "success": True,
        "userId": SNOWFLAKE_ID,
        "created": True,
        "message": "User embedding created.",
    }
    assert first.headers[CORRELATION_HEADER] == "register-correlation"
    assert second.json()["created"] is False
    assert fake.calls == [("ensure_user", SNOWFLAKE_ID), ("ensure_user", SNOWFLAKE_ID)]


def test_post_upsert_and_deletes_follow_canonical_contract(api):
    client, fake = api
    post_id = SNOWFLAKE_ID + 2

    upsert = client.put(
        f"/internal/recommendation/posts/{post_id}/embedding",
        headers=internal_headers(),
        json={"content": "Fakebook AI", "mediaUrls": ["https://example.com/a.jpg"]},
    )
    delete_post = client.delete(
        f"/internal/recommendation/posts/{post_id}/embedding",
        headers=internal_headers(),
    )
    delete_user = client.delete(
        f"/internal/recommendation/users/{SNOWFLAKE_ID}/embedding",
        headers=internal_headers(),
    )

    assert upsert.status_code == 200
    assert upsert.json() == {"success": True, "postId": post_id}
    assert delete_post.status_code == 204
    assert delete_user.status_code == 204
    assert ("upsert_post", post_id, "Fakebook AI", ["https://example.com/a.jpg"]) in fake.calls
    assert ("delete_post", post_id) in fake.calls
    assert ("delete_user", SNOWFLAKE_ID) in fake.calls


def test_post_upsert_accepts_media_without_content(api):
    client, fake = api
    post_id = SNOWFLAKE_ID + 3

    response = client.put(
        f"/internal/recommendation/posts/{post_id}/embedding",
        headers=internal_headers(),
        json={"mediaUrls": ["https://example.com/media-without-extension"]},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "postId": post_id}
    assert ("upsert_post", post_id, "", ["https://example.com/media-without-extension"]) in fake.calls


def test_post_upsert_rejects_empty_content_and_media(api):
    client, fake = api

    response = client.put(
        f"/internal/recommendation/posts/{SNOWFLAKE_ID + 4}/embedding",
        headers=internal_headers(),
        json={"content": "   ", "mediaUrls": ["   "]},
    )

    assert response.status_code == 422
    assert not any(call[0] == "upsert_post" for call in fake.calls)


def test_recommendation_interaction_is_authenticated_and_idempotent(api):
    client, fake = api
    path = f"/internal/recommendation/users/{SNOWFLAKE_ID}/interactions"
    body = {"targetId": SNOWFLAKE_ID + 7, "action": "SAVE"}

    first = client.post(
        path,
        headers={**internal_headers(), "Idempotency-Key": "save-event-1"},
        json=body,
    )
    duplicate = client.post(
        path,
        headers={**internal_headers(), "Idempotency-Key": "save-event-duplicate"},
        json=body,
    )

    assert first.status_code == 200
    assert first.json() == {
        "success": True,
        "applied": True,
        "userId": SNOWFLAKE_ID,
        "targetId": SNOWFLAKE_ID + 7,
        "action": "SAVE",
    }
    assert duplicate.status_code == 200
    assert duplicate.json()["applied"] is False
    assert fake.calls[-2:] == [
        (
            "record_interaction",
            SNOWFLAKE_ID,
            SNOWFLAKE_ID + 7,
            "SAVE",
            "save-event-1",
        ),
        (
            "record_interaction",
            SNOWFLAKE_ID,
            SNOWFLAKE_ID + 7,
            "SAVE",
            "save-event-duplicate",
        ),
    ]


def test_recommendation_interaction_requires_idempotency_key_and_valid_action(api):
    client, fake = api
    path = f"/internal/recommendation/users/{SNOWFLAKE_ID}/interactions"

    missing_key = client.post(
        path,
        headers=internal_headers(),
        json={"targetId": SNOWFLAKE_ID + 7, "action": "LIKE"},
    )
    invalid_action = client.post(
        path,
        headers={**internal_headers(), "Idempotency-Key": "invalid-action"},
        json={"targetId": SNOWFLAKE_ID + 7, "action": "CLICK"},
    )

    assert missing_key.status_code == 422
    assert invalid_action.status_code == 422
    assert not any(call[0] == "record_interaction" for call in fake.calls)


def test_recommendation_interaction_retries_when_target_embedding_is_late(api):
    client, _ = api

    response = client.post(
        f"/internal/recommendation/users/{SNOWFLAKE_ID}/interactions",
        headers={**internal_headers(), "Idempotency-Key": "late-target"},
        json={"targetId": SNOWFLAKE_ID + 99, "action": "COMMENT"},
    )

    assert response.status_code == 425
    assert "not available" in response.json()["detail"]


def test_graphql_recommend_feed_uses_id_scalar_for_snowflakes(api):
    client, fake = api
    response = client.post(
        "/graphql",
        json={
            "query": "query Feed($userId: ID!) { recommendFeed(userId: $userId) { postId } }",
            "variables": {"userId": str(SNOWFLAKE_ID)},
        },
        headers={
            CORRELATION_HEADER: "feed-correlation",
            GATEWAY_SECRET_HEADER: GATEWAY_SHARED_SECRET,
            USER_ID_HEADER: str(SNOWFLAKE_ID),
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["recommendFeed"][0]["postId"] == str(SNOWFLAKE_ID + 1)
    assert fake.calls[-1] == ("recommend", SNOWFLAKE_ID, 0, 20, "feed-correlation")


def test_graphql_recommend_reels_supports_following_mode(api):
    client, fake = api
    response = client.post(
        "/graphql",
        json={
            "query": "query Reels($userId: ID!) { recommendReels(userId: $userId, mode: FOLLOWING) { reelId } }",
            "variables": {"userId": str(SNOWFLAKE_ID)},
        },
        headers={
            CORRELATION_HEADER: "reel-correlation",
            GATEWAY_SECRET_HEADER: GATEWAY_SHARED_SECRET,
            USER_ID_HEADER: str(SNOWFLAKE_ID),
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["recommendReels"][0]["reelId"] == str(SNOWFLAKE_ID + 2)
    assert fake.calls[-1] == (
        "recommend_reels",
        SNOWFLAKE_ID,
        "FOLLOWING",
        0,
        20,
        "reel-correlation",
    )


def test_graphql_recommend_feed_rejects_missing_or_mismatched_trusted_user(api):
    client, fake = api
    query = "query Feed($userId: ID!) { recommendFeed(userId: $userId) { postId } }"

    missing = client.post(
        "/graphql",
        json={"query": query, "variables": {"userId": str(SNOWFLAKE_ID)}},
        headers={GATEWAY_SECRET_HEADER: GATEWAY_SHARED_SECRET},
    )
    mismatched = client.post(
        "/graphql",
        json={"query": query, "variables": {"userId": str(SNOWFLAKE_ID)}},
        headers={
            GATEWAY_SECRET_HEADER: GATEWAY_SHARED_SECRET,
            USER_ID_HEADER: str(SNOWFLAKE_ID + 1),
        },
    )

    assert missing.json()["errors"][0]["extensions"]["code"] == "UNAUTHENTICATED"
    assert mismatched.json()["errors"][0]["extensions"]["code"] == "FORBIDDEN"
    assert not any(call[0] == "recommend" for call in fake.calls)


def test_graphql_rejects_social_graph_service_secret(api):
    client, fake = api
    query = "query Feed($userId: ID!) { recommendFeed(userId: $userId) { postId } }"

    response = client.post(
        "/graphql",
        json={"query": query, "variables": {"userId": str(SNOWFLAKE_ID)}},
        headers={
            RECOMMENDATION_INTERNAL_SECRET_HEADER: SOCIAL_GRAPH_SHARED_SECRET,
            USER_ID_HEADER: str(SNOWFLAKE_ID),
        },
    )

    assert response.json()["errors"][0]["extensions"]["code"] == "FORBIDDEN"
    assert not any(call[0] == "recommend" for call in fake.calls)


def test_graphql_recommend_feed_rejects_non_numeric_user_id(api):
    client, fake = api

    response = client.post(
        "/graphql",
        json={
            "query": "query { recommendFeed(userId: \"not-a-number\") { postId } }",
        },
        headers={
            GATEWAY_SECRET_HEADER: GATEWAY_SHARED_SECRET,
            USER_ID_HEADER: "1",
        },
    )

    assert response.json()["errors"][0]["extensions"]["code"] == "BAD_USER_INPUT"
    assert not any(call[0] == "recommend" for call in fake.calls)


def test_graphql_has_no_embedding_mutation_surface(api):
    client, _ = api
    response = client.post(
        "/graphql",
        json={"query": "mutation { initializeUserEmbedding(userId: 1) { success } }"},
    )

    assert response.status_code == 200
    assert "errors" in response.json()
