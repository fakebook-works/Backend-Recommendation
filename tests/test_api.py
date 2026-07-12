import os

import pytest
from fastapi.testclient import TestClient

from ForFakebook.EmbeddingModel import (
    CORRELATION_HEADER,
    INTERNAL_SECRET_HEADER,
    USER_ID_HEADER,
    app,
)
from ForFakebook.operations import get_operations


SHARED_SECRET = "recommendation-test-shared-secret-at-least-32-bytes"
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

    def recommend_feed(self, user_id, skip, take, correlation_id=None):
        self.calls.append(("recommend", user_id, skip, take, correlation_id))
        return [
            {
                "postId": SNOWFLAKE_ID + 1,
            }
        ]


@pytest.fixture()
def api(monkeypatch):
    fake = FakeOperations()
    monkeypatch.setenv("INTERNAL_SHARED_SECRET", SHARED_SECRET)
    app.dependency_overrides[get_operations] = lambda: fake
    with TestClient(app) as client:
        yield client, fake
    app.dependency_overrides.clear()


def internal_headers(correlation_id=None):
    headers = {INTERNAL_SECRET_HEADER: SHARED_SECRET}
    if correlation_id:
        headers[CORRELATION_HEADER] = correlation_id
    return headers


def test_internal_routes_reject_missing_and_invalid_secret(api):
    client, _ = api
    path = f"/internal/recommendation/users/{SNOWFLAKE_ID}/embedding"

    assert client.put(path).status_code == 403
    assert client.put(path, headers={INTERNAL_SECRET_HEADER: "wrong"}).status_code == 403
    assert client.put(path, headers={INTERNAL_SECRET_HEADER: b"\xffinvalid"}).status_code == 403


def test_internal_auth_matches_path_segment_only(api):
    client, _ = api

    response = client.get("/internalevil")

    assert response.status_code == 404
    assert response.headers[CORRELATION_HEADER]


def test_internal_routes_fail_closed_when_secret_is_not_configured(api, monkeypatch):
    client, _ = api
    monkeypatch.delenv("INTERNAL_SHARED_SECRET", raising=False)

    response = client.put(
        f"/internal/recommendation/users/{SNOWFLAKE_ID}/embedding",
        headers={INTERNAL_SECRET_HEADER: SHARED_SECRET},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "INTERNAL_AUTH_NOT_CONFIGURED"


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
            INTERNAL_SECRET_HEADER: SHARED_SECRET,
            USER_ID_HEADER: str(SNOWFLAKE_ID),
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["recommendFeed"][0]["postId"] == str(SNOWFLAKE_ID + 1)
    assert fake.calls[-1] == ("recommend", SNOWFLAKE_ID, 0, 20, "feed-correlation")


def test_graphql_recommend_feed_rejects_missing_or_mismatched_trusted_user(api):
    client, fake = api
    query = "query Feed($userId: ID!) { recommendFeed(userId: $userId) { postId } }"

    missing = client.post(
        "/graphql",
        json={"query": query, "variables": {"userId": str(SNOWFLAKE_ID)}},
        headers={INTERNAL_SECRET_HEADER: SHARED_SECRET},
    )
    mismatched = client.post(
        "/graphql",
        json={"query": query, "variables": {"userId": str(SNOWFLAKE_ID)}},
        headers={
            INTERNAL_SECRET_HEADER: SHARED_SECRET,
            USER_ID_HEADER: str(SNOWFLAKE_ID + 1),
        },
    )

    assert missing.json()["errors"][0]["extensions"]["code"] == "UNAUTHENTICATED"
    assert mismatched.json()["errors"][0]["extensions"]["code"] == "FORBIDDEN"
    assert not any(call[0] == "recommend" for call in fake.calls)


def test_graphql_recommend_feed_rejects_non_numeric_user_id(api):
    client, fake = api

    response = client.post(
        "/graphql",
        json={
            "query": "query { recommendFeed(userId: \"not-a-number\") { postId } }",
        },
        headers={
            INTERNAL_SECRET_HEADER: SHARED_SECRET,
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
