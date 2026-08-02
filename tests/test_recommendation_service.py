import numpy as np

from ForFakebook import recommendation_service
from ForFakebook.internal_signing import NONCE_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_fetch_candidate_ids_uses_socialgraph_internal_contract():
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse([101, 102, 101])

    result = recommendation_service.fetch_post_candidate_ids(
        9_000_000_000_000_001,
        200,
        "http://socialgraph:1002/",
        "shared-secret-at-least-32-bytes-long",
        "correlation",
        fake_get,
    )

    assert result == [101, 102]
    assert captured["url"] == "http://socialgraph:1002/internal/recommendation/post-candidate-ids"
    assert captured["params"] == {"userId": 9_000_000_000_000_001, "limit": 200}
    assert captured["headers"]["X-Internal-SocialGraphService-Secret"] == "shared-secret-at-least-32-bytes-long"
    assert "X-Gateway-Secret" not in captured["headers"]
    assert captured["headers"]["X-Correlation-ID"] == "correlation"
    assert captured["timeout"] == 10


def test_internal_signing_matches_cross_language_vectors():
    secret = "test-internal-secret-0123456789ab"
    assert sign(
        secret,
        "POST",
        "/internal/users?x=1",
        1_753_500_000,
        "0123456789abcdef0123456789abcdef",
        b'{"userId":42}',
    ) == "e0f96895cf6c2f5b4f075e7f6f36902e591d2ce178321550041d45e6c8726512"
    assert sign(
        secret,
        "GET",
        "/internal/users/42/friend-ids",
        1_753_500_000,
        "ffffffffffffffffffffffffffffffff",
        b"",
    ) == "3ff404655307935abc5825da27bf6fd4b311b0f2034a23a0d7ebbc012aa430c1"


def test_candidate_fetch_can_omit_raw_secret_while_retaining_signature(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse([])

    monkeypatch.setenv("INTERNAL_AUTH_SEND_LEGACY_SECRET", "false")
    recommendation_service.fetch_post_candidate_ids(
        1,
        20,
        "http://socialgraph",
        "shared-secret-at-least-32-bytes-long",
        http_get=fake_get,
    )

    assert "X-Internal-SocialGraphService-Secret" not in captured["headers"]
    assert {TIMESTAMP_HEADER, NONCE_HEADER, SIGNATURE_HEADER}.issubset(captured["headers"])


class FakeResult:
    def __init__(self, one=None, many=None):
        self._one = one
        self._many = many or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class FakeDb:
    def execute(self, statement, parameters):
        sql = str(statement)
        unit = "[" + ",".join(["1"] + ["0"] * 511) + "]"
        if "user_embeddings" in sql:
            return FakeResult(one=(unit,))
        return FakeResult(
            many=[
                (101, unit),
                (102, "[" + ",".join(["0"] * 512) + "]"),
            ]
        )


def test_recommend_feed_ranks_socialgraph_candidates(monkeypatch):
    monkeypatch.setattr(
        recommendation_service,
        "fetch_post_candidate_ids",
        lambda *args, **kwargs: [102, 101],
    )

    result = recommendation_service.recommend_feed_logic(
        FakeDb(),
        user_id=1,
        social_graph_base_url="http://socialgraph",
        shared_secret="shared-secret-at-least-32-bytes-long",
        take=2,
    )

    assert [item["postId"] for item in result] == [101, 102]
    assert result == [{"postId": 101}, {"postId": 102}]


def test_fetch_candidate_ids_rejects_malformed_socialgraph_payload():
    def fake_get(*args, **kwargs):
        return FakeResponse([101, 102.5])

    with np.testing.assert_raises_regex(ValueError, "invalid post candidate ID"):
        recommendation_service.fetch_post_candidate_ids(
            1,
            20,
            "http://socialgraph",
            "shared-secret-at-least-32-bytes-long",
            http_get=fake_get,
        )


def test_fetch_reel_candidates_includes_friends_and_followed_for_following_and_deduplicates():
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            [
                {"id": 201, "authorId": 1, "source": "recent_public", "createdAt": "now"},
                {"id": 203, "authorId": 4, "source": "friend", "createdAt": "now"},
                {"id": 202, "authorId": 2, "source": "followed", "createdAt": "now"},
                {"id": 202, "authorId": 2, "source": "followed", "createdAt": "now"},
                {"id": 204, "authorId": 5, "source": "pending_friend", "createdAt": "now"},
            ]
        )

    result = recommendation_service.fetch_reel_candidate_ids(
        1,
        20,
        "http://socialgraph",
        "shared-secret-at-least-32-bytes-long",
        mode="FOLLOWING",
        http_get=fake_get,
    )

    assert result == [203, 202]
    assert captured["url"].endswith("/internal/recommendation/reel-candidates")
    assert captured["headers"]["X-Internal-SocialGraphService-Secret"] == "shared-secret-at-least-32-bytes-long"
    assert "X-Gateway-Secret" not in captured["headers"]


def test_recommend_reels_ranks_candidates(monkeypatch):
    monkeypatch.setattr(
        recommendation_service,
        "fetch_reel_candidate_ids",
        lambda *args, **kwargs: [102, 101],
    )

    result = recommendation_service.recommend_reels_logic(
        FakeDb(),
        user_id=1,
        social_graph_base_url="http://socialgraph",
        shared_secret="shared-secret-at-least-32-bytes-long",
        take=2,
    )

    assert result == [{"reelId": 101}, {"reelId": 102}]


def test_following_reels_request_the_full_bounded_relationship_pool(monkeypatch):
    captured = {}

    def fake_fetch(user_id, limit, base_url, secret, mode, correlation_id=None):
        captured.update(user_id=user_id, limit=limit, mode=mode)
        return []

    monkeypatch.setattr(recommendation_service, "fetch_reel_candidate_ids", fake_fetch)

    result = recommendation_service.recommend_reels_logic(
        FakeDb(),
        user_id=1,
        social_graph_base_url="http://socialgraph",
        shared_secret="shared-secret-at-least-32-bytes-long",
        mode="FOLLOWING",
        take=20,
    )

    assert result == []
    assert captured == {"user_id": 1, "limit": 500, "mode": "FOLLOWING"}
