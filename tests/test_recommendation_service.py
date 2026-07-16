import numpy as np

from ForFakebook import recommendation_service


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


def test_fetch_reel_candidates_filters_following_and_deduplicates():
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            [
                {"id": 201, "authorId": 1, "source": "recent_public", "createdAt": "now"},
                {"id": 202, "authorId": 2, "source": "followed", "createdAt": "now"},
                {"id": 202, "authorId": 2, "source": "followed", "createdAt": "now"},
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

    assert result == [202]
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
