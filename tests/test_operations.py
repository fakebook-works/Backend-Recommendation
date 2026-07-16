import numpy as np
import pytest

from ForFakebook.operations import (
    InteractionTargetUnavailableError,
    RecommendationOperations,
    apply_interaction_to_embedding,
)


class FakeResult:
    def __init__(self, rowcount=-1, row=None):
        self.rowcount = rowcount
        self._row = row

    def first(self):
        return self._row


class InteractionSession:
    def __init__(self, *, duplicate=False, target="[0,1]", user="[1,0]"):
        self.duplicate = duplicate
        self.target = target
        self.user = user
        self.calls = []
        self.commit_count = 0
        self.rollback_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def execute(self, statement, parameters=None):
        sql = " ".join(str(statement).split())
        self.calls.append((sql, parameters))
        if "INSERT INTO recommendation_interactions" in sql:
            return FakeResult(rowcount=0 if self.duplicate else 1)
        if "SELECT embedding FROM post_embeddings" in sql:
            return FakeResult(row=(self.target,) if self.target is not None else None)
        if "SELECT embedding FROM user_embeddings" in sql:
            return FakeResult(row=(self.user,) if self.user is not None else None)
        return FakeResult()

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


def test_apply_interaction_keeps_unit_length_and_respects_signal_direction():
    current = np.array([1.0, 0.0])
    target = np.array([0.0, 1.0])

    positive = apply_interaction_to_embedding(current, target, 0.35)
    negative = apply_interaction_to_embedding(current, target, -0.20)

    assert np.linalg.norm(positive) == pytest.approx(1.0)
    assert np.linalg.norm(negative) == pytest.approx(1.0)
    assert positive[1] > 0
    assert negative[1] < 0


def test_record_interaction_updates_user_vector_once():
    session = InteractionSession()
    operations = RecommendationOperations(session_factory=lambda: session)

    applied = operations.record_interaction(42, 84, "save", "event-1")

    assert applied is True
    assert session.commit_count == 1
    assert session.rollback_count == 0
    assert any("pg_advisory_xact_lock" in call[0] for call in session.calls)
    upsert = next(call for call in session.calls if "INSERT INTO user_embeddings" in call[0])
    assert upsert[1]["user_id"] == 42
    assert upsert[1]["embedding"].startswith("[")


def test_record_interaction_duplicate_does_not_retrain_vector():
    session = InteractionSession(duplicate=True)
    operations = RecommendationOperations(session_factory=lambda: session)

    applied = operations.record_interaction(42, 84, "LIKE", "event-duplicate")

    assert applied is False
    assert session.commit_count == 1
    assert not any("INSERT INTO user_embeddings" in call[0] for call in session.calls)
    assert not any("pg_advisory_xact_lock" in call[0] for call in session.calls)


def test_record_interaction_rolls_back_when_target_embedding_is_not_ready():
    session = InteractionSession(target=None)
    operations = RecommendationOperations(session_factory=lambda: session)

    with pytest.raises(InteractionTargetUnavailableError):
        operations.record_interaction(42, 84, "COMMENT", "event-late")

    assert session.commit_count == 0
    assert session.rollback_count == 1
    assert not any("INSERT INTO user_embeddings" in call[0] for call in session.calls)


def test_deleting_user_or_post_removes_owned_feedback_ledger_rows():
    user_session = InteractionSession()
    post_session = InteractionSession()
    sessions = iter((user_session, post_session))
    operations = RecommendationOperations(session_factory=lambda: next(sessions))

    operations.delete_user_embedding(42)
    operations.delete_post_embedding(84)

    assert any(
        "DELETE FROM recommendation_interactions WHERE user_id" in call[0]
        for call in user_session.calls
    )
    assert any("DELETE FROM user_embeddings" in call[0] for call in user_session.calls)
    assert any(
        "DELETE FROM recommendation_interactions WHERE target_id" in call[0]
        for call in post_session.calls
    )
    assert any("DELETE FROM post_embeddings" in call[0] for call in post_session.calls)
    assert user_session.commit_count == 1
    assert post_session.commit_count == 1
