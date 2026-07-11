from ForFakebook.database import create_user_embedding_if_missing, vector_literal


class FakeResult:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class FakeDatabase:
    def __init__(self, rowcount):
        self.rowcount = rowcount
        self.statement = None
        self.parameters = None
        self.commit_count = 0

    def execute(self, statement, parameters):
        self.statement = str(statement)
        self.parameters = parameters
        return FakeResult(self.rowcount)

    def commit(self):
        self.commit_count += 1


def test_create_user_embedding_is_atomic_and_does_not_overwrite_existing_vector():
    created_db = FakeDatabase(rowcount=1)
    existing_db = FakeDatabase(rowcount=0)

    assert create_user_embedding_if_missing(created_db, 123, [0.5, -0.5]) is True
    assert create_user_embedding_if_missing(existing_db, 123, [0.5, -0.5]) is False

    assert "ON CONFLICT (user_id) DO NOTHING" in created_db.statement
    assert "DO UPDATE" not in created_db.statement
    assert created_db.parameters == {"user_id": 123, "embedding": "[0.5,-0.5]"}
    assert created_db.commit_count == 1
    assert existing_db.commit_count == 1


def test_vector_literal_uses_compact_pgvector_format():
    assert vector_literal([1, 0.125, -2.5]) == "[1,0.125,-2.5]"
