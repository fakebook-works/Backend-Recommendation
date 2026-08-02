from dataclasses import replace
from pathlib import Path

import pytest

from ForFakebook.migrations import (
    EXPECTED_TABLE_SHAPES,
    MIGRATION_FILES,
    ColumnShape,
    TableShape,
    _validate_migration_shape,
    database_migrations_enabled,
    load_migrations,
    migrate_database_on_startup,
    resolve_migration_database_url,
    run_database_migrations,
)


class FakeResult:
    def __init__(self, *, rows=(), scalar=None):
        self._rows = list(rows)
        self._scalar = scalar

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def scalar_one(self):
        return self._scalar

    def scalar_one_or_none(self):
        return self._scalar


class CatalogConnection:
    def __init__(self, shapes=None, *, interaction_index_valid=True):
        self.shapes = dict(shapes or _expected_shapes_by_table())
        self.interaction_index_valid = interaction_index_valid
        self.shape_reads = {}
        self.index_reads = 0
        self.primary_key_sql = []
        self.index_sql = None

    def execute(self, statement, parameters=None):
        sql = str(statement)
        parameters = parameters or {}
        if "column_row.atttypmod AS type_modifier" in sql:
            table_name = parameters["table_name"]
            self.shape_reads[table_name] = self.shape_reads.get(table_name, 0) + 1
            shape = self.shapes.get(table_name)
            if shape is None:
                return FakeResult()
            return FakeResult(
                rows=(
                    {
                        "relation_kind": shape.relation_kind,
                        "column_name": column.name,
                        "type_schema": column.type_schema,
                        "type_name": column.type_name,
                        "type_modifier": column.type_modifier,
                        "not_null": column.not_null,
                        "default_expression": column.default_expression,
                        "identity": column.identity,
                        "generated": column.generated,
                    }
                    for column in shape.columns
                )
            )
        if "primary_key_column.attname AS column_name" in sql:
            self.primary_key_sql.append(sql)
            shape = self.shapes.get(parameters["table_name"])
            return FakeResult(
                rows=(
                    {"column_name": column_name}
                    for column_name in (() if shape is None else shape.primary_key_columns)
                )
            )
        if "FROM pg_catalog.pg_index index_row" in sql:
            self.index_reads += 1
            self.index_sql = sql
            return FakeResult(scalar=self.interaction_index_valid)
        raise AssertionError(f"Unexpected catalog SQL: {sql}")


class FakeTransaction:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is None:
            self.connection.transaction_commits += 1
        else:
            self.connection.transaction_rollbacks += 1
        return False


class MigrationConnection(CatalogConnection):
    def __init__(self, shapes=None, *, recorded=None, interaction_index_valid=True):
        super().__init__(shapes, interaction_index_valid=interaction_index_valid)
        self.recorded = dict(recorded or {})
        self.driver_sql = []
        self.ledger_inserts = []
        self.transaction_commits = 0
        self.transaction_rollbacks = 0
        self.explicit_commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def begin(self):
        return FakeTransaction(self)

    def commit(self):
        self.explicit_commits += 1

    def exec_driver_sql(self, sql):
        self.driver_sql.append(sql)

    def execute(self, statement, parameters=None):
        sql = str(statement)
        parameters = parameters or {}
        if (
            "column_row.atttypmod AS type_modifier" in sql
            or "primary_key_column.attname AS column_name" in sql
            or "FROM pg_catalog.pg_index index_row" in sql
        ):
            return super().execute(statement, parameters)
        if "pg_advisory_lock(" in sql or "pg_advisory_unlock(" in sql:
            return FakeResult(scalar=True)
        if "FROM pg_available_extensions" in sql:
            return FakeResult(scalar=True)
        if "FROM pg_extension extension_row" in sql:
            return FakeResult(scalar="public")
        if "SELECT checksum_sha256" in sql:
            return FakeResult(scalar=self.recorded.get(parameters["version"]))
        if "INSERT INTO recommendation.schema_migrations" in sql:
            self.ledger_inserts.append(
                (parameters["version"], parameters["checksum"])
            )
            self.recorded[parameters["version"]] = parameters["checksum"]
            return FakeResult()
        if "CREATE SCHEMA IF NOT EXISTS recommendation" in sql:
            return FakeResult()
        if "CREATE TABLE IF NOT EXISTS recommendation.schema_migrations" in sql:
            return FakeResult()
        raise AssertionError(f"Unexpected migration SQL: {sql}")


class FakeEngine:
    def __init__(self, connection):
        self.connection = connection
        self.disposed = False

    def connect(self):
        return self.connection

    def dispose(self):
        self.disposed = True


def _expected_shapes_by_table():
    return {
        table_name: shape for table_name, shape in EXPECTED_TABLE_SHAPES.values()
    }


def _engine_factory_for(connection):
    engine = FakeEngine(connection)

    def factory(*args, **kwargs):
        return engine

    return engine, factory


def test_migration_manifest_uses_all_three_versioned_schema_files_in_order():
    migrations = load_migrations(Path(__file__).resolve().parent.parent)

    assert tuple(f"{migration.version}.sql" for migration in migrations) == MIGRATION_FILES
    assert all(len(migration.checksum_sha256) == 64 for migration in migrations)
    assert "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public" in migrations[0].sql


def test_shape_contract_matches_the_three_authoritative_root_sql_files():
    user_table, user_shape = EXPECTED_TABLE_SHAPES["user_embedding"]
    post_table, post_shape = EXPECTED_TABLE_SHAPES["post_embedding"]
    interaction_table, interaction_shape = EXPECTED_TABLE_SHAPES[
        "recommendation_interactions"
    ]

    assert user_table == "user_embeddings"
    assert user_shape == TableShape(
        "r",
        (
            ColumnShape("user_id", "pg_catalog", "int8", -1, True),
            ColumnShape("embedding", "public", "vector", 512, True),
        ),
        ("user_id",),
    )
    assert post_table == "post_embeddings"
    assert post_shape == TableShape(
        "r",
        (
            ColumnShape("post_id", "pg_catalog", "int8", -1, True),
            ColumnShape("embedding", "public", "vector", 512, True),
        ),
        ("post_id",),
    )
    assert interaction_table == "recommendation_interactions"
    assert interaction_shape.columns == (
        ColumnShape("idempotency_key", "pg_catalog", "varchar", 132, True),
        ColumnShape("user_id", "pg_catalog", "int8", -1, True),
        ColumnShape("target_id", "pg_catalog", "int8", -1, True),
        ColumnShape("action", "pg_catalog", "varchar", 20, True),
        ColumnShape("weight", "pg_catalog", "float4", -1, True),
        ColumnShape(
            "created_at",
            "pg_catalog",
            "timestamptz",
            -1,
            True,
            default_expression="CURRENT_TIMESTAMP",
        ),
    )
    assert interaction_shape.primary_key_columns == ("idempotency_key",)


def test_exact_shapes_and_interaction_index_are_accepted():
    connection = CatalogConnection()

    for version in EXPECTED_TABLE_SHAPES:
        _validate_migration_shape(connection, version)

    assert connection.index_reads == 1
    assert all("primary_index.indisprimary" in sql for sql in connection.primary_key_sql)
    assert all("primary_index.indisvalid" in sql for sql in connection.primary_key_sql)
    assert all("primary_index.indisready" in sql for sql in connection.primary_key_sql)
    for required_index_contract in (
        "access_method.amname = 'btree'",
        "index_row.indisvalid",
        "index_row.indisready",
        "index_row.indislive",
        "NOT index_row.indisunique",
        "index_row.indpred IS NULL",
        "index_row.indexprs IS NULL",
        "index_row.indnkeyatts = 2",
        "index_row.indnatts = 2",
        "index_row.indkey[0] = user_column.attnum",
        "index_row.indkey[1] = created_column.attnum",
        "index_row.indoption[0]::integer = 0",
        "index_row.indoption[1]::integer = 3",
    ):
        assert required_index_contract in connection.index_sql


@pytest.mark.parametrize(
    "bad_shape",
    (
        TableShape(
            "r",
            (ColumnShape("user_id", "pg_catalog", "int8", -1, True),),
            ("user_id",),
        ),
        TableShape(
            "r",
            (
                ColumnShape("user_id", "pg_catalog", "int8", -1, True),
                ColumnShape("embedding", "public", "vector", 1536, True),
            ),
            ("user_id",),
        ),
        TableShape(
            "r",
            (
                ColumnShape("user_id", "pg_catalog", "int8", -1, False),
                ColumnShape("embedding", "public", "vector", 512, True),
            ),
            ("user_id",),
        ),
        TableShape(
            "r",
            (
                ColumnShape("user_id", "pg_catalog", "int8", -1, True),
                ColumnShape("embedding", "public", "vector", 512, True),
                ColumnShape("legacy", "pg_catalog", "int8", -1, False),
            ),
            ("user_id",),
        ),
        TableShape(
            "r",
            (
                ColumnShape("user_id", "pg_catalog", "int8", -1, True),
                ColumnShape("embedding", "public", "vector", 512, True),
            ),
            (),
        ),
    ),
)
def test_partial_or_incompatible_embedding_table_is_rejected(bad_shape):
    shapes = _expected_shapes_by_table()
    shapes["user_embeddings"] = bad_shape

    with pytest.raises(RuntimeError, match="incompatible shape"):
        _validate_migration_shape(CatalogConnection(shapes), "user_embedding")


@pytest.mark.parametrize(
    ("column_index", "replacement"),
    (
        (
            0,
            ColumnShape("idempotency_key", "pg_catalog", "varchar", 64 + 4, True),
        ),
        (4, ColumnShape("weight", "pg_catalog", "float8", -1, True)),
        (4, ColumnShape("weight", "pg_catalog", "float4", -1, False)),
        (5, ColumnShape("created_at", "pg_catalog", "timestamptz", -1, True)),
    ),
)
def test_interaction_types_lengths_nullability_and_default_are_strict(
    column_index, replacement
):
    shapes = _expected_shapes_by_table()
    expected = shapes["recommendation_interactions"]
    wrong_columns = list(expected.columns)
    wrong_columns[column_index] = replacement
    shapes["recommendation_interactions"] = replace(
        expected,
        columns=tuple(wrong_columns),
    )

    with pytest.raises(RuntimeError, match="incompatible shape"):
        _validate_migration_shape(
            CatalogConnection(shapes), "recommendation_interactions"
        )


def test_interaction_primary_key_and_index_are_strict():
    shapes = _expected_shapes_by_table()
    shapes["recommendation_interactions"] = replace(
        shapes["recommendation_interactions"],
        primary_key_columns=("user_id",),
    )

    with pytest.raises(RuntimeError, match="incompatible shape"):
        _validate_migration_shape(
            CatalogConnection(shapes), "recommendation_interactions"
        )

    with pytest.raises(RuntimeError, match="valid non-unique btree index"):
        _validate_migration_shape(
            CatalogConnection(interaction_index_valid=False),
            "recommendation_interactions",
        )


def test_partial_legacy_table_rolls_back_before_a_ledger_row_is_inserted():
    shapes = _expected_shapes_by_table()
    shapes["user_embeddings"] = TableShape(
        "r",
        (ColumnShape("user_id", "pg_catalog", "int8", -1, True),),
        ("user_id",),
    )
    connection = MigrationConnection(shapes)
    engine, engine_factory = _engine_factory_for(connection)

    with pytest.raises(RuntimeError, match="incompatible shape"):
        run_database_migrations("postgresql://migration-owner/db", engine_factory=engine_factory)

    assert len(connection.driver_sql) == 1
    assert connection.ledger_inserts == []
    assert connection.transaction_rollbacks == 1
    assert engine.disposed is True


def test_existing_ledger_row_does_not_bypass_shape_validation():
    first_migration = load_migrations()[0]
    shapes = _expected_shapes_by_table()
    shapes["user_embeddings"] = TableShape(
        "r",
        (ColumnShape("user_id", "pg_catalog", "int8", -1, True),),
        ("user_id",),
    )
    connection = MigrationConnection(
        shapes,
        recorded={first_migration.version: first_migration.checksum_sha256},
    )
    _, engine_factory = _engine_factory_for(connection)

    with pytest.raises(RuntimeError, match="incompatible shape"):
        run_database_migrations("postgresql://migration-owner/db", engine_factory=engine_factory)

    assert connection.driver_sql == []
    assert connection.ledger_inserts == []
    assert connection.transaction_rollbacks == 1


def test_each_new_migration_is_validated_before_insert_and_again_at_the_end():
    connection = MigrationConnection()
    _, engine_factory = _engine_factory_for(connection)

    run_database_migrations("postgresql://migration-owner/db", engine_factory=engine_factory)

    assert [version for version, _ in connection.ledger_inserts] == [
        "user_embedding",
        "post_embedding",
        "recommendation_interactions",
    ]
    assert connection.shape_reads == {
        "user_embeddings": 2,
        "post_embeddings": 2,
        "recommendation_interactions": 2,
    }
    assert connection.index_reads == 2


def test_startup_migrations_are_enabled_by_default_and_have_strict_opt_out():
    assert database_migrations_enabled({}) is True
    assert database_migrations_enabled({"RECOMMENDATION_DB_MIGRATIONS_ENABLED": "false"}) is False

    with pytest.raises(RuntimeError, match="must be true or false"):
        database_migrations_enabled({"RECOMMENDATION_DB_MIGRATIONS_ENABLED": "sometimes"})


def test_dedicated_migration_connection_is_preferred_without_exposing_it():
    url, dedicated = resolve_migration_database_url(
        {
            "DATABASE_URL": "postgresql://runtime:runtime-secret@db/fakebook",
            "RECOMMENDATION_MIGRATION_DATABASE_URL": "postgresql://owner:owner-secret@db/fakebook",
        }
    )

    assert dedicated is True
    assert url == "postgresql://owner:owner-secret@db/fakebook"


def test_disabled_startup_does_not_invoke_migration_runner():
    calls = []

    migrate_database_on_startup(
        environ={"RECOMMENDATION_DB_MIGRATIONS_ENABLED": "off"},
        runner=calls.append,
    )

    assert calls == []


def test_enabled_startup_invokes_runner_with_dedicated_connection():
    calls = []

    migrate_database_on_startup(
        environ={
            "RECOMMENDATION_DB_MIGRATIONS_ENABLED": "true",
            "DATABASE_URL": "postgresql://runtime@db/fakebook",
            "RECOMMENDATION_MIGRATION_DATABASE_URL": "postgresql://owner@db/fakebook",
        },
        runner=calls.append,
    )

    assert calls == ["postgresql://owner@db/fakebook"]
