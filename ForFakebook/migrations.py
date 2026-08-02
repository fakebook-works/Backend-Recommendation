from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


LOGGER = logging.getLogger(__name__)
ADVISORY_LOCK_KEY = 0x46425245434F4D4D  # "FBRECOMM"
DEFAULT_DATABASE_URL = "postgresql://postgres@localhost:5432/fakebook"
MIGRATION_FILES = (
    "user_embedding.sql",
    "post_embedding.sql",
    "recommendation_interactions.sql",
)


@dataclass(frozen=True)
class Migration:
    version: str
    sql: str
    checksum_sha256: str


@dataclass(frozen=True)
class ColumnShape:
    name: str
    type_schema: str
    type_name: str
    type_modifier: int
    not_null: bool
    default_expression: str | None = None
    identity: str = ""
    generated: str = ""


@dataclass(frozen=True)
class TableShape:
    relation_kind: str
    columns: tuple[ColumnShape, ...]
    primary_key_columns: tuple[str, ...]


EXPECTED_TABLE_SHAPES = {
    "user_embedding": (
        "user_embeddings",
        TableShape(
            relation_kind="r",
            columns=(
                ColumnShape("user_id", "pg_catalog", "int8", -1, True),
                ColumnShape("embedding", "public", "vector", 512, True),
            ),
            primary_key_columns=("user_id",),
        ),
    ),
    "post_embedding": (
        "post_embeddings",
        TableShape(
            relation_kind="r",
            columns=(
                ColumnShape("post_id", "pg_catalog", "int8", -1, True),
                ColumnShape("embedding", "public", "vector", 512, True),
            ),
            primary_key_columns=("post_id",),
        ),
    ),
    "recommendation_interactions": (
        "recommendation_interactions",
        TableShape(
            relation_kind="r",
            columns=(
                ColumnShape(
                    "idempotency_key", "pg_catalog", "varchar", 128 + 4, True
                ),
                ColumnShape("user_id", "pg_catalog", "int8", -1, True),
                ColumnShape("target_id", "pg_catalog", "int8", -1, True),
                ColumnShape("action", "pg_catalog", "varchar", 16 + 4, True),
                ColumnShape("weight", "pg_catalog", "float4", -1, True),
                ColumnShape(
                    "created_at",
                    "pg_catalog",
                    "timestamptz",
                    -1,
                    True,
                    default_expression="CURRENT_TIMESTAMP",
                ),
            ),
            primary_key_columns=("idempotency_key",),
        ),
    ),
}


def _read_table_shape(connection: object, table_name: str) -> TableShape | None:
    column_rows = (
        connection.execute(
            text(
                """
                SELECT
                    relation_row.relkind AS relation_kind,
                    column_row.attname AS column_name,
                    type_schema.nspname AS type_schema,
                    type_row.typname AS type_name,
                    column_row.atttypmod AS type_modifier,
                    column_row.attnotnull AS not_null,
                    pg_catalog.pg_get_expr(
                        default_row.adbin, default_row.adrelid
                    ) AS default_expression,
                    column_row.attidentity AS identity,
                    column_row.attgenerated AS generated
                FROM pg_catalog.pg_class relation_row
                JOIN pg_catalog.pg_namespace relation_schema
                  ON relation_schema.oid = relation_row.relnamespace
                JOIN pg_catalog.pg_attribute column_row
                  ON column_row.attrelid = relation_row.oid
                JOIN pg_catalog.pg_type type_row
                  ON type_row.oid = column_row.atttypid
                JOIN pg_catalog.pg_namespace type_schema
                  ON type_schema.oid = type_row.typnamespace
                LEFT JOIN pg_catalog.pg_attrdef default_row
                  ON default_row.adrelid = relation_row.oid
                 AND default_row.adnum = column_row.attnum
                WHERE relation_schema.nspname = 'recommendation'
                  AND relation_row.relname = :table_name
                  AND column_row.attnum > 0
                  AND NOT column_row.attisdropped
                ORDER BY column_row.attnum
                """
            ),
            {"table_name": table_name},
        )
        .mappings()
        .all()
    )
    if not column_rows:
        return None

    primary_key_rows = (
        connection.execute(
            text(
                """
                SELECT primary_key_column.attname AS column_name
                FROM pg_catalog.pg_class relation_row
                JOIN pg_catalog.pg_namespace relation_schema
                  ON relation_schema.oid = relation_row.relnamespace
                JOIN pg_catalog.pg_constraint constraint_row
                  ON constraint_row.conrelid = relation_row.oid
                 AND constraint_row.contype = 'p'
                 AND constraint_row.convalidated
                JOIN pg_catalog.pg_index primary_index
                  ON primary_index.indexrelid = constraint_row.conindid
                 AND primary_index.indrelid = relation_row.oid
                 AND primary_index.indisprimary
                 AND primary_index.indisunique
                 AND primary_index.indisvalid
                 AND primary_index.indisready
                 AND primary_index.indislive
                 AND primary_index.indpred IS NULL
                 AND primary_index.indexprs IS NULL
                 AND primary_index.indnkeyatts =
                     pg_catalog.cardinality(constraint_row.conkey)
                 AND primary_index.indnatts =
                     pg_catalog.cardinality(constraint_row.conkey)
                JOIN pg_catalog.pg_class primary_index_relation
                  ON primary_index_relation.oid = primary_index.indexrelid
                JOIN pg_catalog.pg_am primary_index_method
                  ON primary_index_method.oid = primary_index_relation.relam
                 AND primary_index_method.amname = 'btree'
                JOIN LATERAL unnest(constraint_row.conkey) WITH ORDINALITY
                  AS key_column(attribute_number, position) ON TRUE
                JOIN pg_catalog.pg_attribute primary_key_column
                  ON primary_key_column.attrelid = relation_row.oid
                 AND primary_key_column.attnum = key_column.attribute_number
                 AND primary_index.indkey[
                     (key_column.position - 1)::integer
                 ] = key_column.attribute_number
                WHERE relation_schema.nspname = 'recommendation'
                  AND relation_row.relname = :table_name
                ORDER BY key_column.position
                """
            ),
            {"table_name": table_name},
        )
        .mappings()
        .all()
    )

    return TableShape(
        relation_kind=str(column_rows[0]["relation_kind"]),
        columns=tuple(
            ColumnShape(
                name=str(row["column_name"]),
                type_schema=str(row["type_schema"]),
                type_name=str(row["type_name"]),
                type_modifier=int(row["type_modifier"]),
                not_null=bool(row["not_null"]),
                default_expression=(
                    None
                    if row["default_expression"] is None
                    else str(row["default_expression"])
                ),
                identity=str(row["identity"]),
                generated=str(row["generated"]),
            )
            for row in column_rows
        ),
        primary_key_columns=tuple(
            str(row["column_name"]) for row in primary_key_rows
        ),
    )


def _interaction_index_is_valid(connection: object) -> bool:
    return bool(
        connection.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_index index_row
                    JOIN pg_catalog.pg_class table_row
                      ON table_row.oid = index_row.indrelid
                    JOIN pg_catalog.pg_namespace table_schema
                      ON table_schema.oid = table_row.relnamespace
                    JOIN pg_catalog.pg_class index_relation
                      ON index_relation.oid = index_row.indexrelid
                    JOIN pg_catalog.pg_namespace index_schema
                      ON index_schema.oid = index_relation.relnamespace
                    JOIN pg_catalog.pg_am access_method
                      ON access_method.oid = index_relation.relam
                    JOIN pg_catalog.pg_attribute user_column
                      ON user_column.attrelid = table_row.oid
                     AND user_column.attname = 'user_id'
                     AND NOT user_column.attisdropped
                    JOIN pg_catalog.pg_attribute created_column
                      ON created_column.attrelid = table_row.oid
                     AND created_column.attname = 'created_at'
                     AND NOT created_column.attisdropped
                    WHERE table_schema.nspname = 'recommendation'
                      AND table_row.relname = 'recommendation_interactions'
                      AND table_row.relkind = 'r'
                      AND index_schema.oid = table_schema.oid
                      AND access_method.amname = 'btree'
                      AND index_row.indisvalid
                      AND index_row.indisready
                      AND index_row.indislive
                      AND NOT index_row.indisunique
                      AND index_row.indpred IS NULL
                      AND index_row.indexprs IS NULL
                      AND index_row.indnkeyatts = 2
                      AND index_row.indnatts = 2
                      AND index_row.indkey[0] = user_column.attnum
                      AND index_row.indkey[1] = created_column.attnum
                      AND index_row.indoption[0]::integer = 0
                      AND index_row.indoption[1]::integer = 3
                )
                """
            )
        ).scalar_one()
    )


def _validate_migration_shape(connection: object, version: str) -> None:
    try:
        table_name, expected_shape = EXPECTED_TABLE_SHAPES[version]
    except KeyError:
        raise RuntimeError(
            f"Recommendation migration {version} has no schema-shape contract."
        ) from None

    actual_shape = _read_table_shape(connection, table_name)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Recommendation database object recommendation.{table_name} has an incompatible shape for migration {version}; refusing to trust or record that migration."
        )

    if version == "recommendation_interactions" and not _interaction_index_is_valid(
        connection
    ):
        raise RuntimeError(
            "Recommendation interactions index is missing or incompatible; expected a valid non-unique btree index on (user_id, created_at DESC)."
        )


def database_migrations_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = source.get("RECOMMENDATION_DB_MIGRATIONS_ENABLED", "true").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        "RECOMMENDATION_DB_MIGRATIONS_ENABLED must be true or false."
    )


def resolve_migration_database_url(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, bool]:
    source = os.environ if environ is None else environ
    dedicated = source.get("RECOMMENDATION_MIGRATION_DATABASE_URL", "")
    if dedicated.strip():
        return dedicated, True
    return source.get("DATABASE_URL", DEFAULT_DATABASE_URL), False


def load_migrations(root: Path | None = None) -> tuple[Migration, ...]:
    migration_root = root or Path(__file__).resolve().parent.parent
    migrations = []
    for filename in MIGRATION_FILES:
        path = migration_root / filename
        try:
            sql = path.read_text(encoding="utf-8")
        except OSError as exception:
            raise RuntimeError(
                f"Required Recommendation migration file '{filename}' is unavailable."
            ) from exception
        normalized_sql = sql.replace("\r\n", "\n").replace("\r", "\n")
        migrations.append(
            Migration(
                version=path.stem,
                sql=sql,
                checksum_sha256=hashlib.sha256(
                    normalized_sql.encode("utf-8")
                ).hexdigest(),
            )
        )
    return tuple(migrations)


def run_database_migrations(
    database_url: str,
    *,
    engine_factory: Callable[..., object] = create_engine,
) -> None:
    migrations = load_migrations()
    try:
        migration_engine = engine_factory(
            database_url,
            pool_pre_ping=True,
            poolclass=NullPool,
        )
    except Exception:
        raise RuntimeError(
            "Recommendation migration connection configuration is invalid."
        ) from None
    lock_acquired = False

    try:
        with migration_engine.connect() as connection:
            connection.execute(
                text("SELECT pg_advisory_lock(:lock_key)"),
                {"lock_key": ADVISORY_LOCK_KEY},
            )
            connection.commit()
            lock_acquired = True

            try:
                with connection.begin():
                    vector_available = connection.execute(
                        text(
                            """
                            SELECT EXISTS (
                                SELECT 1 FROM pg_available_extensions WHERE name = 'vector'
                            )
                            """
                        )
                    ).scalar_one()
                    if not vector_available:
                        raise RuntimeError(
                            "PostgreSQL pgvector is not installed on the server. Install the vector extension before starting Recommendation."
                        )

                    installed_vector_schema = connection.execute(
                        text(
                            """
                            SELECT schema_row.nspname
                            FROM pg_extension extension_row
                            JOIN pg_namespace schema_row
                              ON schema_row.oid = extension_row.extnamespace
                            WHERE extension_row.extname = 'vector'
                            """
                        )
                    ).scalar_one_or_none()
                    if installed_vector_schema not in {None, "public"}:
                        raise RuntimeError(
                            "The pgvector extension must be installed in PostgreSQL schema public for the Recommendation runtime role."
                        )

                    connection.execute(text("CREATE SCHEMA IF NOT EXISTS recommendation"))
                    connection.execute(
                        text(
                            """
                            CREATE TABLE IF NOT EXISTS recommendation.schema_migrations (
                                version TEXT PRIMARY KEY,
                                checksum_sha256 CHAR(64) NOT NULL,
                                applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                            )
                            """
                        )
                    )

                for migration in migrations:
                    with connection.begin():
                        recorded_checksum = connection.execute(
                            text(
                                """
                                SELECT checksum_sha256
                                FROM recommendation.schema_migrations
                                WHERE version = :version
                                """
                            ),
                            {"version": migration.version},
                        ).scalar_one_or_none()

                        if recorded_checksum is not None:
                            if recorded_checksum.strip().lower() != migration.checksum_sha256:
                                raise RuntimeError(
                                    f"Recommendation migration {migration.version} does not match its recorded checksum; published migrations are immutable."
                                )
                            _validate_migration_shape(connection, migration.version)
                            continue

                        connection.exec_driver_sql(migration.sql)
                        _validate_migration_shape(connection, migration.version)
                        connection.execute(
                            text(
                                """
                                INSERT INTO recommendation.schema_migrations
                                    (version, checksum_sha256)
                                VALUES (:version, :checksum)
                                """
                            ),
                            {
                                "version": migration.version,
                                "checksum": migration.checksum_sha256,
                            },
                        )
                        LOGGER.info(
                            "Applied Recommendation database migration %s.",
                            migration.version,
                        )

                with connection.begin():
                    for migration in migrations:
                        _validate_migration_shape(connection, migration.version)
            finally:
                if lock_acquired:
                    try:
                        connection.execute(
                            text("SELECT pg_advisory_unlock(:lock_key)"),
                            {"lock_key": ADVISORY_LOCK_KEY},
                        )
                        connection.commit()
                    except Exception:
                        LOGGER.warning(
                            "Could not explicitly release the Recommendation migration lock; closing the connection will release it."
                        )
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError(
            "Recommendation database migration failed. Configure RECOMMENDATION_MIGRATION_DATABASE_URL with a PostgreSQL migration-owner role that can create pgvector and the recommendation schema."
        ) from None
    finally:
        migration_engine.dispose()


def migrate_database_on_startup(
    *,
    environ: Mapping[str, str] | None = None,
    runner: Callable[[str], None] | None = None,
) -> None:
    if not database_migrations_enabled(environ):
        LOGGER.warning("Recommendation startup database migrations are disabled.")
        return

    database_url, dedicated = resolve_migration_database_url(environ)
    if dedicated:
        LOGGER.info("Using the dedicated Recommendation migration connection.")
    else:
        LOGGER.warning(
            "RECOMMENDATION_MIGRATION_DATABASE_URL is not configured; startup migrations will use DATABASE_URL. The runtime role must have DDL privileges."
        )

    (runner or run_database_migrations)(database_url)
