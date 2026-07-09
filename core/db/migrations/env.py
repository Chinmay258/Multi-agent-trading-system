"""
core/db/migrations/env.py
--------------------------
Alembic migration environment.

Uses a SYNC psycopg2 connection (Alembic's runner is synchronous).
The database URL comes from settings.database.sync_url so that the same
env-var configuration drives both the ORM (asyncpg) and migrations (psycopg2).
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make project root importable when alembic is invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import core.db.models  # noqa: E402, F401 — registers all ORM models with Base.metadata
from core.config import get_settings  # noqa: E402
from core.db.connection import Base  # noqa: E402

# ---------------------------------------------------------------------------
# Alembic Config object (provides access to alembic.ini values)
# ---------------------------------------------------------------------------

config = context.config

if config.config_file_name:
    fileConfig(config.config_file_name)

# Override the URL from settings so env-vars drive everything.
config.set_main_option("sqlalchemy.url", get_settings().database.sync_url)

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Migration runners
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL without a live DB)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
