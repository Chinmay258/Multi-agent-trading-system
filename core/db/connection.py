"""
core/db/connection.py
---------------------
Async SQLAlchemy engine and session management.

All database I/O in this system is async (asyncpg driver). SQLAlchemy's
async session is used for all ORM operations; raw SQL is reserved for
TimescaleDB-specific queries (hypertable creation, time_bucket aggregations).

Design decisions:
- Single engine per process: engines are heavyweight, sessions are cheap.
  get_engine() is called once at startup, reused everywhere.
- get_session() is an async context manager — always use it with `async with`
  to guarantee commit/rollback and connection return to the pool.
- FastAPI integration: get_db_session() is a dependency that yields a session
  per request and rolls back on exception.
- Alembic migrations use a SYNC connection (psycopg2) — async doesn't work
  with Alembic's migration runner. Both URLs are available from config.

Usage:
    # In an agent
    async with get_session() as session:
        session.add(trade_record)
        await session.commit()

    # In FastAPI
    async def endpoint(db: AsyncSession = Depends(get_db_session)):
        result = await db.execute(select(Trade))
        return result.scalars().all()
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.config import get_settings
from core.logging import get_logger

logger = get_logger("db")

# ---------------------------------------------------------------------------
# SQLAlchemy declarative base — all ORM models inherit from this
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.
    Import from here: from core.db.connection import Base
    """

    pass


# ---------------------------------------------------------------------------
# Engine (singleton per process)
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """
    Return the process-level async SQLAlchemy engine.

    Called once at startup. Subsequent calls return the cached instance.
    Call dispose_engine() during shutdown to cleanly close connections.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        db = settings.database

        _engine = create_async_engine(
            db.url,
            pool_size=db.pool_size,
            max_overflow=db.max_overflow,
            echo=db.echo_sql,
            pool_pre_ping=True,  # verify connections before use
            pool_recycle=3600,  # recycle connections every hour
        )
        logger.info(
            "database_engine_created",
            host=db.host,
            port=db.port,
            database=db.name,
            pool_size=db.pool_size,
        )
    return _engine


async def dispose_engine() -> None:
    """Gracefully close all pooled connections. Call during agent shutdown."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("database_engine_disposed")


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the async session factory (created once from the engine)."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,  # avoid lazy-load issues in async context
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    Async context manager providing a database session with automatic
    commit on success and rollback on exception.

    Usage:
        async with get_session() as session:
            session.add(record)
            await session.commit()  # explicit commit required
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency: yields a session per request.
    Rolls back automatically if the endpoint raises an exception.

    Usage in FastAPI:
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with get_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Schema management helpers
# ---------------------------------------------------------------------------


async def create_all_tables() -> None:
    """
    Create all tables defined via SQLAlchemy ORM models.
    For development/testing only — production uses Alembic migrations.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("all_tables_created")


async def drop_all_tables() -> None:
    """
    Drop all tables. DESTRUCTIVE — testing only.
    Raises RuntimeError in production environment.
    """
    settings = get_settings()
    if settings.is_production:
        raise RuntimeError("drop_all_tables() is forbidden in production")

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("all_tables_dropped")


async def check_connection() -> bool:
    """
    Verify database connectivity. Used by health checks.
    Returns True if the database is reachable, False otherwise.
    """
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))
        return False
