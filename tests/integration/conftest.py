"""
tests/integration/conftest.py
-----------------------------
Shared fixtures for integration tests.

These fixtures are scoped to the integration suite so unit tests (which run
without I/O at all) are unaffected. The fixtures are deliberately minimal
and composable — every E2E test builds the exact scenario it needs from
these primitives rather than depending on a fat "make me a full system"
fixture.

Fixture inventory
-----------------
- ``mock_settings``  — a fresh ``Settings`` object pinned to paper mode,
                       1 symbol and 1 timeframe; clears the lru_cache so
                       tests see a stable view of config.
- ``fake_redis``     — an in-process ``fakeredis.aioredis`` instance for
                       tests that exercise the messaging layer without a
                       real Redis. Most E2E tests do not need this and
                       instead drive components directly.
- ``in_memory_db``   — an aiosqlite engine with all ORM tables created.
- ``paper_broker``   — a connected ``PaperBroker`` ready to fill orders.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from core.config import Settings, TradingMode, get_settings

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> Settings:
    """
    Return a Settings instance forced into a deterministic test profile.

    We patch the environment for the duration of the test, clear the lru_cache
    that backs ``get_settings``, and yield a fresh settings object. On
    teardown the cache is cleared again so the next test rebuilds from the
    real environment (or its own patched values).
    """
    env_overrides = {
        "TRADING_MODE": "paper",
        "MARKET_DATA_SYMBOLS": '["BTC/USDT"]',
        "MARKET_DATA_OHLCV_TIMEFRAMES": '["1m"]',
        "PAPER_INITIAL_BALANCE_USD": "10000",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        get_settings.cache_clear()
        settings = get_settings()
        try:
            yield settings
        finally:
            get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fake Redis (for tests that want a bus-level mock; most don't need it)
# ---------------------------------------------------------------------------


@pytest.fixture
async def fake_redis() -> AsyncIterator:
    """
    Yield a ``fakeredis.aioredis.FakeRedis`` connection.

    Tests that drive components directly (without the MessageBus) do not need
    this fixture; it exists so future tests can swap in a fake without
    touching infrastructure code. The connection is closed cleanly on
    teardown to avoid event-loop warnings.
    """
    import fakeredis.aioredis

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# In-memory database
# ---------------------------------------------------------------------------


@pytest.fixture
async def in_memory_db() -> AsyncIterator:
    """
    Create an aiosqlite engine, build the ORM schema, and yield the engine.

    SQLite cannot model TimescaleDB hypertables, but it does honour the same
    column types and constraints, so the repository layer behaves identically
    for the operations the tests exercise. The engine is disposed on teardown
    so each test starts from an empty schema.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    # Importing the models module registers all tables on Base.metadata.
    import core.db.models  # noqa: F401  (side-effect: registers tables)
    from core.db.connection import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield engine
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------


@pytest.fixture
async def paper_broker(mock_settings: Settings) -> AsyncIterator:
    """
    Yield a connected ``PaperBroker`` instance.

    The broker calls ``assert_paper_mode`` on ``connect()`` and ``place_order``,
    so we deliberately route this through ``mock_settings`` which guarantees
    ``TRADING_MODE=paper``. Without that the broker would refuse to start.
    """
    from agents.execution.paper_broker import PaperBroker

    # Make sure the broker sees a paper-mode settings object even if a test
    # ran earlier that mutated the lru_cache.
    assert mock_settings.trading_mode == TradingMode.PAPER

    broker = PaperBroker(mock_settings)
    await broker.connect()
    try:
        yield broker
    finally:
        await broker.disconnect()
