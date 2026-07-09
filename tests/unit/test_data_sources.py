"""
tests/unit/test_data_sources.py
-------------------------------
Unit tests for the pluggable market-data layer (``data_sources/``).

Covered:
- ``get_data_source`` selection logic (public default, explicit public/mt5, auto,
  unknown → fallback).
- ``PublicExchangeSource`` delegates to an injected fetcher and reports keyless
  capabilities.
- ``MT5Source`` availability guard, symbol mapping, and clean failure when the
  MetaTrader5 package is absent (never required, never imported on this host).

No network, no Redis, no exchange, no MT5 terminal.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from core.config import get_settings
from core.models.market import OHLCVCandle, Ticker
from data_sources import PublicExchangeSource, get_data_source
from data_sources.base import DataSource
from data_sources.mt5_source import MT5Source


def _with_data_source(value: str):
    """Context manager: pin DATA_SOURCE and rebuild the settings cache."""
    return patch.dict(os.environ, {"DATA_SOURCE": value}, clear=False)


# ---------------------------------------------------------------------------
# Factory selection
# ---------------------------------------------------------------------------


class TestFactory:
    def test_default_is_public(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATA_SOURCE", None)
            get_settings.cache_clear()
            try:
                src = get_data_source()
            finally:
                get_settings.cache_clear()
        assert isinstance(src, PublicExchangeSource)
        assert src.capabilities.is_keyless is True
        assert src.capabilities.requires_local_terminal is False

    def test_explicit_public(self) -> None:
        with _with_data_source("public"):
            get_settings.cache_clear()
            try:
                src = get_data_source()
            finally:
                get_settings.cache_clear()
        assert isinstance(src, PublicExchangeSource)

    def test_unknown_falls_back_to_public(self) -> None:
        with _with_data_source("nonsense"):
            get_settings.cache_clear()
            try:
                src = get_data_source()
            finally:
                get_settings.cache_clear()
        assert isinstance(src, PublicExchangeSource)

    def test_explicit_mt5_constructs_mt5_source(self) -> None:
        # Constructing MT5Source does not import MetaTrader5 or connect.
        with _with_data_source("mt5"):
            get_settings.cache_clear()
            try:
                src = get_data_source()
            finally:
                get_settings.cache_clear()
        assert isinstance(src, MT5Source)
        assert src.source_name == "mt5"

    def test_auto_falls_back_when_mt5_unavailable(self) -> None:
        # MetaTrader5 is not installed on this host, so auto → public.
        with _with_data_source("auto"):
            get_settings.cache_clear()
            try:
                src = get_data_source()
            finally:
                get_settings.cache_clear()
        assert isinstance(src, (PublicExchangeSource, MT5Source))
        # On a host without MetaTrader5 this must be the public source.
        if not MT5Source.is_available():
            assert isinstance(src, PublicExchangeSource)


# ---------------------------------------------------------------------------
# PublicExchangeSource delegation
# ---------------------------------------------------------------------------


class _FakeFetcher:
    """Minimal stand-in for ExchangeFetcher to verify delegation without CCXT."""

    def __init__(self) -> None:
        self.connected = False
        self.is_sandbox = False
        self.calls: list[str] = []

    async def connect(self) -> None:
        self.connected = True
        self.calls.append("connect")

    async def disconnect(self) -> None:
        self.connected = False
        self.calls.append("disconnect")

    async def fetch_ohlcv(self, symbol, timeframe, limit=500, since_ms=None):
        self.calls.append(f"ohlcv:{symbol}:{timeframe}:{limit}")
        return [
            OHLCVCandle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("5"),
            )
        ]

    async def fetch_ticker(self, symbol):
        self.calls.append(f"ticker:{symbol}")
        return Ticker(
            symbol=symbol,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            last=Decimal("100"),
        )

    async def ping(self) -> bool:
        return True


class TestPublicExchangeSource:
    def test_is_a_data_source(self) -> None:
        assert issubclass(PublicExchangeSource, DataSource)

    async def test_delegates_lifecycle_and_fetches(self) -> None:
        fake = _FakeFetcher()
        src = PublicExchangeSource(fetcher=fake)

        async with src:
            assert fake.connected is True
            candles = await src.fetch_ohlcv("BTC/USDT", "1m", limit=10)
            assert len(candles) == 1 and candles[0].symbol == "BTC/USDT"
            ticker = await src.fetch_ticker("BTC/USDT")
            assert ticker.last == Decimal("100")
            assert await src.ping() is True

        assert fake.connected is False
        assert "connect" in fake.calls and "disconnect" in fake.calls
        assert src.is_sandbox is False


# ---------------------------------------------------------------------------
# MT5Source guards & mapping (never connects to a live terminal in tests)
# ---------------------------------------------------------------------------


class TestMT5Source:
    def test_symbol_mapping(self) -> None:
        src = MT5Source()
        assert src.map_symbol("BTC/USDT") == "BTCUSD"
        assert src.map_symbol("ETH/USDT") == "ETHUSD"
        # Unknown symbols fall back to a slash-stripped form.
        assert src.map_symbol("XRP/USDT") == "XRPUSDT"

    def test_capabilities_not_keyless(self) -> None:
        caps = MT5Source().capabilities
        assert caps.requires_local_terminal is True
        assert caps.is_keyless is False

    @pytest.mark.skipif(MT5Source.is_available(), reason="MetaTrader5 is installed here")
    async def test_connect_raises_without_metatrader5(self) -> None:
        from core.exceptions import ExchangeConnectionError

        with pytest.raises(ExchangeConnectionError):
            await MT5Source().connect()
