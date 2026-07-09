"""
tests/unit/test_core.py
------------------------
Unit tests for core infrastructure: config, normaliser, channels.
No I/O — all tests run without Redis, DB, or exchange.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from core.config import TradingMode, get_settings
from core.messaging import Channels
from core.models.market import OHLCVCandle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def setup_method(self):
        """Clear settings cache before each test."""
        get_settings.cache_clear()

    def teardown_method(self):
        get_settings.cache_clear()

    def test_default_trading_mode_is_paper(self):
        with patch.dict(os.environ, {}, clear=False):
            settings = get_settings()
            assert settings.trading_mode == TradingMode.PAPER
            assert settings.is_paper_trading is True

    def test_assert_paper_mode_raises_in_live(self):
        get_settings.cache_clear()
        with patch.dict(os.environ, {"TRADING_MODE": "live"}):
            get_settings.cache_clear()
            settings = get_settings()
            with pytest.raises(RuntimeError, match="paper trading mode"):
                settings.assert_paper_mode()

    def test_log_summary_contains_no_secrets(self):
        settings = get_settings()
        summary = settings.log_summary()
        summary_str = str(summary)
        assert "password" not in summary_str.lower()
        assert "secret" not in summary_str.lower()
        assert "api_key" not in summary_str.lower()

    def test_database_url_format(self):
        settings = get_settings()
        url = settings.database.url
        assert url.startswith("postgresql+asyncpg://")
        assert settings.database.name in url

    def test_redis_url_no_password(self):
        settings = get_settings()
        url = settings.redis.url
        assert url.startswith("redis://")

    def test_risk_fraction_validation(self):
        from core.config import RiskSettings

        with pytest.raises(ValueError):
            RiskSettings(max_position_pct=1.5)
        with pytest.raises(ValueError):
            RiskSettings(max_daily_loss_pct=0.0)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


class TestChannels:
    def test_ohlcv_channel_replaces_slash(self):
        ch = Channels.ohlcv("BTC/USDT", "1m")
        assert ch == "market.ohlcv.BTC-USDT.1m"
        assert "/" not in ch

    def test_ohlcv_channel_different_timeframes(self):
        assert Channels.ohlcv("ETH/USDT", "5m") == "market.ohlcv.ETH-USDT.5m"
        assert Channels.ohlcv("BTC/USDT", "1h") == "market.ohlcv.BTC-USDT.1h"

    def test_orderbook_channel(self):
        assert Channels.orderbook("BTC/USDT") == "market.orderbook.BTC-USDT"

    def test_technical_signal_channel(self):
        assert Channels.technical_signal("BTC/USDT") == "signal.technical.BTC-USDT"

    def test_constants_are_strings(self):
        assert isinstance(Channels.DECISION_PROPOSAL, str)
        assert isinstance(Channels.SYSTEM_HEARTBEAT, str)
        assert isinstance(Channels.SYSTEM_RISK_OVERRIDE, str)


# ---------------------------------------------------------------------------
# OHLCVNormaliser
# ---------------------------------------------------------------------------


class TestOHLCVNormaliser:
    def _make_candle(self, ts_offset_seconds: int = 0, symbol="BTC/USDT") -> OHLCVCandle:
        base_ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        from datetime import timedelta

        ts = base_ts + timedelta(seconds=ts_offset_seconds)
        return OHLCVCandle(
            symbol=symbol,
            timeframe="1m",
            timestamp=ts,
            open=Decimal("42000"),
            high=Decimal("42500"),
            low=Decimal("41800"),
            close=Decimal("42200"),
            volume=Decimal("100"),
        )

    def test_single_candle_passes_through(self):
        from agents.market_data.normalizer import OHLCVNormaliser

        n = OHLCVNormaliser()
        candle = self._make_candle()
        result = n.process_batch([candle])
        assert len(result) == 1
        assert result[0].timestamp == candle.timestamp

    def test_duplicate_candle_is_dropped(self):
        from agents.market_data.normalizer import OHLCVNormaliser

        n = OHLCVNormaliser()
        candle = self._make_candle()
        # First pass — accepted
        result1 = n.process_batch([candle])
        assert len(result1) == 1
        # Second pass — same timestamp, should be dropped
        result2 = n.process_batch([candle])
        assert len(result2) == 0

    def test_batch_sorted_by_timestamp(self):
        from agents.market_data.normalizer import OHLCVNormaliser

        n = OHLCVNormaliser()
        # Submit out-of-order
        candles = [
            self._make_candle(ts_offset_seconds=120),
            self._make_candle(ts_offset_seconds=0),
            self._make_candle(ts_offset_seconds=60),
        ]
        result = n.process_batch(candles)
        assert len(result) == 3
        timestamps = [c.timestamp for c in result]
        assert timestamps == sorted(timestamps)

    def test_empty_batch_returns_empty(self):
        from agents.market_data.normalizer import OHLCVNormaliser

        n = OHLCVNormaliser()
        assert n.process_batch([]) == []

    def test_stats_track_duplicates(self):
        from agents.market_data.normalizer import OHLCVNormaliser

        n = OHLCVNormaliser()
        candle = self._make_candle()
        n.process_batch([candle])
        n.process_batch([candle])  # duplicate
        stats = n.get_stats()
        key = "BTC/USDT/1m"
        assert stats[key]["duplicates"] == 1
        assert stats[key]["published"] == 1

    def test_get_last_candle(self):
        from agents.market_data.normalizer import OHLCVNormaliser

        n = OHLCVNormaliser()
        assert n.get_last_candle("BTC/USDT", "1m") is None
        candle = self._make_candle()
        n.process_batch([candle])
        assert n.get_last_candle("BTC/USDT", "1m") is not None

    def test_reset_clears_state(self):
        from agents.market_data.normalizer import OHLCVNormaliser

        n = OHLCVNormaliser()
        candle = self._make_candle()
        n.process_batch([candle])
        n.reset("BTC/USDT", "1m")
        # After reset, the same candle should be accepted again
        result = n.process_batch([candle])
        assert len(result) == 1
