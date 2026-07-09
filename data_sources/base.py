"""
data_sources/base.py
---------------------
The ``DataSource`` abstraction — the system's single seam for market data.

Every agent that needs OHLCV candles or live prices goes through a ``DataSource``,
never through CCXT or MetaTrader directly. This mirrors the execution side, where
``ExecutionBroker`` decouples order placement from any specific venue.

Two implementations ship today:
- ``PublicExchangeSource`` — keyless, read-only public market data via CCXT
  (Binance/Bybit/... public endpoints). **This is the default** and the only source
  the public/cloud paper-trading demo ever uses. No API key required.
- ``MT5Source`` — local-only, reads candles/prices from a running MetaTrader 5
  terminal. Optional, never required, and read-only (it never places orders).

Selection is config-driven (``DATA_SOURCE``), resolved by ``get_data_source()`` in
``data_sources/__init__.py``.

Contract: a ``DataSource`` is an async context manager that yields typed
``OHLCVCandle`` / ``Ticker`` models — exactly the contracts the rest of the system
already speaks. It does **not** publish to Redis, compute indicators, or place orders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.models.market import OHLCVCandle, Ticker


@dataclass(frozen=True)
class DataSourceCapabilities:
    """Static description of what a data source can do and what it costs to use."""

    name: str  # Short id, e.g. "binance" or "mt5"
    is_live: bool  # True if it returns real, current market prices
    is_keyless: bool  # True if usable with zero credentials
    requires_local_terminal: bool  # True if it needs software running on this host (MT5)


class DataSource(ABC):
    """
    Abstract market-data source.

    Subclasses must implement the lifecycle (``connect``/``disconnect``) and the two
    fetch methods. The async-context-manager and ``ping`` helpers are provided here.
    """

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def capabilities(self) -> DataSourceCapabilities:
        """Return this source's static capabilities."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short human-readable name used in logs/heartbeats (e.g. 'binance', 'mt5')."""
        ...

    @property
    def is_sandbox(self) -> bool:
        """
        Whether the source is in a sandbox/testnet mode.

        Defaults to False; ``PublicExchangeSource`` overrides this to reflect the
        exchange's testnet flag. Kept for backwards-compatibility with the market
        data agent's startup logging.
        """
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Open the underlying connection (network session or terminal handle)."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the underlying connection and release resources."""
        ...

    async def __aenter__(self) -> DataSource:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since_ms: int | None = None,
    ) -> list[OHLCVCandle]:
        """Fetch OHLCV candles, oldest first."""
        ...

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch the latest ticker (last price + 24h stats) for a symbol."""
        ...

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """
        Lightweight reachability check. Default implementation attempts a tiny
        OHLCV fetch on a best-effort symbol; subclasses may override with something
        cheaper.
        """
        try:
            await self.fetch_ohlcv(self._ping_symbol(), self._ping_timeframe(), limit=1)
            return True
        except Exception:
            return False

    def _ping_symbol(self) -> str:
        return "BTC/USDT"

    def _ping_timeframe(self) -> str:
        return "1m"
