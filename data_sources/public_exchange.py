"""
data_sources/public_exchange.py
-------------------------------
``PublicExchangeSource`` — the default, keyless market-data source.

It wraps the existing ``ExchangeFetcher`` (CCXT) so all the hardening that already
lives there — retries, rate limiting, typed normalisation, connection guards — is
reused unchanged. The only thing this class adds is the ``DataSource`` contract on
top, so the rest of the system can stay venue-agnostic.

Keyless by design: market data (OHLCV, ticker, order book) needs no API key on
Binance and most exchanges. Credentials are only required for *trading*, which this
source never does. That is exactly what makes the public paper-trading demo runnable
by anyone with zero secrets.
"""

from __future__ import annotations

from agents.market_data.fetcher import ExchangeFetcher
from core.config import get_settings
from core.models.market import OHLCVCandle, Ticker
from data_sources.base import DataSource, DataSourceCapabilities


class PublicExchangeSource(DataSource):
    """Keyless public-exchange market data via CCXT (default source)."""

    def __init__(self, fetcher: ExchangeFetcher | None = None) -> None:
        # Allow injecting a fetcher (tests); otherwise build the default one.
        self._fetcher = fetcher or ExchangeFetcher()
        self._settings = get_settings()

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name=self._settings.exchange.name,
            is_live=True,
            is_keyless=True,
            requires_local_terminal=False,
        )

    @property
    def source_name(self) -> str:
        return self._settings.exchange.name

    @property
    def is_sandbox(self) -> bool:
        return self._fetcher.is_sandbox

    async def connect(self) -> None:
        await self._fetcher.connect()

    async def disconnect(self) -> None:
        await self._fetcher.disconnect()

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since_ms: int | None = None,
    ) -> list[OHLCVCandle]:
        return await self._fetcher.fetch_ohlcv(symbol, timeframe, limit=limit, since_ms=since_ms)

    async def fetch_ticker(self, symbol: str) -> Ticker:
        return await self._fetcher.fetch_ticker(symbol)

    async def ping(self) -> bool:
        return await self._fetcher.ping()
