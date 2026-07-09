"""
agents/market_data/fetcher.py
------------------------------
CCXT exchange abstraction layer.

All direct exchange API calls live here and nowhere else. Every other module
in the system talks to this class — never to CCXT directly. This means:
- Swapping exchange (Binance → OKX) touches one file.
- Adding retry logic touches one file.
- Mocking in tests touches one file.

Design decisions:
- Async CCXT (ccxt.async_support) throughout — no blocking I/O.
- Tenacity for retry logic with exponential backoff. Exchange APIs are
  unreliable; every fetch must be retry-safe.
- ExchangeFetcher is a context manager. Always use `async with` to ensure
  the underlying aiohttp session is properly closed.
- Rate limiting is handled by CCXT internally (enableRateLimit=True), but
  we add our own conservative floor via EXCHANGE_RATE_LIMIT_MS.
- Paper trading: fetcher still hits the real exchange for market data
  (prices, order books). Only order placement is mocked. This is correct —
  you want real prices in paper trading, just no real orders.
- Sandbox mode: uses the exchange's own testnet. Separate from paper trading.

Usage:
    async with ExchangeFetcher() as fetcher:
        candles = await fetcher.fetch_ohlcv("BTC/USDT", "1m", limit=100)
        book = await fetcher.fetch_order_book("BTC/USDT")
        ticker = await fetcher.fetch_ticker("BTC/USDT")
"""

from __future__ import annotations

import ccxt.async_support as ccxt
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings
from core.exceptions import (
    ExchangeAuthError,
    ExchangeConnectionError,
    ExchangeRateLimitError,
    InsufficientDataError,
)
from core.logging import get_logger
from core.models.market import OHLCVCandle, OrderBook, Ticker

logger = get_logger("exchange_fetcher")


# ---------------------------------------------------------------------------
# Retry policies
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient exchange errors that are safe to retry."""
    retryable = (
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
        ExchangeRateLimitError,
    )
    return isinstance(exc, retryable)


RETRY_POLICY = dict(
    retry=retry_if_exception_type(
        (
            ccxt.NetworkError,
            ccxt.RequestTimeout,
            ccxt.ExchangeNotAvailable,
            ccxt.DDoSProtection,
        )
    ),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


# ---------------------------------------------------------------------------
# ExchangeFetcher
# ---------------------------------------------------------------------------


class ExchangeFetcher:
    """
    Async wrapper around CCXT providing typed, retry-safe market data fetching.

    Responsibilities:
    - Initialise and configure the CCXT exchange client.
    - Fetch OHLCV, order book, ticker data.
    - Normalise raw CCXT responses into our typed models.
    - Handle rate limiting, retries, and connection errors.
    - Report health status.

    NOT responsible for:
    - Placing orders (that's ExecutionAgent / LiveExchange / PaperExchange).
    - Publishing to Redis (that's MarketDataAgent).
    - Indicator calculation (that's TechnicalAnalysisAgent).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._exchange: ccxt.Exchange | None = None
        self._markets: dict = {}
        self._connected = False

    async def connect(self) -> None:
        """
        Initialise the CCXT exchange client and load markets.
        Validates API credentials if provided.
        """
        cfg = self._settings.exchange
        exchange_class = getattr(ccxt, cfg.name, None)

        if exchange_class is None:
            raise ExchangeConnectionError(
                f"Unknown exchange: '{cfg.name}'. "
                f"Check EXCHANGE_NAME in config. Available: {ccxt.exchanges[:10]}..."
            )

        exchange_config: dict = {
            "enableRateLimit": True,
            "rateLimit": cfg.rate_limit_ms,
            "timeout": cfg.request_timeout_ms,
            "options": {"defaultType": "spot"},
        }

        # Only include credentials if provided — fetcher works without them
        # for public endpoints (OHLCV, order book, ticker)
        if cfg.api_key and cfg.api_secret:
            exchange_config["apiKey"] = cfg.api_key.get_secret_value()
            exchange_config["secret"] = cfg.api_secret.get_secret_value()
            if cfg.api_passphrase:
                exchange_config["password"] = cfg.api_passphrase.get_secret_value()

        if cfg.sandbox:
            exchange_config["sandbox"] = True

        self._exchange = exchange_class(exchange_config)

        try:
            self._markets = await self._exchange.load_markets()
            self._connected = True
            logger.info(
                "exchange_connected",
                exchange=cfg.name,
                sandbox=cfg.sandbox,
                market_count=len(self._markets),
            )
        except ccxt.AuthenticationError as e:
            raise ExchangeAuthError(f"Invalid API credentials for {cfg.name}: {e}") from e
        except ccxt.NetworkError as e:
            raise ExchangeConnectionError(f"Cannot connect to {cfg.name}: {e}") from e

    async def disconnect(self) -> None:
        """Close the underlying aiohttp session."""
        if self._exchange:
            await self._exchange.close()
            self._connected = False
            logger.info("exchange_disconnected", exchange=self._settings.exchange.name)

    async def __aenter__(self) -> ExchangeFetcher:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Public data fetching methods
    # ------------------------------------------------------------------

    @retry(**RETRY_POLICY)
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since_ms: int | None = None,
    ) -> list[OHLCVCandle]:
        """
        Fetch OHLCV candles for a symbol and timeframe.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT"
            timeframe: Candle interval, e.g. "1m", "5m", "1h"
            limit: Number of candles to fetch (max varies by exchange)
            since_ms: Fetch candles from this timestamp (milliseconds UTC)

        Returns:
            List of OHLCVCandle objects, oldest first.

        Raises:
            ExchangeConnectionError: Cannot reach the exchange.
            InsufficientDataError: Exchange returned fewer candles than needed.
        """
        self._assert_connected()
        self._assert_symbol_supported(symbol)

        try:
            raw = await self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_ms,
            )
        except ccxt.RateLimitExceeded as e:
            raise ExchangeRateLimitError(str(e)) from e
        except ccxt.BadSymbol as e:
            raise ExchangeConnectionError(f"Bad symbol {symbol}: {e}") from e
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            raise ExchangeConnectionError(str(e)) from e

        if not raw:
            raise InsufficientDataError(
                required=1,
                available=0,
                indicator=f"OHLCV[{symbol}/{timeframe}]",
            )

        candles = [OHLCVCandle.from_ccxt(row, symbol, timeframe) for row in raw]

        logger.debug(
            "ohlcv_fetched",
            symbol=symbol,
            timeframe=timeframe,
            candles=len(candles),
            first=str(candles[0].timestamp),
            last=str(candles[-1].timestamp),
        )
        return candles

    @retry(**RETRY_POLICY)
    async def fetch_order_book(
        self,
        symbol: str,
        depth: int | None = None,
    ) -> OrderBook:
        """
        Fetch L2 order book snapshot.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT"
            depth: Number of levels (bids/asks) to return. None = exchange default.

        Returns:
            OrderBook with bids and asks.
        """
        self._assert_connected()
        self._assert_symbol_supported(symbol)
        depth = depth or self._settings.market_data.orderbook_depth

        try:
            raw = await self._exchange.fetch_order_book(symbol, limit=depth)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            raise ExchangeConnectionError(str(e)) from e

        book = OrderBook.from_ccxt(raw, symbol)
        logger.debug(
            "orderbook_fetched",
            symbol=symbol,
            bids=len(book.bids),
            asks=len(book.asks),
            spread_pct=book.spread_pct,
        )
        return book

    @retry(**RETRY_POLICY)
    async def fetch_ticker(self, symbol: str) -> Ticker:
        """
        Fetch 24-hour rolling ticker.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT"

        Returns:
            Ticker with last price, 24h high/low/volume.
        """
        self._assert_connected()
        self._assert_symbol_supported(symbol)

        try:
            raw = await self._exchange.fetch_ticker(symbol)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            raise ExchangeConnectionError(str(e)) from e

        ticker = Ticker.from_ccxt(raw)
        logger.debug(
            "ticker_fetched",
            symbol=symbol,
            last=float(ticker.last),
            change_pct=ticker.change_24h_pct,
        )
        return ticker

    @retry(**RETRY_POLICY)
    async def fetch_balance(self) -> dict:
        """
        Fetch account balance. Requires API credentials.
        Used by Risk agent to verify available funds.

        Returns:
            Raw CCXT balance dict (total, free, used per currency).
        """
        self._assert_connected()
        if not self._settings.exchange.api_key:
            raise ExchangeAuthError("fetch_balance requires API credentials")

        try:
            balance = await self._exchange.fetch_balance()
            logger.debug("balance_fetched", currencies=list(balance.get("total", {}).keys()))
            return balance
        except ccxt.AuthenticationError as e:
            raise ExchangeAuthError(str(e)) from e
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            raise ExchangeConnectionError(str(e)) from e

    # ------------------------------------------------------------------
    # Symbol utilities
    # ------------------------------------------------------------------

    def get_supported_symbols(self) -> list[str]:
        """Return all trading pairs available on this exchange."""
        return list(self._markets.keys())

    def is_symbol_supported(self, symbol: str) -> bool:
        return symbol in self._markets

    def get_min_order_size(self, symbol: str) -> float | None:
        """Return the minimum order size for a symbol (base asset amount)."""
        market = self._markets.get(symbol, {})
        return market.get("limits", {}).get("amount", {}).get("min")

    def get_price_precision(self, symbol: str) -> int | None:
        """Return the number of decimal places for prices on this symbol."""
        market = self._markets.get(symbol, {})
        return market.get("precision", {}).get("price")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """
        Lightweight connectivity check.
        Returns True if the exchange API is reachable.
        """
        try:
            await self._exchange.fetch_time()
            return True
        except Exception:
            return False

    @property
    def exchange_name(self) -> str:
        return self._settings.exchange.name

    @property
    def is_sandbox(self) -> bool:
        return self._settings.exchange.sandbox

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _assert_connected(self) -> None:
        if not self._connected or self._exchange is None:
            raise ExchangeConnectionError(
                "ExchangeFetcher not connected. Use 'async with ExchangeFetcher() as fetcher'"
            )

    def _assert_symbol_supported(self, symbol: str) -> None:
        if self._markets and symbol not in self._markets:
            raise ExchangeConnectionError(
                f"Symbol '{symbol}' not available on {self.exchange_name}. "
                f"Check MARKET_DATA_SYMBOLS in config."
            )
