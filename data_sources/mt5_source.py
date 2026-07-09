"""
data_sources/mt5_source.py
--------------------------
``MT5Source`` — optional, local-only, READ-ONLY market data from MetaTrader 5.

This source reads candles and prices from a MetaTrader 5 terminal running on the
*same machine*. It is never required: the public/cloud demo uses
``PublicExchangeSource`` and never imports the ``MetaTrader5`` package. ``MT5Source``
is only constructed when ``DATA_SOURCE=mt5`` (or ``auto`` with MT5 available).

Hard guarantees:
- **Read-only.** This class only fetches data (``copy_rates_*``, ``symbol_info_tick``).
  It never opens, modifies, or closes a position. Order placement lives entirely in
  ``agents/execution`` behind the ``ExecutionBroker`` interface and is out of scope
  here.
- **Lazy import.** ``import MetaTrader5`` happens inside ``connect()``/``is_available()``
  so that environments without the package (Linux, CI, the cloud demo) are unaffected.
- **Symbol mapping.** Exchange-style ``BTC/USDT`` is mapped to a broker symbol such as
  ``BTCUSD``. The mapping is intentionally small and local; brokers vary, so override
  ``MT5_SYMBOL_MAP`` if your broker uses different names.

Because activating this requires a live terminal, it is not exercised by CI. Its unit
tests cover the availability guard and symbol/timeframe mapping only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from core.config import get_settings
from core.exceptions import ExchangeConnectionError
from core.logging import get_logger
from core.models.market import OHLCVCandle, Ticker
from data_sources.base import DataSource, DataSourceCapabilities

logger = get_logger("mt5_source")


# Default exchange-symbol → broker-symbol map. Brokers differ; extend as needed.
_DEFAULT_SYMBOL_MAP: dict[str, str] = {
    "BTC/USDT": "BTCUSD",
    "ETH/USDT": "ETHUSD",
    "BTC/USD": "BTCUSD",
    "ETH/USD": "ETHUSD",
}


class MT5Source(DataSource):
    """Read-only market data from a local MetaTrader 5 terminal."""

    def __init__(self, symbol_map: dict[str, str] | None = None) -> None:
        self._settings = get_settings()
        self._mt5: Any = None  # the MetaTrader5 module, imported lazily
        self._connected = False
        self._symbol_map = symbol_map or dict(_DEFAULT_SYMBOL_MAP)

    # ------------------------------------------------------------------
    # Availability (used by the factory's "auto" mode)
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """True if the MetaTrader5 package can be imported on this host."""
        try:
            import MetaTrader5  # noqa: F401

            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> DataSourceCapabilities:
        return DataSourceCapabilities(
            name="mt5",
            is_live=True,
            is_keyless=False,  # needs a logged-in local terminal
            requires_local_terminal=True,
        )

    @property
    def source_name(self) -> str:
        return "mt5"

    # ------------------------------------------------------------------
    # Symbol / timeframe mapping
    # ------------------------------------------------------------------

    def map_symbol(self, symbol: str) -> str:
        """Map an exchange symbol (BTC/USDT) to a broker symbol (BTCUSD)."""
        return self._symbol_map.get(symbol, symbol.replace("/", ""))

    def _mt5_timeframe(self, timeframe: str) -> int:
        """Map a string timeframe to the MetaTrader5 TIMEFRAME_* constant."""
        mt5 = self._require_module()
        table = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        if timeframe not in table:
            raise ExchangeConnectionError(f"MT5Source: unsupported timeframe '{timeframe}'")
        return table[timeframe]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _require_module(self) -> Any:
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # type: ignore
            except Exception as exc:  # pragma: no cover - import guard
                raise ExchangeConnectionError(
                    "MetaTrader5 package is not installed. MT5Source is local-only; "
                    "use DATA_SOURCE=public for the keyless demo."
                ) from exc
            self._mt5 = mt5
        return self._mt5

    async def connect(self) -> None:
        mt5 = self._require_module()
        # initialize() attaches to a running, logged-in terminal. We pass no
        # credentials — this is read-only and relies on an already-authenticated
        # terminal (the operator's session). We never place trades.
        if not mt5.initialize():
            code, msg = mt5.last_error()
            raise ExchangeConnectionError(f"MT5 initialize() failed ({code}): {msg}")
        self._connected = True
        logger.info("mt5_source_connected", read_only=True)

    async def disconnect(self) -> None:
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("mt5_source_disconnected")

    # ------------------------------------------------------------------
    # Data fetching (read-only)
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since_ms: int | None = None,
    ) -> list[OHLCVCandle]:
        mt5 = self._require_module()
        broker_symbol = self.map_symbol(symbol)
        tf = self._mt5_timeframe(timeframe)

        rates = mt5.copy_rates_from_pos(broker_symbol, tf, 0, limit)
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise ExchangeConnectionError(
                f"MT5 copy_rates_from_pos returned no data for {broker_symbol} ({code}): {msg}"
            )

        candles: list[OHLCVCandle] = []
        for r in rates:
            ts = datetime.fromtimestamp(int(r["time"]), tz=UTC)
            # MT5 reports tick_volume; real_volume is often 0 on CFD/crypto symbols.
            vol = r["real_volume"] if r["real_volume"] else r["tick_volume"]
            candles.append(
                OHLCVCandle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=Decimal(str(r["open"])),
                    high=Decimal(str(r["high"])),
                    low=Decimal(str(r["low"])),
                    close=Decimal(str(r["close"])),
                    volume=Decimal(str(vol)),
                )
            )
        return candles

    async def fetch_ticker(self, symbol: str) -> Ticker:
        mt5 = self._require_module()
        broker_symbol = self.map_symbol(symbol)
        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            code, msg = mt5.last_error()
            raise ExchangeConnectionError(
                f"MT5 symbol_info_tick failed for {broker_symbol} ({code}): {msg}"
            )
        last = tick.last or tick.bid or tick.ask
        return Ticker(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            last=Decimal(str(last)),
            bid=Decimal(str(tick.bid)) if tick.bid else None,
            ask=Decimal(str(tick.ask)) if tick.ask else None,
        )
