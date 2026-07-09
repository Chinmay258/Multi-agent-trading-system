"""
agents/execution/symbol_mapper.py
-----------------------------------
Bidirectional symbol mapping between Python/CCXT format and MetaTrader 5 format.

Python format uses a slash separator (e.g. "BTC/USDT").
MT5 format is a single concatenated string without a separator (e.g. "BTCUSD").

The mapper is intentionally strict: unknown symbols raise ValueError rather than
silently passing through, preventing silent symbol mismatches from causing bad trades.
"""

from __future__ import annotations

_DEFAULT_MAPPINGS: dict[str, str] = {
    "BTC/USDT": "BTCUSD",
    "ETH/USDT": "ETHUSD",
    "BNB/USDT": "BNBUSD",
    "SOL/USDT": "SOLUSD",
    "XRP/USDT": "XRPUSD",
}


class SymbolMapper:
    """
    Bidirectional mapping between Python (CCXT) symbol format and MT5 symbol format.

    Python → MT5:  "BTC/USDT" → "BTCUSD"
    MT5 → Python:  "BTCUSD"   → "BTC/USDT"

    Usage:
        mapper = SymbolMapper()
        mapper.to_mt5("BTC/USDT")   # "BTCUSD"
        mapper.to_python("BTCUSD")  # "BTC/USDT"

        # With extra pairs:
        mapper = SymbolMapper(extra={"DOGE/USDT": "DOGEUSD"})
    """

    def __init__(self, extra: dict[str, str] | None = None) -> None:
        self._python_to_mt5: dict[str, str] = dict(_DEFAULT_MAPPINGS)
        if extra:
            self._python_to_mt5.update(extra)
        # Build the reverse mapping from the combined dict
        self._mt5_to_python: dict[str, str] = {
            mt5: python for python, mt5 in self._python_to_mt5.items()
        }
        # Tickmill may name the ETH contract ETHUSDT rather than ETHUSD.
        # Add as a fallback so to_python() handles both without changing the
        # primary forward mapping (ETH/USDT → ETHUSD).
        self._mt5_to_python.setdefault("ETHUSDT", "ETH/USDT")

    def to_mt5(self, symbol: str) -> str:
        """Convert Python format symbol to MT5 format.

        Args:
            symbol: Python/CCXT symbol, e.g. "BTC/USDT"

        Returns:
            MT5 symbol string, e.g. "BTCUSD"

        Raises:
            ValueError: Symbol not in known mappings.
        """
        result = self._python_to_mt5.get(symbol)
        if result is None:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Add it to SymbolMapper or pass via extra=."
            )
        return result

    def to_python(self, symbol: str) -> str:
        """Convert MT5 format symbol to Python format.

        Args:
            symbol: MT5 symbol string, e.g. "BTCUSD"

        Returns:
            Python/CCXT symbol, e.g. "BTC/USDT"

        Raises:
            ValueError: Symbol not in known mappings.
        """
        result = self._mt5_to_python.get(symbol)
        if result is None:
            raise ValueError(
                f"Unknown MT5 symbol '{symbol}'. Add it to SymbolMapper or pass via extra=."
            )
        return result
