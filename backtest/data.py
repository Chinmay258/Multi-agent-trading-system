"""
backtest/data.py
----------------
Historical OHLCV loading for the backtest, keyless and reproducible.

Default source is the bundled, version-controlled dataset under ``data/sample/``
(zero external calls). Optionally, fresh history can be fetched from the keyless
public exchange and cached under ``data/cache/`` for longer backtests.
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

from core.models.market import OHLCVCandle

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = _REPO_ROOT / "data" / "sample"
_CACHE_DIR = _REPO_ROOT / "data" / "cache"


def _csv_path(base_dir: Path, symbol: str, timeframe: str) -> Path:
    return base_dir / f"{symbol.replace('/', '-')}_{timeframe}.csv"


def _read_csv(path: Path, symbol: str, timeframe: str) -> list[OHLCVCandle]:
    candles: list[OHLCVCandle] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            candles.append(
                OHLCVCandle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=row["timestamp"],
                    open=Decimal(row["open"]),
                    high=Decimal(row["high"]),
                    low=Decimal(row["low"]),
                    close=Decimal(row["close"]),
                    volume=Decimal(row["volume"]),
                )
            )
    return candles


def load_candles(
    symbol: str,
    timeframe: str,
    source: str = "bundled",
) -> list[OHLCVCandle]:
    """
    Load chronological candles for a symbol/timeframe.

    source="bundled" (default): read ``data/sample/`` — offline, zero external calls.
    source="cache":             read ``data/cache/`` (populated by ``fetch_and_cache``).
    """
    base = _SAMPLE_DIR if source == "bundled" else _CACHE_DIR
    path = _csv_path(base, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"No {source} data for {symbol} {timeframe} at {path}. "
            f"Bundled sets: {[p.name for p in _SAMPLE_DIR.glob('*.csv')]}"
        )
    candles = _read_csv(path, symbol, timeframe)
    candles.sort(key=lambda c: c.timestamp)
    return candles


async def fetch_and_cache(
    symbol: str,
    timeframe: str,
    limit: int = 1000,
) -> list[OHLCVCandle]:
    """
    Fetch fresh history keylessly from the public exchange and cache it under
    ``data/cache/``. Used for longer backtests; never required for the default run.
    """
    from data_sources import get_data_source

    src = get_data_source()
    await src.connect()
    try:
        candles = await src.fetch_ohlcv(symbol, timeframe, limit=limit)
    finally:
        await src.disconnect()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _csv_path(_CACHE_DIR, symbol, timeframe)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.timestamp.isoformat(), c.open, c.high, c.low, c.close, c.volume])
    return candles


def to_arrays(candles: list[OHLCVCandle]) -> dict:
    """Convert candles to parallel float arrays + timestamps for numeric code."""
    import numpy as np

    return {
        "opens": np.array([float(c.open) for c in candles], dtype=np.float64),
        "highs": np.array([float(c.high) for c in candles], dtype=np.float64),
        "lows": np.array([float(c.low) for c in candles], dtype=np.float64),
        "closes": np.array([float(c.close) for c in candles], dtype=np.float64),
        "volumes": np.array([float(c.volume) for c in candles], dtype=np.float64),
        "timestamps": [c.timestamp for c in candles],
    }


def available_datasets() -> list[tuple[str, str]]:
    """List (symbol, timeframe) pairs available in the bundled dataset."""
    out: list[tuple[str, str]] = []
    for p in sorted(_SAMPLE_DIR.glob("*.csv")):
        sym_part, _, tf = p.stem.rpartition("_")
        out.append((sym_part.replace("-", "/"), tf))
    return out
