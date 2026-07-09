"""
scripts/seed_historical.py
---------------------------
Fetch historical OHLCV data from the exchange and upsert into the database.

Connects to the configured exchange via ExchangeFetcher, fetches N days of
OHLCV for each (symbol, timeframe) pair, and upserts via CandleRepository.
Requires a running PostgreSQL instance (docker-compose up -d postgres).

Usage:
    python scripts/seed_historical.py
    python scripts/seed_historical.py --days 30 --symbols "BTC/USDT,ETH/USDT"
    python scripts/seed_historical.py --days 90 --timeframes "1h,4h,1d"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.market_data.fetcher import ExchangeFetcher
from core.config import get_settings
from core.db.connection import get_session
from core.db.repositories.candle_repo import CandleRepository
from core.models.market import OHLCVCandle

# CCXT returns up to 1000 candles per request; use 500 to stay conservative.
_BATCH_SIZE = 500

_TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


async def seed_symbol_timeframe(
    fetcher: ExchangeFetcher,
    symbol: str,
    timeframe: str,
    days: int,
) -> int:
    """
    Fetch and upsert all candles for one (symbol, timeframe) pair.

    Returns the total number of candles upserted.
    """
    timeframe_ms = _TIMEFRAME_MS.get(timeframe, 60_000)
    start_ms = int((datetime.now(UTC) - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    since_ms = start_ms
    total = 0

    print(f"  {symbol} {timeframe}  ({days} days) ...", flush=True)

    while since_ms < now_ms:
        raw = await fetcher._exchange.fetch_ohlcv(
            symbol, timeframe, since=since_ms, limit=_BATCH_SIZE
        )
        if not raw:
            break

        candles = [OHLCVCandle.from_ccxt(row, symbol, timeframe) for row in raw]

        async with get_session() as session:
            repo = CandleRepository(session)
            for candle in candles:
                await repo.upsert_candle(candle)
            await session.commit()

        since_ms = raw[-1][0] + timeframe_ms  # advance past last candle
        total += len(raw)

        if total % 5000 == 0:
            pct = (since_ms - start_ms) / (now_ms - start_ms) * 100
            print(f"  {total:,} candles ({pct:.1f}%) ...", flush=True)

        await asyncio.sleep(0.3)

    print(f"  -> {total:,} candles upserted")
    return total


async def run_seed(
    days: int,
    symbols: list[str],
    timeframes: list[str],
) -> None:
    """Connect to exchange and seed all requested (symbol, timeframe) pairs."""
    settings = get_settings()
    fetcher = ExchangeFetcher()

    print(f"Connecting to {settings.exchange.name} (sandbox={settings.exchange.sandbox}) ...")
    await fetcher.connect()

    try:
        grand_total = 0
        for symbol in symbols:
            for timeframe in timeframes:
                count = await seed_symbol_timeframe(fetcher, symbol, timeframe, days)
                grand_total += count

        print(f"\nDone.  {grand_total:,} total candles upserted.")
    finally:
        await fetcher.disconnect()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV from exchange and upsert into DB."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of history to fetch (default: 90)",
    )
    parser.add_argument(
        "--symbols",
        default="BTC/USDT,ETH/USDT",
        help='Comma-separated symbols (default: "BTC/USDT,ETH/USDT")',
    )
    parser.add_argument(
        "--timeframes",
        default="1h,4h",
        help='Comma-separated timeframes (default: "1h,4h")',
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    if not symbols:
        print("Error: --symbols cannot be empty.")
        sys.exit(1)
    if not timeframes:
        print("Error: --timeframes cannot be empty.")
        sys.exit(1)

    print(f"Seeding {args.days} days of OHLCV for {symbols} x {timeframes}")
    await run_seed(days=args.days, symbols=symbols, timeframes=timeframes)


if __name__ == "__main__":
    asyncio.run(_main())
