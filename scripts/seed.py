"""
scripts/seed.py
---------------
OFFLINE database seeder. Loads the bundled sample OHLCV dataset under ``data/sample/``
into TimescaleDB so the demo, tests, and backtests can run with **zero external
calls**.

This is the offline counterpart to ``scripts/seed_historical.py`` (which fetches
fresh data from the live public exchange). Use this one when you want a fully
reproducible, network-free seed:

    python scripts/seed.py                 # load every bundled CSV into the DB
    python scripts/seed.py --dry-run       # parse + count only, no DB needed
    python scripts/seed.py --symbol BTC/USDT --timeframe 1h
    python scripts/seed.py --data-dir data/sample

The bundled CSVs were captured keylessly from the public exchange and are checked
into the repo, so a fresh clone has data immediately. Loading is idempotent
(``session.merge`` on the (symbol, timeframe, timestamp) primary key), so re-running
never duplicates rows.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from decimal import Decimal
from pathlib import Path

# Allow `python scripts/seed.py` from the repo root (mirrors the other scripts).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.logging import configure_logging, get_logger
from core.models.market import OHLCVCandle

logger = get_logger("seed")

# Repo-root-relative default location of the bundled dataset.
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def _parse_filename(path: Path) -> tuple[str, str]:
    """
    Recover (symbol, timeframe) from a file named like ``BTC-USDT_1h.csv``.
    The '-' in the symbol is converted back to '/'.
    """
    stem = path.stem  # e.g. "BTC-USDT_1h"
    sym_part, _, tf = stem.rpartition("_")
    symbol = sym_part.replace("-", "/")
    return symbol, tf


def load_csv_candles(path: Path) -> list[OHLCVCandle]:
    """Parse one bundled CSV into a list of OHLCVCandle (no DB, no network)."""
    symbol, timeframe = _parse_filename(path)
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


def discover_files(
    data_dir: Path,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> list[Path]:
    """Return the sample CSVs under ``data_dir`` matching optional filters."""
    files = sorted(data_dir.glob("*.csv"))
    selected: list[Path] = []
    for f in files:
        sym, tf = _parse_filename(f)
        if symbol and sym != symbol:
            continue
        if timeframe and tf != timeframe:
            continue
        selected.append(f)
    return selected


async def _seed_file(path: Path) -> int:
    """Bulk-load one CSV into the DB in a single transaction. Returns row count."""
    from core.db.connection import get_session
    from core.db.models import OHLCVCandleRow

    candles = load_csv_candles(path)
    async with get_session() as session:
        for c in candles:
            await session.merge(
                OHLCVCandleRow(
                    symbol=c.symbol,
                    timeframe=c.timeframe,
                    timestamp=c.timestamp,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                    quote_volume=c.quote_volume,
                    received_at=c.received_at,
                )
            )
        await session.commit()
    return len(candles)


async def main_async(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("data_dir_missing", path=str(data_dir))
        return 1

    files = discover_files(data_dir, args.symbol, args.timeframe)
    if not files:
        logger.error(
            "no_matching_files",
            data_dir=str(data_dir),
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
        return 1

    total = 0
    for f in files:
        sym, tf = _parse_filename(f)
        candles = load_csv_candles(f)
        if args.dry_run:
            print(
                f"  {sym:10} {tf:4}  {len(candles):>5} candles  "
                f"{candles[0].timestamp.date()} -> {candles[-1].timestamp.date()}  (dry-run)"
            )
            total += len(candles)
            continue
        n = await _seed_file(f)
        total += n
        print(f"  {sym:10} {tf:4}  {n:>5} candles seeded")
        logger.info("seeded_file", file=f.name, symbol=sym, timeframe=tf, candles=n)

    if not args.dry_run:
        # Close the pooled engine cleanly so the script exits without warnings.
        from core.db.connection import dispose_engine

        await dispose_engine()

    print(
        f"\nDone: {total} candles across {len(files)} file(s)"
        f"{' (dry-run, nothing written)' if args.dry_run else ''}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed the DB from the bundled offline sample dataset."
    )
    parser.add_argument(
        "--data-dir",
        default=str(_DEFAULT_DATA_DIR),
        help=f"directory of sample CSVs (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--symbol", default=None, help="only seed this symbol, e.g. BTC/USDT")
    parser.add_argument("--timeframe", default=None, help="only seed this timeframe, e.g. 1h")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and count only; no database connection required",
    )
    args = parser.parse_args()

    configure_logging()
    print(f"Seeding from {args.data_dir}{' (dry-run)' if args.dry_run else ''}:")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
