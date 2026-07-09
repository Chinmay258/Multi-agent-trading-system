"""
scripts/evaluate_model.py
--------------------------
Compare the ML signal model against the rule-based generator on out-of-sample
historical data and report a simplistic simulated PnL.

Replay logic mirrors ``scripts/backtest.py`` but is deliberately lightweight:
no Decision/Risk pipeline — each generator emits a direction per bar, and we
account PnL as the next-bar close move. Useful as a smoke check, NOT as a
realistic strategy backtest.

Usage:
    python scripts/evaluate_model.py \\
        --symbol BTC/USDT \\
        --timeframe 1m \\
        --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Make the project root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.technical_analysis.candle_buffer import CandleBufferRegistry
from agents.technical_analysis.ml.ml_signal_generator import MLSignalGenerator
from agents.technical_analysis.ml.model_registry import ModelRegistry
from agents.technical_analysis.signal_generator import SignalGenerator
from core.db.connection import get_session
from core.db.repositories.candle_repo import CandleRepository
from core.logging import configure_logging
from core.models.signals import SignalDirection

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the ML model vs. the rule-based generator on recent data."
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model path. Defaults to the latest registry entry.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------


def _direction_label(direction: SignalDirection | str) -> str:
    d = direction.value if isinstance(direction, SignalDirection) else str(direction)
    if d.endswith("buy"):
        return "buy"
    if d.endswith("sell"):
        return "sell"
    return "neutral"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


async def _evaluate(symbol: str, timeframe: str, days: int, model_path: str | None) -> int:
    configure_logging()
    now = datetime.now(UTC)
    since = now - timedelta(days=days)

    print(f"\nLoading {symbol} {timeframe} candles  {since.date()} -> {now.date()} ...")
    async with get_session() as session:
        repo = CandleRepository(session)
        candles = await repo.get_candles(symbol, timeframe, since=since, until=now, limit=200_000)

    if not candles:
        print("ERROR: No candles found. Run `scripts/seed_historical.py` first.")
        return 1

    print(f"Loaded {len(candles)} candles. Replaying both generators...\n")

    # Resolve the model path.
    if model_path is None:
        latest = ModelRegistry().get_latest_path(symbol, timeframe)
        if latest is None:
            print(
                "ERROR: No trained model found. Train one first: `python scripts/train_model.py`."
            )
            return 1
        model_path = str(latest)

    ml_gen = MLSignalGenerator(model_path)
    if not ml_gen.is_loaded():
        print(f"ERROR: Failed to load model at {model_path}.")
        return 1
    rule_gen = SignalGenerator()

    ml_registry = CandleBufferRegistry()
    rule_registry = CandleBufferRegistry()

    ml_signals = {"buy": 0, "sell": 0, "neutral": 0, "total": 0}
    rule_signals = {"buy": 0, "sell": 0, "neutral": 0, "total": 0}
    ml_directions: list[tuple[int, str]] = []  # (bar_index, direction)
    rule_directions: list[tuple[int, str]] = []
    agreement = 0
    disagreement = 0
    closes = [float(c.close) for c in candles]

    for i, candle in enumerate(candles):
        ml_buf = ml_registry.add(candle)
        rule_buf = rule_registry.add(candle)
        if not ml_buf.is_warm:
            continue

        ml_sig = ml_gen.generate(ml_buf)
        rule_sig = rule_gen.generate(rule_buf)

        if ml_sig is not None:
            label = _direction_label(ml_sig.direction)
            ml_signals[label] += 1
            ml_signals["total"] += 1
            ml_directions.append((i, label))

        if rule_sig is not None:
            label = _direction_label(rule_sig.direction)
            rule_signals[label] += 1
            rule_signals["total"] += 1
            rule_directions.append((i, label))

        if ml_sig is not None and rule_sig is not None:
            if _direction_label(ml_sig.direction) == _direction_label(rule_sig.direction):
                agreement += 1
            else:
                disagreement += 1

    # ---- Simulated PnL (next-bar return). No transaction costs. ----
    def _pnl(directions: list[tuple[int, str]]) -> float:
        total = 0.0
        for bar_idx, label in directions:
            if bar_idx + 1 >= len(closes):
                continue
            ret = (closes[bar_idx + 1] - closes[bar_idx]) / closes[bar_idx]
            if label == "buy":
                total += ret
            elif label == "sell":
                total -= ret
        return total * 100  # percent

    ml_pnl = _pnl(ml_directions)
    rule_pnl = _pnl(rule_directions)
    buy_hold_pnl = (closes[-1] - closes[0]) / closes[0] * 100 if closes else 0.0

    ml_total = max(ml_signals["total"], 1)
    both_total = max(agreement + disagreement, 1)

    def _pct(n: int, total: int) -> str:
        return f"{n:>4}  ({(n / total * 100):4.0f}%)" if total > 0 else f"{n:>4}"

    print(f"=== ML Model Performance (last {days} days) ===")
    print(f"Signals generated:     {ml_signals['total']}")
    print(f"Buy signals:           {_pct(ml_signals['buy'], ml_total)}")
    print(f"Sell signals:          {_pct(ml_signals['sell'], ml_total)}")
    print(f"Neutral/suppressed:    {_pct(ml_signals['neutral'], ml_total)}")

    print("\n=== vs Rule-Based System ===")
    print(f"Rule signals generated: {rule_signals['total']}")
    print(
        f"Agreement rate:         {(agreement / both_total * 100):.1f}%  (both agree on direction)"
    )

    print("\n=== Simulated PnL (no costs) ===")
    print(f"ML model:    {ml_pnl:+.2f}%")
    print(f"Rule-based:  {rule_pnl:+.2f}%")
    print(f"Buy & hold:  {buy_hold_pnl:+.2f}%")
    return 0


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(
        asyncio.run(_evaluate(args.symbol, args.timeframe, args.days, args.model_path))
    )
