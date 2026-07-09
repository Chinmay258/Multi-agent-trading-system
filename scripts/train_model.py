"""
scripts/train_model.py
-----------------------
CLI: train an XGBoost ML signal model from historical OHLCV in the DB.

Usage:
    python scripts/train_model.py \\
        --symbol BTC/USDT \\
        --timeframe 1m \\
        --days 365 \\
        --horizon 12 \\
        --threshold 0.003
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Make the project root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.technical_analysis.ml.model_registry import ModelRegistry
from agents.technical_analysis.ml.model_trainer import CLASS_NAMES, ModelTrainer
from core.logging import configure_logging, get_logger

logger = get_logger("train_model_cli")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost ML signal model from historical OHLCV."
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="1m", help="Candle timeframe (default: 1m)")
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Look-back window in days (default: 365)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forward-return horizon in bars (default: 12)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.003,
        help="Label threshold (default: 0.003 = +/-0.3%%)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _print_distribution(dist: dict, total: int) -> None:
    print("\nLabel distribution:")
    for key in ("sell", "hold", "buy"):
        count = dist.get(key, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {key.upper():<4}  {count:>8,}  ({pct:5.1f}%)")
    print(f"  TOTAL {dist.get('total', total):>8,}")

    # Warn if any class is < 15% of total — model will likely degenerate.
    if total > 0:
        for key in ("sell", "hold", "buy"):
            pct = (dist.get(key, 0) / total) * 100
            if pct < 15.0:
                print(
                    f"  !! WARNING: class {key.upper()} is only {pct:.1f}% of samples. "
                    "Consider widening the date range or lowering --threshold."
                )


def _print_metrics(metrics: dict) -> None:
    print("\nMetrics on held-out test set (last 20%):")
    accuracy = metrics.get("accuracy", 0.0)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  {'Class':<6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10}")
    per = metrics.get("per_class", {})
    for name in CLASS_NAMES:
        m = per.get(name, {})
        prec = m.get("precision", 0.0)
        rec = m.get("recall", 0.0)
        f1 = m.get("f1", 0.0)
        print(f"  {name:<6} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")


def _print_feature_importance(metrics: dict) -> None:
    feats = metrics.get("top_features") or []
    if not feats:
        return
    print("\nTop features by gain:")
    print(f"  {'#':<3} {'feature':<24} {'gain':>12}")
    print(f"  {'-' * 3} {'-' * 24} {'-' * 12}")
    for i, item in enumerate(feats, start=1):
        name = item.get("feature", "?")
        gain = float(item.get("gain", 0.0))
        print(f"  {i:<3} {name:<24} {gain:>12.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _main() -> int:
    args = _parse_args()
    configure_logging()

    now = datetime.now(UTC)
    since = now - timedelta(days=args.days)

    print(
        f"Training ML model for {args.symbol} {args.timeframe}  "
        f"{since.date()} -> {now.date()}  "
        f"horizon={args.horizon}bars  threshold={args.threshold}"
    )

    trainer = ModelTrainer(
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizon=args.horizon,
        threshold=args.threshold,
    )

    X, y = await trainer.load_data_from_db(since=since, until=now)
    if X.shape[0] == 0:
        print("\nERROR: No candles available in DB for the requested window.")
        print("       Run `python scripts/seed_historical.py` to populate data.")
        return 1

    if X.shape[0] < 500:
        print(
            f"\nWARNING: Only {X.shape[0]} bars available. "
            "Recommend 10,000+ for reliable model.\n"
            "         Training anyway for testing purposes."
        )

    dist = trainer.label_gen.label_distribution(y)
    _print_distribution(dist, total=int(X.shape[0]))

    model, metrics = trainer.train(X, y)
    _print_metrics(metrics)
    _print_feature_importance(metrics)

    registry = ModelRegistry()
    saved_path = registry.save_model(
        model=model,
        metrics=metrics,
        feature_names=trainer.engineer.feature_names(),
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    print(f"\nModel saved to: {saved_path}")
    print("To activate: docker-compose restart technical_analysis_agent")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
