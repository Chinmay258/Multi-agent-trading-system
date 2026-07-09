"""
backtest/run.py
---------------
Orchestrates a full evaluation run and regenerates all artifacts:

  backtest/results/baseline_metrics.json
  backtest/results/EVALUATION_REPORT.html
  backtest/results/EVALUATION_REPORT.pdf

Run via ``make eval`` (``python -m backtest.run``). Keyless and offline by default
(uses the bundled ``data/sample/`` dataset).

    python -m backtest.run --symbol BTC/USDT --timeframe 1d
    python -m backtest.run --source cache   # use freshly fetched/cached data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow `python backtest/run.py` as well as `python -m backtest.run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("backtest_run")

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _serialise(payload: dict) -> dict:
    """Strip non-JSON objects (curves/trades/figures) for the metrics JSON file."""
    cfg = payload["config"]
    out = {
        "config": cfg,
        "strategy": payload["strategy"]["metrics"],
        "buy_and_hold": payload["buy_hold"]["metrics"],
        "random_entry": payload.get("random") or {},
        "ml_evaluation": payload.get("ml") or {},
    }
    imp = payload.get("improved")
    if imp:
        out["improved_strategy"] = {
            "metrics": imp["metrics"],
            "walkforward_params": imp["walkforward_params"],
            "vs_baseline": imp["vs_baseline"],
            "beats_baseline": imp["beats_baseline"],
            "kept": imp["kept"],
        }
    return out


async def _maybe_fetch(symbol: str, timeframe: str, source: str, limit: int) -> None:
    if source == "cache":
        from backtest.data import fetch_and_cache

        logger.info("backtest_fetching", symbol=symbol, timeframe=timeframe, limit=limit)
        await fetch_and_cache(symbol, timeframe, limit=limit)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation harness (keyless, offline).")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1d", help="default 1d (longest bundled span)")
    parser.add_argument("--source", choices=["bundled", "cache"], default="bundled")
    parser.add_argument(
        "--fetch-limit", type=int, default=1000, help="candles to fetch when --source cache"
    )
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--no-report", action="store_true", help="write JSON only, skip HTML/PDF")
    parser.add_argument("--ml-horizon", type=int, default=10)
    parser.add_argument("--ml-threshold", type=float, default=0.01)
    parser.add_argument(
        "--no-improved",
        action="store_true",
        help="skip the Phase-5 walk-forward ML strategy (baseline only)",
    )
    parser.add_argument(
        "--wf-no-calibrate", action="store_true", help="disable isotonic calibration"
    )
    args = parser.parse_args()

    configure_logging()

    # Make the live SignalGenerator use the same confidence threshold as the backtest.
    os.environ["TA_MIN_SIGNAL_CONFIDENCE"] = str(args.min_confidence)
    from core.config import get_settings

    get_settings.cache_clear()

    from backtest.benchmarks import buy_and_hold, random_entry
    from backtest.config import BacktestConfig
    from backtest.data import load_candles
    from backtest.engine import run_strategy
    from backtest.metrics import compute_metrics
    from backtest.ml_eval import evaluate_ml

    if args.source == "cache":
        asyncio.run(_maybe_fetch(args.symbol, args.timeframe, args.source, args.fetch_limit))

    cfg = BacktestConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        min_confidence=args.min_confidence,
    )

    print(f"Loading {cfg.symbol} {cfg.timeframe} ({args.source}) ...")
    candles = load_candles(cfg.symbol, cfg.timeframe, source=args.source)
    if len(candles) < 80:
        print(f"ERROR: only {len(candles)} candles — too few to evaluate.")
        return 1

    print(f"Replaying {len(candles)} candles through the rule-based pipeline (no lookahead) ...")
    run = run_strategy(candles, cfg)
    strat_metrics = compute_metrics(
        run.curve, run.trades, cfg.periods_per_year, cfg.initial_balance
    )

    bh_curve, bh_metrics = buy_and_hold(candles, cfg)
    rnd = (
        random_entry(candles, cfg, target_trades=strat_metrics["num_trades"])
        if strat_metrics["num_trades"]
        else {}
    )

    print("Evaluating the ML classifier out-of-sample ...")
    ml = evaluate_ml(
        cfg.symbol, cfg.timeframe, horizon=args.ml_horizon, threshold=args.ml_threshold
    )

    # Phase 5: the walk-forward ML strategy (retrain on past only, predict forward).
    improved = None
    if not args.no_improved:
        from backtest.walkforward import WalkForwardParams, run_walkforward_ml

        print("Running walk-forward ML strategy (retraining on past data only) ...")
        wf = WalkForwardParams(
            horizon=args.ml_horizon,
            threshold=args.ml_threshold,
            calibrate=not args.wf_no_calibrate,
        )
        wf_run, wf_diag = run_walkforward_ml(candles, cfg, wf)
        imp_metrics = compute_metrics(
            wf_run.curve, wf_run.trades, cfg.periods_per_year, cfg.initial_balance
        )
        # Honest verdict: did the ML strategy beat the rule baseline out-of-sample?
        beat = (
            imp_metrics["total_return_pct"] > strat_metrics["total_return_pct"]
            and imp_metrics["sharpe"] > strat_metrics["sharpe"]
        )
        improved = {
            "metrics": imp_metrics,
            "curve": wf_run.curve,
            "trades": wf_run.trades,
            "walkforward_params": wf_diag,
            "vs_baseline": {
                "return_delta_pct": round(
                    imp_metrics["total_return_pct"] - strat_metrics["total_return_pct"], 4
                ),
                "sharpe_delta": round(imp_metrics["sharpe"] - strat_metrics["sharpe"], 4),
            },
            "beats_baseline": beat,
            "kept": beat,  # we only adopt changes that hold up out-of-sample
        }

    payload = {
        "config": {
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "since": str(candles[0].timestamp.date()),
            "until": str(candles[-1].timestamp.date()),
            "initial_balance": cfg.initial_balance,
            "fee_pct": cfg.fee_pct,
            "slippage_pct": cfg.slippage_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "take_profit_pct": cfg.take_profit_pct,
            "max_position_pct": cfg.max_position_pct,
            "min_confidence": cfg.min_confidence,
            "signal_source": cfg.signal_source,
        },
        "strategy": {"metrics": strat_metrics, "curve": run.curve, "trades": run.trades},
        "buy_hold": {"metrics": bh_metrics, "curve": bh_curve},
        "random": rnd,
        "ml": ml,
        "improved": improved,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "baseline_metrics.json"
    json_path.write_text(json.dumps(_serialise(payload), indent=2), encoding="utf-8")
    print(f"\nWrote {json_path}")

    if improved is not None:
        improved_out = {
            "config": payload["config"],
            "walkforward_params": improved["walkforward_params"],
            "metrics": improved["metrics"],
            "baseline_metrics": strat_metrics,
            "vs_baseline": improved["vs_baseline"],
            "beats_baseline": improved["beats_baseline"],
            "kept": improved["kept"],
        }
        imp_path = _RESULTS_DIR / "improved_metrics.json"
        imp_path.write_text(json.dumps(improved_out, indent=2), encoding="utf-8")
        print(f"Wrote {imp_path}")

    if not args.no_report:
        from backtest.report import generate_report

        paths = generate_report(_RESULTS_DIR, payload)
        for kind, p in paths.items():
            print(f"Wrote {p}  ({kind})")

    # Honest verdict to stdout.
    print(
        f"\nVerdict: baseline(rules) {strat_metrics['total_return_pct']}%  vs  "
        f"buy&hold {bh_metrics['total_return_pct']}%  vs  random "
        f"{rnd.get('mean_total_return_pct', 'n/a')}%  |  Sharpe {strat_metrics['sharpe']}  |  "
        f"ML acc {ml.get('accuracy', 'n/a')} (random {ml.get('random_baseline_accuracy', '0.33')})"
    )
    if improved is not None:
        verdict = (
            "BEATS baseline - adopt"
            if improved["beats_baseline"]
            else "did NOT beat baseline - keep rules"
        )
        print(
            f"Phase 5: walk-forward ML {improved['metrics']['total_return_pct']}% "
            f"(Sharpe {improved['metrics']['sharpe']}, {improved['metrics']['num_trades']} trades)  "
            f"vs baseline {strat_metrics['total_return_pct']}% (Sharpe {strat_metrics['sharpe']}) "
            f"-> {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
