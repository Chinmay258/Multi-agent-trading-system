# Evaluation

This project is evaluated **honestly**. The guiding rule: *if a number looks too good,
assume a bug (lookahead / fees / survivorship) until proven otherwise. A credible mediocre
result beats a fake good one.*

Regenerate everything with `make eval` →
`backtest/results/{baseline_metrics.json, improved_metrics.json, EVALUATION_REPORT.html, .pdf}`.

## Methodology

- **No lookahead.** A signal is computed at bar *i* (reading close[i]); the order fills at
  bar **i+1's open** — never the same bar's close. SL/TP brackets are checked intrabar
  against each later bar's high/low.
- **Realistic costs.** 0.1% taker fee + 0.05% slippage **per side** — matched exactly to the
  live `PaperBroker`.
- **Benchmarks.** Every strategy result is compared against **buy-and-hold** and a
  **random-entry** baseline (200 seeds, trade-frequency matched).
- **Walk-forward for ML.** The ML strategy retrains on an expanding window of **past bars
  only** (a training sample's label window must fully close before the prediction bar), so no
  future return ever leaks into training.
- Metrics: total return, CAGR, monthly returns, Sharpe, Sortino, Calmar, max drawdown +
  duration, win rate, profit factor, expectancy, avg win/loss, # trades, exposure.

The harness lives in [`backtest/`](../backtest) and is unit-tested — including an explicit
**no-lookahead** test (`tests/unit/test_backtest.py`).

## Headline results (BTC/USDT, 1d, ~2 years, out-of-sample)

| | Total return | Sharpe | Win rate | # trades |
|---|---|---|---|---|
| **Rule-based strategy (default)** | **+0.21%** | **+0.26** | 32% | 34 |
| Buy & hold | −13.0% | +0.07 | — | — |
| Random entry (mean of 200) | −0.21% | −0.32 | 29% | — |
| Walk-forward ML (Phase 5 attempt) | −0.24% | −0.18 | 38% | 137 |

**The honest finding: no demonstrable edge.** The strategy's returns are tiny and
statistically **indistinguishable from random entry**. It "beats" buy-and-hold only by
sitting out most of a falling market (low exposure ≠ predictive skill), and the conservative
2% position sizing caps both risk and reward — so judge it on Sharpe / vs-random, not dollars.

## ML classifier (out-of-sample)

On a 3-class problem where random scores **0.333**, the model lands at **~0.42** accuracy,
and the **HOLD class is essentially unlearned** (0 precision). Directional precision is close
to a coin flip. The [report](../backtest/results/EVALUATION_REPORT.html) includes the
confusion matrix, per-class precision/recall/F1, class balance, calibration curve, and
feature importance.

## Phase 5 — did retraining help? No, and we say so.

We tried to improve the system by enabling the ML signal path with proper walk-forward
retraining + isotonic calibration. It **did not beat the rule baseline** out-of-sample
(Δ return −0.45 pp, Δ Sharpe −0.44) and overtrades badly (137 trades vs 34, bleeding on
fees). So we **kept the simpler rule-based default**. A documented negative result is a
legitimate, valuable outcome. Full write-up: [MODEL_CHANGES.md](MODEL_CHANGES.md).

## Limitations

- Single asset, single timeframe, modest bundled history — results are indicative, not a
  multi-year, multi-regime validation.
- Paper mode has no native SL/TP, so the backtest applies an explicit, documented SL/TP
  bracket; live paper behaviour can differ.
- No survivorship/regime adjustments; flat per-side cost model (real thin-book fills would be
  worse). See the "Limitations" section embedded in the generated report.
