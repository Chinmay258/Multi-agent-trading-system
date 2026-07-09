# Model Changes — Phase 5 (Retrain & Improve)

**TL;DR — an honest negative result.** We tried to make the system better by enabling the
ML signal path with a rigorous **walk-forward** methodology. Evaluated out-of-sample, the
ML strategy **did not beat the rule-based baseline** — it overtrades and bleeds on fees
without a real directional edge. We therefore **keep the rule-based baseline** and ship it
as the default (`TA_USE_ML_SIGNALS=false`). A documented "we tried X and it didn't beat
baseline" is a legitimate, valuable outcome — and far more useful than a cherry-picked
backtest.

All numbers below are reproducible: `make eval` regenerates
`backtest/results/{baseline_metrics.json, improved_metrics.json, EVALUATION_REPORT.html}`.

---

## 1. Starting point (Phase 4 baseline)

The Phase-4 baseline is the system's **rule-based** pipeline (RSI/MACD/Bollinger/EMA →
weighted score), backtested with **no lookahead** (signal at bar *i*, fill at bar *i+1*'s
open), realistic fees (0.1%/side) + slippage (0.05%/side), SL/TP brackets, and 2% sizing.

It has **no demonstrable edge**: returns are tiny and statistically indistinguishable from
random entry. The deployed XGBoost classifier scores ~0.42–0.49 accuracy on a 3-class
problem where random is 0.33, and the HOLD class is essentially unlearned.

## 2. What we tried

1. **Walk-forward ML strategy** (`backtest/walkforward.py`). The deployed model can't be
   backtested in-sample (it would peek at its own training data), so we retrain repeatedly
   on an **expanding window of past bars only**:
   - A model retrained at bar `t` trains only on samples whose label window has fully
     closed by `t` (`b + horizon <= t`) — **no future return ever leaks into training**.
   - That model predicts bars `[t, t+retrain_every)`; orders still fill at the next bar's
     open. Retrain cadence: every 30 bars; initial train window: 200 bars.
2. **Probability calibration** — isotonic calibration (`CalibratedClassifierCV`, `cv="prefit"`)
   on a held-out, strictly-past slice of each training window.
3. **Directional-threshold tuning** — only emit BUY/SELL when the top class probability
   clears a threshold (tried 0.40 / 0.45), else treat as HOLD (no trade).
4. **Class weighting** — inverse-frequency sample weights to counter the rare-HOLD imbalance.

Methodology guards: chronological splits only (no shuffle), label horizon respected on
every retrain (no leakage), no parameter was tuned on the final test window to flatter
results.

## 3. Before vs. after (BTC/USDT, 1d, 2024-07 → 2026-06)

| Variant | Total return | Sharpe | Win rate | # trades | Max DD |
|---|---|---|---|---|---|
| **Baseline — rules (kept)** | **+0.21%** | **+0.26** | 32% | 34 | 0.7% |
| Walk-forward ML, thr 0.40 | −0.94% | −0.72 | 41% | 166 | 1.1% |
| Walk-forward ML, thr 0.45 | −1.11% | −0.86 | 41% | 158 | 1.2% |
| Walk-forward ML, thr 0.40, **calibrated** | −0.24% | −0.18 | 38% | 137 | 1.5% |

Sanity benchmarks over the same window: **buy & hold −13.0%** (Sharpe +0.07),
**random entry −0.21%** (mean of 200 seeds).

Cross-check on **4h** (1000 bars): rules −0.14% (Sharpe −0.49) vs. calibrated walk-forward
ML −0.08% (Sharpe −0.24) — effectively tied, both break-even-ish, neither with an edge.

The classifier's own out-of-sample metrics (chronological 80/20): **accuracy 0.42 vs 0.33
random**, with **HOLD precision/recall = 0** — it predicts a direction on nearly every bar.

## 4. Why the ML path loses

- **Overtrading.** The model rarely predicts HOLD, so the strategy is in the market far
  more often (137–166 trades vs. 34) and pays ~0.3% round-trip cost each time. A near-coin-flip
  signal traded frequently is a slow bleed.
- **No directional edge.** ~0.42 accuracy with a broken HOLD class means BUY/SELL precision
  is close to random; calibration reduces the bleeding (fewer confident trades) but can't
  manufacture skill.
- **Poor calibration of the raw model.** High-confidence predictions were not more accurate
  than low-confidence ones (see the calibration chart in the report) — isotonic calibration
  helped but didn't change the verdict.

## 5. Decision

- **Keep the rule-based baseline.** Default flipped to `TA_USE_ML_SIGNALS=false`
  (`.env.example`, `.env`, and the `core/config.py` default), with a comment pointing here.
  The ML code paths and trained models remain in the repo — opt back in with
  `TA_USE_ML_SIGNALS=true` to experiment.
- The walk-forward harness (`backtest/walkforward.py`) stays as the *correct* way to evaluate
  any future ML strategy, so this result is easy to revisit.

## 6. What would actually be needed to do better (future work)

- **More and longer data**, across multiple regimes (the bundled sample is modest).
- **Better labels** — fixed-horizon ±threshold labelling creates a near-useless HOLD class;
  triple-barrier or volatility-scaled labels would likely help class balance.
- **Trade-frequency control** — a regime/volatility filter and a meaningfully higher
  confidence gate to stop overtrading.
- **Richer features / stronger model** only *after* the labelling and cost problems are
  fixed — otherwise it's polishing noise.

> Honesty note: none of the above is claimed to work here. They are hypotheses for future
> work. As of Phase 5, the rule-based baseline is the system's default because it is the
> simplest thing that wasn't beaten out-of-sample.
