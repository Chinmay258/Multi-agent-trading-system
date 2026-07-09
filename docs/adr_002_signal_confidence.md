# ADR-002 — Composite signal confidence scoring

- **Status:** Accepted
- **Date:** 2026-05-21
- **Owners:** Technical Analysis + Decision agents

> ADR-001 used a `.py` module docstring because it documents an imported
> constant. ADR-002 is markdown because no code needs to read it — the
> formula lives in `agents/technical_analysis/signal_generator.py`.

---

## Context

The Technical Analysis agent computes several independent indicators per
candle (RSI, MACD, Bollinger Bands, EMA cross) and must turn them into a
single actionable signal for the Decision agent. We need:

1. **A single confidence number** in `[0, 1]` so the Decision agent can
   gate proposals with a threshold.
2. **A formula that rewards agreement** — five indicators all saying
   "BUY" should be more credible than three saying "BUY" and two saying
   "SELL".
3. **A formula that rewards conviction** — a marginal score barely past
   neutral should be less confident than a score deep into the BUY zone.
4. **A volume sanity check** so a signal that fires on disappearing
   volume is downgraded.

Without explicit reasoning about confidence, the natural choice ("just
average the indicator weights") collapses both axes and produces
overconfident signals on noisy or thin markets — exactly the conditions
we most want to avoid.

## Decision

The confidence score is computed in
[agents/technical_analysis/signal_generator.py](../agents/technical_analysis/signal_generator.py)
as:

```
agreement       = 1 - std_dev(indicator_scalars)            # [0, 1]
score_magnitude = |composite_score|                          # [0, 1]
base_confidence = 0.6 × agreement + 0.4 × score_magnitude    # [0, 1]

confidence = base_confidence
if volume_ratio ≥ 1.5:   confidence = min(1.0, confidence + 0.10)
if volume_ratio ≤ 0.6:   confidence = max(0.0, confidence - 0.10)
```

The Decision agent's `SignalAggregator` then drops any signal with
`confidence < min_signal_confidence` (default **0.6**).

### Why 60/40, not 50/50

Agreement weighs more than magnitude because **disagreement among
indicators is a stronger signal that we don't understand what's
happening** than weak magnitude is a sign that nothing's happening.
A 50/50 split was tried in early prototypes and produced more
false-positive signals on choppy days.

### Why ±0.10 for volume

A multiplicative volume term made the score unstable when indicators
near the threshold flipped sign. A flat ±0.10 additive adjustment is
calibrated so volume can lift a borderline signal into actionable range
(or knock a borderline signal out) without dominating the formula.

### Why threshold 0.6

This is the parameter we tuned hardest. We picked 0.6 because:

- **0.5 or lower** lets through too many low-conviction signals during
  oscillating markets. In offline replays, lowering the threshold
  doubled the trade rate but barely moved realised PnL.
- **0.7 or higher** suppresses signals during legitimately fast trends
  where one indicator is briefly noisy. The system stayed flat through
  rallies that other systems caught.
- **0.6** sits in the sweet spot where false positives drop sharply but
  the system still trades through normal breakout patterns.

The threshold lives in `TechnicalAnalysisSettings.min_signal_confidence`
so it can be tuned per environment without code changes.

## Consequences

### Positive

- Signals carry a single, interpretable confidence number that drives a
  single, interpretable threshold.
- The formula is auditable: every TechnicalSignal carries its
  `indicators[]` list, so for any rejected trade we can reconstruct
  agreement and magnitude after the fact.
- Volume gating prevents the most embarrassing class of false signals
  (indicator alignment on no actual market activity).

### Negative

- The 0.6 weight, 0.10 volume bonus, and 0.6 threshold are **calibrated
  to current indicator weights** (MACD 0.30, RSI 0.25, EMA 0.25, BB 0.20).
  Re-weighting indicators requires re-tuning these constants.
- The score is uncalibrated against realised PnL. A 0.8 confidence
  signal does **not** mean an 80% win rate; it means the indicators
  agreed strongly. The historical correlation between confidence and
  outcome is a Phase 7+ ML question (see below).

## Known limitations

### Degenerate test data

A constant-price series produces perfect agreement (every indicator
returns NEUTRAL → agreement = 1.0) but zero magnitude
(composite_score = 0.0). The formula collapses to `confidence = 0.6 × 1.0
+ 0.4 × 0.0 = 0.6`, which sits exactly at the threshold — and the
direction is NEUTRAL anyway, which `ProposalBuilder` discards. **This is
the intended behaviour**: a flat market should produce no trade, not a
loud HOLD signal.

This explains why
`tests/integration/test_e2e_paper_trading.py::test_candle_to_signal_pipeline`
uses a synthetic *trending* series rather than constant prices, and
contains a `pytest.skip` guard for the case where the synthetic series
still doesn't breach the threshold. The test is verifying *contract*
(if a signal is produced it has all required fields), not magnitude.

### Single timeframe

Confidence is computed per primary timeframe (default 1m). Multi-
timeframe confluence — where a 1m signal is boosted if the 5m and 1h
trends agree — is not yet implemented. Phase 5 backtests suggest this
would raise the threshold's discrimination power; it is a planned
extension, not a fix to the current formula.

### Indicator failure handling

If an indicator computation fails (e.g. NaN from TA-Lib on insufficient
data), it is silently dropped from the readings list. We require at
least two surviving indicators to produce a signal. This is correct but
makes confidence less comparable across calls — a 4-indicator signal
and a 2-indicator signal both produce confidence in `[0, 1]` even
though the 4-indicator one is more evidence.

## Future work

- **ML-calibrated confidence (Phase 7+).** Train a small classifier on
  historical features (indicator readings, volume regime, recent
  realised vol) to predict outcome probability. Use that probability as
  the published `confidence` field; the manual formula above becomes a
  fallback for when the model can't score.
- **Per-symbol thresholds.** Different markets have different signal
  hygiene; a uniform 0.6 cut-off is a simplification. Phase 7 will let
  the threshold come from a per-symbol config map.
- **Confidence calibration plot.** Add a back-office dashboard that
  bins signals by confidence decile and shows realised win-rate per
  bin. If the curve is flat the formula is broken; if it slopes
  upward, the threshold is doing its job.

## References

- [agents/technical_analysis/signal_generator.py](../agents/technical_analysis/signal_generator.py) — implementation
- [agents/decision/signal_aggregator.py](../agents/decision/signal_aggregator.py) — confidence gating
- [agents/technical_analysis/agent.py](../agents/technical_analysis/agent.py) — Prometheus `trading_signal_confidence` gauge
- [docs/adr_001_mt5_hybrid_execution.py](adr_001_mt5_hybrid_execution.py) — prior ADR (note: different format)
