"""
backtest/walkforward.py
-----------------------
Walk-forward ML strategy — the honest way to backtest the ML signal path.

The deployed model can't be backtested in-sample (it would peek at its own training
data). Instead we retrain repeatedly on an **expanding window of past bars only** and
predict forward:

- A model retrained at bar ``t`` is trained on samples whose label window has fully
  closed by ``t`` (``b + horizon <= t``) — so no future return ever leaks into training.
- That model predicts bars ``[t, t+retrain_every)``; orders still fill at the *next*
  bar's open via the shared engine, so there is no same-bar lookahead either.

The result is a single out-of-sample equity curve directly comparable to the Phase-4
rule-based baseline, buy & hold, and random entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.technical_analysis.ml.feature_engineer import FeatureEngineer
from agents.technical_analysis.ml.label_generator import LabelGenerator
from backtest.config import BacktestConfig
from backtest.data import to_arrays
from backtest.engine import BacktestRun, run_with_decider
from core.models.market import OHLCVCandle

# SELL=0, HOLD=1, BUY=2 (mirrors model_trainer).
_LABEL_TO_CLASS = {-1.0: 0, 0.0: 1, 1.0: 2}


@dataclass(frozen=True)
class WalkForwardParams:
    """Knobs for the walk-forward ML strategy."""

    horizon: int = 10  # forward-return horizon for labels (bars)
    threshold: float = 0.01  # label threshold (+/- forward return)
    initial_train: int = 200  # min bars before the first model is trained
    retrain_every: int = 30  # retrain cadence (bars)
    min_train_samples: int = 120  # don't train on fewer than this
    directional_threshold: float = 0.40  # min class prob to emit BUY/SELL (else HOLD)
    calibrate: bool = False  # isotonic-calibrate probabilities on a holdout


def _xgb_params() -> dict[str, Any]:
    return dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )


class _WalkForwardModel:
    """Holds the current model + retrain bookkeeping, closing over full history."""

    def __init__(self, candles: list[OHLCVCandle], wf: WalkForwardParams) -> None:
        self.wf = wf
        eng = FeatureEngineer()
        self.start = eng.MIN_BARS_REQUIRED - 1
        arr = to_arrays(candles)
        self.closes = arr["closes"]
        self.n = len(self.closes)
        self.X_full, self.names = eng.build_feature_matrix(
            closes=arr["closes"],
            highs=arr["highs"],
            lows=arr["lows"],
            volumes=arr["volumes"],
            timestamps=arr["timestamps"],
            opens=arr["opens"],
        )
        self.y_full = LabelGenerator(horizon=wf.horizon, threshold=wf.threshold).generate(
            arr["closes"]
        )
        self._model: Any = None
        self._last_train = -(10**9)
        self.retrains = 0

    def _row_for_bar(self, bar: int) -> int | None:
        r = bar - self.start
        if 0 <= r < self.X_full.shape[0]:
            return r
        return None

    def _maybe_retrain(self, i: int) -> None:
        if self._model is not None and (i - self._last_train) < self.wf.retrain_every:
            return
        # Training cutoff: labels must be fully realised by bar i (no leakage).
        b_max = i - self.wf.horizon
        last_row = b_max - self.start
        if last_row < 0:
            return
        X = self.X_full[: last_row + 1]
        y = self.y_full[self.start : b_max + 1]
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        if X.shape[0] < self.wf.min_train_samples:
            return
        y_cls = np.vectorize(_LABEL_TO_CLASS.get)(y).astype(np.int64)
        if len(np.unique(y_cls)) < 2:
            return

        import xgboost as xgb

        counts = np.bincount(y_cls, minlength=3).astype(np.float64)
        inv = np.where(counts > 0, counts.sum() / (3 * counts), 0.0)
        sample_weight = inv[y_cls]

        model: Any = xgb.XGBClassifier(**_xgb_params())
        model.fit(X, y_cls, sample_weight=sample_weight, verbose=False)

        if self.wf.calibrate and X.shape[0] >= 200:
            try:
                from sklearn.calibration import CalibratedClassifierCV

                # Calibrate on the most recent 20% (still strictly past data).
                cut = int(X.shape[0] * 0.8)
                base = xgb.XGBClassifier(**_xgb_params())
                base.fit(X[:cut], y_cls[:cut], sample_weight=sample_weight[:cut], verbose=False)
                cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
                cal.fit(X[cut:], y_cls[cut:])
                model = cal
            except Exception:
                pass

        self._model = model
        self._last_train = i
        self.retrains += 1

    def decide(self, _buf: Any, i: int) -> str | None:
        if i < self.wf.initial_train:
            return None
        self._maybe_retrain(i)
        if self._model is None:
            return None
        r = self._row_for_bar(i)
        if r is None:
            return None
        proba = self._model.predict_proba(self.X_full[r : r + 1])[0]
        cls = int(np.argmax(proba))
        if proba[cls] < self.wf.directional_threshold:
            return None
        if cls == 2:
            return "buy"
        if cls == 0:
            return "sell"
        return None  # HOLD


def run_walkforward_ml(
    candles: list[OHLCVCandle],
    config: BacktestConfig,
    wf: WalkForwardParams | None = None,
) -> tuple[BacktestRun, dict]:
    """Run the walk-forward ML strategy. Returns (run, diagnostics)."""
    wf = wf or WalkForwardParams()
    wfm = _WalkForwardModel(candles, wf)
    run = run_with_decider(candles, config, wfm.decide)
    diag = {
        "horizon": wf.horizon,
        "threshold": wf.threshold,
        "retrain_every": wf.retrain_every,
        "initial_train": wf.initial_train,
        "directional_threshold": wf.directional_threshold,
        "calibrated": wf.calibrate,
        "retrains": wfm.retrains,
    }
    return run, diag
