"""
agents/technical_analysis/ml/label_generator.py
-------------------------------------------------
Create training labels from a forward-return horizon.

Label semantics:
    +1 (BUY)   — forward return strictly above +threshold.
     0 (HOLD)  — forward return inside (-threshold, +threshold).
    -1 (SELL)  — forward return strictly below -threshold.

The last ``horizon`` bars in the input series have no future to look at,
so their labels are NaN. The trainer is expected to drop those rows when
aligning features with labels.

Design decisions:
- A 3-class formulation (rather than regression) lets the model focus on
  directional decisions, which is what the downstream Decision agent
  consumes — it never sees expected-return magnitudes.
- The threshold is symmetric. Crypto is roughly directionally symmetric
  intraday; asymmetric thresholds would bake in a long bias that should
  be a strategy choice, not a default.
- ``label_distribution`` is exposed so the trainer can warn on severely
  imbalanced classes — a model that predicts all-HOLD is useless.
"""

from __future__ import annotations

import numpy as np


class LabelGenerator:
    """
    Turn a 1-D close-price array into a {-1, 0, +1} label array.

    Usage:
        gen = LabelGenerator(horizon=12, threshold=0.003)
        labels = gen.generate(closes)               # NaN for last `horizon` bars
        dist = gen.label_distribution(labels)
    """

    def __init__(self, horizon: int = 12, threshold: float = 0.003) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self.horizon = horizon
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Label construction
    # ------------------------------------------------------------------

    def generate(self, closes: np.ndarray) -> np.ndarray:
        """
        Produce a float array of labels (so we can store NaN for the tail).

        Returns:
            np.ndarray of shape (len(closes),), dtype float64.
            Values are {-1.0, 0.0, +1.0} for the first len-horizon bars,
            and NaN for the trailing ``horizon`` bars.
        """
        n = len(closes)
        labels = np.full(n, np.nan, dtype=np.float64)
        if n <= self.horizon:
            return labels

        closes_f = closes.astype(np.float64)
        future = closes_f[self.horizon :]
        present = closes_f[: n - self.horizon]
        # Guard against zero prices in case of bad data.
        with np.errstate(divide="ignore", invalid="ignore"):
            forward_returns = np.where(present > 0, (future - present) / present, 0.0)

        out = np.zeros_like(forward_returns)
        out[forward_returns > self.threshold] = 1.0
        out[forward_returns < -self.threshold] = -1.0
        labels[: n - self.horizon] = out
        return labels

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def label_distribution(labels: np.ndarray) -> dict:
        """
        Count per-class occurrences. NaN entries are ignored.

        Returns:
            dict with keys: "buy", "hold", "sell", "total".
        """
        valid = labels[~np.isnan(labels)]
        buy = int(np.sum(valid == 1.0))
        hold = int(np.sum(valid == 0.0))
        sell = int(np.sum(valid == -1.0))
        return {
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "total": int(valid.size),
        }
