"""
agents/technical_analysis/ml/model_trainer.py
-----------------------------------------------
Train an XGBoost classifier on historical OHLCV data from the database.

Pipeline:

    DB candles
        → FeatureEngineer.build_feature_matrix   (X)
        → LabelGenerator.generate                 (y)
        → time-series split (no shuffle — temporal order matters)
        → XGBoost classifier with early stopping
        → metrics (accuracy, per-class P/R/F1, top-k feature importance)
        → save via ModelRegistry

Design decisions:
- TIME-SERIES split, never random shuffle. Random splitting on time-series
  data leaks the future into the training set and produces falsely
  optimistic accuracy. We take the first 80% chronologically as train,
  the last 20% as test.
- ``scale_pos_weight`` is computed from class counts to handle imbalance.
  Crypto data is typically dominated by HOLD; without weighting the model
  collapses to a constant prediction.
- ``early_stopping_rounds`` is supported across XGBoost versions: 1.6+
  takes it on the estimator, 2.x prefers it on ``.fit()``. We pass it in
  the constructor where supported and fall back to ``.fit()`` otherwise.
- Labels are remapped from {-1, 0, +1} → {0, 1, 2} for sklearn-style
  ``predict_proba``. The signal generator handles the reverse mapping.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from agents.technical_analysis.ml.feature_engineer import FeatureEngineer
from agents.technical_analysis.ml.label_generator import LabelGenerator
from core.db.connection import get_session
from core.db.repositories.candle_repo import CandleRepository
from core.logging import get_logger
from core.models.market import OHLCVCandle

logger = get_logger("model_trainer")


# Internal class encoding: SELL=0, HOLD=1, BUY=2.
# This matches scikit-learn's expectation that classes are non-negative
# consecutive ints. The MLSignalGenerator decodes back into directions.
LABEL_TO_CLASS = {-1.0: 0, 0.0: 1, 1.0: 2}
CLASS_NAMES = ["SELL", "HOLD", "BUY"]


class ModelTrainer:
    """Train and evaluate a per-symbol/timeframe XGBoost classifier."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        horizon: int = 12,
        threshold: float = 0.003,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.horizon = horizon
        self.threshold = threshold
        self.engineer = FeatureEngineer()
        self.label_gen = LabelGenerator(horizon=horizon, threshold=threshold)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def load_data_from_db(
        self,
        since: datetime,
        until: datetime,
        limit: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load candles from the DB, build aligned (X, y) arrays.

        Returns:
            X: float64 array of shape (n_samples, n_features).
            y: float64 array of shape (n_samples,) with values in {-1.0, 0.0, +1.0}.
        """
        async with get_session() as session:
            repo = CandleRepository(session)
            candles: list[OHLCVCandle] = await repo.get_candles(
                self.symbol,
                self.timeframe,
                since=since,
                until=until,
                limit=limit,
            )

        if not candles:
            logger.warning(
                "model_trainer_no_candles",
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=str(since),
                until=str(until),
            )
            return (
                np.zeros((0, len(self.engineer.feature_names())), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )

        opens = np.array([float(c.open) for c in candles], dtype=np.float64)
        highs = np.array([float(c.high) for c in candles], dtype=np.float64)
        lows = np.array([float(c.low) for c in candles], dtype=np.float64)
        closes = np.array([float(c.close) for c in candles], dtype=np.float64)
        volumes = np.array([float(c.volume) for c in candles], dtype=np.float64)
        timestamps = [c.timestamp for c in candles]

        X_all, _ = self.engineer.build_feature_matrix(
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            timestamps=timestamps,
            opens=opens,
        )
        y_all = self.label_gen.generate(closes)

        # X corresponds to bars [MIN_BARS_REQUIRED-1 .. n-1]; labels are
        # NaN for the trailing ``horizon`` bars. Align both.
        start = self.engineer.MIN_BARS_REQUIRED - 1
        n = len(closes)
        end = n - self.horizon  # exclusive — labels valid in [0, end)

        if end <= start:
            logger.warning(
                "model_trainer_insufficient_history",
                bars=n,
                required_min=start + self.horizon + 1,
            )
            return (
                np.zeros((0, len(self.engineer.feature_names())), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )

        # Number of feature rows already produced by build_feature_matrix:
        # the function yields one row per i in [start, n-1]. We trim from
        # the END to drop rows whose labels are NaN.
        keep_count = end - start
        X = X_all[:keep_count]
        y = y_all[start:end]

        dist = self.label_gen.label_distribution(y)
        logger.info(
            "model_trainer_data_loaded",
            symbol=self.symbol,
            timeframe=self.timeframe,
            candles=n,
            feature_rows=int(X.shape[0]),
            feature_cols=int(X.shape[1]),
            label_buy=dist["buy"],
            label_hold=dist["hold"],
            label_sell=dist["sell"],
        )
        return X.astype(np.float64), y.astype(np.float64)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> tuple[Any, dict]:
        """
        Fit the XGBoost classifier on (X, y) with a time-series split.

        Args:
            X: feature matrix (n_samples, n_features), float64.
            y: labels in {-1.0, 0.0, +1.0}, length n_samples.

        Returns:
            (model, metrics_dict)
        """
        if X.shape[0] < 20:
            raise ValueError(
                f"Insufficient training data: only {X.shape[0]} samples. "
                "Need at least ~20 bars to fit a model."
            )

        # Map labels {-1, 0, 1} -> {0, 1, 2} for the classifier.
        y_class = np.vectorize(LABEL_TO_CLASS.get)(y).astype(np.int64)

        # Time-series split (chronological, no shuffle).
        split = int(X.shape[0] * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_class[:split], y_class[split:]

        logger.info(
            "model_trainer_split",
            train_size=int(X_train.shape[0]),
            test_size=int(X_test.shape[0]),
        )

        # Class weight (handles imbalance).
        sample_weight = self._compute_sample_weights(y_train)

        # Lazy import keeps the module importable even when xgboost isn't
        # installed (e.g. CI runs only the rule-based path).
        import xgboost as xgb

        params: dict[str, Any] = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
        )
        try:
            params["use_label_encoder"] = False
        except Exception:  # pragma: no cover — defensive
            pass

        model = xgb.XGBClassifier(**params)

        # Fit. ``early_stopping_rounds`` is accepted on .fit() in xgboost<2,
        # and on the constructor in xgboost>=2. Handle both for portability.
        fit_kwargs: dict[str, Any] = dict(
            eval_set=[(X_test, y_test)] if X_test.shape[0] > 0 else None,
            sample_weight=sample_weight,
            verbose=False,
        )
        try:
            model.fit(X_train, y_train, early_stopping_rounds=50, **fit_kwargs)
        except TypeError:
            # xgboost >= 2: early_stopping_rounds is a constructor param.
            model.set_params(early_stopping_rounds=50)
            model.fit(X_train, y_train, **fit_kwargs)

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        metrics = self._evaluate(model, X_test, y_test) if X_test.shape[0] > 0 else {}
        metrics["train_size"] = int(X_train.shape[0])
        metrics["test_size"] = int(X_test.shape[0])
        metrics["horizon"] = self.horizon
        metrics["threshold"] = self.threshold
        metrics["params"] = {k: params[k] for k in sorted(params.keys())}

        # Feature importance (top 10 by gain).
        importance = self._top_feature_importance(model, top_k=10)
        metrics["top_features"] = importance

        return model, metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sample_weights(y_train: np.ndarray) -> np.ndarray:
        """
        Per-sample weight = (n_total / (n_classes * n_in_class)).
        Mirrors sklearn's 'balanced' weighting. Stops dominant classes
        from dictating predictions.
        """
        n = len(y_train)
        weights = np.ones(n, dtype=np.float64)
        if n == 0:
            return weights
        classes, counts = np.unique(y_train, return_counts=True)
        n_classes = max(len(classes), 1)
        for cls, cnt in zip(classes, counts):
            if cnt == 0:
                continue
            w = n / (n_classes * cnt)
            weights[y_train == cls] = w
        return weights

    def _evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute accuracy and per-class precision/recall/F1 on the test fold."""
        from sklearn.metrics import precision_recall_fscore_support

        preds = model.predict(X_test)
        accuracy = float(np.mean(preds == y_test))

        precision, recall, f1, _support = precision_recall_fscore_support(
            y_test,
            preds,
            labels=[0, 1, 2],
            zero_division=0,
        )
        per_class = {}
        for i, name in enumerate(CLASS_NAMES):
            per_class[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
        return {
            "accuracy": accuracy,
            "per_class": per_class,
        }

    def _top_feature_importance(self, model: Any, top_k: int = 10) -> list[dict]:
        """Return the top-k features by gain."""
        names = self.engineer.feature_names()
        try:
            booster = model.get_booster()
            gains = booster.get_score(importance_type="gain")
            # XGBoost names features as f0, f1, ... when raw arrays are used.
            scored: list[tuple[str, float]] = []
            for k, v in gains.items():
                # Convert "f12" -> index 12 -> name
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    label = names[idx] if 0 <= idx < len(names) else k
                else:
                    label = k
                scored.append((label, float(v)))
            scored.sort(key=lambda t: t[1], reverse=True)
            return [{"feature": n, "gain": g} for n, g in scored[:top_k]]
        except Exception as exc:
            logger.warning("model_trainer_importance_failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model: Any, metrics: dict, path: str) -> None:
        """
        Persist model + sidecar JSON files alongside ``path``.

        path/<base>.joblib            — model
        path/<base>_metrics.json      — metrics
        path/<base>_features.json     — ordered list of feature names

        Use ModelRegistry.save_model() in normal flows — this helper is
        provided for callers that want to write to a custom path.
        """
        import json
        from pathlib import Path

        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, p)
        base = p.with_suffix("")
        Path(f"{base}_metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )
        Path(f"{base}_features.json").write_text(
            json.dumps(self.engineer.feature_names(), indent=2), encoding="utf-8"
        )
