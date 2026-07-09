"""
agents/technical_analysis/ml/ml_signal_generator.py
-----------------------------------------------------
Inference-time signal generator backed by a trained XGBoost classifier.

Drop-in replacement for the rule-based SignalGenerator.generate(buffer):
returns a TechnicalSignal (or None when the model is unsure / data is
insufficient). The signal carries class probabilities and the top features
in its ``metadata`` for downstream debugging.

Design decisions:
- The model file is loaded lazily once at construction. We also load the
  feature-names sidecar and verify length matches what FeatureEngineer
  currently produces — guards against running an old model after the
  feature set changes.
- Class encoding mirrors ModelTrainer: 0=SELL, 1=HOLD, 2=BUY.
- Confidence is the max of (p_sell, p_buy) — i.e. confidence in the
  directional call. p_hold doesn't count; if HOLD wins, direction is
  NEUTRAL and confidence is the runner-up directional probability.
- A signal is suppressed (returns None) when confidence falls below
  ``min_signal_confidence`` from config — matching the rule-based agent's
  behaviour so downstream filters don't have to special-case ML output.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from agents.technical_analysis.candle_buffer import CandleBuffer
from agents.technical_analysis.ml.feature_engineer import FeatureEngineer
from core.config import get_settings
from core.logging import get_logger
from core.models.signals import (
    IndicatorReading,
    SignalDirection,
    SignalSource,
    TechnicalSignal,
)

logger = get_logger("ml_signal_generator")


# Class index → display name
CLASS_NAMES = ["SELL", "HOLD", "BUY"]


class MLSignalGenerator:
    """
    Wrap a saved XGBoost classifier and emit TechnicalSignal events.

    Usage:
        gen = MLSignalGenerator("models/btc_usdt_1m_v1.joblib")
        if gen.is_loaded():
            signal = gen.generate(buffer)
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)
        self._model: Any | None = None
        self._feature_names: list[str] = []
        self._metrics: dict = {}
        self._engineer = FeatureEngineer()
        self._loaded = False
        self._top_features_cached: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load model + sidecars. Sets ``_loaded`` to True on success."""
        if not self._model_path.exists():
            logger.warning("ml_model_file_missing", path=str(self._model_path))
            return

        try:
            import joblib

            self._model = joblib.load(self._model_path)

            base = self._model_path.with_suffix("")
            metrics_path = Path(f"{base}_metrics.json")
            features_path = Path(f"{base}_features.json")

            if features_path.exists():
                self._feature_names = json.loads(features_path.read_text(encoding="utf-8"))
            if metrics_path.exists():
                self._metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            # Validate feature compatibility.
            current = self._engineer.feature_names()
            if self._feature_names and self._feature_names != current:
                logger.warning(
                    "ml_feature_mismatch",
                    model_features=len(self._feature_names),
                    current_features=len(current),
                    first_diff_index=self._first_diff(self._feature_names, current),
                )

            # Cache top feature names for inclusion in signal metadata.
            self._top_features_cached = [
                t.get("feature", "?") for t in (self._metrics.get("top_features") or [])[:3]
            ]

            self._loaded = True
            logger.info(
                "ml_model_loaded",
                path=str(self._model_path),
                accuracy=self._metrics.get("accuracy"),
                feature_count=len(self._feature_names),
                top_features=self._top_features_cached,
            )
        except Exception as exc:
            logger.error("ml_model_load_failed", path=str(self._model_path), error=str(exc))
            self._loaded = False
            self._model = None

    @staticmethod
    def _first_diff(a: list[str], b: list[str]) -> int | None:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        if len(a) != len(b):
            return min(len(a), len(b))
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """True if the model was loaded successfully and is ready for inference."""
        return self._loaded and self._model is not None

    def generate(self, buffer: CandleBuffer) -> TechnicalSignal | None:
        """
        Run inference and produce a TechnicalSignal.

        Returns None when:
        - the model isn't loaded,
        - the buffer doesn't have MIN_BARS_REQUIRED candles,
        - confidence is below the configured ``min_signal_confidence``.
        """
        if not self.is_loaded():
            return None

        features = self._engineer.build_features(buffer)
        if features is None:
            return None

        # Extract ema_50_dist for trend-filter metadata consumed by DecisionAgent.
        _feature_names = self._feature_names or self._engineer.feature_names()
        _ema_50_dist = 0.0
        if "ema_50_dist" in _feature_names:
            _ema_50_dist = float(features[_feature_names.index("ema_50_dist")])

        # XGBoost expects 2-D input.
        X = features.reshape(1, -1).astype(np.float64)

        try:
            proba = self._model.predict_proba(X)[0]  # shape: (3,)
        except Exception as exc:
            logger.error("ml_predict_failed", error=str(exc), symbol=buffer.symbol)
            return None

        if len(proba) != 3:
            logger.warning(
                "ml_proba_shape_unexpected",
                expected=3,
                got=int(len(proba)),
                symbol=buffer.symbol,
            )
            return None

        p_sell = float(proba[0])
        p_hold = float(proba[1])
        p_buy = float(proba[2])

        argmax = int(np.argmax(proba))

        # Emit a directional signal if buy or sell probability clears the
        # threshold — even when p_hold is the argmax.  A model output of
        # p_buy=0.40, p_hold=0.42, p_sell=0.18 is a mild buy signal, not
        # neutral.  confidence is set to the directional class probability so
        # downstream filters (min_signal_confidence) use the same scale.
        _DIRECTIONAL_THRESHOLD = 0.35
        if p_buy >= _DIRECTIONAL_THRESHOLD and p_buy > p_sell:
            direction = SignalDirection.BUY
            confidence = p_buy
        elif p_sell >= _DIRECTIONAL_THRESHOLD and p_sell > p_buy:
            direction = SignalDirection.SELL
            confidence = p_sell
        else:
            direction = SignalDirection.NEUTRAL
            confidence = max(p_buy, p_sell)

        # Apply min-confidence filter — match the rule-based generator's contract.
        min_conf = get_settings().technical_analysis.min_signal_confidence
        if confidence < min_conf:
            logger.debug(
                "ml_signal_below_confidence_threshold",
                symbol=buffer.symbol,
                confidence=round(confidence, 4),
                min_required=min_conf,
            )
            return None

        # Build the signal envelope.
        ttl = get_settings().technical_analysis.signal_ttl_seconds
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
        current_price = float(buffer.latest_close or 0.0)

        metadata: dict = {
            "model": "xgboost",
            "model_path": str(self._model_path),
            "p_buy": round(p_buy, 4),
            "p_sell": round(p_sell, 4),
            "p_hold": round(p_hold, 4),
            "top_features": self._top_features_cached,
            "argmax_class": CLASS_NAMES[argmax],
            "ema_50_dist": round(_ema_50_dist, 6),
        }

        # Carry forward the training accuracy if available — useful context
        # for the Decision agent when blending ML and rule signals.
        if "accuracy" in self._metrics:
            metadata["training_accuracy"] = self._metrics["accuracy"]

        # Empty indicators list — ML doesn't decompose into named indicator
        # readings. The metadata carries the relevant inference info instead.
        readings: list[IndicatorReading] = []

        signal = TechnicalSignal(
            source=SignalSource.ML,
            symbol=buffer.symbol,
            timeframe=buffer.timeframe,
            expires_at=expires_at,
            direction=direction,
            confidence=round(confidence, 4),
            indicators=readings,
            price=current_price,
            metadata=metadata,
        )

        logger.info(
            "ml_signal_generated",
            symbol=buffer.symbol,
            timeframe=buffer.timeframe,
            direction=direction.value,
            confidence=round(confidence, 4),
            p_buy=round(p_buy, 4),
            p_sell=round(p_sell, 4),
            p_hold=round(p_hold, 4),
        )
        return signal
