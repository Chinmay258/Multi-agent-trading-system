"""
tests/unit/test_ml.py
----------------------
Zero-I/O unit tests for the ML signal engine.

These tests rely only on synthetic OHLCV data and use the FeatureEngineer,
LabelGenerator, MLSignalGenerator, and ModelRegistry directly. The model
under test is a hand-rolled stub returning fixed probabilities — we never
spin up a real XGBoost training run inside the unit suite.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from agents.technical_analysis.candle_buffer import CandleBuffer
from agents.technical_analysis.ml.feature_engineer import FeatureEngineer
from agents.technical_analysis.ml.label_generator import LabelGenerator
from agents.technical_analysis.ml.ml_signal_generator import MLSignalGenerator
from agents.technical_analysis.ml.model_registry import ModelRegistry
from core.models.market import OHLCVCandle
from core.models.signals import SignalDirection

# ---------------------------------------------------------------------------
# Synthetic candle helpers
# ---------------------------------------------------------------------------


def _make_candles(
    n: int,
    start_price: float = 100.0,
    drift: float = 0.05,
    timeframe: str = "1m",
    symbol: str = "BTC/USDT",
    seed: int = 7,
) -> list[OHLCVCandle]:
    """
    Build a list of synthetic OHLCV candles with deterministic noise.

    The price walks up with small jitter so indicators move past their
    warmup regions and labels are not trivially constant.
    """
    rng = np.random.default_rng(seed)
    candles: list[OHLCVCandle] = []
    price = float(start_price)
    base_ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    for i in range(n):
        next_price = price + drift + float(rng.normal(0.0, 0.5))
        next_price = max(next_price, 1.0)
        high = max(price, next_price) + abs(float(rng.normal(0.0, 0.2)))
        low = min(price, next_price) - abs(float(rng.normal(0.0, 0.2)))
        low = max(low, 0.5)
        volume = float(rng.uniform(50.0, 150.0))
        candles.append(
            OHLCVCandle(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=base_ts + timedelta(minutes=i),
                open=Decimal(str(round(price, 4))),
                high=Decimal(str(round(high, 4))),
                low=Decimal(str(round(low, 4))),
                close=Decimal(str(round(next_price, 4))),
                volume=Decimal(str(round(volume, 4))),
            )
        )
        price = next_price
    return candles


def _fill_buffer(candles: Iterable[OHLCVCandle]) -> CandleBuffer:
    buf = CandleBuffer("BTC/USDT", "1m")
    for c in candles:
        buf.add(c)
    return buf


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


class TestFeatureEngineer:
    def test_returns_correct_shape(self):
        engineer = FeatureEngineer()
        buf = _fill_buffer(_make_candles(100))
        feats = engineer.build_features(buf)
        assert feats is not None
        assert feats.ndim == 1
        assert feats.shape[0] == len(engineer.feature_names())

    def test_returns_none_insufficient_data(self):
        engineer = FeatureEngineer()
        buf = _fill_buffer(_make_candles(30))
        assert engineer.build_features(buf) is None

    def test_no_nan_or_inf(self):
        engineer = FeatureEngineer()
        buf = _fill_buffer(_make_candles(120))
        feats = engineer.build_features(buf)
        assert feats is not None
        assert np.all(np.isfinite(feats)), "Feature vector contains non-finite values"

    def test_feature_names_match_vector_length(self):
        engineer = FeatureEngineer()
        buf = _fill_buffer(_make_candles(120))
        feats = engineer.build_features(buf)
        assert feats is not None
        assert len(engineer.feature_names()) == feats.shape[0]

    def test_build_feature_matrix_shape(self):
        engineer = FeatureEngineer()
        candles = _make_candles(200)
        closes = np.array([float(c.close) for c in candles])
        highs = np.array([float(c.high) for c in candles])
        lows = np.array([float(c.low) for c in candles])
        opens = np.array([float(c.open) for c in candles])
        volumes = np.array([float(c.volume) for c in candles])
        timestamps = [c.timestamp for c in candles]

        X, names = engineer.build_feature_matrix(
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            timestamps=timestamps,
            opens=opens,
        )
        # n_samples = n - (MIN_BARS_REQUIRED - 1).
        expected_rows = len(candles) - (engineer.MIN_BARS_REQUIRED - 1)
        assert X.shape == (expected_rows, len(names))
        assert names == engineer.feature_names()
        assert np.all(np.isfinite(X))


# ---------------------------------------------------------------------------
# LabelGenerator
# ---------------------------------------------------------------------------


class TestLabelGenerator:
    def test_correct_shape(self):
        gen = LabelGenerator(horizon=5, threshold=0.001)
        closes = np.linspace(100.0, 110.0, 50)
        labels = gen.generate(closes)
        assert labels.shape == closes.shape

    def test_distribution_only_known_classes(self):
        gen = LabelGenerator(horizon=5, threshold=0.001)
        closes = np.linspace(100.0, 110.0, 60)
        labels = gen.generate(closes)
        valid = labels[~np.isnan(labels)]
        unique = set(np.unique(valid).tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_horizon_nans(self):
        horizon = 7
        gen = LabelGenerator(horizon=horizon, threshold=0.001)
        closes = np.linspace(100.0, 130.0, 40)
        labels = gen.generate(closes)
        # Last ``horizon`` entries must be NaN
        assert np.all(np.isnan(labels[-horizon:]))
        # And the entries before that are valid
        assert not np.any(np.isnan(labels[:-horizon]))

    def test_label_distribution_counts(self):
        gen = LabelGenerator(horizon=2, threshold=0.001)
        # Up, up, neutral, down, up — long enough for horizon=2
        closes = np.array([100.0, 101.0, 100.0, 99.0, 100.0, 100.0, 100.0])
        labels = gen.generate(closes)
        dist = gen.label_distribution(labels)
        assert dist["total"] == len(closes) - 2
        assert dist["buy"] + dist["hold"] + dist["sell"] == dist["total"]


# ---------------------------------------------------------------------------
# MLSignalGenerator (with stub model)
# ---------------------------------------------------------------------------


class _StubModel:
    """
    Minimal stand-in for an XGBoost classifier.

    ``predict_proba(X)`` returns a fixed (1, 3) probability vector regardless
    of input. The signal generator only uses ``predict_proba``.
    """

    def __init__(self, probs: list[float]) -> None:
        self._probs = np.array([probs], dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._probs


def _write_stub_model(
    tmp_path: Path,
    probs: list[float],
    feature_names: list[str],
    accuracy: float = 0.55,
) -> Path:
    """Persist a stub model + sidecar JSON files to disk."""
    import joblib

    model_path = tmp_path / "stub_v1.joblib"
    joblib.dump(_StubModel(probs), model_path)
    base = model_path.with_suffix("")
    (Path(f"{base}_features.json")).write_text(json.dumps(feature_names), encoding="utf-8")
    (Path(f"{base}_metrics.json")).write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "top_features": [
                    {"feature": "returns_1", "gain": 12.3},
                    {"feature": "rsi_14", "gain": 9.7},
                    {"feature": "macd_hist", "gain": 5.1},
                ],
            }
        ),
        encoding="utf-8",
    )
    return model_path


class TestMLSignalGenerator:
    def test_generate_buy(self, tmp_path):
        engineer = FeatureEngineer()
        model_path = _write_stub_model(
            tmp_path,
            probs=[0.10, 0.20, 0.70],  # SELL, HOLD, BUY
            feature_names=engineer.feature_names(),
        )
        gen = MLSignalGenerator(str(model_path))
        assert gen.is_loaded()

        buf = _fill_buffer(_make_candles(120))
        signal = gen.generate(buf)
        assert signal is not None
        assert signal.direction == SignalDirection.BUY.value
        assert signal.confidence == pytest.approx(0.70, abs=1e-4)
        meta = signal.metadata
        assert meta["model"] == "xgboost"
        assert meta["p_buy"] == pytest.approx(0.70, abs=1e-4)
        assert meta["p_sell"] == pytest.approx(0.10, abs=1e-4)
        assert meta["top_features"] == ["returns_1", "rsi_14", "macd_hist"]

    def test_generate_sell(self, tmp_path):
        engineer = FeatureEngineer()
        model_path = _write_stub_model(
            tmp_path,
            probs=[0.65, 0.25, 0.10],
            feature_names=engineer.feature_names(),
        )
        gen = MLSignalGenerator(str(model_path))
        buf = _fill_buffer(_make_candles(120))
        signal = gen.generate(buf)
        assert signal is not None
        assert signal.direction == SignalDirection.SELL.value
        assert signal.confidence == pytest.approx(0.65, abs=1e-4)

    def test_generate_neutral_when_hold_dominates(self, tmp_path, monkeypatch):
        engineer = FeatureEngineer()
        model_path = _write_stub_model(
            tmp_path,
            probs=[0.30, 0.65, 0.05],  # HOLD wins → direction NEUTRAL
            feature_names=engineer.feature_names(),
        )
        # Make sure low confidence still passes the threshold for this test —
        # confidence = max(p_sell, p_buy) = 0.30. Force the min threshold to
        # a small value so we get a signal back (NEUTRAL).
        from core.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("TA_MIN_SIGNAL_CONFIDENCE", "0.0")

        gen = MLSignalGenerator(str(model_path))
        buf = _fill_buffer(_make_candles(120))
        signal = gen.generate(buf)
        assert signal is not None
        assert signal.direction == SignalDirection.NEUTRAL.value

        get_settings.cache_clear()

    def test_generate_returns_none_on_insufficient_data(self, tmp_path):
        engineer = FeatureEngineer()
        model_path = _write_stub_model(
            tmp_path,
            probs=[0.1, 0.1, 0.8],
            feature_names=engineer.feature_names(),
        )
        gen = MLSignalGenerator(str(model_path))
        buf = _fill_buffer(_make_candles(30))
        assert gen.generate(buf) is None

    def test_generate_suppressed_below_confidence(self, tmp_path, monkeypatch):
        engineer = FeatureEngineer()
        model_path = _write_stub_model(
            tmp_path,
            probs=[0.20, 0.55, 0.25],  # max directional prob is 0.25 → below threshold
            feature_names=engineer.feature_names(),
        )
        from core.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("TA_MIN_SIGNAL_CONFIDENCE", "0.6")

        gen = MLSignalGenerator(str(model_path))
        buf = _fill_buffer(_make_candles(120))
        assert gen.generate(buf) is None

        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_save_and_load(self, tmp_path):
        registry = ModelRegistry(models_dir=tmp_path)
        feature_names = FeatureEngineer().feature_names()
        model = _StubModel([0.1, 0.2, 0.7])
        metrics = {"accuracy": 0.6, "per_class": {}}

        saved = registry.save_model(
            model=model,
            metrics=metrics,
            feature_names=feature_names,
            symbol="BTC/USDT",
            timeframe="1m",
        )
        assert saved.exists()
        assert saved.name == "btc_usdt_1m_v1.joblib"

        latest = registry.get_latest_path("BTC/USDT", "1m")
        assert latest == saved

        # A second save bumps the version and updates the pointer.
        saved2 = registry.save_model(
            model=model,
            metrics={"accuracy": 0.7, "per_class": {}},
            feature_names=feature_names,
            symbol="BTC/USDT",
            timeframe="1m",
        )
        assert saved2.name == "btc_usdt_1m_v2.joblib"
        latest2 = registry.get_latest_path("BTC/USDT", "1m")
        assert latest2 == saved2

        # list_models surfaces both versions.
        listing = registry.list_models()
        assert len(listing) >= 2
        versions = sorted([m["version"] for m in listing if m["base"].startswith("btc_usdt_1m")])
        assert versions == [1, 2]

    def test_get_latest_path_returns_none_when_empty(self, tmp_path):
        registry = ModelRegistry(models_dir=tmp_path)
        assert registry.get_latest_path("ETH/USDT", "1h") is None
