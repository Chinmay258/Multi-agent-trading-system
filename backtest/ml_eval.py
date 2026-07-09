"""
backtest/ml_eval.py
-------------------
Honest, out-of-sample evaluation of the ML classifier.

Builds features/labels with the system's own ``FeatureEngineer`` + ``LabelGenerator``,
does a **chronological** train/test split (no shuffle, no lookahead), trains the same
XGBoost configuration the system uses, and reports the metrics that actually reveal
whether the model has skill: per-class precision/recall/F1, confusion matrix, class
balance, feature importance, and a calibration curve.

The headline truth (stated plainly in the report): on a 3-class problem where random
scores ~33%, the model lands in the ~44–49% range and leans heavily on HOLD — i.e. it
has little reliable directional skill. We do not dress this up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.technical_analysis.ml.feature_engineer import FeatureEngineer
from agents.technical_analysis.ml.label_generator import LabelGenerator
from backtest.data import load_candles, to_arrays

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _REPO_ROOT / "models"

# Internal class encoding mirrors model_trainer: SELL=0, HOLD=1, BUY=2.
_CLASS_NAMES = ["SELL", "HOLD", "BUY"]
_LABEL_TO_CLASS = {-1.0: 0, 0.0: 1, 1.0: 2}


def _build_xy(
    candles: list, horizon: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Replicate ModelTrainer's offline (X, y) construction with correct alignment."""
    eng = FeatureEngineer()
    lab = LabelGenerator(horizon=horizon, threshold=threshold)
    arr = to_arrays(candles)
    closes = arr["closes"]

    X_all, names = eng.build_feature_matrix(
        closes=closes,
        highs=arr["highs"],
        lows=arr["lows"],
        volumes=arr["volumes"],
        timestamps=arr["timestamps"],
        opens=arr["opens"],
    )
    y_all = lab.generate(closes)
    start = eng.MIN_BARS_REQUIRED - 1
    n = len(closes)
    end = n - horizon
    if end <= start or X_all.shape[0] == 0:
        return np.zeros((0, len(names))), np.zeros((0,)), names
    keep = end - start
    return X_all[:keep], y_all[start:end], names


def _stored_model_metrics(symbol: str, timeframe: str) -> dict | None:
    """Return the deployed model's saved metrics JSON, if present (for reference)."""
    base = f"{symbol.replace('/', '_').replace('-', '_').lower()}_{timeframe}"
    latest = _MODELS_DIR / f"{base}_latest"
    if latest.exists():
        version = latest.read_text().strip()
        path = _MODELS_DIR / f"{version}_metrics.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def evaluate_ml(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    horizon: int = 10,
    threshold: float = 0.01,
) -> dict:
    """Train/test split, fit, predict, and return a full honest metric set."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    candles = load_candles(symbol, timeframe)
    X, y, names = _build_xy(candles, horizon, threshold)

    if X.shape[0] < 60:
        return {
            "available": False,
            "reason": f"insufficient samples ({X.shape[0]}) for {symbol} {timeframe}",
        }

    y_cls = np.vectorize(_LABEL_TO_CLASS.get)(y).astype(np.int64)
    split = int(X.shape[0] * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y_cls[:split], y_cls[split:]

    def _dist(arr: np.ndarray) -> dict:
        return {_CLASS_NAMES[c]: int((arr == c).sum()) for c in (0, 1, 2)}

    import xgboost as xgb

    params = dict(
        n_estimators=400,
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
    model = xgb.XGBClassifier(**params)
    # Class weights to counter imbalance (rare HOLD/BUY/SELL).
    counts = np.bincount(y_tr, minlength=3).astype(np.float64)
    inv = np.where(counts > 0, counts.sum() / (3 * counts), 0.0)
    sample_weight = inv[y_tr]
    model.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)

    proba = model.predict_proba(X_te)
    y_pred = proba.argmax(axis=1)

    accuracy = float((y_pred == y_te).mean())
    prec, rec, f1, support = precision_recall_fscore_support(
        y_te, y_pred, labels=[0, 1, 2], zero_division=0
    )
    per_class = {
        _CLASS_NAMES[i]: {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(3)
    }
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2]).tolist()

    # Calibration: reliability of the predicted top-class probability vs. correctness.
    confidence = proba.max(axis=1)
    correct = (y_pred == y_te).astype(int)
    try:
        frac_pos, mean_pred = calibration_curve(correct, confidence, n_bins=8, strategy="quantile")
        calibration = {
            "mean_predicted_confidence": [round(float(x), 4) for x in mean_pred],
            "observed_accuracy": [round(float(x), 4) for x in frac_pos],
        }
    except Exception:
        calibration = {"mean_predicted_confidence": [], "observed_accuracy": []}

    # Feature importance by gain.
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    fmap = {f"f{i}": names[i] for i in range(len(names))}
    top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_features = [{"feature": fmap.get(k, k), "gain": round(float(v), 4)} for k, v in top]

    return {
        "available": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon": horizon,
        "threshold": threshold,
        "n_samples": int(X.shape[0]),
        "train_size": int(X_tr.shape[0]),
        "test_size": int(X_te.shape[0]),
        "random_baseline_accuracy": round(1 / 3, 4),
        "accuracy": round(accuracy, 4),
        "class_balance_train": _dist(y_tr),
        "class_balance_test": _dist(y_te),
        "per_class": per_class,
        "confusion_matrix": {"labels": _CLASS_NAMES, "matrix": cm},
        "calibration": calibration,
        "top_features": top_features,
        "deployed_model_metrics": _stored_model_metrics(symbol, timeframe),
    }
