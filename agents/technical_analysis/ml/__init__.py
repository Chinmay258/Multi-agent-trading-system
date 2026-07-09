"""
agents/technical_analysis/ml
-----------------------------
ML signal engine for the Technical Analysis agent.

Replaces (or augments) the rule-based SignalGenerator with an XGBoost
classifier trained on historical OHLCV data. The rule-based system
remains the fallback when no trained model is available.

Components:
    FeatureEngineer    — build feature vectors from candle buffers.
    LabelGenerator     — derive {-1, 0, +1} training labels from forward returns.
    ModelTrainer       — load OHLCV from DB, fit XGBoost, evaluate, save.
    MLSignalGenerator  — load a saved model and emit TechnicalSignal at inference time.
    ModelRegistry      — manage versioned model files on disk.
"""

from agents.technical_analysis.ml.feature_engineer import FeatureEngineer
from agents.technical_analysis.ml.label_generator import LabelGenerator
from agents.technical_analysis.ml.ml_signal_generator import MLSignalGenerator
from agents.technical_analysis.ml.model_registry import ModelRegistry

__all__ = [
    "FeatureEngineer",
    "LabelGenerator",
    "MLSignalGenerator",
    "ModelRegistry",
]
