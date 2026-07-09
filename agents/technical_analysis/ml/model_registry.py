"""
agents/technical_analysis/ml/model_registry.py
------------------------------------------------
Manage trained model files on disk.

Disk layout:

    models/
        btc_usdt_1m_v1.joblib            — joblib-serialised XGBoost classifier
        btc_usdt_1m_v1_metrics.json      — training metrics (accuracy, P/R/F1, importances)
        btc_usdt_1m_v1_features.json     — list of feature names (validated at inference)
        btc_usdt_1m_latest               — plain-text pointer to the latest base name
                                           (e.g. contents: "btc_usdt_1m_v1")

Design decisions:
- File names normalise the symbol: "BTC/USDT" → "btc_usdt". Slashes break
  paths on every OS we care about and the case is irrelevant to the
  filesystem on Windows.
- "Latest" is a plain text file rather than a symlink because symlinks on
  Windows require elevated permissions or Developer Mode. A text pointer
  is portable, atomic to update, and easy to inspect.
- The registry exposes paths only — actual loading/saving lives with the
  Trainer and the SignalGenerator. The registry stays I/O-light and easy
  to unit test with ``tmp_path``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Default location, relative to the repo root. Override via constructor.
DEFAULT_MODELS_DIR = Path("models")


def _normalise(symbol: str, timeframe: str) -> str:
    """
    Build the file-system-safe base used for a (symbol, timeframe) pair.

    "BTC/USDT", "1m" → "btc_usdt_1m"
    Non-alphanumeric characters in symbols collapse to "_".
    """
    sym = symbol.lower().replace("/", "_")
    sym = re.sub(r"[^a-z0-9_]+", "_", sym).strip("_")
    tf = timeframe.lower()
    return f"{sym}_{tf}"


class ModelRegistry:
    """
    Tracks versioned model files for each (symbol, timeframe) pair.

    Usage:
        reg = ModelRegistry()  # uses ./models
        path = reg.get_latest_path("BTC/USDT", "1m")
        if path:
            generator = MLSignalGenerator(str(path))

        # After training:
        saved = reg.save_model(model, metrics, feature_names, "BTC/USDT", "1m")
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir: Path = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _latest_pointer(self, symbol: str, timeframe: str) -> Path:
        return self.models_dir / f"{_normalise(symbol, timeframe)}_latest"

    def _next_version(self, symbol: str, timeframe: str) -> int:
        """Return the next integer version (1 if none exist)."""
        prefix = _normalise(symbol, timeframe)
        pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.joblib$")
        versions: list[int] = []
        if self.models_dir.exists():
            for p in self.models_dir.iterdir():
                m = pattern.match(p.name)
                if m:
                    versions.append(int(m.group(1)))
        return max(versions, default=0) + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest_path(self, symbol: str, timeframe: str) -> Path | None:
        """
        Return the path to the latest model file for the given pair,
        or None if no model has been trained yet.
        """
        pointer = self._latest_pointer(symbol, timeframe)
        if pointer.exists():
            base = pointer.read_text(encoding="utf-8").strip()
            candidate = self.models_dir / f"{base}.joblib"
            if candidate.exists():
                return candidate

        # Fallback: scan for the highest existing version even if the
        # pointer file is missing (e.g. someone deleted it).
        prefix = _normalise(symbol, timeframe)
        pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.joblib$")
        best: tuple[int, Path] | None = None
        if self.models_dir.exists():
            for p in self.models_dir.iterdir():
                m = pattern.match(p.name)
                if m:
                    v = int(m.group(1))
                    if best is None or v > best[0]:
                        best = (v, p)
        return best[1] if best else None

    def save_model(
        self,
        model: Any,
        metrics: dict,
        feature_names: list[str],
        symbol: str,
        timeframe: str,
    ) -> Path:
        """
        Save model + sidecars + update the "latest" pointer atomically.

        Returns the path to the saved model file.
        """
        import joblib  # local import: keep registry importable without ML deps

        self._ensure_dir()
        version = self._next_version(symbol, timeframe)
        base = f"{_normalise(symbol, timeframe)}_v{version}"

        model_path = self.models_dir / f"{base}.joblib"
        metrics_path = self.models_dir / f"{base}_metrics.json"
        features_path = self.models_dir / f"{base}_features.json"

        joblib.dump(model, model_path)

        full_metrics = {
            **metrics,
            "symbol": symbol,
            "timeframe": timeframe,
            "version": version,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        metrics_path.write_text(json.dumps(full_metrics, indent=2, default=str), encoding="utf-8")
        features_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

        # Update the "latest" text pointer last so a partial failure above
        # doesn't leave the pointer aimed at a half-written model.
        self._latest_pointer(symbol, timeframe).write_text(base, encoding="utf-8")

        return model_path

    def list_models(self) -> list[dict]:
        """
        Enumerate every model file in the registry with its metrics summary.

        Returns a list sorted by (symbol, timeframe, version desc).
        """
        out: list[dict] = []
        if not self.models_dir.exists():
            return out

        pattern = re.compile(r"^(?P<base>.+)_v(?P<version>\d+)\.joblib$")
        for p in sorted(self.models_dir.iterdir()):
            m = pattern.match(p.name)
            if not m:
                continue
            base = f"{m.group('base')}_v{m.group('version')}"
            metrics_file = self.models_dir / f"{base}_metrics.json"
            metrics: dict = {}
            if metrics_file.exists():
                try:
                    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metrics = {"error": "metrics file unreadable"}
            out.append(
                {
                    "path": str(p),
                    "base": base,
                    "version": int(m.group("version")),
                    "metrics": metrics,
                }
            )
        # Newest first for each prefix
        out.sort(key=lambda d: (d["base"].rsplit("_v", 1)[0], -d["version"]))
        return out
