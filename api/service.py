"""
api/service.py
==============
The prediction logic, kept separate from the web layer (main.py).

WHY separate? So it can be unit-tested WITHOUT starting a web server (Phase 6),
and so the same logic could be reused by a batch job or another interface.

The crucial job here: reproduce EXACTLY the training-time feature engineering by
reusing features.add_domain_features(). If the API computed features differently
from training, predictions would silently degrade (train/serve skew) — the single
most common production ML bug. We avoid it by importing the same function.
"""

import joblib
import pandas as pd

from config import MODELS_DIR
from features import add_domain_features

# Map the clean JSON field names -> the exact dataset column names the model
# pipeline expects. One source of truth for this mapping.
FIELD_TO_COLUMN = {
    "type": "Type",
    "air_temperature": "Air temperature [K]",
    "process_temperature": "Process temperature [K]",
    "rotational_speed": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}


class PredictionService:
    """Loads the trained model once and serves predictions."""

    def __init__(self, model_path=None):
        self.model_path = model_path or (MODELS_DIR / "best_model.joblib")
        self.pipeline = None
        self.threshold = 0.5
        self.model_name = None
        self.feature_names = None

    def load(self):
        """Load the serialized bundle produced by train.py.

        Called once at startup. If the model file is missing we raise a clear
        error rather than starting a server that returns garbage."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run train.py first.")
        bundle = joblib.load(self.model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.model_name = bundle["model_name"]
        self.feature_names = bundle["feature_names"]
        return self

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def _to_frame(self, readings: list[dict]) -> pd.DataFrame:
        """Turn a list of raw readings into the exact DataFrame the model wants."""
        # rename clean field names -> dataset column names
        rows = [{FIELD_TO_COLUMN[k]: v for k, v in r.items()} for r in readings]
        df = pd.DataFrame(rows)
        # apply the SAME feature engineering used in training
        df = add_domain_features(df)
        # order columns to match training (defensive: avoids silent misalignment)
        return df[self.feature_names]

    def _risk_level(self, prob: float) -> str:
        """Human-friendly bucket for dashboards/alerts."""
        if prob >= self.threshold:
            return "high"
        if prob >= self.threshold / 2:
            return "medium"
        return "low"

    def predict(self, readings: list[dict]) -> list[dict]:
        """Predict for one or many readings. Returns structured dicts."""
        if not self.is_ready:
            raise RuntimeError("Model is not loaded.")
        X = self._to_frame(readings)
        probs = self.pipeline.predict_proba(X)[:, 1]

        out = []
        for p in probs:
            p = float(p)
            out.append({
                "failure_predicted": bool(p >= self.threshold),
                "failure_probability": round(p, 4),
                "risk_level": self._risk_level(p),
                "threshold_used": round(self.threshold, 4),
                "model_name": self.model_name,
            })
        return out
