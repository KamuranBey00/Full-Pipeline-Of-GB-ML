"""Shared test setup: make project modules importable + common fixtures."""

import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "api"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import MODELS_DIR  # noqa: E402

MODEL_EXISTS = (MODELS_DIR / "best_model.joblib").exists()
requires_model = pytest.mark.skipif(not MODEL_EXISTS,
                                    reason="models/best_model.joblib missing - run train.py")

# A reading the model should flag (high torque + high tool wear).
RISKY = {"type": "L", "air_temperature": 298.9, "process_temperature": 309.0,
         "rotational_speed": 1300, "torque": 65.0, "tool_wear": 220}
# A typical healthy reading (the /docs example).
NORMAL = {"type": "L", "air_temperature": 298.1, "process_temperature": 308.6,
          "rotational_speed": 1551, "torque": 42.8, "tool_wear": 108}
