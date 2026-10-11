"""
api/main.py
===========
The web layer: a FastAPI app exposing the model over HTTP.

Endpoints:
  GET  /            -> basic service info
  GET  /health      -> is the model loaded? (used by load balancers / Docker)
  POST /predict     -> score ONE machine
  POST /predict/batch -> score MANY machines in one call

Run locally:
  uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs for the auto-generated interactive UI.
"""

import sys
import pathlib

# --- make project modules importable regardless of layout ---------------------
# This lets the API find config.py / features.py whether they live at the project
# root (flat layout) or under src/. Also lets us import schemas/service by name.
HERE = pathlib.Path(__file__).resolve().parent          # .../api
ROOT = HERE.parent                                       # project root
for _p in (HERE, ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from config import PREDICTION_LOG
from schemas import (SensorReading, BatchRequest, PredictionResponse,
                     HealthResponse)
from service import PredictionService

# ---------------------------------------------------------------------------
# Create the app and load the model ONCE at import time. Loading per-request
# would be far too slow; the model is small and stateless, so we share it.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure risk from live sensor readings.",
    version="1.0.0",
)

service = PredictionService()
try:
    service.load()
except FileNotFoundError as e:
    # Don't crash the server: /health will report the model as not loaded,
    # which is the correct, observable behaviour for an ops team.
    print(f"[api] WARNING: {e}")


def log_predictions(readings: list[dict], results: list[dict]) -> None:
    """Append each scored reading to a JSONL log — the 'current' data that
    monitor.py compares against training data to detect drift (Phase 8)."""
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with PREDICTION_LOG.open("a", encoding="utf-8") as f:
        for r, p in zip(readings, results):
            f.write(json.dumps({"ts": ts, **r,
                                "failure_probability": p["failure_probability"]}) + "\n")


@app.get("/")
def root():
    return {"service": "predictive-maintenance", "docs": "/docs",
            "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness/readiness check. Returns 200 even if the model is missing, but
    reports model_loaded=False so orchestrators can react."""
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        model_name=service.model_name,
        threshold=service.threshold if service.is_ready else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    """Score a single machine. Pydantic has already validated `reading`."""
    if not service.is_ready:
        # 503 = service temporarily unavailable (model not loaded)
        raise HTTPException(status_code=503, detail="Model not loaded.")
    readings = [reading.model_dump()]
    results = service.predict(readings)
    log_predictions(readings, results)
    return results[0]


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchRequest):
    """Score many machines at once (a fleet). Returns results in input order."""
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not request.readings:
        raise HTTPException(status_code=400, detail="No readings provided.")
    readings = [r.model_dump() for r in request.readings]
    results = service.predict(readings)
    log_predictions(readings, results)
    return results
