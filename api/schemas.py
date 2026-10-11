"""
api/schemas.py
==============
Request/response data contracts, validated by Pydantic.

WHY: the API must reject bad input BEFORE it reaches the model. Pydantic turns
the JSON body into a typed, range-checked object automatically and returns a
clear 422 error on violations — no manual validation code, no garbage-in.

We deliberately use clean field names (air_temperature, not "Air temperature [K]")
so the JSON API is tidy; service.py maps them to the dataset's column names.
"""

from typing import Literal, List
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """One machine's live sensor snapshot — the RAW inputs a machine streams.

    The engineered features (power, temp_diff, wear_x_torque) are NOT accepted
    here: they are derived server-side so the client can't get them wrong and so
    training and serving compute them identically.
    """
    # 'Type' is the product quality variant. Literal restricts it to valid values.
    type: Literal["L", "M", "H"] = Field(..., description="Product quality: L/M/H")

    # Ranges below are generous but sane, based on the observed data ranges.
    # ge/le = inclusive bounds; the API returns 422 if a value is outside them.
    air_temperature: float = Field(..., ge=290, le=310, description="Air temp [K]")
    process_temperature: float = Field(..., ge=300, le=320, description="Process temp [K]")
    rotational_speed: float = Field(..., ge=1000, le=3500, description="Speed [rpm]")
    torque: float = Field(..., ge=0, le=100, description="Torque [Nm]")
    tool_wear: float = Field(..., ge=0, le=300, description="Tool wear [min]")

    # A ready-to-use example shown in the auto-generated /docs page.
    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "L", "air_temperature": 298.1, "process_temperature": 308.6,
                "rotational_speed": 1551, "torque": 42.8, "tool_wear": 108,
            }
        }
    }


class BatchRequest(BaseModel):
    """Score many machines in one call (realistic for a fleet)."""
    # Upper bound protects the public API: an unbounded list lets one request
    # exhaust server memory (DoS). 1000 machines per call is plenty for a fleet.
    readings: List[SensorReading] = Field(..., max_length=1000)


class PredictionResponse(BaseModel):
    """Structured, typed output — clients can rely on this shape."""
    failure_predicted: bool          # decision after applying the tuned threshold
    failure_probability: float       # calibrated-ish P(failure) in [0, 1]
    risk_level: Literal["low", "medium", "high"]
    threshold_used: float            # which cut-off produced the decision
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    threshold: float | None = None
