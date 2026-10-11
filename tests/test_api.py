import pytest
from fastapi.testclient import TestClient

from conftest import RISKY, NORMAL, requires_model
from api import main

client = TestClient(main.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] == main.service.is_ready


@requires_model
def test_predict():
    r = client.post("/predict", json=RISKY)
    assert r.status_code == 200
    assert r.json()["failure_predicted"] is True


@requires_model
def test_predict_batch_keeps_order():
    r = client.post("/predict/batch", json={"readings": [NORMAL, RISKY]})
    assert r.status_code == 200
    assert [p["failure_predicted"] for p in r.json()] == [False, True]


@pytest.mark.parametrize("bad", [
    {**RISKY, "type": "X"},            # invalid category
    {**RISKY, "torque": 500},          # out of range
    {k: v for k, v in RISKY.items() if k != "tool_wear"},  # missing field
])
def test_invalid_input_422(bad):
    assert client.post("/predict", json=bad).status_code == 422


@requires_model
def test_empty_batch_400():
    assert client.post("/predict/batch", json={"readings": []}).status_code == 400


def test_model_not_loaded_503(monkeypatch):
    monkeypatch.setattr(main.service, "pipeline", None)
    assert client.post("/predict", json=RISKY).status_code == 503
    assert client.get("/health").json()["status"] == "degraded"


@requires_model
def test_predictions_are_logged(tmp_prediction_log):
    client.post("/predict/batch", json={"readings": [NORMAL, RISKY]})
    lines = tmp_prediction_log.read_text().splitlines()
    assert len(lines) == 2
    assert '"tool_wear": 220' in lines[1] and "failure_probability" in lines[1]


def test_oversized_batch_422():
    assert client.post("/predict/batch",
                       json={"readings": [NORMAL] * 1001}).status_code == 422
