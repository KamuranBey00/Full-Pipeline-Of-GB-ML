import pytest

from conftest import RISKY, NORMAL, requires_model
from service import PredictionService

pytestmark = requires_model


@pytest.fixture(scope="module")
def svc():
    return PredictionService().load()


def test_output_shape_and_ranges(svc):
    out = svc.predict([NORMAL, RISKY])
    assert len(out) == 2
    for r in out:
        assert 0.0 <= r["failure_probability"] <= 1.0
        assert r["failure_predicted"] == (r["failure_probability"] >= svc.threshold)
        assert r["risk_level"] in {"low", "medium", "high"}


def test_risky_vs_normal(svc):
    normal, risky = svc.predict([NORMAL, RISKY])
    assert risky["failure_predicted"] and risky["risk_level"] == "high"
    assert not normal["failure_predicted"]


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PredictionService(model_path=tmp_path / "nope.joblib").load()
