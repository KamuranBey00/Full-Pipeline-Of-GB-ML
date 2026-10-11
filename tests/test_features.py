import math

import pandas as pd

from features import add_domain_features, ENGINEERED


def _raw():
    return pd.DataFrame([{
        "Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100,
    }])


def test_engineered_values():
    row = add_domain_features(_raw()).iloc[0]
    assert row["temp_diff [K]"] == 10.5
    assert math.isclose(row["power [W]"], 40.0 * 1500 * 2 * math.pi / 60, rel_tol=1e-4)
    assert row["wear_x_torque"] == 4000.0


def test_does_not_mutate_input():
    raw = _raw()
    out = add_domain_features(raw)
    assert set(ENGINEERED) <= set(out.columns)
    assert not set(ENGINEERED) & set(raw.columns)
