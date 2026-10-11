"""
features.py
===========
Turn the raw dataframe into model-ready (X, y) and build a preprocessing
transformer. Kept separate from training so the SAME transformation is reused
everywhere (training, evaluation, and later the API) — no drift between them.

Two responsibilities:
  1) add_domain_features(): create physics-informed features from the sensors.
  2) make_xy() + build_preprocessor(): drop leakage/ID columns, one-hot encode.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from config import TARGET, FAILURE_MODES, ID_COLS, CATEGORICAL, NUMERIC


# ---------------------------------------------------------------------------
# 1) Domain feature engineering
# ---------------------------------------------------------------------------
# WHY: the dataset's failure modes are DEFINED by physical relationships between
# raw sensors. Giving the model those same quantities directly (instead of hoping
# it re-discovers them) usually improves recall on rare failures. Each feature
# below maps to a real failure mode described in the dataset documentation.
ENGINEERED = ["temp_diff [K]", "power [W]", "wear_x_torque"]


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Heat Dissipation Failure (HDF) depends on the gap between process and air
    # temperature. A small/abnormal gap => poor heat dissipation. So we encode
    # that gap explicitly instead of leaving two correlated temperatures.
    df["temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Power Failure (PWF) is defined by mechanical power leaving a safe band.
    # Power = torque * angular_speed;  angular_speed(rad/s) = rpm * 2*pi / 60.
    df["power [W]"] = (df["Torque [Nm]"]
                       * df["Rotational speed [rpm]"] * 2 * 3.14159 / 60)

    # Overstrain Failure (OSF) is defined by (tool_wear * torque) exceeding a
    # threshold. We hand the model that exact product.
    df["wear_x_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    return df


# ---------------------------------------------------------------------------
# 2) Build X / y  (drop the columns EDA proved we must remove)
# ---------------------------------------------------------------------------
def make_xy(df: pd.DataFrame):
    df = add_domain_features(df)

    y = df[TARGET].astype(int)                      # binary target

    # Drop three groups, each for a concrete reason proven in EDA:
    #  - FAILURE_MODES : LEAKAGE (they equal the target 99.7% of the time)
    #  - ID_COLS       : pure identifiers, no predictive signal (avoid memorising)
    #  - TARGET        : obviously not an input
    drop_cols = FAILURE_MODES + ID_COLS + [TARGET]
    X = df.drop(columns=drop_cols)
    return X, y


# ---------------------------------------------------------------------------
# 3) Preprocessing transformer
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """One-hot encode the categorical 'Type'; pass numerics through untouched.

    WHY passthrough (no scaling)? Our models are tree-based (Random Forest,
    Gradient Boosting). Trees split on thresholds, so they are invariant to
    feature scale — StandardScaler would add complexity for zero benefit.
    """
    numeric_features = NUMERIC + ENGINEERED
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
