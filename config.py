"""
config.py
=========
Central configuration for the whole project.

WHY a single config file?
- We never hard-code paths, column names, or hyper-parameters inside logic files.
  Keeping them in ONE place makes the project reproducible and easy to change:
  a reviewer (or future you) can see every knob of the system at a glance.
- This is a small but strong "production mindset" signal for a portfolio project.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — everything is relative to the project root, so the repo is portable
# (works on any machine, unlike an absolute path like C:\Users\...).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[0]          # project root folder
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = DATA_DIR / "ai4i2020.csv"                 # downloaded raw dataset

# ---------------------------------------------------------------------------
# Dataset schema — AI4I 2020 Predictive Maintenance (UCI).
# We name columns explicitly so the rest of the code never relies on guesswork.
# ---------------------------------------------------------------------------
TARGET = "Machine failure"                          # binary target: 1 = failure

# The five failure-mode flags. IMPORTANT (leakage!): these are only "true"
# BECAUSE the machine already failed. If we fed them to the model, it would be
# cheating — they are basically the answer. So we use them ONLY to understand
# the data, then DROP them before training.
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Identifier columns carry no predictive signal (they are just row IDs / serials),
# so they are dropped as well to avoid the model "memorising" rows.
ID_COLS = ["UDI", "Product ID"]

# The single categorical feature (product quality variant: Low / Medium / High).
CATEGORICAL = ["Type"]

# The continuous sensor measurements — the real inputs a machine would stream.
NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ---------------------------------------------------------------------------
# Experiment settings — fixed seed so results are reproducible run-to-run.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2                                     # 80/20 train/test split

# ---------------------------------------------------------------------------
# MLflow tracking (Phase 4) — local SQLite backend under the repo, so `mlflow
# ui` works out of the box with zero external server. (Plain './mlruns'
# filesystem tracking is deprecated as of MLflow 3.x — SQLite is the
# supported local, serverless option.) Run artifacts (models, figures) still
# land as files under mlruns/, pointed to via MLRUNS_DIR below.
# Both mlflow.db and mlruns/ are git-ignored, same treatment as data/ and models/.
# ---------------------------------------------------------------------------
MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_DB_PATH = ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "predictive-maintenance"
