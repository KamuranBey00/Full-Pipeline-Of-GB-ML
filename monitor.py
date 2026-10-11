"""
monitor.py
==========
Phase 8 — data drift monitoring with Evidently.

A model is only as good as the match between the data it was trained on and the
data it sees in production. If sensors start reading differently (new machines,
recalibration, worn tooling across the fleet), predictions silently degrade.

This script compares:
  - REFERENCE : the training split (exactly what the model learned from)
  - CURRENT   : the raw readings the API has scored (logs/predictions.jsonl)
and writes an Evidently HTML report + prints which features drifted.

Run:
  python monitor.py              # compare against real API traffic
  python monitor.py --simulate   # no traffic yet? shift the test split to demo drift
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

from config import (RANDOM_STATE, TEST_SIZE, TARGET, NUMERIC, CATEGORICAL,
                    PREDICTION_LOG, DRIFT_REPORT)
from data import load_data
from api.service import FIELD_TO_COLUMN

# Monitor the RAW inputs the API receives; engineered features derive from them.
COLUMNS = CATEGORICAL + NUMERIC


def reference_and_test():
    """Re-create train.py's exact split (same seed + stratify) -> train/test inputs."""
    df = load_data()
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                   random_state=RANDOM_STATE)
    return train[COLUMNS], test[COLUMNS]


def load_current() -> pd.DataFrame:
    if not PREDICTION_LOG.exists():
        raise SystemExit(f"No prediction log at {PREDICTION_LOG}. Call the API "
                         f"first, or run with --simulate.")
    logged = pd.read_json(PREDICTION_LOG, lines=True)
    return logged.rename(columns=FIELD_TO_COLUMN)[COLUMNS]


def simulate_drift(test: pd.DataFrame) -> pd.DataFrame:
    """Shift unseen test data the way a real fleet might drift: hotter ambient
    air and more worn tools. The report should flag exactly these columns."""
    cur = test.copy()
    cur["Air temperature [K]"] += 3.0
    cur["Tool wear [min]"] += 60
    return cur


def run(current: pd.DataFrame, reference: pd.DataFrame) -> dict:
    definition = DataDefinition(numerical_columns=NUMERIC,
                                categorical_columns=CATEGORICAL)
    # Statistical tests (p-values) instead of Evidently's default distance
    # metrics: distances ignore sample size, so a small batch of perfectly normal
    # API traffic was flagged as drift. KS / chi-square account for it.
    report = Report([DataDriftPreset(num_method="ks", cat_method="chisquare")])
    snapshot = report.run(Dataset.from_pandas(current, data_definition=definition),
                          Dataset.from_pandas(reference, data_definition=definition))
    DRIFT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(DRIFT_REPORT))
    return snapshot.dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--simulate", action="store_true",
                        help="use a shifted test split instead of the API log")
    args = parser.parse_args()

    reference, test = reference_and_test()
    current = simulate_drift(test) if args.simulate else load_current()
    print(f"Reference rows: {len(reference)} | Current rows: {len(current)}")

    result = run(current, reference)
    print(f"\n{'feature':<26}{'test':<12}{'p-value':>8}  drift?")
    for m in result["metrics"]:
        cfg = m["config"]
        if "column" not in cfg:                     # the overall summary metric
            summary = m["value"]
            continue
        score = float(m["value"])                   # p-value (KS / chi-square)
        drifted = score < cfg["threshold"]
        print(f"{cfg['column']:<26}{cfg['method']:<12}{score:>8.3f}  "
              f"{'YES' if drifted else 'no'}")
    print(f"\nDrifted features: {int(summary['count'])}/{len(COLUMNS)} "
          f"({summary['share']:.0%})")
    print(f"Report -> {DRIFT_REPORT}")


if __name__ == "__main__":
    main()
