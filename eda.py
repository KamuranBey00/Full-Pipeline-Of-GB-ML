"""
eda.py
======
Exploratory Data Analysis (EDA).

GOAL of EDA in a real project: BEFORE we model, we must understand the data.
Specifically for this dataset we want to answer:
  1. How imbalanced is the target? (drives our metric choice later)
  2. Which failure modes exist and how frequent are they?
  3. Are there data-quality traps (missing values, leakage, useless columns)?
  4. How do sensor features relate to failure? (early feature intuition)

Every finding here is printed AND saved as a figure in reports/figures/,
so the repo tells a story a reviewer can follow without running anything.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend: just save PNGs
import matplotlib.pyplot as plt

from config import (TARGET, FAILURE_MODES, NUMERIC, CATEGORICAL,
                    FIGURES_DIR, REPORTS_DIR)
from data import load_data


def _save(fig, name: str) -> None:
    """Save a figure to reports/figures and close it to free memory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda]   saved figure -> {path.relative_to(REPORTS_DIR.parent)}")


def basic_overview(df: pd.DataFrame) -> None:
    """Shape, dtypes, missing values — the very first thing to check."""
    print("\n=== 1) OVERVIEW ===")
    print(f"Rows x Cols : {df.shape[0]} x {df.shape[1]}")
    print(f"Duplicates  : {df.duplicated().sum()}")   # exact duplicate rows
    missing = df.isna().sum()
    print(f"Missing vals: {int(missing.sum())} total")
    # WHY it matters: if there were missing values we'd need an imputation
    # strategy. AI4I is clean, but we prove it rather than assume it.


def target_balance(df: pd.DataFrame) -> None:
    """How rare are failures? This single number dictates our whole strategy."""
    print("\n=== 2) TARGET BALANCE ===")
    counts = df[TARGET].value_counts().sort_index()
    rate = df[TARGET].mean()
    print(counts.to_string())
    print(f"Failure rate: {rate:.3%}")
    # KEY INSIGHT: ~3.4% failures = severe class imbalance. This is exactly why
    # accuracy is a misleading metric here (predicting 'no failure' always
    # scores ~96.6%). We will instead rely on PR-AUC / recall / F1 later.

    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["No failure (0)", "Failure (1)"], rotation=0)
    ax.set_title(f"Target balance (failure rate = {rate:.1%})")
    ax.set_ylabel("count")
    _save(fig, "01_target_balance.png")


def failure_modes(df: pd.DataFrame) -> None:
    """Break down the five independent failure modes (multi-label nature)."""
    print("\n=== 3) FAILURE MODES ===")
    mode_counts = df[FAILURE_MODES].sum().sort_values(ascending=False)
    print(mode_counts.to_string())
    # NOTE on RNF (Random Failure): by definition it fires with a fixed 0.1%
    # probability REGARDLESS of sensor values. It is pure noise with no learnable
    # signal, so later we may treat it carefully. Seeing it here justifies that.

    fig, ax = plt.subplots(figsize=(6, 4))
    mode_counts.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Failure mode frequency")
    ax.set_ylabel("count")
    _save(fig, "02_failure_modes.png")


def leakage_check(df: pd.DataFrame) -> None:
    """Confirm that the failure-mode flags essentially equal the target.

    WHY: this demonstrates data leakage concretely. If ANY mode flag is 1, the
    machine failed. Feeding these flags to a model would give ~100% 'accuracy'
    that collapses in production (the flags don't exist before failure). Proving
    it here is exactly the kind of reasoning an interviewer wants to see.
    """
    print("\n=== 4) LEAKAGE CHECK ===")
    any_mode = df[FAILURE_MODES].max(axis=1)          # 1 if any mode fired
    agreement = (any_mode == df[TARGET]).mean()
    print(f"'any failure mode' matches target in {agreement:.3%} of rows")
    print("-> These flags are the answer in disguise; we DROP them for modeling.")


def numeric_by_target(df: pd.DataFrame) -> None:
    """Compare sensor distributions for failed vs healthy machines."""
    print("\n=== 5) SENSORS vs FAILURE (mean per group) ===")
    grouped = df.groupby(TARGET)[NUMERIC].mean().T
    grouped.columns = ["healthy(0)", "failed(1)"]
    print(grouped.round(2).to_string())
    # EARLY INTUITION: features whose means differ most between the two groups
    # (e.g. torque, tool wear, rotational speed) are likely the most predictive.

    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(4 * len(NUMERIC), 4))
    for ax, col in zip(axes, NUMERIC):
        for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
            ax.hist(df.loc[df[TARGET] == label, col], bins=30, alpha=0.6,
                    density=True, color=color, label=str(label))
        ax.set_title(col, fontsize=9)
        ax.legend(title="failure")
    fig.suptitle("Sensor distributions: healthy vs failed")
    _save(fig, "03_sensor_distributions.png")


def correlations(df: pd.DataFrame) -> None:
    """Correlation heatmap among numeric sensors + target."""
    print("\n=== 6) CORRELATIONS ===")
    cols = NUMERIC + [TARGET]
    corr = df[cols].corr()
    print(corr[TARGET].sort_values(ascending=False).round(3).to_string())
    # Air & process temperature are physically linked, so we expect them to
    # correlate strongly (possible redundancy -> a feature-engineering idea:
    # use their difference as a single feature).

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation matrix")
    _save(fig, "04_correlation_matrix.png")


def category_effect(df: pd.DataFrame) -> None:
    """Does product quality Type (L/M/H) affect failure rate?"""
    print("\n=== 7) FAILURE RATE BY TYPE ===")
    rate = df.groupby(CATEGORICAL[0])[TARGET].mean().sort_values()
    print((rate * 100).round(2).astype(str).add(" %").to_string())
    # If Low-quality parts fail more, 'Type' is a useful categorical feature
    # we must one-hot encode before modeling.


def run_all() -> None:
    df = load_data()
    basic_overview(df)
    target_balance(df)
    failure_modes(df)
    leakage_check(df)
    numeric_by_target(df)
    correlations(df)
    category_effect(df)
    print("\n[eda] Done. Figures are in reports/figures/.")


if __name__ == "__main__":
    run_all()
