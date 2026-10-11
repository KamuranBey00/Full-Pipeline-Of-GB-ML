"""
explain.py
==========
Phase 3 — Explainability. A model that a business can trust must answer
"WHY did you predict a failure?". We answer at two levels:

  GLOBAL (which features drive the model overall):
    1. Permutation importance — model-agnostic, measured on our real metric
       (PR-AUC). The most honest global ranking.
    2. SHAP summary (beeswarm) — shows direction + magnitude of each feature's
       effect across the test set.

  LOCAL (why this ONE machine was flagged):
    3. SHAP waterfall for a single prediction — the kind of per-alert explanation
       a maintenance engineer would actually receive.

A second goal: verify whether our engineered features (power, wear_x_torque,
temp_diff) actually earned their place, or were just noise.

Run:  python explain.py    (requires a trained models/best_model.joblib)
"""

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from config import RANDOM_STATE, TEST_SIZE, MODELS_DIR, FIGURES_DIR
from data import load_data
from features import make_xy


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain]   saved -> {path}")


def load_artifacts():
    """Load the trained pipeline and reproduce the SAME test split used in train.py.

    We reuse RANDOM_STATE + TEST_SIZE so the test rows are identical — otherwise
    our explanations would be computed on data the model may have trained on.
    """
    bundle = joblib.load(MODELS_DIR / "best_model.joblib")
    pipe = bundle["pipeline"]
    print(f"[explain] Loaded model: {bundle['model_name']} "
          f"(tuned threshold={bundle['threshold']:.3f})")

    df = load_data()
    X, y = make_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return pipe, X_te, y_te, bundle


# ---------------------------------------------------------------------------
# 1) Permutation importance (model-agnostic, on the pipeline, on our metric)
# ---------------------------------------------------------------------------
def permutation_global(pipe, X_te, y_te):
    """Shuffle each ORIGINAL input column and measure how much PR-AUC drops.

    Big drop => the model relies on that feature. This runs on the full pipeline
    with the ORIGINAL columns, so 'Type' counts as one interpretable feature."""
    print("\n=== 1) PERMUTATION IMPORTANCE (drop in PR-AUC when shuffled) ===")
    result = permutation_importance(
        pipe, X_te, y_te, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)

    order = result.importances_mean.argsort()[::-1]
    for i in order:
        print(f"  {X_te.columns[i]:24s} "
              f"{result.importances_mean[i]:+.4f} ± {result.importances_std[i]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(X_te.columns)[order][::-1],
            result.importances_mean[order][::-1],
            xerr=result.importances_std[order][::-1], color="#4C72B0")
    ax.set_xlabel("Mean PR-AUC drop when feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    _save(fig, "07_permutation_importance.png")
    return [X_te.columns[i] for i in order]


# ---------------------------------------------------------------------------
# 2) & 3) SHAP (global beeswarm + local waterfall)
# ---------------------------------------------------------------------------
def shap_explanations(pipe, X_te, y_te):
    """TreeExplainer on the boosted-tree step.

    SHAP needs the matrix the MODEL sees, so we transform X through the fitted
    preprocessor first, and pull the post-encoding feature names to label plots.
    """
    print("\n=== 2) SHAP (global + local) ===")
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    X_trans = prep.transform(X_te)
    feature_names = list(prep.get_feature_names_out())
    # clean the ColumnTransformer prefixes ('num__', 'cat__') for readability
    feature_names = [f.split("__", 1)[-1] for f in feature_names]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)
    # For sklearn GradientBoosting (single output) shap_values is a 2D array;
    # some versions return a list — normalise to a 2D array for the positive class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # --- global beeswarm: direction + magnitude of each feature ---
    fig = plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    _save(plt.gcf(), "08_shap_summary.png")

    # --- global bar: mean |SHAP| ranking ---
    fig = plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names,
                      plot_type="bar", show=False)
    _save(plt.gcf(), "09_shap_bar.png")

    # --- local: explain one real failure the model correctly caught ---
    proba = pipe.predict_proba(X_te)[:, 1]
    y_te_arr = y_te.to_numpy()
    # pick a true positive with the highest confidence (clearest example)
    tp_mask = (y_te_arr == 1) & (proba >= 0.5)
    if tp_mask.any():
        idx = np.where(tp_mask)[0][np.argmax(proba[tp_mask])]
        # expected_value can be a scalar or a 1-element array depending on the
        # SHAP/model version; waterfall needs a plain float, so we coerce it.
        base_val = float(np.ravel(explainer.expected_value)[0])
        expl = shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_trans[idx],
            feature_names=feature_names)
        fig = plt.figure()
        shap.plots.waterfall(expl, show=False)
        _save(plt.gcf(), "10_shap_local_failure.png")
        print(f"  Local example: test row {idx}, model p(failure)={proba[idx]:.3f}")

    # report which features SHAP ranks highest, and flag engineered ones
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs), key=lambda t: t[1], reverse=True)
    print("  SHAP global ranking (mean |SHAP|):")
    engineered = {"power [W]", "wear_x_torque", "temp_diff [K]"}
    for name, val in ranking:
        tag = "  <-- engineered" if name in engineered else ""
        print(f"    {name:24s} {val:.4f}{tag}")
    return ranking


def main():
    pipe, X_te, y_te, bundle = load_artifacts()
    permutation_global(pipe, X_te, y_te)
    shap_explanations(pipe, X_te, y_te)
    print("\n[explain] Done. See reports/figures/07..10 for the plots.")
    print("Takeaway: check whether the engineered features rank highly — if they "
          "do, our physics-informed feature engineering paid off.")


if __name__ == "__main__":
    main()
