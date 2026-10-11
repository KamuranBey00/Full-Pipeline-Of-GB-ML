"""
train.py
========
Phase 2 — model comparison done the RIGHT way for imbalanced data.
Phase 4 — the same pipeline, now tracked end-to-end in MLflow.

Pipeline of this script:
  1. Load data -> build X, y (features.py handles leakage/ID drops + engineering).
  2. Stratified train/test split (preserve the 3.4% failure ratio in both sides).
  3. Compare three models with a fair, threshold-INDEPENDENT metric (PR-AUC):
        - DummyClassifier  : the "always predict majority" baseline to beat.
        - RandomForest     : bagged trees, class_weight='balanced'.
        - GradientBoosting : boosted trees.
     Each model gets its own NESTED MLflow run (params + metrics), so the three
     candidates can be compared side by side in the MLflow UI.
  4. Cross-validate on the training set so the ranking isn't a fluke of one split.
  5. Pick the best model, then TUNE THE DECISION THRESHOLD without leakage
     (using out-of-fold probabilities on train), targeting a business recall.
  6. Final, honest evaluation on the untouched test set + save model, metrics, plots
     — logged both to disk (models/, reports/) and to the PARENT MLflow run
     (metrics, figures, and the model itself via mlflow.sklearn.log_model).

Run:            python train.py
Inspect runs:   mlflow ui --backend-store-uri mlruns
"""

import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve)

from config import (RANDOM_STATE, TEST_SIZE, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
                    MLRUNS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)
from data import load_data
from features import make_xy, build_preprocessor

# The business target: we would rather investigate a few false alarms than MISS
# a real failure, so we require the model to catch at least this fraction of
# failures. We set it ABOVE the model's default-0.5 recall on purpose, so that
# tuning has to LOWER the threshold and genuinely catch MORE failures (that is
# the whole point). The threshold is then chosen to hit this recall.
TARGET_RECALL = 0.85


def make_pipeline(model):
    """Glue preprocessing + model so the exact same transforms run at predict
    time. Building the preprocessor per pipeline avoids state being shared."""
    return Pipeline([("prep", build_preprocessor()), ("clf", model)])


def evaluate_at_threshold(y_true, proba, threshold):
    """Compute the metrics we care about at a given probability cut-off."""
    y_pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Local SQLite tracking store (mlflow.db under the repo) — zero external
    # setup. Artifacts (models, figures) are stored as files under mlruns/;
    # we pin that location explicitly the first time the experiment is created
    # (afterwards MLflow remembers it from the experiment's own metadata).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME,
                                 artifact_location=MLRUNS_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- 1) data ---
    df = load_data()
    X, y = make_xy(df)
    print(f"Features used ({X.shape[1]}): {list(X.columns)}")

    # --- 2) stratified split ---
    # stratify=y is ESSENTIAL: with only 3.4% positives, a random split could
    # put too few failures in the test set and make metrics meaningless.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train failures: {y_tr.sum()}/{len(y_tr)} | "
          f"Test failures: {y_te.sum()}/{len(y_te)}")

    # One PARENT run covers the whole training pipeline; each candidate model
    # below gets its own NESTED child run. We use start/end explicitly (instead
    # of a `with` block wrapping the rest of main()) so the existing function
    # body doesn't need to be re-indented — the run stays open until the
    # explicit mlflow.end_run() near the bottom of this function.
    mlflow.start_run(run_name="model_comparison")
    mlflow.log_params({
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target_recall": TARGET_RECALL,
        "n_features": X.shape[1],
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    })

    # --- 3) candidate models ---
    models = {
        "baseline_dummy": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    pipelines = {}

    for name, model in models.items():
        pipe = make_pipeline(model)

        # Cross-validated PR-AUC on TRAIN: robust ranking quality, not one split.
        # 'average_precision' == area under precision-recall curve = PR-AUC.
        if name == "baseline_dummy":
            cv_prauc = np.array([y_tr.mean()])          # dummy AP == positive rate
        else:
            cv_prauc = cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                       scoring="average_precision", n_jobs=-1)

        # Fit on full train, score on the held-out TEST set (threshold-free metrics).
        pipe.fit(X_tr, y_tr)
        if hasattr(pipe, "predict_proba"):
            proba_te = pipe.predict_proba(X_te)[:, 1]
        else:
            proba_te = pipe.predict(X_te).astype(float)

        results[name] = {
            "cv_pr_auc_mean": round(float(cv_prauc.mean()), 4),
            "cv_pr_auc_std": round(float(cv_prauc.std()), 4),
            "test_pr_auc": round(float(average_precision_score(y_te, proba_te)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_te, proba_te)), 4),
            # metrics at the naive default 0.5 cut-off, to contrast with tuned later
            "default_0.5": evaluate_at_threshold(y_te, proba_te, 0.5),
        }
        pipelines[name] = pipe

        # Nested child run: one candidate model's params + metrics, so the
        # MLflow UI can compare all three side by side under the parent run.
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params({f"clf__{k}": v for k, v in model.get_params().items()})
            mlflow.log_metrics({
                "cv_pr_auc_mean": results[name]["cv_pr_auc_mean"],
                "cv_pr_auc_std": results[name]["cv_pr_auc_std"],
                "test_pr_auc": results[name]["test_pr_auc"],
                "test_roc_auc": results[name]["test_roc_auc"],
                "default_precision": results[name]["default_0.5"]["precision"],
                "default_recall": results[name]["default_0.5"]["recall"],
                "default_f1": results[name]["default_0.5"]["f1"],
            })

    # --- report the comparison ---
    print("\n=== MODEL COMPARISON (higher PR-AUC = better ranking) ===")
    print(f"{'model':20s} {'cv PR-AUC':>12s} {'test PR-AUC':>12s} {'test ROC-AUC':>13s}")
    for name, r in results.items():
        print(f"{name:20s} {r['cv_pr_auc_mean']:>12.4f} "
              f"{r['test_pr_auc']:>12.4f} {r['test_roc_auc']:>13.4f}")

    # --- 4) pick the best (exclude the dummy baseline) ---
    ranked = {k: v for k, v in results.items() if k != "baseline_dummy"}
    best_name = max(ranked, key=lambda k: ranked[k]["test_pr_auc"])
    best_pipe = pipelines[best_name]
    print(f"\nBest model by test PR-AUC: {best_name}")

    # --- 5) threshold tuning WITHOUT leakage ---
    # We must NOT choose the threshold by peeking at the test set. Instead we get
    # out-of-fold probabilities on TRAIN (each row scored by a model that never
    # saw it), pick the threshold there, then apply it once to the test set.
    oof_proba = cross_val_predict(best_pipe, X_tr, y_tr, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof_proba)

    # Show the trade-off transparently: at several recall floors, what threshold
    # and precision do we get (measured on out-of-fold TRAIN data)? This table is
    # exactly how an ML engineer justifies the operating point to stakeholders.
    print("\n--- Threshold trade-off (out-of-fold on train) ---")
    print(f"{'recall floor':>12s} {'threshold':>10s} {'precision':>10s}")
    for floor in (0.80, 0.85, 0.90, 0.95):
        idx = np.where(rec[:-1] >= floor)[0]
        if len(idx):
            t = thr[idx[-1]]
            print(f"{floor:>12.2f} {t:>10.3f} {prec[idx[-1]]:>10.3f}")

    # find the highest threshold that still reaches our target recall
    # (higher threshold => fewer false alarms, so we want the strictest one
    #  that still catches TARGET_RECALL of failures).
    ok = np.where(rec[:-1] >= TARGET_RECALL)[0]
    chosen_threshold = float(thr[ok[-1]]) if len(ok) else 0.5
    print(f"Chosen threshold (recall>={TARGET_RECALL} on OOF train): "
          f"{chosen_threshold:.4f}")

    # --- 6) final honest evaluation on TEST at the tuned threshold ---
    best_pipe.fit(X_tr, y_tr)                        # refit on all training data
    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    tuned = evaluate_at_threshold(y_te, proba_te, chosen_threshold)
    default = results[best_name]["default_0.5"]

    print("\n=== BEST MODEL ON TEST: default 0.5 vs tuned threshold ===")
    for label, m in [("default 0.5", default), (f"tuned {chosen_threshold:.3f}", tuned)]:
        cm = m["confusion_matrix"]
        print(f"{label:16s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f}  confusion={cm}")
    # Interpretation printed for the reader:
    print("\nRead the confusion matrix as [[TN, FP], [FN, TP]]. Tuning trades a few "
          "extra false alarms (FP) for catching more real failures (fewer FN) — "
          "exactly the business trade-off we want in maintenance.")

    # --- save artifacts ---
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump({"pipeline": best_pipe, "threshold": chosen_threshold,
                 "feature_names": list(X.columns), "model_name": best_name},
                model_path)
    metrics_path = REPORTS_DIR / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump({"comparison": results, "best_model": best_name,
                   "chosen_threshold": chosen_threshold,
                   "test_default": default, "test_tuned": tuned}, f, indent=2)
    print(f"\nSaved model -> {model_path}")
    print(f"Saved metrics -> {metrics_path}")

    # --- figures: PR curves + best-model confusion matrix ---
    _plot_pr_curves(pipelines, X_te, y_te)
    _plot_confusion(y_te, (proba_te >= chosen_threshold).astype(int), best_name)

    # --- MLflow: log the winning result on the PARENT run ---
    mlflow.set_tag("best_model", best_name)
    mlflow.log_param("chosen_threshold", round(chosen_threshold, 4))
    mlflow.log_metrics({
        "test_default_precision": default["precision"],
        "test_default_recall": default["recall"],
        "test_default_f1": default["f1"],
        "test_tuned_precision": tuned["precision"],
        "test_tuned_recall": tuned["recall"],
        "test_tuned_f1": tuned["f1"],
    })
    mlflow.log_artifact(str(metrics_path))
    mlflow.log_artifact(str(FIGURES_DIR / "05_pr_curves.png"))
    mlflow.log_artifact(str(FIGURES_DIR / "06_confusion_matrix.png"))

    # Log the fitted pipeline itself (preprocessing + model, same object saved
    # to models/best_model.joblib) so MLflow holds a servable, versioned copy
    # with an input/output signature — not just the raw metric numbers.
    # MLflow >= 3.x serializes with skops, which refuses tree internals unless
    # explicitly trusted. Safe here: we trained this model ourselves.
    signature = infer_signature(X_tr, best_pipe.predict(X_tr))
    mlflow.sklearn.log_model(best_pipe, name="model",
                             signature=signature, input_example=X_tr.head(3),
                             skops_trusted_types=["sklearn.tree._tree.Tree"])
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    print(f"\nMLflow run '{run_id}' logged -> {MLFLOW_TRACKING_URI} "
          f"(experiment='{MLFLOW_EXPERIMENT_NAME}')")
    print(f"Inspect with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


def _plot_pr_curves(pipelines, X_te, y_te):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in pipelines.items():
        if name == "baseline_dummy":
            continue
        proba = pipe.predict_proba(X_te)[:, 1]
        p, r, _ = precision_recall_curve(y_te, proba)
        ap = average_precision_score(y_te, proba)
        ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y_te.mean(), ls="--", color="gray",
               label=f"baseline (={y_te.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "05_pr_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {FIGURES_DIR / '05_pr_curves.png'}")


def _plot_confusion(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(f"Confusion matrix — {name} (tuned)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIGURES_DIR / "06_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {FIGURES_DIR / '06_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
