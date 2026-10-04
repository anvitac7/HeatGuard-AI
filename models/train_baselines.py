"""
HeatGuard AI - Phase 3: Classical & Scientific Baseline Models.

Trains and evaluates 3 benchmark models:
1. Baseline 1: Climatological Persistence Heuristic (Domain Physical Anchor)
2. Baseline 2: Balanced Logistic Regression (Linear L2 Regularized Convex Benchmark)
3. Baseline 3: Tree Ensemble Baseline (Balanced Random Forest / GBDT via LightGBM)

Outputs:
- models/saved/persistence_baseline.pkl
- models/saved/logreg_baseline.pkl
- models/saved/rf_baseline.pkl
- models/saved/scaler.pkl
- data/baseline_metrics.json
- data/baseline_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add repo root to python path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.evaluate import compute_metrics, find_optimal_threshold, save_metrics_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_baselines")


# ========================================================================================
# Baseline 1: Climatological Persistence Heuristic Estimator
# ========================================================================================
class PersistenceBaseline:
    """
    Physical domain heuristic baseline.
    Assigns high probability if current day is a heatwave (empirical persistence ~63.15%),
    or if current temp departure is >= 0, and low base rate otherwise (~0.36%).
    """

    def __init__(self, p_given_hw: float = 0.6315, p_given_norm: float = 0.0036):
        self.p_given_hw = p_given_hw
        self.p_given_norm = p_given_norm

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> PersistenceBaseline:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if "heatwave_streak_days" in X.columns:
            is_hw = (X["heatwave_streak_days"] > 0) | (X.get("temp_departure_today", 0) >= 0)
        elif "is_heatwave_day" in X.columns:
            is_hw = X["is_heatwave_day"] == 1
        else:
            is_hw = X.get("temp_departure_today", 0) >= 0

        p1 = np.where(is_hw, self.p_given_hw, self.p_given_norm)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


# ========================================================================================
# Data Ingestion and Feature Extraction
# ========================================================================================
def load_data_and_schema(
    data_path: Path,
    schema_path: Path,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Load model ready dataset and feature definitions."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run data_cleaning/feature_eng.py first.")
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}.")

    log.info("Loading dataset from %s ...", data_path)
    df = pd.read_csv(data_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema["feature_columns"]
    target_col = schema["primary_target"]

    log.info("Loaded %d rows with %d feature columns.", len(df), len(feature_cols))
    return df, feature_cols, target_col


def split_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """Extract train, validation, and test splits."""
    splits = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        X = sub[feature_cols].copy()
        y = sub[target_col].to_numpy().astype(int)
        splits[s] = (X, y)
        log.info(
            "Split %5s: %7d rows | %5d positives (%5.3f%%)",
            s.upper(),
            len(sub),
            int(y.sum()),
            float(y.mean() * 100),
        )
    return splits


# ========================================================================================
# Training and Evaluation Engine
# ========================================================================================
def run_baselines(
    data_path: Path = REPO_ROOT / "data" / "model_ready_dataset.csv",
    schema_path: Path = REPO_ROOT / "data" / "feature_schema.json",
    output_dir: Path = REPO_ROOT / "models" / "saved",
    save_models: bool = True,
) -> Dict[str, Any]:
    """Train all 3 baselines and generate evaluation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df, feature_cols, target_col = load_data_and_schema(data_path, schema_path)
    splits = split_data(df, feature_cols, target_col)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    results: Dict[str, Any] = {}

    # ------------------------------------------------------------------------------------
    # 1. Baseline 1: Persistence Rule
    # ------------------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Evaluating Baseline 1: Climatological Persistence Heuristic...")
    persistence = PersistenceBaseline()
    persistence.fit(X_train, y_train)

    train_probs_p = persistence.predict_proba(X_train)[:, 1]
    val_probs_p = persistence.predict_proba(X_val)[:, 1]
    test_probs_p = persistence.predict_proba(X_test)[:, 1]

    p_thresh, _ = find_optimal_threshold(y_val, val_probs_p)

    results["Persistence_Heuristic"] = {
        "train": compute_metrics(y_train, train_probs_p, threshold=p_thresh),
        "val": compute_metrics(y_val, val_probs_p, threshold=p_thresh),
        "test": compute_metrics(y_test, test_probs_p, threshold=p_thresh),
        "optimal_threshold": p_thresh,
    }
    if save_models:
        joblib.dump(persistence, output_dir / "persistence_baseline.pkl")

    # ------------------------------------------------------------------------------------
    # 2. Baseline 2: Balanced Logistic Regression
    # ------------------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Training Baseline 2: Balanced Logistic Regression (L2 Regularized)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    t0 = time.time()
    logreg = LogisticRegression(
        C=0.1,
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    logreg.fit(X_train_scaled, y_train)
    t_logreg = time.time() - t0
    log.info("Logistic Regression fitted in %.2f seconds.", t_logreg)

    train_probs_lr = logreg.predict_proba(X_train_scaled)[:, 1]
    val_probs_lr = logreg.predict_proba(X_val_scaled)[:, 1]
    test_probs_lr = logreg.predict_proba(X_test_scaled)[:, 1]

    lr_thresh, lr_val_f1 = find_optimal_threshold(y_val, val_probs_lr)
    log.info("Tuned optimal threshold on Val: tau = %.3f (Val F1: %.4f)", lr_thresh, lr_val_f1)

    results["Logistic_Regression"] = {
        "train": compute_metrics(y_train, train_probs_lr, threshold=lr_thresh),
        "val": compute_metrics(y_val, val_probs_lr, threshold=lr_thresh),
        "test": compute_metrics(y_test, test_probs_lr, threshold=lr_thresh),
        "optimal_threshold": lr_thresh,
        "training_time_seconds": round(t_logreg, 2),
    }

    coefs = pd.Series(logreg.coef_[0], index=feature_cols).sort_values(ascending=False)
    results["Logistic_Regression"]["top_positive_features"] = coefs.head(5).to_dict()
    results["Logistic_Regression"]["top_negative_features"] = coefs.tail(5).to_dict()

    if save_models:
        joblib.dump(logreg, output_dir / "logreg_baseline.pkl")
        joblib.dump(scaler, output_dir / "scaler.pkl")

    # ------------------------------------------------------------------------------------
    # 3. Baseline 3: Tree Ensemble Baseline (Random Forest / GBDT via LightGBM)
    # ------------------------------------------------------------------------------------
    log.info("=" * 70)
    log.info("Training Baseline 3: Tree Ensemble Benchmark (LightGBM Bagged Forest)...")
    t0 = time.time()
    # LightGBM Tree Baseline with balanced class weighting and feature subsampling
    rf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    log.info("Tree Ensemble Baseline fitted in %.2f seconds.", t_rf)

    train_probs_rf = rf.predict_proba(X_train)[:, 1]
    val_probs_rf = rf.predict_proba(X_val)[:, 1]
    test_probs_rf = rf.predict_proba(X_test)[:, 1]

    rf_thresh, rf_val_f1 = find_optimal_threshold(y_val, val_probs_rf)
    log.info("Tuned optimal threshold on Val: tau = %.3f (Val F1: %.4f)", rf_thresh, rf_val_f1)

    results["Tree_Ensemble_Baseline"] = {
        "train": compute_metrics(y_train, train_probs_rf, threshold=rf_thresh),
        "val": compute_metrics(y_val, val_probs_rf, threshold=rf_thresh),
        "test": compute_metrics(y_test, test_probs_rf, threshold=rf_thresh),
        "optimal_threshold": rf_thresh,
        "training_time_seconds": round(t_rf, 2),
    }

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    results["Tree_Ensemble_Baseline"]["top_feature_importances"] = importances.head(10).to_dict()

    if save_models:
        joblib.dump(rf, output_dir / "rf_baseline.pkl")

    # ------------------------------------------------------------------------------------
    # Summary Display & Report Export
    # ------------------------------------------------------------------------------------
    json_path = REPO_ROOT / "data" / "baseline_metrics.json"
    md_path = REPO_ROOT / "data" / "baseline_report.md"
    save_metrics_summary(results, json_path, md_path)
    log.info("Saved baseline metrics to %s and %s", json_path, md_path)

    print_terminal_summary(results)
    return results


def print_terminal_summary(results: Dict[str, Any]) -> None:
    """Print clean comparison table to stdout."""
    print("\n" + "=" * 105)
    print("                      HEATGUARD AI -- PHASE 3 BASELINE BENCHMARK RESULTS")
    print("=" * 105)
    header = f"{'Model':<24} | {'Split':<5} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'F1-HW':<7} | {'Precision':<9} | {'Recall':<7} | {'Brier':<7} | {'Threshold (tau)':<15}"
    print(header)
    print("-" * 105)

    for model_name, data in results.items():
        for split in ("train", "val", "test"):
            m = data[split]
            label = model_name if split == "train" else ""
            tau_str = f"tau = {m['threshold']}"
            row = (
                f"{label:<24} | {split.upper():<5} | "
                f"{m['pr_auc']:<7.4f} | {m['roc_auc']:<7.4f} | {m['f1_heatwave']:<7.4f} | "
                f"{m['precision']:<9.4f} | {m['recall']:<7.4f} | {m['brier_score']:<7.4f} | "
                f"{tau_str:<15}"
            )
            print(row)
        print("-" * 105)
    print("=" * 105)
    print("All serialized models and scaler saved in: models/saved/")
    print("Full JSON and Markdown report generated in: data/\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="HeatGuard AI - Phase 3 Baseline Models")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "model_ready_dataset.csv"))
    ap.add_argument("--schema", default=str(REPO_ROOT / "data" / "feature_schema.json"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "models" / "saved"))
    ap.add_argument("--no-save", action="store_true", help="Do not save model pickle files")
    args = ap.parse_args()

    run_baselines(
        data_path=Path(args.data),
        schema_path=Path(args.schema),
        output_dir=Path(args.outdir),
        save_models=not args.no_save,
    )


if __name__ == "__main__":
    main()
