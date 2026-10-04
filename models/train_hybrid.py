"""
HeatGuard AI - Phase 4: Training & Evaluation Pipeline for Two-Stage Hybrid Engine.

Trains:
- Stage 1: Continuous Departure Regressor (Huber Loss) + Quantile Heads [p10, p90]
- Stage 2: Cost-Sensitive Calibrated LightGBM Classifier
- Stage 3: Physics-Grounded IMD Severity & Risk Tier Mapping

Outputs:
- models/saved/hybrid_predictor.pkl (Production full pipeline)
- models/saved/departure_regressor.pkl
- models/saved/lgbm_classifier.pkl
- models/saved/model_metadata.json
- data/hybrid_metrics.json
- data/hybrid_report.md
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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add repo root to python path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.evaluate import compute_metrics, find_optimal_threshold
from models.hybrid_engine import TwoStageHeatwavePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_hybrid")


def load_dataset(
    data_path: Path,
    schema_path: Path,
) -> Tuple[pd.DataFrame, List[str], str, str]:
    """Load model ready dataset and schemas."""
    log.info("Loading dataset from %s ...", data_path)
    df = pd.read_csv(data_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_cols = schema["feature_columns"]
    clf_target = "target_heatwave_next_day"
    reg_target = "target_departure_next_day"

    log.info("Loaded %d rows with %d features.", len(df), len(feature_cols))
    return df, feature_cols, clf_target, reg_target


def train_hybrid_pipeline(
    data_path: Path = REPO_ROOT / "data" / "model_ready_dataset.csv",
    schema_path: Path = REPO_ROOT / "data" / "feature_schema.json",
    output_dir: Path = REPO_ROOT / "models" / "saved",
    save_models: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate the two-stage hybrid prediction engine."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df, feature_cols, clf_target, reg_target = load_dataset(data_path, schema_path)

    # Split dataset chronologically
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[feature_cols].copy()
    y_train_clf = train_df[clf_target].to_numpy().astype(int)
    y_train_reg = train_df[reg_target].to_numpy().astype(float)

    X_val = val_df[feature_cols].copy()
    y_val_clf = val_df[clf_target].to_numpy().astype(int)
    y_val_reg = val_df[reg_target].to_numpy().astype(float)

    X_test = test_df[feature_cols].copy()
    y_test_clf = test_df[clf_target].to_numpy().astype(int)
    y_test_reg = test_df[reg_target].to_numpy().astype(float)

    log.info("=" * 80)
    log.info("TRAINING TWO-STAGE HYBRID ENGINE (LightGBM Regressor + Calibrated Classifier)...")
    t0 = time.time()

    predictor = TwoStageHeatwavePredictor(
        reg_learning_rate=0.03,
        reg_n_estimators=400,
        reg_num_leaves=31,
        reg_max_depth=10,
        clf_learning_rate=0.03,
        clf_n_estimators=350,
        clf_num_leaves=31,
        clf_max_depth=10,
        scale_pos_weight=15.0,
        calibration_method="isotonic",
        random_state=42,
    )

    predictor.fit(
        X_train=X_train,
        y_train_clf=y_train_clf,
        y_train_reg=y_train_reg,
        X_val=X_val,
        y_val_clf=y_val_clf,
        y_val_reg=y_val_reg,
    )
    t_train = time.time() - t0
    log.info("Two-Stage Engine trained in %.2f seconds.", t_train)

    # ------------------------------------------------------------------------------------
    # Decision Threshold Tuning on Validation Split Strictly
    # ------------------------------------------------------------------------------------
    val_probs = predictor.predict_proba(X_val)[:, 1]
    optimal_tau, val_f1 = find_optimal_threshold(y_val_clf, val_probs)
    predictor.optimal_threshold = optimal_tau
    log.info("Optimized Validation Decision Threshold: tau* = %.3f (Validation F1: %.4f)", optimal_tau, val_f1)

    # ------------------------------------------------------------------------------------
    # Comprehensive Evaluation
    # ------------------------------------------------------------------------------------
    results: Dict[str, Any] = {
        "training_time_seconds": round(t_train, 2),
        "optimal_threshold": optimal_tau,
        "classification": {},
        "regression": {},
        "severity_breakdown": {},
    }

    # 1. Classification Metrics across Splits
    for split_name, (X_s, y_s) in (("train", (X_train, y_train_clf)), ("val", (X_val, y_val_clf)), ("test", (X_test, y_test_clf))):
        probs = predictor.predict_proba(X_s)[:, 1]
        results["classification"][split_name] = compute_metrics(y_s, probs, threshold=optimal_tau)

    # 2. Continuous Departure Regression Metrics across Splits
    for split_name, (X_s, y_s_reg, y_s_clf) in (
        ("train", (X_train, y_train_reg, y_train_clf)),
        ("val", (X_val, y_val_reg, y_val_clf)),
        ("test", (X_test, y_test_reg, y_test_clf)),
    ):
        pred_dep = predictor.predict_departure(X_s)
        rmse = float(np.sqrt(mean_squared_error(y_s_reg, pred_dep)))
        mae = float(mean_absolute_error(y_s_reg, pred_dep))
        r2 = float(r2_score(y_s_reg, pred_dep))

        # Tail Error (MAE strictly on true Heatwave Days)
        hw_mask = y_s_clf == 1
        tail_mae = float(mean_absolute_error(y_s_reg[hw_mask], pred_dep[hw_mask])) if hw_mask.sum() > 0 else 0.0

        results["regression"][split_name] = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "heatwave_day_mae": round(tail_mae, 4),
        }

    # 3. Severity Classification Verification on Test Set
    test_preds_df = predictor.predict_full(X_test, threshold=optimal_tau)
    test_sev_true = test_df["target_severity_next_day"].to_numpy()
    test_sev_pred = test_preds_df["predicted_severity"].to_numpy()

    sev_labels = ["Normal", "Warning", "Severe", "Extreme"]
    sev_counts_true = {k: int((test_sev_true == k).sum()) for k in sev_labels}
    sev_counts_pred = {k: int((test_sev_pred == k).sum()) for k in sev_labels}

    results["severity_breakdown"]["test_true_distribution"] = sev_counts_true
    results["severity_breakdown"]["test_predicted_distribution"] = sev_counts_pred

    # ------------------------------------------------------------------------------------
    # Serialization & Report Generation
    # ------------------------------------------------------------------------------------
    if save_models:
        log.info("Saving model checkpoints to %s ...", output_dir)
        predictor.save(output_dir / "hybrid_predictor.pkl")
        joblib.dump(predictor.regressor, output_dir / "departure_regressor.pkl")
        joblib.dump(predictor.calibrated_classifier, output_dir / "lgbm_classifier.pkl")

        metadata = {
            "model_type": "TwoStageHeatwavePredictor",
            "optimal_threshold": optimal_tau,
            "feature_columns": feature_cols,
            "test_pr_auc": results["classification"]["test"]["pr_auc"],
            "test_roc_auc": results["classification"]["test"]["roc_auc"],
            "test_f1_hw": results["classification"]["test"]["f1_heatwave"],
            "test_reg_rmse": results["regression"]["test"]["rmse"],
            "test_reg_r2": results["regression"]["test"]["r2"],
        }
        with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    json_path = REPO_ROOT / "data" / "hybrid_metrics.json"
    md_path = REPO_ROOT / "data" / "hybrid_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    export_markdown_report(results, md_path)
    log.info("Saved metrics to %s and %s", json_path, md_path)

    print_terminal_summary(results)
    return results


def export_markdown_report(results: Dict[str, Any], md_path: Path) -> None:
    """Generate clean GitHub Markdown performance report."""
    clf = results["classification"]
    reg = results["regression"]
    tau = results["optimal_threshold"]

    lines = [
        "# HeatGuard AI -- Phase 4 Two-Stage Hybrid Engine Performance Report",
        "",
        "## 1. Primary Classification Metrics (Heatwave Event Detection)",
        "",
        "| Split | PR-AUC (Primary) | ROC-AUC | F1 (HW) | Precision | Recall | Brier Score | Decision Threshold (tau) |",
        "|---|---|---|---|---|---|---|---|",
        f"| **TRAIN** | {clf['train']['pr_auc']:.4f} | {clf['train']['roc_auc']:.4f} | {clf['train']['f1_heatwave']:.4f} | {clf['train']['precision']:.4f} | {clf['train']['recall']:.4f} | {clf['train']['brier_score']:.4f} | tau = {tau} |",
        f"| **VAL**   | {clf['val']['pr_auc']:.4f} | {clf['val']['roc_auc']:.4f} | {clf['val']['f1_heatwave']:.4f} | {clf['val']['precision']:.4f} | {clf['val']['recall']:.4f} | {clf['val']['brier_score']:.4f} | tau = {tau} (tuned) |",
        f"| **TEST**  | **{clf['test']['pr_auc']:.4f}** | **{clf['test']['roc_auc']:.4f}** | **{clf['test']['f1_heatwave']:.4f}** | **{clf['test']['precision']:.4f}** | **{clf['test']['recall']:.4f}** | **{clf['test']['brier_score']:.4f}** | tau = {tau} (applied) |",
        "",
        "## 2. Continuous Departure & Temperature Regression Metrics",
        "",
        "| Split | RMSE (°C) | MAE (°C) | R² Score | Heatwave-Day Tail MAE (°C) |",
        "|---|---|---|---|---|",
        f"| **TRAIN** | {reg['train']['rmse']:.4f} | {reg['train']['mae']:.4f} | {reg['train']['r2']:.4f} | {reg['train']['heatwave_day_mae']:.4f} |",
        f"| **VAL**   | {reg['val']['rmse']:.4f} | {reg['val']['mae']:.4f} | {reg['val']['r2']:.4f} | {reg['val']['heatwave_day_mae']:.4f} |",
        f"| **TEST**  | **{reg['test']['rmse']:.4f}** | **{reg['test']['mae']:.4f}** | **{reg['test']['r2']:.4f}** | **{reg['test']['heatwave_day_mae']:.4f}** |",
        "",
        "## 3. Test Set Physical Severity Distribution",
        "",
        "| Severity Tier | Ground Truth Days | Hybrid Engine Predicted Days |",
        "|---|---|---|",
    ]
    s_true = results["severity_breakdown"]["test_true_distribution"]
    s_pred = results["severity_breakdown"]["test_predicted_distribution"]
    for k in ["Normal", "Warning", "Severe", "Extreme"]:
        lines.append(f"| **{k}** | {s_true.get(k, 0):,d} | {s_pred.get(k, 0):,d} |")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_terminal_summary(results: Dict[str, Any]) -> None:
    """Print clean comparison summary to stdout."""
    clf = results["classification"]
    reg = results["regression"]
    tau = results["optimal_threshold"]

    print("\n" + "=" * 105)
    print("                HEATGUARD AI -- PHASE 4 TWO-STAGE HYBRID ENGINE BENCHMARK")
    print("=" * 105)
    print("1. CLASSIFICATION STAGE (Calibrated LightGBM with Isotonic Tuning):")
    print(f"{'Split':<6} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'F1-HW':<8} | {'Precision':<10} | {'Recall':<8} | {'Brier':<8} | {'Threshold (tau)':<15}")
    print("-" * 105)
    for s in ("train", "val", "test"):
        m = clf[s]
        print(f"{s.upper():<6} | {m['pr_auc']:<8.4f} | {m['roc_auc']:<8.4f} | {m['f1_heatwave']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['brier_score']:<8.4f} | tau = {tau}")
    print("-" * 105)

    print("\n2. CONTINUOUS REGRESSION STAGE (Departure & Uncertainty Predictor):")
    print(f"{'Split':<6} | {'RMSE (°C)':<10} | {'MAE (°C)':<10} | {'R2 Score':<10} | {'Heatwave Tail MAE (°C)':<22}")
    print("-" * 105)
    for s in ("train", "val", "test"):
        r = reg[s]
        print(f"{s.upper():<6} | {r['rmse']:<10.4f} | {r['mae']:<10.4f} | {r['r2']:<10.4f} | {r['heatwave_day_mae']:<22.4f}")
    print("-" * 105)

    print("\n3. SEVERITY TIERS ON TEST PERIOD (2021-2024 Climate Surge):")
    s_true = results["severity_breakdown"]["test_true_distribution"]
    s_pred = results["severity_breakdown"]["test_predicted_distribution"]
    print(f"  Normal  -> True: {s_true.get('Normal', 0):>5} | Predicted: {s_pred.get('Normal', 0):>5}")
    print(f"  Warning -> True: {s_true.get('Warning', 0):>5} | Predicted: {s_pred.get('Warning', 0):>5}")
    print(f"  Severe  -> True: {s_true.get('Severe', 0):>5} | Predicted: {s_pred.get('Severe', 0):>5}")
    print(f"  Extreme -> True: {s_true.get('Extreme', 0):>5} | Predicted: {s_pred.get('Extreme', 0):>5}")
    print("=" * 105)
    print("All serialized models, calibrators, and metadata saved in: models/saved/")
    print("Full JSON and Markdown report exported to: data/\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="HeatGuard AI - Phase 4 Two-Stage Hybrid Engine")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "model_ready_dataset.csv"))
    ap.add_argument("--schema", default=str(REPO_ROOT / "data" / "feature_schema.json"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "models" / "saved"))
    ap.add_argument("--no-save", action="store_true", help="Do not save model artifacts")
    args = ap.parse_args()

    train_hybrid_pipeline(
        data_path=Path(args.data),
        schema_path=Path(args.schema),
        output_dir=Path(args.outdir),
        save_models=not args.no_save,
    )


if __name__ == "__main__":
    main()
