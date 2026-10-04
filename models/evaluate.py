"""
HeatGuard AI - Evaluation Engine & Metrics Protocol.

Provides imbalance-aware metrics for heatwave rare-event prediction:
- PR-AUC (Average Precision) - Primary Metric
- ROC-AUC
- Precision, Recall, F1-Score (Macro and Heatwave-specific)
- Recall @ 80% Precision
- Brier Score Calibration
- Confusion Matrix & Cost-sensitive Score
- Automatic Optimal Decision Threshold (tau*) finder on Validation set
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for binary rare-event prediction."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Avoid crash if single class present
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    brier = float(brier_score_loss(y_true, y_prob))
    f1_hw = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Recall @ 80% Precision calculation
    try:
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        valid_recalls = [r for p, r in zip(precisions, recalls) if p >= 0.80]
        recall_at_80_prec = float(max(valid_recalls)) if valid_recalls else 0.0
    except Exception:
        recall_at_80_prec = 0.0

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_heatwave": round(f1_hw, 4),
        "f1_macro": round(f1_macro, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "recall_at_80_precision": round(recall_at_80_prec, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_samples": len(y_true),
        "positive_samples": int(y_true.sum()),
        "positive_rate_pct": round(float(y_true.mean() * 100), 3),
    }


def find_optimal_threshold(
    y_val: np.ndarray,
    y_val_prob: np.ndarray,
    min_thresh: float = 0.05,
    max_thresh: float = 0.90,
    step: float = 0.01,
) -> Tuple[float, float]:
    """
    Search candidate decision thresholds strictly on Validation data to maximize F1-score.
    Returns (optimal_threshold, best_f1_score).
    """
    best_thresh = 0.50
    best_f1 = -1.0

    thresholds = np.arange(min_thresh, max_thresh + step, step)
    for t in thresholds:
        preds = (y_val_prob >= t).astype(int)
        score = f1_score(y_val, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(t)

    return round(best_thresh, 3), round(best_f1, 4)


def save_metrics_summary(
    results: Dict[str, Any],
    json_path: Path,
    md_path: Path | None = None,
) -> None:
    """Save metrics dictionary to JSON and optional GitHub Markdown table."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if md_path:
        md_lines = [
            "# HeatGuard AI — Model Benchmark Report",
            "",
            "| Model | Split | PR-AUC | ROC-AUC | F1 (HW) | Precision | Recall | Brier Score | Decision Threshold (τ) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for model_name, splits in results.items():
            for split_name in ("train", "val", "test"):
                if split_name in splits:
                    m = splits[split_name]
                    md_lines.append(
                        f"| **{model_name}** | {split_name.upper()} | "
                        f"{m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['f1_heatwave']:.4f} | "
                        f"{m['precision']:.4f} | {m['recall']:.4f} | {m['brier_score']:.4f} | "
                        f"τ = {m['threshold']} |"
                    )
        md_lines.append("")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
