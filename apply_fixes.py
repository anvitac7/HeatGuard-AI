"""
HeatGuard AI - apply_fixes.py
Run from the project root (the folder containing app.py):  python apply_fixes.py

What it does (safe to re-run; it reports what it changed and what it skipped):
  1. services/prediction_service.py : reject malformed / out-of-range dates (HTTP 400 via /api/predict)
  2. services/analytics_service.py  : replace stale benchmark numbers with the saved model's real test metrics
  3. services/genai_service.py      : remove false claims (LightGBM / isotonic calibration / tau=0.230 / 97.68%)
Backups are written next to each file as *.bak.
"""
from pathlib import Path
import shutil, sys

ROOT = Path(__file__).resolve().parent

DATE_OLD = '        sub = self.city_dfs[city]\r\n        matched = sub[sub["date_str"] <= date_str]'
DATE_NEW = (
    '        from datetime import datetime as _datetime\r\n'
    '        try:\r\n'
    '            date_str = _datetime.strptime(str(date_str), "%Y-%m-%d").strftime("%Y-%m-%d")\r\n'
    '        except (ValueError, TypeError):\r\n'
    '            raise ValueError(f"Invalid date \'{date_str}\'. Expected format YYYY-MM-DD.")\r\n'
    '\r\n'
    '        sub = self.city_dfs[city]\r\n'
    '        _min_d, _max_d = str(sub["date_str"].min()), str(sub["date_str"].max())\r\n'
    '        if date_str < _min_d or date_str > _max_d:\r\n'
    '            raise ValueError(f"Date {date_str} is outside the available data range {_min_d} to {_max_d} for {city}.")\r\n'
    '        matched = sub[sub["date_str"] <= date_str]'
)

EDITS = {
    "services/prediction_service.py": [(DATE_OLD, DATE_NEW)],
    "services/analytics_service.py": [
        ('"test_accuracy": 97.68,', '"test_accuracy": 96.18,'),
        ('"test_balanced_accuracy": 84.40,', '"test_balanced_accuracy": 94.38,'),
        ('"test_pr_auc": 0.6747,', '"test_pr_auc": 0.7294,'),
        ('"pr_auc_lift": "20.7x",', '"pr_auc_lift": "22.3x",'),
        ('"test_roc_auc": 0.9671,', '"test_roc_auc": 0.9818,'),
    ],
    "services/genai_service.py": [
        ("- Calibrated Heatwave Probability: {prob}%", "- Heatwave Probability (uncalibrated model score): {prob}%"),
        ("Model: Two-Stage Huber + Isotonic LightGBM (97.68% accuracy, \u03c4* = 0.230, PR-AUC 0.6747).",
         "Model: Two-Stage HistGradientBoosting (96.18% accuracy, 94.38% balanced accuracy, \u03c4* = 0.35, PR-AUC 0.7294; probabilities are uncalibrated)."),
        ("Two-Stage Hybrid model (97.68% accuracy)", "Two-Stage Hybrid model (96.18% accuracy)"),
        ("Two-Stage Huber + Isotonic LightGBM architecture (97.68% accuracy)", "Two-Stage HistGradientBoosting architecture (96.18% accuracy)"),
        ("Stage 1 (Continuous Huber Regressor):** A LightGBM continuous model", "Stage 1 (Temperature-Departure Regressor):** A HistGradientBoostingRegressor"),
        ("(achieving an R\u00b2 of 0.8690 and RMSE of 1.52\u00b0C)", "(test R\u00b2 0.8751, RMSE 1.49\u00b0C)"),
        ("Stage 2 (Cost-Sensitive Rare-Event Classifier):** A cost-weighted LightGBM model calibrated with **Isotonic Probability Calibration** (`CalibratedClassifierCV`) tuned to an optimal decision threshold of **\u03c4* = 0.230**.",
         "Stage 2 (Rare-Event Classifier):** A class-weighted HistGradientBoostingClassifier (`class_weight='balanced'`) with a decision threshold of **\u03c4* = 0.35**. Its outputs are uncalibrated scores, not calibrated probabilities."),
        ("the pipeline achieves **97.68% overall accuracy**, **84.40% balanced accuracy**, and a **PR-AUC of 0.6747** (a 20.7x lift over random guessing).",
         "the saved model achieves **96.18% accuracy**, **94.38% balanced accuracy**, a **PR-AUC of 0.7294** (about 22x the 3.26% heatwave base rate), **92.45% recall** and **45.79% precision** at the 0.35 threshold."),
        ("LightGBM + Scikit-Learn", "HistGradientBoosting + Scikit-Learn"),
        ("\u0394T, \u03c4* = 0.230.", "\u0394T, \u03c4* = 0.35."),
        ("Two-Stage Hybrid Predictor (LightGBM)", "Two-Stage Hybrid Predictor (HistGradientBoosting)"),
    ],
}

def main():
    if not (ROOT / "app.py").exists():
        sys.exit("Run this from the project root (folder containing app.py).")
    for rel, edits in EDITS.items():
        p = ROOT / rel
        raw = p.read_bytes().decode("utf-8")
        new, applied, skipped = raw, 0, []
        for old, repl in edits:
            if old in new:
                new = new.replace(old, repl); applied += 1
            else:
                skipped.append(old[:60].replace("\r\n", "\\n"))
        if applied:
            shutil.copy(p, str(p) + ".bak")
            p.write_bytes(new.encode("utf-8"))
        print(f"{rel}: {applied} edit(s) applied, {len(skipped)} skipped")
        for s in skipped:
            print("   skipped (already applied or text differs):", s)
    # leftover check
    gen = (ROOT / "services/genai_service.py").read_text(encoding="utf-8")
    for term in ["LightGBM", "Isotonic", "0.230", "97.68", "Huber", "Calibrated Heatwave"]:
        if term in gen:
            print(f"WARNING: '{term}' still appears in genai_service.py - review manually")

if __name__ == "__main__":
    main()