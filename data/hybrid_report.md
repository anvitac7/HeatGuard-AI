# HeatGuard AI — Evaluation of the Saved Runtime Model

## Evaluation basis

This report describes the model artifact currently loaded by the Flask service, not an older training narrative. On 9 October 2026, the saved `models/saved/hybrid_predictor.pkl` was loaded, the current feature-generation functions were applied in memory to `data/heatguard_clean.csv`, and the 2021–2024 chronological test partition was evaluated. No generated feature CSV was required for this evaluation.

The loaded object is `models.hybrid_engine.TwoStageHeatwavePredictor`. It contains scikit-learn `HistGradientBoostingRegressor` models for departure and 10th/90th quantile bounds, plus a `HistGradientBoostingClassifier` using balanced class weights. Its stored decision threshold is **0.35**. The probability estimator is the base classifier; the current implementation does not fit a probability-calibration wrapper.

## Held-out classification results

| Metric | Test result |
|---|---:|
| Test rows | 8,120 |
| Heatwave positives | 265 (3.264%) |
| Decision threshold | 0.35 |
| Accuracy | 96.18% |
| Balanced accuracy | 94.38% |
| PR-AUC (average precision) | 0.7294 |
| ROC-AUC | 0.9818 |
| Precision | 45.79% |
| Recall | 92.45% |
| F1 (heatwave class) | 0.6125 |
| Brier score | 0.0263 |

### Confusion matrix

|  | Actual normal | Actual heatwave |
|---|---:|---:|
| Predicted normal | 7,565 | 20 |
| Predicted heatwave | 290 | 245 |

The Brier score is included as a scoring metric. It does not demonstrate calibrated probabilities, and the code does not currently calibrate the classifier.

## Held-out departure regression

| Metric | Test result |
|---|---:|
| RMSE | 1.4899 °C |
| MAE | 1.0700 °C |
| R² | 0.8751 |

## Interpretation and scope

These results are from the saved model artifact evaluated with the checked-in feature-generation code and the test-date partition. They are not a live forecast evaluation. Raw accuracy is presented with balanced accuracy and precision/recall because heatwave labels are uncommon in this test period. At its stored 0.35 threshold, the model has high recall and also produces false positives.

The current estimators and output mapping are implemented in `models/hybrid_engine.py`; the evaluation helpers are in `models/evaluate.py`. Treat model scores as prototype outputs. Further probability calibration, independent temporal evaluation, and operational validation have not been completed.
