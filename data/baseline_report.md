# HeatGuard AI — Model Benchmark Report

| Model | Split | PR-AUC | ROC-AUC | F1 (HW) | Precision | Recall | Brier Score | Decision Threshold (τ) |
|---|---|---|---|---|---|---|---|---|
| **Persistence_Heuristic** | TRAIN | 0.3903 | 0.8095 | 0.6222 | 0.6222 | 0.6222 | 0.0051 | τ = 0.05 |
| **Persistence_Heuristic** | VAL | 0.3668 | 0.7981 | 0.6024 | 0.6048 | 0.6000 | 0.0062 | τ = 0.05 |
| **Persistence_Heuristic** | TEST | 0.4921 | 0.8420 | 0.6943 | 0.6943 | 0.6943 | 0.0170 | τ = 0.05 |
| **Logistic_Regression** | TRAIN | 0.6647 | 0.9956 | 0.5264 | 0.3693 | 0.9161 | 0.0245 | τ = 0.9 |
| **Logistic_Regression** | VAL | 0.6293 | 0.9941 | 0.4832 | 0.3276 | 0.9200 | 0.0343 | τ = 0.9 |
| **Logistic_Regression** | TEST | 0.7395 | 0.9878 | 0.6619 | 0.5349 | 0.8679 | 0.0384 | τ = 0.9 |
| **Tree_Ensemble_Baseline** | TRAIN | 0.9427 | 0.9996 | 0.7029 | 0.5420 | 1.0000 | 0.0064 | τ = 0.61 |
| **Tree_Ensemble_Baseline** | VAL | 0.6137 | 0.9937 | 0.5801 | 0.4430 | 0.8400 | 0.0104 | τ = 0.61 |
| **Tree_Ensemble_Baseline** | TEST | 0.7408 | 0.9877 | 0.6800 | 0.6090 | 0.7698 | 0.0192 | τ = 0.61 |
