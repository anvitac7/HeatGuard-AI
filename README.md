# HeatGuard AI: GenAI-Powered Heatwave Prediction & Grounded Advisory System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM%20SOTA-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Backend-Flask%20REST-red.svg)](https://flask.palletsprojects.com/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.68%25-success.svg)]()
[![Balanced Accuracy](https://img.shields.io/badge/Balanced%20Accuracy-84.40%25-blueviolet.svg)]()
[![PR-AUC](https://img.shields.io/badge/Test%20PR--AUC-0.6747%20(20.7x%20Lift)-orange.svg)]()

> **HeatGuard AI** is an operational meteorological early-warning, risk estimation, and GenAI-powered public health advisory platform built on **74 consecutive years (1951–2024)** of daily Indian meteorological data (187,387 records across 7 major metropolitan cities).

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Meteorological Climatology](#-dataset--meteorological-climatology)
3. [The Two-Stage Hybrid Architecture](#-the-two-stage-hybrid-architecture)
4. [Performance Metrics & Evaluation Breakdown](#-performance-metrics--evaluation-breakdown)
5. [Evaluator / Viva Cheat Sheet](#-evaluator--viva-cheat-sheet)
6. [Repository Structure](#-repository-structure)
7. [Installation & Quickstart](#-installation--quickstart)
8. [End-to-End System Workflow](#-end-to-end-system-workflow)

---

## 🌟 Project Overview

Extreme heat events pose grave threats to public health, agriculture, and municipal infrastructure across India. **HeatGuard AI** replaces naive heuristics with a rigorous, zero-leakage, two-stage machine learning engine paired with grounded Generative AI advisories:

- **Zero-Leakage Time-Series Feature Pipeline:** Reconstructs 45 meteorological features (autoregressive lags, 3d/7d rolling dynamics, diurnal temperature range, trigonometric cyclical seasonality) respecting 40 calendar gaps across 74 years.
- **Two-Stage Hybrid ML Engine:** Combines a continuous temperature departure regressor ($R^2 = 0.8690$, $\text{MAE} = 1.08^\circ\text{C}$) with an isotonically calibrated cost-sensitive LightGBM classifier ($\text{Brier} = 0.0164$, $\text{PR-AUC} = 0.6747$).
- **Physics-Grounded IMD Severity Mapping:** Maps continuous departures to IMD risk tiers (`Normal`, `Warning`, `Severe`, `Extreme`) avoiding class-imbalance collapse on rare extremes.
- **Multi-Stakeholder GenAI Advisories:** Translates verified ML predictions into hallucination-free, targeted actionable guidance for 4 distinct personas: **Citizens**, **Farmers**, **Health Agencies**, and **Municipal Authorities**.
- **Interactive Web Platform:** Glassmorphism dashboard with Leaflet.js interactive geospatial map, Chart.js 74-year climate analytics, and a grounded context-aware meteorological chatbot.

---

## 📊 Dataset & Meteorological Climatology

| Characteristic | Specification |
|---|---|
| **Data Source** | `heatguard_clean.csv` (Segmented, cleaned, verified 0 duplicates) |
| **Observation Period** | January 1, 1951 to June 21, 2024 (**74 consecutive years**) |
| **Total Observations** | **187,387 daily records** (187,022 rows in feature matrix after 7-day lag warmup) |
| **Cities Covered (7)** | Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai, Pune |
| **Class Distribution** | 185,596 Normal (99.04%), 1,663 Warning (0.89%), 126 Severe (0.07%), 2 Extreme (0.001%) |
| **Temporal Split** | **Train:** 1951–2015 (166,131 rows) · **Val:** 2016–2020 (12,771 rows) · **Test:** 2021–2024 (8,120 rows) |

### Key Climatological Discoveries:
1. **Spatial Concentration:** Delhi (722 days), Ahmedabad (619 days), and Chennai (388 days) account for **96.5%** of all recorded heatwaves. Bengaluru's static 40.0°C threshold was never breached in 74 years.
2. **Rainfall Missingness Etiology:** Chennai (95.62% null) and Mumbai (95.74% null) had zero rainfall records from 1951 to 2020 (recordings began strictly in 2021). Handled with zero-fill + `rain_is_recorded` binary indicator.
3. **Decadal Climate Surge:** The 4.5 years of 2020–2024 recorded **284 heatwave days** and both extreme events (May 2024 in Ahmedabad), exceeding any full 10-year decade since 1951.
4. **Thermal Persistence:** $P(\text{Heatwave}_{t+1} \mid \text{Heatwave}_t) = 63.15\%$ vs $0.36\%$ on normal days ($175\times$ persistence multiplier).

---

## 🧠 The Two-Stage Hybrid Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │       Day t Meteorological Feature Vector X_t            │
                    │   (45 engineered lag, rolling, cyclical & spatial inputs)│
                    └─────────────────────────────┬────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┴────────────────────────────┐
                    ▼                                                          ▼
     ┌──────────────────────────────┐                         ┌─────────────────────────────────┐
     │           STAGE 1            │                         │             STAGE 2             │
     │ Continuous Huber Regressor   │                         │  Cost-Sensitive Classifier      │
     │     (LightGBM Regressor)     │                         │ (LightGBM + Isotonic Calibration)│
     └──────────────┬───────────────┘                         └────────────────┬────────────────┘
                    │                                                          │
                    ▼                                                          ▼
     ┌──────────────────────────────┐                         ┌─────────────────────────────────┐
     │ Predicted Departure ΔT(t+1)  │                         │ Calibrated Probability          │
     │  & Quantile Bounds [p10,p90] │                         │       P(Heatwave t+1)           │
     └──────────────┬───────────────┘                         └────────────────┬────────────────┘
                    │                                                          │
                    └─────────────────────────────┬────────────────────────────┘
                                                  ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │               STAGE 3: DECISION ENGINE                   │
                    │               Is P(Heatwave t+1) >= 0.230?               │
                    └─────────────────────────────┬────────────────────────────┘
                                     ┌────────────┴────────────┐
                                     │ NO                      │ YES
                                     ▼                         ▼
                              NORMAL DAY                IMD PHYSICAL DEPARTURE
                                                        ┌───────────────────────┐
                                                        │ ΔT < 2.0°C  → WARNING │
                                                        │ 2-4°C       → SEVERE  │
                                                        │ >= 4.0°C    → EXTREME │
                                                        └───────────────────────┘
```

### Why Did We Build a Two-Stage Hybrid Model?
1. **Solves the Extreme Class Imbalance:** Pure classifiers struggle because only ~0.96% of days are heatwaves. Stage 1 trains on **all 187k continuous rows**, learning complete seasonal warming curves, nighttime thermal retention, and diurnal heat dynamics.
2. **Produces Honest, Calibrated Probabilities:** Using `scale_pos_weight=15.0` and `CalibratedClassifierCV` (Isotonic regression), the Stage 2 classifier produces genuine posterior probabilities ($\text{Brier Score} = 0.0164$) suitable for risk communication.
3. **Overcomes the "Extreme" Category Data Scarcity:** Direct 4-class multi-class classifiers cannot learn the "Extreme" class because only **2 extreme rows exist in 74 years**. Stage 3 maps Stage 1's continuous predicted departure directly to IMD physical definitions ($0–2^\circ\text{C}$ Warning, $2–4^\circ\text{C}$ Severe, $\ge 4^\circ\text{C}$ Extreme).

---

## 📈 Performance Metrics & Evaluation Breakdown

### Test Set Performance (2021–2024 Climate Surge — 8,120 Out-of-Sample Days):

| Evaluation Metric | Test Result | Interpretation & Significance |
|---|---|---|
| **Raw Overall Accuracy** | **97.68%** | $7,932 / 8,120$ days correctly classified across the entire test set. |
| **Balanced Accuracy** | **84.40%** | Average of accuracy on normal days ($98.61\%$) and heatwave days ($70.19\%$). |
| **PR-AUC (Primary Metric)** | **0.6747** | **$20.7\times$ lift** over the random guessing baseline ($3.26\%$). |
| **ROC-AUC** | **0.9671** | Exceptional discriminatory separation across all confidence thresholds. |
| **Precision** | **63.05%** | $186 / 295$ issued heatwave warnings are true heatwaves (low false alarm rate). |
| **Recall / Sensitivity** | **70.19%** | Successfully catches $>70\%$ of all heatwaves 24 hours in advance. |
| **Brier Score** | **0.0164** | Tight probability calibration (near 0.0 is perfect reliability). |
| **Departure Regression $R^2$** | **0.8690** | Explains **86.9%** of day-to-day temperature departure variance. |
| **Departure Regression MAE** | **1.0807 °C** | Temperature predictions within $\approx 1^\circ\text{C}$ average error over 4 years. |

### Test Confusion Matrix Breakdown:
```
                             ACTUAL NORMAL (7,855)      ACTUAL HEATWAVE (265)
PREDICTED NORMAL (7,825):        7,746 (TN)                  79 (FN)
PREDICTED HEATWAVE (295):          109 (FP)                 186 (TP)
```
- **Normal Day Accuracy (Specificity):** $\frac{7746}{7855} = \mathbf{98.61\%}$
- **Heatwave Day Accuracy (Recall):** $\frac{186}{265} = \mathbf{70.19\%}$
- **Balanced Accuracy:** $\frac{98.61\% + 70.19\%}{2} = \mathbf{84.40\%}$

---

## 🎓 Evaluator / Viva Cheat Sheet

### Q1: "What are the 2 models in your hybrid system?"
> **Answer:** 
> 1. **Model 1 (Stage 1 Regressor):** A LightGBM continuous regressor with Huber loss that predicts the exact numerical maximum temperature and departure ($\Delta T = T_{\max} - \text{threshold}$) for tomorrow ($t+1$).
> 2. **Model 2 (Stage 2 Classifier):** A cost-sensitive LightGBM classifier (`scale_pos_weight=15.0`) with Isotonic Probability Calibration (`CalibratedClassifierCV`) that calculates the calibrated probability $P(\text{Heatwave}_{t+1})$.
> 
> *The decision engine activates heatwave alerts when $P \ge 0.230$, and assigns IMD severity (`Warning`, `Severe`, `Extreme`) directly using Stage 1's physical departure.*

### Q2: "What is your final accuracy?"
> **Answer:** 
> - **Overall Raw Accuracy is 97.68%** ($7,932 / 8,120$ days correct on the 2021–2024 test set).
> - Because heatwaves are rare (3.26% of test days), we also evaluate **Balanced Accuracy, which is 84.40%** (98.61% accuracy on normal days and 70.19% detection accuracy on heatwaves).

### Q3: "What is PR-AUC and isn't 0.6747 low?"
> **Answer:** 
> - **No, 0.6747 is an outstanding result.** In standard balanced datasets, random guessing gives a PR-AUC of 0.50. But on our imbalanced test dataset, heatwaves occur on only **3.26%** of days, meaning a random guessing model has a PR-AUC of only **0.0326**.
> - Achieving **0.6747** represents a **$20.7\times$ performance lift** over random guessing, demonstrating high precision (63.05%) while maintaining strong recall (70.19%).

### Q4: "Why did you split by year instead of random k-fold cross-validation?"
> **Answer:** 
> Meteorological weather sequences exhibit strong autoregression and decadal climate trends. Random shuffling causes severe **temporal data leakage** (using tomorrow's weather to predict yesterday). We enforced a strict chronological split: **1951–2015 (Train)**, **2016–2020 (Val)**, and **2021–2024 (Test)**.

---

## 📁 Repository Structure

```text
HeatGuard-AI/
├── data/
│   ├── heatguard_raw.csv                # Ingested 74-year daily dataset
│   ├── heatguard_clean.csv              # Cleaned dataset (0 nulls, 40 segments)
│   ├── model_ready_dataset.csv          # Feature matrix (187,022 rows x 45 features)
│   ├── feature_schema.json              # Grouped feature definitions & meta columns
│   ├── cleaning_report.json             # Validation report from data cleaning
│   ├── baseline_report.md               # Phase 3 baseline evaluation report
│   ├── baseline_metrics.json            # Phase 3 raw baseline metrics
│   ├── hybrid_report.md                 # Phase 4 Two-Stage Hybrid Engine report
│   └── hybrid_metrics.json              # Phase 4 serialized test metrics
│
├── data_cleaning/
│   ├── Data_cleaner.py                  # Preprocessing, missingness & calendar reindexing
│   └── feature_eng.py                   # Zero-leakage 45-feature pipeline & 12 test assertions
│
├── models/
│   ├── evaluate.py                      # PR-AUC, ROC-AUC, Brier score, CM evaluation module
│   ├── train_baselines.py               # Persistence, Balanced LogReg, Tree Baselines
│   ├── hybrid_engine.py                 # TwoStageHeatwavePredictor production class
│   ├── train_hybrid.py                  # Two-stage training, calibration & threshold tuning
│   └── saved/
│       ├── hybrid_predictor.pkl         # Production TwoStageHeatwavePredictor
│       ├── departure_regressor.pkl      # Stage 1 Huber Regressor
│       ├── lgbm_classifier.pkl          # Stage 2 Calibrated Classifier
│       ├── persistence_baseline.pkl     # Baseline 1
│       ├── logreg_baseline.pkl          # Baseline 2
│       ├── rf_baseline.pkl              # Baseline 3
│       ├── scaler.pkl                   # Feature standardizer
│       └── model_metadata.json          # Deployment parameters & feature schema
│
├── notebooks/
│   ├── 03_baseline_models.ipynb         # Interactive baseline modeling notebook
│   └── 04_two_stage_hybrid_engine.ipynb # Interactive two-stage hybrid engine notebook
│
├── services/                            # (Phase 5 Flask Services)
│   ├── prediction_service.py            # Live lag feature builder & model inference
│   ├── analytics_service.py             # 74-year climate trends & decadal analytics
│   └── genai_service.py                 # Grounded 4-persona advisory generator & chatbot
│
├── templates/                           # (Phase 6 Jinja2 HTML Pages)
│   ├── index.html                       # Landing page
│   ├── dashboard.html                   # Live command center & Leaflet map
│   ├── advisory.html                    # Multi-stakeholder advisory studio
│   ├── chatbot.html                     # Grounded meteorological assistant
│   └── analytics.html                   # Historical climate explorer
│
├── static/                              # (Phase 6 CSS & JS Assets)
│   ├── css/style.css                    # Glassmorphism dark-mode styling
│   └── js/                              # Interactive Leaflet & Chart.js controllers
│
├── app.py                               # Flask application entrypoint
├── requirements.txt                     # Project dependencies
├── progress.md                          # Team progress tracker & work log
├── HeatGuard_AI_Implementation_Plan_Dataset_Specific.md  # Architectural plan
└── README.md                            # Comprehensive project guide
```

---

## ⚡ Installation & Quickstart

### 1. Clone & Set Up Virtual Environment:
```bash
git clone https://github.com/anvitac7/HeatGuard-AI.git
cd HeatGuard-AI
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts:
```bash
# 1. Clean & segment data (generates data/heatguard_clean.csv)
python data_cleaning/Data_cleaner.py

# 2. Build 45 zero-leakage features (generates data/model_ready_dataset.csv)
python data_cleaning/feature_eng.py

# 3. Train scientific baselines (Persistence, LogReg, Tree Forest)
python models/train_baselines.py

# 4. Train production Two-Stage Hybrid Engine
python models/train_hybrid.py
```

### 4. Launch the Web Application:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000/`.

---

## 🔄 End-to-End System Workflow

```text
[User Selects City & Date] 
       │
       ▼
[Dynamic Lag Retrieval] ──► Extracts preceding 7-day weather series from heatguard_clean.csv
       │
       ▼
[Feature Pipeline]      ──► Constructs 45 zero-leakage lag, rolling, and cyclical features
       │
       ▼
[Two-Stage Engine]      ──► Stage 1 predicts Departure ΔT (+1.2°C)
                        ──► Stage 2 calculates Calibrated Risk P (81.4%)
                        ──► Stage 3 maps Severity Tier (Warning / High Risk)
       │
       ▼
[Dashboard UI]          ──► Animates Leaflet Map Marker, Circular Gauge, and KPI Cards
       │
       ▼
[Advisory Engine]       ──► Formats ML context into persona prompts (Farmer, Citizen, Health, City)
                        ──► Outputs grounded, actionable, non-hallucinatory safety checklists
```

---

## 👥 Authors & Acknowledgments
- **Anvita & Team** — HeatGuard AI Team
- **Built for:** Operational Heatwave Risk Early Warning & Public Safety Communication in India.
- **Reference Dataset:** Indian Meteorological Daily Observations (1951–2024).
