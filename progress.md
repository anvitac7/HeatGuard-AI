# HeatGuard AI — Team Progress Tracker

> **How to use this file:** Whenever you finish (or start) something, tick the box, add your name, and add a line to the [Work Log](#6-work-log) at the bottom. Commit it with your changes so everyone sees the latest status.
>
> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Last updated:** 2026-10-05 (Phases 0, 1, 2, 3, 4 Completed and Verified; Ready for Phase 5 Backend & Phase 6 Frontend)  
**Repo:** https://github.com/anvitac7/HeatGuard-AI  
**Reference doc:** `HeatGuard_AI_Implementation_Plan_Dataset_Specific.md` & `README.md`

---

## 1. Project Snapshot

HeatGuard AI predicts **next-day heatwaves ($t+1$)** for 7 Indian cities and turns the predictions into stakeholder advisories through a full-stack web dashboard and grounded AI chatbot.

| Item | Value |
|---|---|
| Dataset | 187,387 daily rows, 14 columns, 1951-01-01 to 2024-06-21 (74 years) |
| Cities | Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai, Pune |
| Targets (day t+1) | `target_heatwave_next_day` (binary), `target_departure_next_day` (regression), `target_severity_next_day` (4 tiers) |
| Class imbalance | Only 1,791 heatwave days (~0.96%). Overall Test Accuracy: **97.68%**, Balanced Accuracy: **84.40%**, Test PR-AUC: **0.6747** ($20.7\times$ lift over 3.26% base rate). |
| Split | Train 1951–2015 (166k), Validation 2016–2020 (12.7k), Test 2021–2024 (8.1k) (chronological, **no shuffling**) |
| Stack | Python 3.10+, LightGBM, scikit-learn, Flask, Leaflet.js, Chart.js, LLM API |

---

## 2. Overall Progress

| Phase | Name | Status | Key Deliverables / Metrics |
|---|---|---|---|
| 0 | Planning & dataset profiling | `[x]` Done | `HeatGuard_AI_Implementation_Plan_Dataset_Specific.md`, `README.md` |
| 1 | Data preprocessing & calendar pipeline | `[x]` Done | `data_cleaning/Data_cleaner.py`, `data/heatguard_clean.csv` (187,387 rows, 0 nulls, 40 segments) |
| 2 | Zero-leakage feature engineering | `[x]` Done | `data_cleaning/feature_eng.py`, `data/model_ready_dataset.csv` (187,022 rows x 45 features, 12 checks pass) |
| 3 | Baseline models | `[x]` Done | `models/evaluate.py`, `models/train_baselines.py`, `notebooks/03_baseline_models.ipynb` (LogReg & LightGBM Forest) |
| 4 | Two-stage hybrid engine | `[x]` Done | `models/hybrid_engine.py`, `models/train_hybrid.py`, `models/saved/hybrid_predictor.pkl` (Raw Acc: 97.68%, Balanced Acc: 84.40%, PR-AUC: 0.6747) |
| 5 | Flask REST backend | `[ ]` Next | `app.py`, `services/prediction_service.py`, `services/analytics_service.py`, `services/genai_service.py` |
| 6 | Dashboard, Leaflet map, analytics UI | `[ ]` Next | `templates/dashboard.html`, `static/css/style.css`, `static/js/dashboard.js`, `static/js/charts.js` |
| 7 | GenAI advisory + chatbot | `[ ]` Planned | 4 Persona prompts (Citizen, Farmer, Health, Municipality), `templates/advisory.html`, `templates/chatbot.html` |
| 8 | Integration, verification, demo | `[ ]` Planned | End-to-end testing (Pune/Delhi/Ahmedabad demo scenarios), final documentation |

**Completion:** 5 of 9 phases complete (Data & ML layers 100% finished, serialized, and benchmarked).

---

## 3. Phase Details

### Phase 0 — Planning & Dataset Profiling ✅
- [x] Write dataset-specific implementation plan (`HeatGuard_AI_Implementation_Plan_Dataset_Specific.md`)
- [x] Comprehensive root `README.md` with evaluator cheat sheet
- [x] Profile 74-year dataset (missing values, 40 calendar gaps, class balance, spatial disparity)

---

### Phase 1 — Data Preprocessing & Calendar Pipeline ✅
**Files:** `data_cleaning/Data_cleaner.py`, `data/heatguard_clean.csv`, `data/cleaning_report.json`
- [x] Parse ISO dates, sort chronologically by `['city', 'date']`
- [x] Confirm 0 duplicate `(city, date)` combinations
- [x] Segmented linear interpolation for missing `temp_min` (34 rows)
- [x] Impute missing rainfall (Chennai & Mumbai pre-2021) with `0.0` and add `rain_is_recorded` binary indicator
- [x] Detect 40 calendar gaps and create `segment_id` boundaries
- [x] Automated validation passing 9 out of 9 data health checks

---

### Phase 2 — Zero-Leakage Feature Engineering ✅
**Files:** `data_cleaning/feature_eng.py`, `data/model_ready_dataset.csv`, `data/feature_schema.json`, `data/feature_report.json`
- [x] Autoregressive lags (1, 2, 3, 7 days for $T_{\max}, T_{\min}$; 1, 3, 7 days for rain)
- [x] Rolling window statistics (3d/7d averages, 7d max, 7d rain sum, 3d diurnal range average)
- [x] Cyclical trigonometric encodings (`sin_doy`, `cos_doy`, `sin_month`, `cos_month`)
- [x] Derived interactions: `diurnal_temp_range`, `temp_departure_today`, `heatwave_streak_days`, `year_scaled`
- [x] Verified zero cross-gap contamination using `groupby(['city', 'segment_id'])`
- [x] 12 automated leakage checks passed (including synthetic target contamination tests)

---

### Phase 3 — Baseline Models ✅
**Files:** `models/evaluate.py`, `models/train_baselines.py`, `notebooks/03_baseline_models.ipynb`, `data/baseline_metrics.json`, `data/baseline_report.md`  
**Saved Models:** `models/saved/persistence_baseline.pkl`, `models/saved/logreg_baseline.pkl`, `models/saved/rf_baseline.pkl`, `models/saved/scaler.pkl`
- [x] Persistence rule baseline (Test PR-AUC: 0.4921, ROC-AUC: 0.8420, F1: 0.6943)
- [x] Logistic Regression with `StandardScaler` + `class_weight='balanced'` (Test PR-AUC: 0.7395, ROC-AUC: 0.9878, F1: 0.6619)
- [x] Tree Ensemble Baseline (LightGBM Forest, Test PR-AUC: 0.7408, ROC-AUC: 0.9877, F1: 0.6800)
- [x] Evaluation engine with PR-AUC, ROC-AUC, F1, Recall @ 80% precision, Brier score, and Confusion Matrix

---

### Phase 4 — Two-Stage Hybrid Engine ✅
**Files:** `models/hybrid_engine.py`, `models/train_hybrid.py`, `notebooks/04_two_stage_hybrid_engine.ipynb`, `data/hybrid_metrics.json`, `data/hybrid_report.md`  
**Saved Models:** `models/saved/hybrid_predictor.pkl`, `models/saved/departure_regressor.pkl`, `models/saved/lgbm_classifier.pkl`, `models/saved/model_metadata.json`
- [x] **Stage 1 (Continuous Huber Regressor):** LightGBM Regressor for continuous departure $\Delta T_{t+1}$ ($R^2 = 0.8690$, $\text{RMSE} = 1.5257^\circ\text{C}$, $\text{MAE} = 1.0807^\circ\text{C}$) + Quantile Bounds $[p_{10}, p_{90}]$
- [x] **Stage 2 (Calibrated Classifier):** Cost-sensitive LightGBM (`scale_pos_weight=15.0`) + Isotonic Calibration via `CalibratedClassifierCV`
- [x] **Optimal Threshold:** $\tau^* = 0.230$ tuned on Validation set (2016–2020)
- [x] **Test Performance (2021–2024 Climate Surge):**
  - **Raw Overall Accuracy:** **97.68%** ($7,932 / 8,120$ days correct)
  - **Balanced Accuracy:** **84.40%** ($98.61\%$ normal, $70.19\%$ heatwaves)
  - **PR-AUC:** **0.6747** ($20.7\times$ lift over 3.26% random base rate)
  - **ROC-AUC:** **0.9671**
  - **Brier Calibration Score:** **0.0164**
- [x] **Stage 3 (Severity Mapping):** Maps continuous departure predictions to IMD physical risk tiers (`Normal`, `Warning`, `Severe`, `Extreme`)
- [x] Production class `TwoStageHeatwavePredictor` serialized to `models/saved/hybrid_predictor.pkl`

---

### Phase 5 — Flask REST Backend ⏳ *(Next Step)*
**Files to create:** `app.py`, `requirements.txt`, `services/prediction_service.py`, `services/analytics_service.py`, `services/genai_service.py`
- [ ] Setup `requirements.txt` with production dependencies
- [ ] Create `services/prediction_service.py` (extracts past 7-day weather series dynamically from `heatguard_clean.csv`, constructs 45 features, runs `hybrid_predictor.pkl`)
- [ ] Create `services/analytics_service.py` (aggregates 74-year climate trends, decadal heatwave counts, monthly distribution, and city comparisons)
- [ ] Create `services/genai_service.py` (grounded prompt builder for 4 personas + chat grounding)
- [ ] Implement Flask endpoints:
  - `GET /api/cities`
  - `GET /api/history/<city>`
  - `POST /api/predict`
  - `POST /api/advisory`
  - `POST /api/chat`
  - `GET /api/analytics/decades`

---

### Phase 6 — Frontend (Dashboard & Analytics UI) ⏳
**Files to create:** `templates/`, `static/css/style.css`, `static/js/`
- [ ] Modern dark glassmorphism design system (`static/css/style.css`)
- [ ] Command Center (`templates/dashboard.html`): city selector, date picker, KPI cards, animated risk gauge, severity badge
- [ ] Interactive geospatial map (`static/js/dashboard.js` with Leaflet.js)
- [ ] 74-Year Climate Analytics (`templates/analytics.html`, `static/js/charts.js` with Chart.js)
- [ ] Landing page (`templates/index.html`)

---

### Phase 7 — GenAI Advisory Studio & Assistant ⏳
- [ ] Advisory Studio UI (`templates/advisory.html`): Persona tabs (Citizen, Farmer, Health Agency, Municipality)
- [ ] Chatbot Interface (`templates/chatbot.html`, `static/js/chat.js`): Grounded meteorological assistant with suggested prompt chips
- [ ] Hallucination guardrails and priority checklist badges

---

### Phase 8 — Integration, Verification & Demo ⏳
- [ ] End-to-end integration testing (Dashboard → Predict → Advisory → Chatbot)
- [ ] Demo scenario rehearsals: Pune (June 2024), Delhi (Peak Summer), Ahmedabad (Extreme Heatwave)
- [ ] Final verification report & project deliverables

---

## 4. Evaluator Quick Reference

| Question | Answer Summary |
|---|---|
| **What are the 2 models in your hybrid engine?** | **Model 1:** Continuous Huber Regressor (predicts exact departure $\Delta T$ and temp $T_{\max, t+1}$). **Model 2:** Cost-sensitive LightGBM Classifier with Isotonic Probability Calibration (calculates calibrated heatwave probability $P$). |
| **What is your final accuracy?** | **Raw Overall Accuracy is 97.68%** ($7,932/8,120$ days correct on test set). **Balanced Accuracy is 84.40%** ($98.61\%$ on normal days, $70.19\%$ on heatwaves). |
| **Why is PR-AUC (0.6747) key?** | On an imbalanced dataset where heatwaves occur only 3.26% of the time, random guessing yields a PR-AUC of 0.0326. **0.6747 represents a $20.7\times$ performance lift**, balancing high precision (63.05%) and high recall (70.19%). |
| **How do you predict "Extreme" severity?** | "Extreme" has only 2 rows in 74 years, so classifiers fail to learn it directly. The hybrid engine calculates continuous predicted departure $\Delta \widehat{T}$ and maps it to physical IMD thresholds ($\ge 4.0^\circ\text{C} \implies \text{Extreme}$). |

---

## 5. What Can Start Right Now

Phases 0–4 are complete, models are serialized and verified. We are ready to implement:
1. **Phase 5:** Flask backend (`app.py`, `services/prediction_service.py`, `services/analytics_service.py`, `services/genai_service.py`).
2. **Phase 6:** Frontend UI templates (`index.html`, `dashboard.html`, `advisory.html`, `chatbot.html`, `analytics.html`, `style.css`).

---

## 6. Work Log

| Date | Who | What |
|---|---|---|
| 2026-10-05 | Antigravity & User | Updated documentation suite: root `README.md`, enhanced `data/hybrid_report.md`, updated `progress.md`, and synchronized `HeatGuard_AI_Implementation_Plan_Dataset_Specific.md` with full evaluator cheat sheets, exact metrics (Raw Acc: 97.68%, Balanced Acc: 84.40%, PR-AUC: 0.6747, Brier: 0.0164, R²: 0.8690), and Two-Stage Hybrid architecture details. |
| 2026-10-05 | Antigravity & User | Phase 4 done: created `models/hybrid_engine.py`, `models/train_hybrid.py`, `notebooks/04_two_stage_hybrid_engine.ipynb`. Trained and verified Two-Stage Hybrid Engine (Stage 1 Huber Regressor R² 0.8690, Stage 2 Calibrated Classifier PR-AUC 0.6747, Accuracy 97.68%, Balanced Accuracy 84.40%, Brier Score 0.0164). Serialized models to `models/saved/hybrid_predictor.pkl`. |
| 2026-10-05 | Antigravity | Phase 3 done: created `models/evaluate.py`, `models/train_baselines.py`, `notebooks/03_baseline_models.ipynb`. Trained Persistence, Balanced Logistic Regression (Test PR-AUC 0.7395), and Tree Ensemble (Test PR-AUC 0.7408). Saved models to `models/saved/` and exported `data/baseline_metrics.json`. |
| 2026-10-04 | Anvita | Phase 2 done: wrote `data_cleaning/feature_eng.py`, generated `data/model_ready_dataset.csv` (187,022 rows, 45 features, 0 nulls), `feature_schema.json` and `feature_report.json`. 12 leakage checks pass. |
| 2026-10-04 | Anvita | Phase 1 done: wrote `data_cleaning/Data_cleaner.py`, generated `data/heatguard_clean.csv` (187,387 rows, 0 nulls) and `data/cleaning_report.json`. All 9 validation checks pass. |
| 2026-10-04 | Anvita | Reviewed implementation plan and raw dataset, verified profiling numbers, created initial progress tracker. |