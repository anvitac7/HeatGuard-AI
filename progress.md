# HeatGuard AI — Team Progress Tracker

> **How to use this file:** Whenever you finish (or start) something, tick the box, add your name, and add a line to the [Work Log](#work-log) at the bottom. Commit it with your changes so everyone sees the latest status.
>
> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Last updated:** 2026-10-04 (Phase 1 code complete, pending review)
**Repo:** https://github.com/anvitac7/HeatGuard-AI
**Reference doc:** `HeatGuard_AI_Implementation_Plan_Dataset_Specific.md`

---

## 1. Project Snapshot

HeatGuard AI predicts **next-day heatwaves** for 7 Indian cities and turns the predictions into stakeholder advisories through a web dashboard and chatbot.

| Item | Value |
|---|---|
| Dataset | 187,387 daily rows, 14 columns, 1951-01-01 to 2024-06-21 |
| Cities | Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai, Pune |
| Targets (day t+1) | `is_heatwave_day` (binary), `temp_max - threshold` (regression), `severity` (4 classes) |
| Class imbalance | Only 1,791 heatwave days (~0.96%), so accuracy is misleading. Use PR-AUC. |
| Split | Train 1951–2015, Validation 2016–2020, Test 2021–2024 (chronological, **no shuffling**) |
| Stack | Python, scikit-learn, LightGBM/CatBoost, Flask, Leaflet.js, Chart.js, LLM API |

---

## 2. Team & Roles

Roles are **not decided yet**. Suggested split below; fill in names once agreed.

| Workstream | Covers phases |
|---|---|---|
| **A. Data & Features** | 1, 2 |
| **B. ML Modeling** | 3, 4 | 
| **C. Backend + GenAI** | 5, 7 | 
| **D. Frontend + Integration** | 6, 8 |

> With 3 people, one person will need to take two workstreams. A natural pairing is **C + D** (or B + A). Decide this before Phase 3 starts.

---

## 3. Overall Progress

| Phase | Name | Status | Owner |
|---|---|---|---|
| 0 | Planning & dataset profiling | `[x]` Done | 
| 1 | Data preprocessing & calendar pipeline | `[x]` Code done, needs teammate review |
| 2 | Zero-leakage feature engineering | `[ ]` Not started |
| 3 | Baseline models | `[ ]` Not started |
| 4 | LightGBM / CatBoost + two-stage hybrid engine | `[ ]` Not started | 
| 5 | Flask REST backend | `[ ]` Not started | 
| 6 | Dashboard, Leaflet map, analytics UI | `[ ]` Not started |
| 7 | GenAI advisory + chatbot | `[ ]` Not started | 
| 8 | Integration, verification, demo | `[ ]` Not started |

**Rough completion:** 2 of 9 phases done (Phase 1 awaiting review).

---

## 4. Phase Details

### Phase 0 — Planning & dataset profiling ✅
- [x] Write implementation plan (`HeatGuard_AI_Implementation_Plan_Dataset_Specific.md`)
- [x] Add raw dataset to repo (`heatguard_raw.csv`)
- [x] Profile dataset (missing values, gaps, class balance, city climatology)

---

### Phase 1 — Data Preprocessing & Calendar Pipeline ✅ *(Anvita — code complete, needs review)*
**Files:** `preprocessing/data_cleaner.py`, `data/heatguard_clean.csv`, `data/cleaning_report.json`
**Run it:** `python preprocessing/data_cleaner.py` (from repo root; reads `heatguard_raw.csv`)
**Unblocks:** Phase 2 (and therefore everything after it).

- [x] Create folder structure (`data/`, `preprocessing/`)
- [x] Load CSV, parse `date` as datetime, sort by `['city', 'date']`
- [x] Confirm 0 duplicate `(city, date)` pairs (script drops any it finds)
- [x] Impute missing `temp_min` by linear interpolation **within each city and segment** (34 values: 33 NaN + 1 zero placeholder)
- [x] Handle `rain`: NaN filled with `0.0` and `rain_is_recorded` flag added (0 = not measured)
- [x] Detect the 40 calendar gaps and add `days_since_prev`, `gap_before`, `segment_id`
- [x] Sanity-check outliers and decide (see decisions below)
- [x] Save cleaned file `data/heatguard_clean.csv`
- [x] Automated validation inside the script (9 checks, run fails loudly if any break)
- [ ] **Teammate review** of the cleaning decisions below (5 min read)
- [ ] Move `heatguard_raw.csv` into `data/` (the script finds it in either place)

**Cleaning decisions made (please sanity-check these):**

| Issue | What we found | What we did |
|---|---|---|
| `temp_min` NaN (33) | 2019-01-11 and 2024-04-16 are missing in several cities, plus 20 scattered in Chennai | Linear interpolation inside a segment, max 3 days in a row. All imputed values lie between their neighbours. Flagged in `temp_min_imputed`. |
| `temp_min == 0.0` (1) | Bengaluru 2019-01-11, the **same date** all 6 other cities are NaN, so it is a missing-data placeholder | Treated as NaN, then interpolated (to 14.1 °C) |
| `rain` NaN (51,594) | Chennai and Mumbai have no rain data before 2021 | Filled with 0.0 and `rain_is_recorded = 0`. Chennai is 4.4% recorded, Mumbai 4.3%. |
| `rain` 1014.5 / 1011.7 mm | Kolkata and Mumbai on 2023-12-01 (December, neighbouring days are 0 mm) | Impossible values, so set to "not recorded" (0.0 and flag 0). Rule: >300 mm outside Jun–Sep is invalid. |
| `rain` 303–414 mm (4 rows) | Delhi 1957, Kolkata 1978, Pune 1967 and 2005, all in monsoon months | Kept, since they are plausible heavy-monsoon days. |
| Calendar gaps (40) | 29 of 2 days, 6 of 3 days, 5 of 84 days (ends 2024-02-23) | **Rows are not inserted.** Each city is split into segments (`segment_id`) so Phase 2 can reset lags. |
| Labels | `is_heatwave_day` and `severity` still match the departure rule on all rows | Untouched |

**Output schema:** the original 14 columns plus 5 new ones: `days_since_prev`, `gap_before`, `segment_id`, `temp_min_imputed`, `rain_is_recorded`. The cleaned file still has 187,387 rows and 0 nulls.

**⚠️ For whoever does Phase 2:**
1. Compute **every** lag and rolling feature with `groupby(['city', 'segment_id'])`, never just `groupby('city')`.
2. The next-day target must only exist when the next row is exactly 1 day later. Use `days_since_prev` of the following row, and drop the target if it is not 1.
3. `rain_is_recorded` is a real feature and should go into the model.
4. Do **not** use `temp_min_imputed`, `gap_before` or `segment_id` as model inputs. They are bookkeeping columns.

**Things to flag to the team:**
1. The repo file is named **`heatguard_raw.csv`**, but the plan refers to `heatguard_clean.csv`. Our cleaning step now produces the "clean" file.
2. The Gantt chart in the plan shows some tasks as `:done` / `:active`. **That does not reflect reality**, so this file is the source of truth for status.
3. Only 2 "Extreme" rows exist, so the severity classifier cannot learn that class. Severity should come from the regressor's predicted departure (as the plan describes), not from direct classification.

---

### Phase 2 — Zero-Leakage Feature Engineering
**Needs:** Phase 1 output. **Output:** `preprocessing/feature_engineering.py`, `data/model_ready_dataset.csv`.

- [ ] Lag features: `temp_max` / `temp_min` lags 1, 2, 3, 7; `rain` lags 1, 3, 7
- [ ] Rolling features: 3d/7d averages, 7d max, 7d rain sum, 3d diurnal range average
- [ ] Cyclical encodings: `sin/cos` of day-of-year and month
- [ ] Derived: `diurnal_temp_range`, `temp_departure_today`, `heatwave_streak_days`, `year_scaled`
- [ ] One-hot encode `city` and `season`
- [ ] Reset lags/rolling windows across calendar gaps (no cross-gap contamination)
- [ ] Create the 3 next-day targets (shifted by −1 within each city); drop the last row per city
- [ ] Verify there is no leakage: Day-t `is_heatwave_day`, `severity` and `heatwave_threshold` are **not** used as raw inputs
- [ ] Chronological split: Train ≤2015 / Val 2016–2020 / Test 2021–2024

---

### Phase 3 — Baseline Models
**Needs:** Phase 2. **Output:** `models/train_models.py` (baseline part), metrics table.

- [ ] Persistence rule baseline
- [ ] Logistic Regression (`class_weight='balanced'`)
- [ ] Random Forest (300 trees, `balanced_subsample`)
- [ ] Evaluation helper: PR-AUC, F1 (macro and heatwave), Recall @ 80% precision, Brier score, ROC-AUC
- [ ] Record all results in one comparison table

---

### Phase 4 — Advanced Models & Two-Stage Hybrid Engine
**Needs:** Phase 3 (evaluation helper).

- [ ] LightGBM classifier (`scale_pos_weight` or focal loss), tuned on the validation set
- [ ] CatBoost comparison
- [ ] LightGBM regressor for next-day `temp_max` / departure
- [ ] Probability calibration (`CalibratedClassifierCV`) and threshold tuning (~0.35 starting point)
- [ ] Severity tier mapper (Warning / Severe / Extreme from predicted departure)
- [ ] Save `heatwave_lgbm_model.pkl`, `departure_regressor.pkl`, `model_metadata.json`
- [ ] Write `heatguard_evaluation_report.md`
- [ ] *(Stretch)* EBM, TFT, PatchTST, Chronos / TimesFM / TabPFN. Only attempt if time allows.

---

### Phase 5 — Flask REST Backend
**Needs:** saved models from Phase 4. *Can start early using stub/dummy predictions.*

- [ ] `app.py` skeleton + `requirements.txt`
- [ ] `GET /api/cities`
- [ ] `GET /api/history/<city>`
- [ ] `POST /api/predict` (builds lag features on the fly, runs two-stage model)
- [ ] `GET /api/analytics/decades`
- [ ] `POST /api/advisory` and `POST /api/chat` (wired up in Phase 7)
- [ ] `services/prediction_service.py` and `services/analytics_service.py`

---

### Phase 6 — Frontend (Dashboard & Analytics)
**Needs:** API contract from Phase 5. *Can start early against mock JSON.*

- [ ] Base layout + `style.css` (dark glassmorphism theme)
- [ ] Dashboard: city selector, date picker, KPI cards, risk gauge, severity badge
- [ ] Leaflet map with 7 city markers coloured by risk
- [ ] Analytics page: decadal surge chart, city comparison, monthly seasonality
- [ ] Landing page (`index.html`)

---

### Phase 7 — GenAI Advisory & Chatbot
**Needs:** `/api/predict` output format (can be mocked). *Can start in parallel with Phases 3–4.*

- [ ] Choose LLM provider and set up API key handling (use environment variables, **never commit keys**)
- [ ] Grounded prompt template with 4 personas: Citizen, Farmer, Health Agency, Municipality
- [ ] `services/genai_service.py`
- [ ] Guardrails: numbers come only from model output, no diagnosis, no emergency declarations
- [ ] Advisory Studio page (`advisory.html`)
- [ ] Chatbot backend (city-history retrieval) and `chatbot.html` / `chat.js`

---

### Phase 8 — Integration, Verification & Demo
- [ ] End-to-end test: Dashboard → Predict → Advisory → Chat
- [ ] Rehearse the demo scenario (Pune / Delhi / Ahmedabad)
- [ ] Final README with setup instructions
- [ ] Final report / presentation slides

---

## 5. What Can Start Right Now (for teammates)

Phase 1 is done (pending review), so **Phase 2 can start now**. Beyond that, these are also unblocked:

| Teammate could start… | Why it's unblocked |
|---|---|
| **Phase 7:** prompt templates, persona design, LLM provider setup | Only needs the JSON shape `{city, date, temp_max, prob, risk, severity}`, which the plan already defines |
| **Phase 6:** HTML/CSS layout, Leaflet map, charts with mock data | Needs only the API contract from the plan |
| **Phase 5:** Flask skeleton and `/api/cities`, `/api/history`, `/api/analytics/decades` | These only read the raw data, no models needed |
| **Phase 3:** write the evaluation-metrics helper and model-training scaffolding | Can be tested on dummy data until Phase 2 lands |

**Blocked until Phase 2 is finished:** actual model training (Phases 3–4) and real `/api/predict`.

---

## 6. Work Log

Newest entries first. Format: `date — name — what was done`.

| Date | Who | What |
|---|---|---|
| 2026-10-04 | Anvita | Phase 1 done: wrote `preprocessing/data_cleaner.py`, generated `data/heatguard_clean.csv` (187,387 rows, 0 nulls) and `data/cleaning_report.json`. All 9 validation checks pass. Needs teammate review. |
| 2026-10-04 | Anvita | Reviewed implementation plan and raw dataset, verified profiling numbers, created this progress tracker. |

---

## 7. Open Decisions

- [ ] Who owns which workstream (see §2)
- [ ] Which LLM provider for the advisory/chatbot (Gemini / OpenAI / Anthropic / Hugging Face)
- [x] How to treat suspicious values: decided in Phase 1 (see table). Teammates can still challenge it in review.
- [ ] Whether to attempt stretch models (TFT, Chronos, TimesFM) or stop at LightGBM/CatBoost
- [ ] Git workflow: branch per person + pull requests, or commit directly to `main`