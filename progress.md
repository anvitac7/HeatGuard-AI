# HeatGuard AI — Team Progress Tracker

> **How to use this file:** Whenever you finish (or start) something, tick the box, add your name, and add a line to the [Work Log](#work-log) at the bottom. Commit it with your changes so everyone sees the latest status.
>
> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Last updated:** 2026-10-04 (Phases 1 and 2 code complete, pending review)
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
| 2 | Zero-leakage feature engineering | `[x]` Code done, needs teammate review | 
| 3 | Baseline models | `[ ]` Not started | 
| 4 | LightGBM / CatBoost + two-stage hybrid engine | `[ ]` Not started | 
| 5 | Flask REST backend | `[ ]` Not started | 
| 6 | Dashboard, Leaflet map, analytics UI | `[ ]` Not started | 
| 7 | GenAI advisory + chatbot | `[ ]` Not started | 
| 8 | Integration, verification, demo | `[ ]` Not started | 

**Rough completion:** 3 of 9 phases done (Phases 1 and 2 awaiting review).

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

### Phase 2 — Zero-Leakage Feature Engineering ✅ *(Anvita — code complete, needs review)*
**Files:** `preprocessing/feature_engineering.py`, `data/model_ready_dataset.csv`, `data/feature_schema.json`, `data/feature_report.json`
**Run it:** `python preprocessing/feature_engineering.py` (reads `data/heatguard_clean.csv`, takes a few minutes at most)
**Unblocks:** Phases 3 and 4 (model training).

- [x] Lag features: `temp_max` / `temp_min` lags 1, 2, 3, 7; `rain` lags 1, 3, 7
- [x] Rolling features: 3d/7d averages, 7d max, 7d rain sum, 3d diurnal range average
- [x] Cyclical encodings: `sin/cos` of day-of-year and month
- [x] Derived: `diurnal_temp_range`, `temp_departure_today`, `heatwave_streak_days`, `year_scaled`
- [x] One-hot encode `city` and `season` (string columns kept too, for CatBoost)
- [x] Lags and rolling windows computed inside `(city, segment_id)`, so none cross a calendar gap
- [x] Create the 3 next-day targets plus `target_temp_max_next_day` for the Stage 1 regressor
- [x] Leakage verification: 12 automated checks (see below), all passing
- [x] Chronological split label in a `split` column
- [ ] **Teammate review** of the decisions below

**Output:** 187,022 rows, 57 columns, 0 nulls. 45 of the columns are model features. The full list, grouped, is in `data/feature_schema.json`, which is what Phase 3 should read to know what to feed the models.

**Split (by the date of the target day, so no training row has a target in the validation period):**

| Split | Rows | Heatwave next day | Positive rate |
|---|---|---|---|
| Train (to 2015) | 166,131 | 1,395 | 0.84% |
| Val (2016–2020) | 12,771 | 125 | 0.98% |
| Test (2021–2024) | 8,120 | 265 | 3.26% |

**How leakage is checked:** the script recomputes a random sample of 3,000 rows from the clean data using plain date lookups (independent of the pandas `shift` code) and compares lags, rolling windows, the heatwave streak and all targets. It also checks that no forbidden column is a feature, the target date is exactly t+1, the split is chronological, and the labels agree with each other. I also fed it deliberately leaky data (tomorrow's temperature as "lag1", and today's label as the target) and it caught both.

**Decisions made (please sanity-check these):**

| Decision | Why |
|---|---|
| Rows without a full 7-day history are dropped (318 rows), plus rows with no valid next-day target (47 rows) | Keeps every window inside one segment. 365 rows lost in total (0.2%). |
| `is_heatwave_day`, `severity`, `heatwave_threshold` are kept in the file as **reference columns, never features** | The persistence baseline and evaluation need them, and they are listed under `reference_columns_never_inputs` in the schema |
| **Added** `threshold_next_day` and `gap_to_next_threshold` as features (not in the original plan) | The IMD threshold follows a fixed calendar per city and day of year (Delhi goes from 40 to 44.6 °C across the season), so tomorrow's threshold is known today. This is not leakage, but it is an addition to the plan, so **please confirm you are happy with it**. Drop both if not. |
| Float features rounded to 4 decimals | Removes float32 noise and cuts the file from 98 MB to 59 MB. Targets and reference columns are left exact. |
| `model_ready_dataset.csv` is in `.gitignore` | It is regenerable in one command and is 59 MB. The clean CSV (19 MB) is still committed. |

**⚠️ For whoever does Phase 3 and 4:**
1. **The test period looks different from training.** The test positive rate is 3.26% against 0.84% in training, which is the 2021–2024 climate surge the plan describes. Expect the test PR-AUC to differ from validation, and tune thresholds on validation only.
2. **Extreme severity exists only in the test set** (2 rows). Train and validation have 0, so no classifier can learn it. Derive severity from the regressor's predicted departure.
3. **The test set has 8,120 rows**, not the ~10,600 the plan estimated. The plan's number was a rough guess, and the real count is what the dataset contains.
4. Sanity check passed: P(heatwave tomorrow | heatwave today) is 63.15%, matching the plan's 63.11%.
5. Select features from `feature_schema.json` (`feature_columns`). Never use the `reference_columns_never_inputs`, any `target_*` column, or the meta columns.

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

Phases 1 and 2 are done (pending review), so **Phases 3 and 4 (model training) can start now**. These are also unblocked:

| Teammate could start… | Why it's unblocked |
|---|---|
| **Phase 7:** prompt templates, persona design, LLM provider setup | Only needs the JSON shape `{city, date, temp_max, prob, risk, severity}`, which the plan already defines |
| **Phase 6:** HTML/CSS layout, Leaflet map, charts with mock data | Needs only the API contract from the plan |
| **Phase 5:** Flask skeleton and `/api/cities`, `/api/history`, `/api/analytics/decades` | These only read the raw data, no models needed |
| **Phase 3:** baselines and the evaluation helper | `data/model_ready_dataset.csv` and `feature_schema.json` are ready |

**Blocked until Phase 4 is finished:** the real `/api/predict` (needs the saved models).

---

## 6. Work Log

Newest entries first. Format: `date — name — what was done`.

| Date | Who | What |
|---|---|---|
| 2026-10-04 | Anvita | Phase 2 done: wrote `preprocessing/feature_engineering.py`, generated `data/model_ready_dataset.csv` (187,022 rows, 45 features, 0 nulls), `feature_schema.json` and `feature_report.json`. 12 leakage checks pass. Needs teammate review. |
| 2026-10-04 | Anvita | Phase 1 done: wrote `preprocessing/data_cleaner.py`, generated `data/heatguard_clean.csv` (187,387 rows, 0 nulls) and `data/cleaning_report.json`. All 9 validation checks pass. Needs teammate review. |
| 2026-10-04 | Anvita | Reviewed implementation plan and raw dataset, verified profiling numbers, created this progress tracker. |

---

## 7. Open Decisions

- [ ] Who owns which workstream (see §2)
- [ ] Which LLM provider for the advisory/chatbot (Gemini / OpenAI / Anthropic / Hugging Face)
- [x] How to treat suspicious values: decided in Phase 1 (see table). Teammates can still challenge it in review.
- [ ] Whether to attempt stretch models (TFT, Chronos, TimesFM) or stop at LightGBM/CatBoost
- [ ] Confirm the two extra features `threshold_next_day` and `gap_to_next_threshold` (Phase 2 addition)
- [ ] Git workflow: branch per person + pull requests, or commit directly to `main`