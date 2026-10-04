# HeatGuard AI — Team Progress Tracker

> **How to use this file:** Whenever you finish (or start) something, tick the box, add your name, and add a line to the [Work Log](#work-log) at the bottom. Commit it with your changes so everyone sees the latest status.
>
> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

**Last updated:** 2026-10-04
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

| Workstream | Covers phases | Owner |
|---|---|---|
| **A. Data & Features** | 1, 2 | _TBD_ (currently: **Anvita**, starting preprocessing) |
| **B. ML Modeling** | 3, 4 | _TBD_ |
| **C. Backend + GenAI** | 5, 7 | _TBD_ |
| **D. Frontend + Integration** | 6, 8 | _TBD_ |

> With 3 people, one person will need to take two workstreams. A natural pairing is **C + D** (or B + A). Decide this before Phase 3 starts.

---

## 3. Overall Progress

| Phase | Name | Status | Owner |
|---|---|---|---|
| 0 | Planning & dataset profiling | `[x]` Done | Team |
| 1 | Data preprocessing & calendar pipeline | `[~]` Starting now | Anvita |
| 2 | Zero-leakage feature engineering | `[ ]` Not started | _TBD_ |
| 3 | Baseline models | `[ ]` Not started | _TBD_ |
| 4 | LightGBM / CatBoost + two-stage hybrid engine | `[ ]` Not started | _TBD_ |
| 5 | Flask REST backend | `[ ]` Not started | _TBD_ |
| 6 | Dashboard, Leaflet map, analytics UI | `[ ]` Not started | _TBD_ |
| 7 | GenAI advisory + chatbot | `[ ]` Not started | _TBD_ |
| 8 | Integration, verification, demo | `[ ]` Not started | _TBD_ |

**Rough completion:** ~1 of 9 phases done.

---

## 4. Phase Details

### Phase 0 — Planning & dataset profiling ✅
- [x] Write implementation plan (`HeatGuard_AI_Implementation_Plan_Dataset_Specific.md`)
- [x] Add raw dataset to repo (`heatguard_raw.csv`)
- [x] Profile dataset (missing values, gaps, class balance, city climatology)

---

### Phase 1 — Data Preprocessing & Calendar Pipeline 🔨 *(Anvita — in progress)*
**Output:** `preprocessing/data_cleaner.py` + a cleaned CSV that Phase 2 can read.
**Blocks:** Phase 2 (and therefore everything after it).

- [ ] Create folder structure (`data/`, `preprocessing/`, `models/`)
- [ ] Load CSV, parse `date` as datetime, sort by `['city', 'date']`
- [ ] Confirm 0 duplicate `(city, date)` pairs (already verified once on the raw file)
- [ ] Impute 33 missing `temp_min` values by linear interpolation **within each city**
- [ ] Handle `rain`: fill NaN with `0.0` and add `rain_is_recorded` flag (0 = not measured)
- [ ] Detect the 40 calendar gaps and add a `gap_before` / `days_since_prev` column so Phase 2 can reset lag windows
- [ ] Sanity-check outliers (see findings below) and decide: keep, cap, or flag
- [ ] Save cleaned file (e.g. `data/heatguard_clean.csv`)
- [ ] Write a short validation script/notebook: row counts, null counts, gap counts before vs after

**Known data findings (verified on `heatguard_raw.csv`):**

| Check | Result |
|---|---|
| Shape | 187,387 rows × 14 columns ✅ matches the plan |
| Duplicates on `(city, date)` | 0 |
| Missing `temp_min` | 33 |
| Missing `rain` | 51,594 (27.5%), mostly Chennai & Mumbai before 2021 |
| Calendar gaps (>1 day) | 40, largest is **84 days** ending 2024-02-23 (same in several cities) |
| Class counts | Normal 185,596 · Warning 1,663 · Severe 126 · Extreme 2 |
| `temp_min == 0.0` | 1 row, **suspicious** (probably a placeholder, check it) |
| `rain > 300 mm` | 6 rows, max 1014.5 mm, **check whether these are real** |
| Thresholds | Bengaluru, Kolkata, Mumbai, Pune are constant at 40.0 °C. Delhi, Ahmedabad and Chennai vary by year. |

**⚠️ Things to flag to the team:**
1. The repo file is named **`heatguard_raw.csv`**, but the plan refers to `heatguard_clean.csv`. Our cleaning step is what produces the "clean" file, so the plan wording is just ahead of the work.
2. The Gantt chart in the plan shows some tasks as `:done` / `:active`. **That does not reflect reality**, since no code has been written yet. This file is the source of truth for status.
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

Phase 1 is in progress, but not everything is blocked by it:

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
| 2026-10-04 | Anvita | Reviewed implementation plan and raw dataset, verified profiling numbers, created this progress tracker. Starting Phase 1 (preprocessing). |

---

## 7. Open Decisions

- [ ] Who owns which workstream (see §2)
- [ ] Which LLM provider for the advisory/chatbot (Gemini / OpenAI / Anthropic / Hugging Face)
- [ ] How to treat suspicious values (`rain > 300 mm`, `temp_min == 0`): keep, cap, or flag
- [ ] Whether to attempt stretch models (TFT, Chronos, TimesFM) or stop at LightGBM/CatBoost
- [ ] Git workflow: branch per person + pull requests, or commit directly to `main`