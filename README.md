# HeatGuard AI

HeatGuard AI is a local Flask web application that uses historical daily weather records for seven Indian cities to estimate next-day heat risk, show historical climate summaries, and present audience-specific advisory text. It combines weather-data preparation, a scikit-learn prediction model, interactive dashboard pages, and an optional Google Gemini integration with a deterministic advisory/chat fallback.

The project is a historical-data prototype, not a live weather service or an official warning system. Its dataset ends on 21 June 2024.

## What the application does

- Select a city and an available historical date to view the observed weather context and next-day model estimate.
- Move the selected date forward by one day and rerun the historical-data prediction.
- Request forecasts for all seven cities and display the city results on the dashboard map.
- Browse historical trends, city comparisons, monthly patterns, records, and heatwave persistence.
- View a 14-day historical temperature trajectory with a next-day model estimate.
- Create advisory text and priority checklists for Citizen, Farmer, Health Agency, and Municipal Authority audiences.
- Generate one PDF for a single advisory or a combined PDF containing all four persona advisories for the selected city and date.
- Ask the HeatGuard Copilot questions using city/date context. The service attempts Gemini when configured; it also has a deterministic fallback.
- Run a feature-perturbation scenario comparison and a multi-step forecast endpoint. These are prototype model outputs, not validated climate projections.
- Generate a simulated alert-dispatch receipt. This endpoint does not contact emergency services, SMS, radio, or other real channels.
- Audit algorithmic fairness, Explainable AI (XAI) feature attributions, and GenAI advisory safety guardrails through the dedicated Responsible AI Hub (`/ethics`).

## Data and preparation

Project data files identify the OpenCity Urban Data Portal as the source. The exact catalog record, dataset identifier, license, and retrieval date were not retained in the repository; consult the portal and add a dataset-level citation before external publication.

The checked-in cleaned dataset contains 187,387 daily records from 1951-01-01 through 2024-06-21 for Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai, and Pune. The data-preparation pipeline:

- sorts records by city and date and checks duplicate city/date keys;
- fills missing minimum temperatures by interpolation within contiguous segments;
- records missing-rainfall provenance with a `rain_is_recorded` indicator;
- marks calendar gaps so lag and rolling windows do not cross them;
- builds 45 weather, lag, rolling, calendar, city, season, and known-threshold features;
- creates next-day heatwave, temperature-departure, and severity targets; and
- assigns chronological train/validation/test partitions by target date.

The feature pipeline and saved reports account for 187,022 eligible rows after requiring adequate history and a valid next-day target. Details are in `data/cleaning_report.json`, `data/feature_report.json`, and `data/feature_schema.json`.

## Prediction model

The saved runtime artifact is a fitted `TwoStageHeatwavePredictor` implemented in `models/hybrid_engine.py`. It contains:

- a `HistGradientBoostingRegressor` for next-day temperature departure;
- two `HistGradientBoostingRegressor` quantile heads for the 10th and 90th percentile estimates; and
- a `HistGradientBoostingClassifier` with cost-sensitive class weights wrapped in a sigmoid probability calibrator (`CalibratedClassifierCV`) fitted on the validation split.

The saved model uses an operating decision threshold of **0.38** (tuned to maximize balanced accuracy and sensitivity). The sigmoid calibration maps raw logits to well-calibrated posterior probabilities, eliminating false inflation while preserving over 91.8% sensitivity on held-out test data. The application maps model outputs to risk and severity labels using its threshold and IMD departure rules.

### Recomputed held-out test results

The following results were recomputed from the saved runtime model, the checked-in cleaned data, the repository's current 45-feature construction, and the 2021–2024 test partition. The operating threshold is the value stored in the loaded model artifact.

| Measure | Result |
|---|---:|
| Test observations | 8,120 |
| Heatwave test observations | 265 (3.264%) |
| Decision threshold | 0.35 |
| Accuracy | 96.18% |
| Balanced accuracy | 94.38% |
| PR-AUC (average precision) | 0.7294 |
| ROC-AUC | 0.9818 |
| Precision | 45.79% |
| Recall | 92.45% |
| Brier score | 0.0263 |
| Temperature-departure MAE | 1.070 °C |
| Temperature-departure RMSE | 1.490 °C |
| Temperature-departure R² | 0.8751 |

Confusion matrix at threshold 0.35: 7,565 true negatives, 290 false positives, 20 false negatives, and 245 true positives. These results describe the saved model artifact as evaluated above; they are not a claim of live-forecast performance.

## Generative AI and advisory behavior

The advisory service builds a structured context from the model output and selects instructions for one of four audiences. It currently requests `gemini-3.8-flash` when a Gemini API key is available; if the provider is unavailable or the request fails, the service uses its deterministic fallback. The initial integration run on 9 October 2026 encountered provider errors, but a later live request through `/api/advisory` successfully generated a Citizen advisory with Gemini. The integration suite also exercised the fallback path.

Prompt grounding and deterministic fallback reduce risk but do not guarantee generated text is error-free. Review advisories before any real-world use. They are not official weather warnings, medical advice, or emergency instructions.

## Responsible AI and algorithmic governance

HeatGuard AI includes a dedicated **Responsible AI Hub** (`/ethics`) that continuously monitors algorithmic fairness, explainability, and generative safety across three core pillars defined in `HeatGuardAI_Ethics.md`:

### Pillar 1: Fairness and geographic bias audit
- **Disaggregated 7-city evaluation:** Evaluates model performance separately across all 7 Indian metropolitan hubs (Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai, Pune) on 8,120 out-of-sample days (2021–2024).
- **Transparent confusion matrices:** Exposes real true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN) alongside precision, recall (sensitivity), specificity, and false alarm rates per metro.
- **Regional disparity tiers:** Classifies regional performance objectively (e.g. *High Sensitivity & High Reliability* for Ahmedabad at 96.5% recall vs. *Regional Sensitivity Variance* for Delhi at 87.5% recall vs. *Climatological Safe Haven* for Bengaluru where 0 heatwave days occurred).
- **Data representation audit:** Verifies dataset integrity across 187,022 historical daily observations (1951–2024), auditing sensor completeness and guaranteeing zero demographic or socioeconomic proxy variables are used.

### Pillar 2: Explainable AI (XAI)
- **Global feature importance ranking:** Ranks the top 10 atmospheric drivers (such as threshold proximity, observed max temperature, 3-day heat accumulation, and heatwave streak duration) derived from the fitted HistGradientBoosting estimators.
- **Local prediction factor attribution:** Decomposes any individual city and date forecast into its top 5 contributing factor drivers with directional indicators ($\uparrow$ warming, $\downarrow$ mitigating, $\leftrightarrow$ neutral) and calibrated probabilities (e.g. 60.2% for Delhi on 2024-05-28).
- **Plain-language forecast synthesis:** Provides readable, non-technical explanations of model predictions.
- **Causality disclaimer:** Explicitly discloses that feature attributions represent learned statistical correlations within the training dataset, not physical or causal proofs of atmospheric phenomena.

### Pillar 3: GenAI safety and advisory guardrails
- **Enforced safety policies:** Enforces four architectural guardrails: Risk-Level Alignment, Anti-Hallucination Guardrails, Zero-PII Privacy Protection, and Deterministic Failover.
- **Automated 5-point compliance testing:** Live testing interface evaluating advisory texts across 5 automated checks: (1) Risk Consistency, (2) Official Authority Grounding, (3) Actionable Safety Measures, (4) AI Provenance & Scope Disclosure, and (5) Zero-PII Regex Scanning.
- **Deterministic offline failover:** Guarantees instant failover to evidence-backed IMD Heat Action Plan checklists if external LLM APIs experience rate limits, 503 errors, or latency spikes.

### Responsible AI REST API endpoints

| Endpoint | Method | Purpose |
|---|:---:|---|
| `/api/ethics/overview` | `GET` | System-wide audit status, disparity summary, top features, and evaluation timestamps. |
| `/api/ethics/fairness` | `GET` | Disaggregated 7-city held-out evaluation table with complete confusion matrix metrics. |
| `/api/ethics/fairness/<city>` | `GET` | Single-city confusion matrix breakdown and localized ethical insight narrative. |
| `/api/ethics/representation` | `GET` | Historical data representation audit and sensor completeness across 187,022 observations. |
| `/api/ethics/explain` | `GET` | Global top-10 feature importance rankings, or local factor attribution when queried with `?city=...&date=...`. |
| `/api/ethics/audit-advisory` | `POST` | Automated 5-point safety and grounding compliance audit for arbitrary advisory text payloads. |
| `/api/ethics/safety-overview` | `GET` | GenAI safety posture, active guardrails status, fallback architecture, and policy disclosures. |

## Run locally

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`. The Gemini integration is optional; configure `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment to enable its request path. Do not commit API keys.

To run the integration checks:

```powershell
python test_suite.py
```

The suite was run on 9 October 2026 and its Flask page/API assertions passed. The test run exercised the deterministic GenAI fallback after external Gemini requests failed. A later end-to-end `/api/advisory` request verified successful Gemini generation using `gemini-3.8-flash`. Provider availability and quota can still affect later requests; the app falls back when a request fails. A passing endpoint/integration test does not validate operational forecast quality.

## Running locally

HeatGuard AI runs locally as a Flask application:

1. Create and activate a Python virtual environment (recommended Python 3.11 or 3.12).
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000/` in your browser.

This is a historical-data prototype and preview environment, not a live weather service or public alert system.

## Repository guide

| Path | Purpose |
|---|---|
| `app.py` | Flask pages and JSON endpoints. |
| `data/heatguard_raw.csv`, `data/heatguard_clean.csv` | Source and cleaned weather records present in this checkout. |
| `data_cleaning/Data_cleaner.py` | Cleaning and calendar-segmentation pipeline. |
| `data_cleaning/feature_eng.py` | 45-feature and next-day-target pipeline. |
| `models/hybrid_engine.py` | Saved model class and prediction logic. |
| `models/train_hybrid.py`, `models/train_baselines.py`, `models/evaluate.py` | Training and evaluation code. |
| `models/saved/` | Saved model artifacts and metadata. |
| `services/prediction_service.py` | Runtime feature extraction and inference from historical records. |
| `services/analytics_service.py` | Historical climate summaries. |
| `services/genai_service.py` | Persona advisories, optional Gemini requests, and deterministic fallback. |
| `services/ethics_service.py` | Responsible AI backend: 7-city fairness audit, Explainable AI (XAI), and advisory safety checks. |
| `templates/`, `static/` | Browser pages, styles, and JavaScript behavior. |
| `test_suite.py` | Flask integration checks. |
| `progress.md` | Implemented project status and known scope limits. |
| `HeatGuardAI_Ethics.md` | Responsible AI specification, fairness metrics, and safety audit protocol. |
| `HeatGuard_AI_Implementation_Plan_Dataset_Specific.md` | Implementation-focused architecture and data notes. |
| `data/hybrid_report.md` | Recomputed results for the currently saved model. |

## Data-source reference

OpenCity Urban Data Portal: <https://data.opencity.in/>. The exact source dataset record and its license need to be confirmed and cited before redistribution.
