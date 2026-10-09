# HeatGuard AI — Implemented System Overview

This document describes the project that exists in the repository, not proposed model tiers or unimplemented future work. HeatGuard AI is a historical-data prototype built as a local Flask application. It forecasts one day ahead from weather history available in the checked-in dataset and displays audience-oriented advisory content.

## 1. Project purpose and boundaries

The project brings together:

1. Daily weather-data cleaning and calendar-gap handling.
2. City-specific temporal feature engineering and next-day labels.
3. A saved scikit-learn model for temperature departure, uncertainty bounds, and heatwave risk scoring.
4. A Flask application with prediction, map, trajectory, and historical-analytics pages.
5. Four persona advisory formats and a context-aware chat interface, with optional Gemini requests and deterministic fallback behavior.

This is not a live weather ingest system, official warning service, medical decision tool, or real emergency dispatch integration. The dataset ends on 2024-06-21. The exact OpenCity dataset catalog record and license were not retained in the project files.

## 2. Dataset and cleaning

The cleaned data has 187,387 daily records dated 1951-01-01 through 2024-06-21 for Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai, and Pune. It contains daily maximum/minimum temperature, rainfall, city/date and calendar information, heatwave threshold, heatwave indicator, and severity label.

The cleaning implementation in `data_cleaning/Data_cleaner.py` parses and orders city/date rows, handles missing temperature and rainfall values, records rainfall observability, identifies date gaps, creates segment boundaries, and writes validation results. The checked-in report records 40 calendar gaps and interpolation for 34 missing minimum-temperature values. Missing rainfall is represented with a `rain_is_recorded` field so filled values are distinguishable from observations.

## 3. Feature and target construction

`data_cleaning/feature_eng.py` creates 45 model inputs. They include:

- current maximum/minimum temperature, rain, and diurnal range;
- temperature and rain lags, plus rolling temperature/rain statistics;
- heatwave persistence up to the observation date;
- cyclical day-of-year and month features, scaled year, city and season indicators;
- rainfall-observation status; and
- the known next-day climatological threshold and its gap from current temperature.

Windows are grouped by city and contiguous segment. The pipeline requires full recent history and a valid next calendar day. Targets are next-day heatwave status, temperature departure, severity, and maximum temperature. The report records 187,022 eligible rows from the 187,387 cleaned observations. Split assignment is chronological by target date: train through 2015, validation 2016–2020, and test 2021–2024.

## 4. Prediction implementation

The checked-in runtime artifact is `models/saved/hybrid_predictor.pkl`, an instance of `TwoStageHeatwavePredictor` in `models/hybrid_engine.py`. It contains:

- one `HistGradientBoostingRegressor` using squared-error loss for temperature departure;
- two `HistGradientBoostingRegressor` quantile heads for the 10th and 90th percentile departure estimates; and
- one `HistGradientBoostingClassifier` with `class_weight="balanced"` for event probabilities.

The artifact's stored decision threshold is 0.35. `predict_full` combines the binary model score with the departure estimate and quantile bounds to return risk and severity labels. The current `fit()` implementation uses the base classifier directly; it does not fit an isotonic or sigmoid calibration wrapper. Therefore, documentation and UI should call its probability output an estimated score/probability, not a calibrated probability.

The following figures were recomputed by loading the saved artifact, regenerating the current feature matrix in memory from `data/heatguard_clean.csv`, applying the chronological 2021–2024 test split, and measuring the saved artifact at its stored threshold:

| Test measure | Result |
|---|---:|
| Rows / heatwave positives | 8,120 / 265 (3.264%) |
| Threshold | 0.35 |
| Accuracy / balanced accuracy | 96.18% / 94.38% |
| PR-AUC / ROC-AUC | 0.7294 / 0.9818 |
| Precision / recall | 45.79% / 92.45% |
| Brier score | 0.0263 |
| Departure MAE / RMSE / R² | 1.070 °C / 1.490 °C / 0.8751 |
| Confusion matrix (TN, FP, FN, TP) | 7,565; 290; 20; 245 |

The test-set values describe this saved artifact and the current feature pipeline. They are not evidence of performance on live weather data.

## 5. Flask services and pages

`app.py` serves Jinja pages and JSON endpoints. `services/prediction_service.py` loads the historical records/model, constructs inference features for a city and date, and returns predictions. `services/analytics_service.py` computes historical summaries. `services/genai_service.py` formats persona advisories and chat responses.

Implemented pages:

- **Home:** project introduction and navigation.
- **Dashboard:** historical city/date selection, next-day model result, temperature context, map, trajectory chart, all-city synchronization, and simulated scenario/dispatch controls.
- **Analytics:** city comparisons, historical totals, decadal/monthly summaries, records, and persistence.
- **Advisory Studio:** four audience selections, generated advisory and checklist, copy action, and PDF export.
- **Chatbot:** conversational questions with city/date context, external Gemini attempt when configured, and a deterministic fallback.

Dashboard controls include a date-advance action that changes the selected historical date by one day, all-seven-city inference for the selected date, and a combined PDF export that requests Citizen, Farmer, Health Agency, and Municipal Authority advisories for the selected city/date. The individual Advisory Studio can export the selected persona's advisory to a PDF.

## 6. Endpoint inventory

| Endpoint | Implemented purpose |
|---|---|
| `/api/cities`, `/api/presets` | City metadata and demo presets. |
| `/api/predict`, `/api/predict-all` | One-city or all-seven-city historical-data model inference. |
| `/api/history/<city>`, `/api/heatmap-data` | Observed trajectory plus next-day output and map payload. |
| `/api/analytics/summary`, `/api/analytics/decades`, `/api/analytics/city-comparison`, `/api/analytics/monthly`, `/api/analytics/records`, `/api/analytics/persistence` | Historical climate summaries. |
| `/api/advisory`, `/api/chat` | Persona advisory/checklist and conversational response. |
| `/api/simulate` | Model response after changing selected input features. It is a prototype scenario comparison. |
| `/api/forecast/<city>` | Multi-step forecast endpoint using the service's iterative procedure. |
| `/api/alerts/dispatch` | Returns a simulated dispatch receipt; no message is sent to external recipients. |
| `/api/health`, `/api/export/<city>` | Local service status and historical city CSV export. |

## 7. GenAI behavior

Advisories use model-provided city/date, risk, severity, temperature, departure, and uncertainty context and select a prompt/checklist appropriate to one of four personas: Citizen, Farmer, Health Agency, and Municipal Authority. The chat service can use structured prediction context and historical summaries. If `GEMINI_API_KEY` or `GOOGLE_API_KEY` is configured, the service attempts a Google Gemini request. When a request cannot complete, deterministic response logic is used.

The integration suite verified that the advisory and chat endpoints return responses through the fallback path. The initial 9 October 2026 run encountered provider errors/timeouts. After the app model was updated to `gemini-3.8-flash`, a later live request through `/api/advisory` successfully generated an LLM-backed response. Provider availability and quota can still interrupt future calls, and the fallback remains in place. Grounding controls reduce but do not eliminate the possibility of inaccurate text. Human review is required before advice is used outside a demonstration.

## 8. Verification and known limits

`test_suite.py` was run on 9 October 2026. Its page and API assertions passed for page rendering, city and preset data, city and all-city predictions, history, map data, four persona advisories, chat, six analytics endpoints, a scenario, multi-step forecast response, simulated dispatch, and local health status.

Known limits:

- The stored observations stop at 2024-06-21; model outputs should be treated as historical-data demonstrations.
- OpenCity source-page, license, and retrieval metadata need a dataset-level citation.
- Gemini generation was verified through `/api/advisory` on 9 October 2026 with the updated model. Provider availability and quota can still cause fallback responses.
- The multi-step forecast and feature perturbation outputs have not been validated as meteorological forecast products or climate projections.
- The alert endpoint returns a demo receipt only.
- The model's output probabilities are not calibrated by the current fitting code.

## 9. Main implementation files

| File | Responsibility |
|---|---|
| `data_cleaning/Data_cleaner.py` | Cleaning and calendar segmentation. |
| `data_cleaning/feature_eng.py` | Feature construction, targets, splits, validation. |
| `models/hybrid_engine.py` | Current estimator and inference mapping. |
| `models/train_hybrid.py`, `models/train_baselines.py`, `models/evaluate.py` | Training and evaluation code. |
| `models/saved/hybrid_predictor.pkl` | Fitted runtime predictor artifact. |
| `models/saved/model_metadata.json` | Current model features, threshold, and estimator description. |
| `services/prediction_service.py` | Historical feature extraction and predictions. |
| `services/analytics_service.py` | Historical analytics. |
| `services/genai_service.py` | Advisory, chat, external Gemini attempt, deterministic fallback. |
| `app.py` | Flask page and API handlers. |
| `templates/`, `static/` | User interface and browser behavior. |
| `test_suite.py` | Flask integration checks. |
