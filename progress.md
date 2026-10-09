# HeatGuard AI — Project Status

**Last updated:** 2026-10-09
**Scope:** Implemented local historical-data prototype; no live weather feed or real emergency alert delivery.

## Project summary

HeatGuard AI loads historical daily weather observations for seven Indian cities, prepares city/date features, estimates next-day heat risk, and presents forecasts and historical summaries through a Flask web application. Advisory text is tailored to four audiences. The app can attempt an external Gemini request when configured and uses a deterministic fallback when the request is not available.

## Implemented components

| Area | Implemented behavior | Evidence |
|---|---|---|
| Weather dataset | Cleaned daily records for Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai, and Pune, spanning 1951-01-01 to 2024-06-21. | `data/heatguard_clean.csv`, `data/cleaning_report.json` |
| Data preparation | City/date sorting, duplicate checks, missing-value handling, calendar-gap segmentation, and cleaning validation. | `data_cleaning/Data_cleaner.py` |
| Feature engineering | 45 lagged, rolling, current-weather, calendar, city/season, and threshold-context features; next-day targets; chronological splits. | `data_cleaning/feature_eng.py`, `data/feature_schema.json`, `data/feature_report.json` |
| Prediction | Departure estimate, quantile bounds, heatwave score, risk label, and severity label using saved scikit-learn HistGradientBoosting estimators. | `models/hybrid_engine.py`, `models/saved/hybrid_predictor.pkl` |
| Baselines and evaluation | Persistence, logistic-regression, and tree baseline artifacts plus evaluation helpers. | `models/train_baselines.py`, `models/evaluate.py`, `data/baseline_report.md` |
| Historical analytics | Decadal counts, city comparisons, monthly distribution, records, and persistence summaries. | `services/analytics_service.py`, `/api/analytics/*` |
| Browser application | Dashboard, city/date controls, next-day date advance, all-city predictions/map, trajectory chart, analytics page, advisory studio, and chat page. | `templates/`, `static/` |
| Advisory and chat | Four persona-specific responses, action checklists, optional Gemini request, deterministic fallback. | `services/genai_service.py`, `/api/advisory`, `/api/chat` |
| PDF outputs | Single-persona PDF from the Advisory Studio and combined four-persona PDF from the dashboard. | `static/js/advisory.js`, `static/js/dashboard.js` |
| Scenario and forecast APIs | Feature-perturbation scenario comparison and a multi-step forecast endpoint. | `/api/simulate`, `/api/forecast/<city>` |
| Simulated dispatch | Creates a demo receipt from the prediction; no external alert is transmitted. | `/api/alerts/dispatch` |
| Integration checks | Flask routes and prediction/analytics/advisory/chat/scenario APIs were exercised by the test suite. | `test_suite.py` |

## Current saved-model evaluation

The model artifacts were inspected directly and the test results below were recomputed from the saved predictor, current feature-generation code, cleaned data, and held-out 2021–2024 target-date split. The saved predictor's operating threshold is **0.35**.

| Measure | Recomputed test result |
|---|---:|
| Observations / heatwave positives | 8,120 / 265 |
| Accuracy / balanced accuracy | 96.18% / 94.38% |
| PR-AUC / ROC-AUC | 0.7294 / 0.9818 |
| Precision / recall | 45.79% / 92.45% |
| Brier score | 0.0263 |
| Departure MAE / RMSE / R² | 1.070 °C / 1.490 °C / 0.8751 |

The stored probability outputs are not wrapped in a calibration estimator. The Brier score is reported as an evaluation metric; it does not establish that probabilities are calibrated.

## Verification performed 2026-10-09

- `python test_suite.py`: all page and API assertions passed.
- Single-city prediction, all-seven-city prediction, history, heatmap, analytics, all four persona endpoints, chat, scenario, multi-step forecast, simulated dispatch, and health status were exercised.
- The initial integration run encountered Gemini provider errors/timeouts, and the deterministic fallback supplied its advisory/chat responses. After updating the configured model to `gemini-3.8-flash`, a later `/api/advisory` request successfully generated an LLM-backed response.
- The current `.pkl` artifact was loaded and evaluated. Its estimators are `HistGradientBoostingRegressor` (departure and quantile heads) and `HistGradientBoostingClassifier` with balanced class weights.

## Scope limits and follow-up

- Weather observations in this checkout end on 21 June 2024; predictions are based on historical records, not current observed weather.
- The exact OpenCity dataset catalog page, license, and retrieval date still need to be recorded.
- Gemini generation was verified through `/api/advisory` with the current model on 2026-10-09. Provider availability and quota can still cause failures, in which case the service uses its deterministic fallback.
- The multi-step forecast and feature-perturbation scenario are prototype outputs, not validated weather forecasts or climate projections.
- Alert dispatch is a local simulation; it does not contact municipal, medical, SMS, radio, or public-siren services.
- The saved benchmark outputs were refreshed against the currently loaded model artifact in `data/hybrid_report.md` and `data/hybrid_metrics.json`.

## Work log

| Date | Work |
|---|---|
| 2026-10-09 | Added dashboard generation of a combined PDF for Citizen, Farmer, Health Agency, and Municipal Authority advisories. |
| 2026-10-09 | Added single-persona advisory PDF export, all-city map synchronization, and selected-date advancement. |
| 2026-10-09 | Ran the Flask integration suite and recomputed held-out test metrics from the saved runtime model. |
| 2026-10-09 | Revised project documentation and showcase report to describe implemented behavior and its current limits. |
