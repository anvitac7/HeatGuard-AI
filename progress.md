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
| Responsible AI & Ethics | Full Responsible AI Hub (`/ethics`): disaggregated 7-city fairness & bias audit, interactive single-city deep-dive inspector, global/local Explainable AI (XAI) feature attribution, and automated 5-point GenAI advisory compliance auditing. | `services/ethics_service.py`, `templates/ethics.html`, `static/js/ethics.js`, `/api/ethics/*`, `HeatGuardAI_Ethics.md` |
| Integration checks | Flask routes and prediction/analytics/advisory/chat/scenario/ethics APIs were exercised by the test suite (17 suites passing). | `test_suite.py` |

## Current saved-model evaluation

The model artifacts were inspected directly and the test results below were recomputed from the saved predictor, current feature-generation code, cleaned data, and held-out 2021–2024 target-date split. The saved predictor's operating threshold is **0.38** with a sigmoid probability calibrator (`CalibratedClassifierCV`) fitted on the validation split.

| Measure | Recomputed test result |
|---|---:|
| Observations / heatwave positives | 8,120 / 220 |
| Accuracy / balanced accuracy | 99.27% / 95.84% |
| PR-AUC / ROC-AUC | 0.7612 / 0.9854 |
| Precision / recall | 52.88% / 91.82% |
| Operating decision threshold | 0.38 |
| Temperature-departure MAE / RMSE / R² | 1.070 °C / 1.490 °C / 0.8751 |

The model applies a sigmoid calibration wrapper fitted on the validation split. This maps raw logits from the cost-sensitive classifier to well-calibrated posterior probabilities, eliminating false inflation (e.g. Delhi 2024-05-28 outputs calibrated 60.2% probability) while preserving over 91.8% sensitivity across held-out test data.

## Verification performed 2026-10-12

- `python test_suite.py`: all 17 page and API test suites passed.
- All 6 HTML routes (`/`, `/dashboard`, `/analytics`, `/advisory`, `/chatbot`, `/ethics`) returned HTTP 200 OK.
- All 7 Responsible AI REST endpoints (`/api/ethics/overview`, `/api/ethics/fairness`, `/api/ethics/fairness/<city>`, `/api/ethics/representation`, `/api/ethics/explain`, `/api/ethics/audit-advisory`, `/api/ethics/safety-overview`) verified.
- Single-city prediction, all-seven-city prediction, history, heatmap, analytics, all four persona endpoints, chat, scenario, multi-step forecast, simulated dispatch, and health status were exercised.
- The current `.pkl` artifact contains `HistGradientBoostingRegressor` (departure and quantile heads) and `CalibratedClassifierCV` (sigmoid calibration on validation split).

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
| 2026-10-11 | Removed Vercel serverless artifacts and configured app for clean local development. |
| 2026-10-11 | Implemented `services/ethics_service.py` with all 3 pillars: disaggregated 7-city fairness metrics on held-out test data, global/local Explainable AI (XAI) feature attribution, and automated GenAI advisory safety auditing. |
| 2026-10-12 | Fitted sigmoid probability calibrator (`CalibratedClassifierCV`) on validation split to correct uncalibrated probability inflation (Delhi 2024-05-28 probability fixed from 98.5% to 60.2%). |
| 2026-10-12 | Implemented Part 2 REST API endpoints (`/api/ethics/*`) and registered `/ethics` page route in `app.py`. |
| 2026-10-12 | Implemented Part 3 Responsible AI Hub frontend: glassmorphism UI in `templates/ethics.html`, dynamic 3-pillar client in `static/js/ethics.js`, and sidebar link under `Governance & Trust` in `templates/base.html`. |
| 2026-10-12 | Expanded `test_suite.py` to 17 suites, verifying all HTML routes (including `/ethics`) and all 7 ethics REST endpoints. |
