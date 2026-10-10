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
- a `HistGradientBoostingClassifier` using `class_weight="balanced"` for heatwave probability scores.

The saved model uses a decision threshold of **0.35**. The current implementation does not apply a probability-calibration wrapper; probability values should not be described as calibrated. The application maps the model outputs to risk and severity labels using its current threshold and departure rules.

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

## Deployment and hosting

HeatGuard AI is designed to run both locally and as a lightweight serverless deployment:

- Local development: run the Flask app directly with `python app.py`, then open `http://127.0.0.1:5000/`.
- Serverless hosting: the project includes `api/index.py` and `vercel.json`, which expose the same Flask app through a Vercel rewrite and keep the project root on the Python import path.
- Static assets: browser UI assets are served from `static/`, with `public/` kept as an additional fallback for deployment scenarios.

This is a historical-data prototype and preview environment, not a live weather service or public alert system.

## Repository guide

| Path | Purpose |
|---|---|
| `app.py` | Flask pages and JSON endpoints. |
| `api/index.py` | Vercel serverless entrypoint that loads the Flask app. |
| `vercel.json` | Vercel rewrite and runtime configuration for the app. |
| `data/heatguard_raw.csv`, `data/heatguard_clean.csv` | Source and cleaned weather records present in this checkout. |
| `data_cleaning/Data_cleaner.py` | Cleaning and calendar-segmentation pipeline. |
| `data_cleaning/feature_eng.py` | 45-feature and next-day-target pipeline. |
| `models/hybrid_engine.py` | Saved model class and prediction logic. |
| `models/train_hybrid.py`, `models/train_baselines.py`, `models/evaluate.py` | Training and evaluation code. |
| `models/saved/` | Saved model artifacts and metadata. |
| `services/prediction_service.py` | Runtime feature extraction and inference from historical records. |
| `services/analytics_service.py` | Historical climate summaries. |
| `services/genai_service.py` | Persona advisories, optional Gemini requests, and deterministic fallback. |
| `templates/`, `static/`, `public/` | Browser pages, styles, JavaScript behavior, and deployment assets. |
| `test_suite.py` | Flask integration checks. |
| `progress.md` | Implemented project status and known scope limits. |
| `HeatGuard_AI_Implementation_Plan_Dataset_Specific.md` | Implementation-focused architecture and data notes. |
| `data/hybrid_report.md` | Recomputed results for the currently saved model. |

## Data-source reference

OpenCity Urban Data Portal: <https://data.opencity.in/>. The exact source dataset record and its license need to be confirmed and cited before redistribution.
