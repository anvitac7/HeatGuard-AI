"""
HeatGuard AI - Main Flask Web Application & REST API Server.

Serves the interactive glassmorphism frontend (Dashboard, Leaflet Maps,
74-Year Climate Analytics, Advisory Studio, AI Chatbot) and provides REST APIs
for dynamic feature extraction, Two-Stage Hybrid ML inference, and GenAI grounding.
"""
from __future__ import annotations

import sys
import types

# NumPy 1.x <-> 2.x unpickling cross-version compatibility bridge
try:
    import numpy as np
    if not hasattr(np, "_core") and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
except Exception:
    pass

import logging
import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from services.analytics_service import AnalyticsService
from services.genai_service import GenAIService
from services.prediction_service import PredictionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("heatguard_app")

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "heatguard-ai-secret-2026")


class StripApiPrefixMiddleware:
    """WSGI middleware to normalize PATH_INFO from Vercel serverless rewrites."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path_info = environ.get("PATH_INFO", "")
        for prefix in ["/api/index.py", "/api/index"]:
            if path_info == prefix or path_info == prefix + "/":
                environ["PATH_INFO"] = "/"
                break
            elif path_info.startswith(prefix + "/"):
                environ["PATH_INFO"] = path_info[len(prefix):]
                break
        return self.wsgi_app(environ, start_response)


app.wsgi_app = StripApiPrefixMiddleware(app.wsgi_app)

# Lazy-loaded services
_pred_service: PredictionService = None
_analytics_service: AnalyticsService = None
_genai_service: GenAIService = None


def get_prediction_service() -> PredictionService:
    global _pred_service
    if _pred_service is None:
        _pred_service = PredictionService.get_instance()
    return _pred_service


def get_analytics_service() -> AnalyticsService:
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AnalyticsService.get_instance()
    return _analytics_service


def get_genai_service() -> GenAIService:
    global _genai_service
    if _genai_service is None:
        _genai_service = GenAIService.get_instance()
    return _genai_service


# ----------------------------------------------------------------------------------------
# HTML Page Routes
# ----------------------------------------------------------------------------------------
@app.route("/")
@app.route("/dashboard")
@app.route("/api/index")
@app.route("/api/index.py")
def dashboard_view():
    """Live Heatwave Command Center."""
    pred_svc = get_prediction_service()
    cities = pred_svc.get_cities()
    presets = pred_svc.get_demo_presets()
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        cities=cities,
        presets=presets,
    )


@app.route("/analytics")
def analytics_view():
    """74-Year Historical Climate Analytics Explorer."""
    analytics_svc = get_analytics_service()
    summary = analytics_svc.get_summary()
    return render_template(
        "analytics.html",
        active_page="analytics",
        summary=summary,
    )


@app.route("/advisory")
def advisory_view():
    """Multi-Stakeholder GenAI Advisory Studio."""
    pred_svc = get_prediction_service()
    cities = pred_svc.get_cities()
    presets = pred_svc.get_demo_presets()
    return render_template(
        "advisory.html",
        active_page="advisory",
        cities=cities,
        presets=presets,
    )


@app.route("/chatbot")
def chatbot_view():
    """Dedicated Heatwave AI Assistant Page."""
    pred_svc = get_prediction_service()
    cities = pred_svc.get_cities()
    return render_template(
        "chatbot.html",
        active_page="chatbot",
        cities=cities,
    )


# ----------------------------------------------------------------------------------------
# REST API Endpoints: Prediction & Weather
# ----------------------------------------------------------------------------------------
@app.route("/api/cities", methods=["GET"])
def api_get_cities():
    """Returns all 7 cities with metadata and historical summaries."""
    pred_svc = get_prediction_service()
    return jsonify({"status": "success", "cities": pred_svc.get_cities()})


@app.route("/api/presets", methods=["GET"])
def api_get_presets():
    """Returns curated demo scenarios for quick testing."""
    pred_svc = get_prediction_service()
    return jsonify({"status": "success", "presets": pred_svc.get_demo_presets()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Run Two-Stage Hybrid inference for a single city on a specific observation date.
    Payload: {"city": "Pune", "date": "2024-06-18"}
    """
    data = request.get_json(force=True, silent=True) or {}
    city = data.get("city", "Delhi")
    date_str = data.get("date", "2024-05-28")

    pred_svc = get_prediction_service()
    try:
        result = pred_svc.predict_city(city, date_str)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        log.error("Prediction error for %s on %s: %s", city, date_str, e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/predict-all", methods=["POST"])
def api_predict_all():
    """
    Returns predictions for all 7 metropolitan hubs on the requested date
    to populate the Leaflet map and heat layer simultaneously.
    Payload: {"date": "2024-05-28"}
    """
    data = request.get_json(force=True, silent=True) or {}
    date_str = data.get("date", "2024-05-28")

    pred_svc = get_prediction_service()
    try:
        results = pred_svc.predict_all_cities(date_str)
        return jsonify({"status": "success", "date": date_str, "cities": results})
    except Exception as e:
        log.error("Error predicting all cities on %s: %s", date_str, e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/history/<city>", methods=["GET"])
def api_get_history(city: str):
    """
    Returns 14-day historical trajectory plus the next-day prediction
    for Chart.js rendering.
    Query params: ?date=2024-05-28&days=14
    """
    date_str = request.args.get("date", "2024-05-28")
    n_days = int(request.args.get("days", 14))

    pred_svc = get_prediction_service()
    try:
        traj = pred_svc.get_city_trajectory(city, date_str, n_days=n_days)
        return jsonify({"status": "success", "trajectory": traj})
    except Exception as e:
        log.error("Error retrieving history for %s: %s", city, e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/heatmap-data", methods=["GET"])
def api_get_heatmap_data():
    """
    Provides normalized thermal intensity points [lat, lon, intensity]
    for Leaflet.heat visualization.
    Query param: ?date=2024-05-28
    """
    date_str = request.args.get("date", "2024-05-28")
    pred_svc = get_prediction_service()
    try:
        predictions = pred_svc.predict_all_cities(date_str)
        points = []
        for p in predictions:
            pred = p["prediction"]
            # Intensity normalized between 25°C and 48°C
            temp = pred["predicted_temp_max"]
            intensity = max(0.1, min(1.0, (temp - 25.0) / 23.0))
            points.append({
                "city": p["city"],
                "lat": p["latitude"],
                "lng": p["longitude"],
                "temp": temp,
                "intensity": round(intensity, 3),
                "departure": pred["predicted_departure"],
                "risk": pred["risk_level"],
                "severity": pred["severity"],
                "color": pred["alert_color"],
            })
        return jsonify({"status": "success", "date": date_str, "points": points})
    except Exception as e:
        log.error("Error generating heatmap points: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 400


# ----------------------------------------------------------------------------------------
# REST API Endpoints: GenAI Advisory & Chatbot
# ----------------------------------------------------------------------------------------
@app.route("/api/advisory", methods=["POST"])
def api_generate_advisory():
    """
    Generates a persona-tailored advisory grounded in verified ML predictions.
    Payload: {"city": "Pune", "date": "2024-06-18", "audience": "Farmer"}
    """
    data = request.get_json(force=True, silent=True) or {}
    city = data.get("city", "Delhi")
    date_str = data.get("date", "2024-05-28")
    audience = data.get("audience", "Citizen")

    pred_svc = get_prediction_service()
    genai_svc = get_genai_service()

    try:
        context = pred_svc.predict_city(city, date_str)
        advisory = genai_svc.generate_advisory(context, audience=audience)
        return jsonify({"status": "success", "advisory": advisory})
    except Exception as e:
        log.error("Error generating advisory: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Context-aware natural language assistant grounded in Two-Stage ML predictions and 74-year climate history.
    Payload: {"message": "tell me the weather of delhi tomorrow"}
    Optional: {"city": "Delhi", "date": "2024-05-28"}
    """
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    explicit_city = data.get("city")
    explicit_date = data.get("date")
    history = data.get("history", [])

    if not message:
        return jsonify({"status": "error", "message": "Message is required"}), 400

    pred_svc = get_prediction_service()
    genai_svc = get_genai_service()

    try:
        reply = genai_svc.chat_interactive(
            message=message,
            explicit_city=explicit_city,
            explicit_date=explicit_date,
            chat_history=history,
            pred_svc=pred_svc,
        )
        return jsonify({"status": "success", "chat": reply})
    except Exception as e:
        log.error("Error in chat assistant: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# ----------------------------------------------------------------------------------------
# REST API Endpoints: Historical Climate Analytics
# ----------------------------------------------------------------------------------------
@app.route("/api/analytics/summary", methods=["GET"])
def api_analytics_summary():
    svc = get_analytics_service()
    return jsonify({"status": "success", "data": svc.get_summary()})


@app.route("/api/analytics/decades", methods=["GET"])
def api_analytics_decades():
    svc = get_analytics_service()
    return jsonify({"status": "success", "data": svc.get_decades()})


@app.route("/api/analytics/city-comparison", methods=["GET"])
def api_analytics_city_comparison():
    svc = get_analytics_service()
    return jsonify({"status": "success", "data": svc.get_city_comparison()})


@app.route("/api/analytics/monthly", methods=["GET"])
def api_analytics_monthly():
    svc = get_analytics_service()
    return jsonify({"status": "success", "data": svc.get_monthly()})


@app.route("/api/analytics/records", methods=["GET"])
def api_analytics_records():
    svc = get_analytics_service()
    return jsonify({"status": "success", "data": svc.get_records()})


@app.route("/api/analytics/persistence", methods=["GET"])
def api_analytics_persistence():
    svc = get_analytics_service()
    return jsonify({"status": "success", "data": svc.get_persistence()})


# ----------------------------------------------------------------------------------------
# Advanced REST API Endpoints: Climate Stress Simulator & Forecasting
# ----------------------------------------------------------------------------------------
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Counterfactual climate stress simulation:
    Payload: {
        "city": "Ahmedabad",
        "date": "2024-05-23",
        "temp_offset": 2.0,
        "night_temp_offset": 1.5,
        "rain_offset": -5.0
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    city = data.get("city", "Delhi")
    date_str = data.get("date", "2024-05-28")
    temp_offset = float(data.get("temp_offset", 0.0))
    night_temp_offset = float(data.get("night_temp_offset", 0.0))
    rain_offset = float(data.get("rain_offset", 0.0))

    pred_svc = get_prediction_service()
    try:
        sim = pred_svc.simulate_scenario(
            city=city,
            date_str=date_str,
            temp_offset=temp_offset,
            night_temp_offset=night_temp_offset,
            rain_offset=rain_offset,
        )
        return jsonify({"status": "success", "simulation": sim})
    except Exception as e:
        log.error("Simulation error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/forecast/<city>", methods=["GET"])
def api_forecast(city: str):
    """
    Autoregressive multi-horizon forecast (1 to 7 days forward).
    Query params: ?date=2024-05-28&days=7
    """
    date_str = request.args.get("date", "2024-05-28")
    days = int(request.args.get("days", 7))
    pred_svc = get_prediction_service()
    try:
        fc = pred_svc.forecast_multi_day(city, date_str, days=days)
        return jsonify({"status": "success", "forecast": fc})
    except Exception as e:
        log.error("Multi-day forecast error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/alerts/dispatch", methods=["POST"])
def api_dispatch_alert():
    """
    Simulates sending instant emergency alerts via SMS / Municipal radio / Public sirens.
    Payload: {"city": "Ahmedabad", "date": "2024-05-23", "channels": ["SMS", "Hospital_Surge", "Radio"]}
    """
    data = request.get_json(force=True, silent=True) or {}
    city = data.get("city", "Ahmedabad")
    date_str = data.get("date", "2024-05-23")
    channels = data.get("channels", ["Municipal_Disaster_Cell", "SMS_Broadcast", "Hospital_Trauma_Units"])

    pred_svc = get_prediction_service()
    try:
        pred_res = pred_svc.predict_city(city, date_str)
        pred = pred_res["prediction"]

        import uuid, datetime
        dispatch_receipt = {
            "dispatch_id": f"HG-DISP-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.datetime.now().isoformat(),
            "target_city": city,
            "forecast_date": pred_res["target_date"],
            "risk_level": pred["risk_level"],
            "severity": pred["severity"],
            "predicted_temp": pred["predicted_temp_max"],
            "channels_broadcast": channels,
            "status": "DISPATCHED",
            "message": f"[HEAT EMERGENCY] HeatGuard AI alert for {city}: {pred['predicted_temp_max']}°C predicted. Risk: {pred['risk_level']}."
        }
        return jsonify({"status": "success", "receipt": dispatch_receipt})
    except Exception as e:
        log.error("Dispatch alert error: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/health", methods=["GET"])
def api_health():
    """System health, loaded model status, and telemetry."""
    pred_svc = get_prediction_service()
    status = pred_svc.get_health_status()
    return jsonify({"status": "success", "health": status})


@app.route("/api/export/<city>", methods=["GET"])
def api_export_city(city: str):
    """Exports historical city weather records as CSV."""
    from flask import Response
    pred_svc = get_prediction_service()
    if city not in pred_svc.city_dfs:
        return jsonify({"status": "error", "message": f"Unknown city {city}"}), 404

    df_city = pred_svc.city_dfs[city].copy()
    csv_data = df_city.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=HeatGuard_{city}_1951_2024.csv"}
    )



@app.errorhandler(500)
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    """Returns JSON error with diagnostic details instead of unhandled crash."""
    if isinstance(e, HTTPException):
        return e
    import traceback
    log.exception("Unhandled server exception: %s", e)
    return jsonify({
        "status": "error",
        "error_type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }), 500


# ----------------------------------------------------------------------------------------
# Application Startup
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info("Starting HeatGuard AI Flask Server on http://127.0.0.1:%d ...", port)
    app.run(host="0.0.0.0", port=port, debug=True)
