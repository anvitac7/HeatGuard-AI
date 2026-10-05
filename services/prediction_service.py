"""
HeatGuard AI - Prediction Service.

Extracts dynamic historical lag features and executes the Two-Stage Hybrid
Engine for next-day heatwave prediction, departure regression, uncertainty
quantification, and physics-grounded severity classification.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.hybrid_engine import TwoStageHeatwavePredictor

log = logging.getLogger("prediction_service")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "heatguard_clean.csv"
MODEL_PATH = REPO_ROOT / "models" / "saved" / "hybrid_predictor.pkl"

CITIES = ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Kolkata", "Mumbai", "Pune"]
SEASONS = ["Monsoon", "Post-Monsoon", "Summer", "Winter"]

CITY_COORDINATES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "threshold": 40.0, "state": "NCT of Delhi"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "threshold": 40.0, "state": "Gujarat"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "threshold": 40.0, "state": "Tamil Nadu"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "threshold": 40.0, "state": "West Bengal"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "threshold": 40.0, "state": "Maharashtra"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "threshold": 37.0, "state": "Maharashtra"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "threshold": 40.0, "state": "Karnataka"},
}

DEMO_PRESETS = [
    {
        "id": "ahmedabad-extreme",
        "city": "Ahmedabad",
        "date": "2024-05-23",
        "label": "Ahmedabad 46.6°C Extreme Heatwave (May 2024)",
        "badge": "EXTREME",
        "color": "#ef4444",
        "description": "Historical peak event: Ahmedabad touched 46.6°C (+4.77°C above IMD threshold) during the unprecedented 2024 climate surge.",
    },
    {
        "id": "delhi-peak",
        "city": "Delhi",
        "date": "2024-05-28",
        "label": "Delhi 44.3°C Severe Summer Peak (May 2024)",
        "badge": "HIGH RISK",
        "color": "#f97316",
        "description": "North Indian heat dome: Delhi endured multi-week heatwave conditions exceeding 44°C with dry westerly winds.",
    },
    {
        "id": "pune-june",
        "city": "Pune",
        "date": "2024-06-18",
        "label": "Pune Pre-Monsoon Heat Stress (June 2024)",
        "badge": "ELEVATED",
        "color": "#f59e0b",
        "description": "High pre-monsoon temperature departure before the arrival of the southwest monsoon.",
    },
    {
        "id": "chennai-summer",
        "city": "Chennai",
        "date": "2024-05-15",
        "label": "Chennai Coastal Heat & Humidity (May 2024)",
        "badge": "WARNING",
        "color": "#f97316",
        "description": "Coastal thermal accumulation with high nighttime temperatures and humid heat index.",
    },
    {
        "id": "bengaluru-mild",
        "city": "Bengaluru",
        "date": "2024-05-28",
        "label": "Bengaluru Climatological Safe Haven (May 2024)",
        "badge": "NORMAL",
        "color": "#10b981",
        "description": "Plateau elevation: Bengaluru's temperatures remained under 32°C, well below the 40°C threshold.",
    },
]


class PredictionService:
    """Singleton service for dataset access and two-stage model inference."""

    _instance: Optional[PredictionService] = None

    def __init__(self):
        log.info("Initializing PredictionService...")
        self.df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        self.df["date_str"] = self.df["date"].dt.strftime("%Y-%m-%d")
        self.df.sort_values(["city", "date"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Build fast lookup indexes per city
        self.city_dfs: Dict[str, pd.DataFrame] = {}
        self.city_date_sets: Dict[str, set] = {}
        for c in CITIES:
            sub = self.df[self.df["city"] == c].copy().reset_index(drop=True)
            self.city_dfs[c] = sub
            self.city_date_sets[c] = set(sub["date_str"].values)

        log.info("Loading TwoStageHeatwavePredictor from %s...", MODEL_PATH)
        self.predictor = TwoStageHeatwavePredictor.load(MODEL_PATH)
        log.info("PredictionService ready with %d rows across %d cities.", len(self.df), len(CITIES))

    @classmethod
    def get_instance(cls) -> PredictionService:
        if cls._instance is None:
            cls._instance = PredictionService()
        return cls._instance

    def get_cities(self) -> List[Dict[str, Any]]:
        """Return city list with coordinates, thresholds, and data range."""
        result = []
        for city in CITIES:
            sub = self.city_dfs[city]
            coords = CITY_COORDINATES.get(city, {"lat": 20.0, "lon": 78.0, "threshold": 40.0, "state": "India"})
            hw_count = int(sub["is_heatwave_day"].sum())
            max_temp = float(sub["temp_max"].max())
            avg_temp = float(sub["temp_max"].mean())

            result.append({
                "city": city,
                "state": coords["state"],
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "threshold": coords["threshold"],
                "min_date": sub["date_str"].min(),
                "max_date": sub["date_str"].max(),
                "total_records": len(sub),
                "total_heatwave_days": hw_count,
                "all_time_peak_temp": round(max_temp, 2),
                "historical_avg_temp": round(avg_temp, 2),
            })
        return result

    def get_demo_presets(self) -> List[Dict[str, Any]]:
        return DEMO_PRESETS

    def extract_features(self, city: str, date_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Dynamically extract the 45 leak-free features for the given city and date.
        Also returns context metadata (observed current conditions, streak, thresholds).
        """
        if city not in self.city_dfs:
            raise ValueError(f"Unknown city: {city}. Valid cities: {CITIES}")

        sub = self.city_dfs[city]
        matched = sub[sub["date_str"] <= date_str]
        if len(matched) < 8:
            # Fallback to the first 8 rows if date is before dataset start
            matched = sub.head(8)

        t_row = matched.iloc[-1]
        actual_date_str = str(t_row["date_str"])

        # Fetch recent 8 rows ending at day t
        recent = matched.tail(8).reset_index(drop=True)
        idx = len(recent) - 1

        t_max = recent["temp_max"].values
        t_min = recent["temp_min"].values
        rain = recent["rain"].values

        cur_threshold = float(t_row["heatwave_threshold"])
        cur_tmax = float(t_max[idx])
        cur_tmin = float(t_min[idx])
        cur_rain = float(rain[idx])

        # Compute rolling window statistics
        t_max_3d_avg = float(np.mean(t_max[max(0, idx - 2):idx + 1]))
        t_max_7d_avg = float(np.mean(t_max[max(0, idx - 6):idx + 1]))
        t_max_7d_max = float(np.max(t_max[max(0, idx - 6):idx + 1]))
        t_min_3d_avg = float(np.mean(t_min[max(0, idx - 2):idx + 1]))
        t_min_7d_avg = float(np.mean(t_min[max(0, idx - 6):idx + 1]))
        rain_7d_sum = float(np.sum(rain[max(0, idx - 6):idx + 1]))
        diurnal_range_3d_avg = float(np.mean(
            t_max[max(0, idx - 2):idx + 1] - t_min[max(0, idx - 2):idx + 1]
        ))

        # Heatwave streak up to day t
        hw_flags = recent["is_heatwave_day"].values
        streak = 0
        for k in range(idx, -1, -1):
            if hw_flags[k] == 1:
                streak += 1
            else:
                break

        # Next day threshold lookup
        next_rows = sub[sub["date_str"] > actual_date_str]
        if len(next_rows) > 0:
            next_row = next_rows.iloc[0]
            next_date_str = str(next_row["date_str"])
            next_threshold = float(next_row["heatwave_threshold"])
        else:
            # Climatological roll-forward by 1 day
            next_dt = t_row["date"] + pd.Timedelta(days=1)
            next_date_str = next_dt.strftime("%Y-%m-%d")
            next_threshold = cur_threshold

        # Construct exact 45 features required by hybrid_predictor
        feat: Dict[str, Any] = {
            "temp_max": cur_tmax,
            "temp_min": cur_tmin,
            "rain": cur_rain,
            "diurnal_temp_range": cur_tmax - cur_tmin,
            "temp_departure_today": cur_tmax - cur_threshold,

            # Lags
            "temp_max_lag1": float(t_max[idx - 1]),
            "temp_max_lag2": float(t_max[idx - 2]),
            "temp_max_lag3": float(t_max[idx - 3]),
            "temp_max_lag7": float(t_max[idx - 7]),

            "temp_min_lag1": float(t_min[idx - 1]),
            "temp_min_lag2": float(t_min[idx - 2]),
            "temp_min_lag3": float(t_min[idx - 3]),
            "temp_min_lag7": float(t_min[idx - 7]),

            "rain_lag1": float(rain[idx - 1]),
            "rain_lag3": float(rain[idx - 3]),
            "rain_lag7": float(rain[idx - 7]),

            # Rolling stats
            "temp_max_3d_avg": t_max_3d_avg,
            "temp_max_7d_avg": t_max_7d_avg,
            "temp_max_7d_max": t_max_7d_max,
            "temp_min_3d_avg": t_min_3d_avg,
            "temp_min_7d_avg": t_min_7d_avg,
            "rain_7d_sum": rain_7d_sum,
            "diurnal_range_3d_avg": diurnal_range_3d_avg,

            # Streak & Cyclical
            "heatwave_streak_days": streak,
            "sin_doy": float(np.sin(2 * np.pi * t_row["day_of_year"] / 365.25)),
            "cos_doy": float(np.cos(2 * np.pi * t_row["day_of_year"] / 365.25)),
            "sin_month": float(np.sin(2 * np.pi * t_row["month"] / 12.0)),
            "cos_month": float(np.cos(2 * np.pi * t_row["month"] / 12.0)),
            "year_scaled": float((t_row["year"] - 1951) / (2024 - 1951)),

            # Spatial & Missingness
            "latitude": float(t_row["latitude"]),
            "longitude": float(t_row["longitude"]),
            "rain_is_recorded": int(t_row["rain_is_recorded"]),

            # Known future targets
            "threshold_next_day": next_threshold,
            "gap_to_next_threshold": cur_tmax - next_threshold,
        }

        # One-hot city & season
        for c in CITIES:
            feat[f"city_{c}"] = int(city == c)
        for s in SEASONS:
            feat[f"season_{s}"] = int(t_row["season"] == s)

        # Context metadata
        context = {
            "city": city,
            "observation_date": actual_date_str,
            "target_date": next_date_str,
            "current_temp_max": round(cur_tmax, 2),
            "current_temp_min": round(cur_tmin, 2),
            "current_departure": round(cur_tmax - cur_threshold, 2),
            "current_threshold": round(cur_threshold, 2),
            "next_threshold": round(next_threshold, 2),
            "temp_max_3d_avg": round(t_max_3d_avg, 2),
            "temp_max_7d_avg": round(t_max_7d_avg, 2),
            "heatwave_streak_days": streak,
            "is_heatwave_today": int(t_row["is_heatwave_day"]),
            "rain": round(cur_rain, 2),
            "rain_is_recorded": int(t_row["rain_is_recorded"]),
            "latitude": CITY_COORDINATES[city]["lat"],
            "longitude": CITY_COORDINATES[city]["lon"],
            "state": CITY_COORDINATES[city]["state"],
        }

        return feat, context

    def predict_city(self, city: str, date_str: str) -> Dict[str, Any]:
        """Run two-stage model prediction for a single city on a specific date."""
        feat, context = self.extract_features(city, date_str)
        pred = self.predictor.predict_single(feat)

        prob_pct = round(pred["prob_heatwave"] * 100, 1)
        departure = pred["predicted_departure"]
        pred_temp = pred["predicted_temp_max"]
        risk_level = pred["risk_level"]
        severity = pred["predicted_severity"]

        # Alert color mapping
        color_map = {
            "LOW": "#10b981",       # Green
            "MODERATE": "#f59e0b",  # Amber
            "HIGH": "#f97316",      # Fiery orange
            "VERY HIGH": "#ef4444", # Red
            "EXTREME": "#ec4899",   # Neon magenta
        }

        # Trend direction
        temp_diff_3d = context["current_temp_max"] - context["temp_max_3d_avg"]
        if temp_diff_3d > 0.5:
            trend = "Rising (+{:.1f}°C vs 3d avg)".format(temp_diff_3d)
            trend_icon = "trending-up"
        elif temp_diff_3d < -0.5:
            trend = "Cooling ({:.1f}°C vs 3d avg)".format(temp_diff_3d)
            trend_icon = "trending-down"
        else:
            trend = "Stable (±0.5°C vs 3d avg)"
            trend_icon = "minus"

        return {
            **context,
            "prediction": {
                "probability": pred["prob_heatwave"],
                "probability_pct": prob_pct,
                "is_heatwave": bool(pred["is_heatwave_predicted"]),
                "predicted_departure": departure,
                "predicted_temp_max": pred_temp,
                "temp_lower_p10": pred["temp_lower_p10"],
                "temp_upper_p90": pred["temp_upper_p90"],
                "risk_level": risk_level,
                "severity": severity,
                "alert_color": color_map.get(risk_level, "#f59e0b"),
                "trend": trend,
                "trend_icon": trend_icon,
                "threshold_applied": self.predictor.optimal_threshold,
                "model_version": "TwoStageHybrid-LightGBM-v1.0",
            }
        }

    def predict_all_cities(self, date_str: str) -> List[Dict[str, Any]]:
        """Run predictions for all 7 cities to populate the Leaflet map and heat layers."""
        results = []
        for city in CITIES:
            try:
                res = self.predict_city(city, date_str)
                results.append(res)
            except Exception as e:
                log.error("Failed prediction for %s on %s: %s", city, date_str, e)
        return results

    def get_city_trajectory(self, city: str, end_date_str: str, n_days: int = 14) -> Dict[str, Any]:
        """
        Return the recent n-day historical trajectory plus the next-day prediction
        for seamless visualization in Chart.js.
        """
        if city not in self.city_dfs:
            raise ValueError(f"Unknown city: {city}")

        sub = self.city_dfs[city]
        matched = sub[sub["date_str"] <= end_date_str].tail(n_days).reset_index(drop=True)

        dates = matched["date_str"].tolist()
        t_max = matched["temp_max"].round(2).tolist()
        t_min = matched["temp_min"].round(2).tolist()
        thresholds = matched["heatwave_threshold"].round(2).tolist()
        is_hw = matched["is_heatwave_day"].tolist()

        # Compute next-day prediction
        pred_res = self.predict_city(city, end_date_str)
        pred_info = pred_res["prediction"]

        # Append next-day projection
        dates.append(pred_res["target_date"] + " (Tomorrow)")
        t_max_with_pred = t_max + [pred_info["predicted_temp_max"]]
        thresholds_with_pred = thresholds + [pred_res["next_threshold"]]

        return {
            "city": city,
            "dates": dates,
            "temp_max": t_max,
            "temp_min": t_min,
            "thresholds": thresholds,
            "is_heatwave": is_hw,
            "predicted_date": pred_res["target_date"],
            "predicted_temp_max": pred_info["predicted_temp_max"],
            "predicted_p10": pred_info["temp_lower_p10"],
            "predicted_p90": pred_info["temp_upper_p90"],
            "predicted_severity": pred_info["severity"],
            "predicted_probability_pct": pred_info["probability_pct"],
            "full_dates": dates,
            "full_temp_max": t_max_with_pred,
            "full_thresholds": thresholds_with_pred,
        }
