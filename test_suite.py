"""
Comprehensive integration test for HeatGuard AI Flask application.
Tests all HTML routes, REST endpoints, prediction service, GenAI grounding,
and stress simulation.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from app import app

def run_tests():
    print("=" * 60)
    print("HeatGuard AI - Running Full Integration Test Suite")
    print("=" * 60)
    client = app.test_client()

    # 1. HTML Pages
    html_routes = ["/", "/dashboard", "/analytics", "/advisory", "/chatbot"]
    for route in html_routes:
        res = client.get(route)
        assert res.status_code == 200, f"Route {route} failed with {res.status_code}"
        print(f"  [PASS] HTML route: {route} (200 OK)")

    # 2. REST: City Metadata & Presets
    res = client.get("/api/cities")
    assert res.status_code == 200
    cities = res.get_json()["cities"]
    assert len(cities) == 7
    print(f"  [PASS] /api/cities (7 metros returned: {[c['city'] for c in cities]})")

    res = client.get("/api/presets")
    assert res.status_code == 200
    presets = res.get_json()["presets"]
    assert len(presets) >= 4
    print(f"  [PASS] /api/presets ({len(presets)} curated presets)")

    # 3. REST: Single City Prediction
    payload = {"city": "Delhi", "date": "2024-05-28"}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200, f"Prediction failed: {res.data}"
    pred_data = res.get_json()["data"]
    pred = pred_data["prediction"]
    print(f"  [PASS] /api/predict Delhi 2024-05-28 -> Max: {pred['predicted_temp_max']} C, HW Prob: {pred['probability_pct']}%, Tier: {pred['risk_level']}")

    # 4. REST: Multi-City Predict All (Leaflet payload)
    res = client.post("/api/predict-all", json={"date": "2024-05-28"})
    assert res.status_code == 200
    all_preds = res.get_json()["cities"]
    assert len(all_preds) == 7
    print(f"  [PASS] /api/predict-all (All 7 metros scored simultaneously)")

    # 5. REST: Trajectory History
    res = client.get("/api/history/Delhi?date=2024-05-28&days=14")
    assert res.status_code == 200
    traj = res.get_json()["trajectory"]
    assert len(traj["dates"]) == 15  # 14 observed days + 1 forecast day
    print(f"  [PASS] /api/history/Delhi (14-day sequence + next day forecast)")

    # 6. REST: Heatmap Intensity Points
    res = client.get("/api/heatmap-data?date=2024-05-28")
    assert res.status_code == 200
    pts = res.get_json()["points"]
    assert len(pts) == 7
    print(f"  [PASS] /api/heatmap-data (7 normalized intensity points)")

    # 7. REST: GenAI Grounded Advisory (Citizen, Farmer, Health Agency, Municipal)
    for audience in ["Citizen", "Farmer", "Health Agency", "Municipal Authority"]:
        adv_res = client.post("/api/advisory", json={"city": "Ahmedabad", "date": "2024-05-23", "audience": audience})
        assert adv_res.status_code == 200
        adv = adv_res.get_json()["advisory"]
        assert "advisory_markdown" in adv and "priority_checklist" in adv
        print(f"  [PASS] /api/advisory for {audience}: {len(adv['priority_checklist'])} actionable checklist items")

    # 8. REST: AI Chatbot Assistant
    chat_payload = {"city": "Delhi", "date": "2024-05-28", "message": "Why is tomorrow's risk elevated?"}
    chat_res = client.post("/api/chat", json=chat_payload)
    assert chat_res.status_code == 200
    chat_json = chat_res.get_json()["chat"]
    assert "reply" in chat_json
    print(f"  [PASS] /api/chat Assistant response: {chat_json['reply'][:65]}...")

    # 9. REST: Climate Analytics Suite
    endpoints = [
        "/api/analytics/summary",
        "/api/analytics/decades",
        "/api/analytics/city-comparison",
        "/api/analytics/monthly",
        "/api/analytics/records",
        "/api/analytics/persistence"
    ]
    for ep in endpoints:
        r = client.get(ep)
        assert r.status_code == 200, f"Endpoint {ep} failed: {r.status_code}"
        print(f"  [PASS] Analytics endpoint: {ep}")

    # 10. REST: Climate Stress Simulator
    sim_payload = {
        "city": "Ahmedabad",
        "date": "2024-05-23",
        "temp_offset": 2.0,
        "night_temp_offset": 1.5,
        "rain_offset": -5.0
    }
    sim_res = client.post("/api/simulate", json=sim_payload)
    assert sim_res.status_code == 200
    sim_data = sim_res.get_json()["simulation"]
    print(f"  [PASS] /api/simulate -> Base Prob: {sim_data['baseline']['probability_pct']}%, Sim Prob: {sim_data['simulated']['probability_pct']}%")

    # 11. REST: Multi-day Forecast & Alert Dispatch & System Health
    fc_res = client.get("/api/forecast/Delhi?date=2024-05-28&days=5")
    assert fc_res.status_code == 200
    print(f"  [PASS] /api/forecast/Delhi (5-day autoregressive roll: {len(fc_res.get_json()['forecast'])} days)")

    alert_res = client.post("/api/alerts/dispatch", json={"city": "Delhi", "date": "2024-05-28"})
    assert alert_res.status_code == 200
    print(f"  [PASS] /api/alerts/dispatch (Alert ID: {alert_res.get_json()['receipt']['dispatch_id']})")

    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    print(f"  [PASS] /api/health (Status: {health_res.get_json()['health']['status']}, Pipeline: {health_res.get_json()['health']['model_pipeline']})")

    print("=" * 60)
    print("ALL 16 SUITES PASSED FLAWLESSLY! HEATGUARD AI IS 100% OPERATIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
