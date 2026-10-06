/**
 * HeatGuard AI - Dashboard & Map Controller
 * Handles Leaflet.js geospatial mapping, dynamic KPI cards, animated risk gauge,
 * Chart.js trajectory visualization, and real-time API synchronization.
 */

let map = null;
let markersLayer = null;
let heatLayer = null;
let activeLayer = 'markers'; // 'markers' or 'heatmap'
let trajectoryChart = null;
let currentCityData = null;

// City coordinates mapping
const CITY_COORDS = {
  "Delhi": [28.6139, 77.2090],
  "Ahmedabad": [23.0225, 72.5714],
  "Chennai": [13.0827, 80.2707],
  "Kolkata": [22.5726, 88.3639],
  "Pune": [18.5204, 73.8567],
  "Mumbai": [19.0760, 72.8777],
  "Bengaluru": [12.9716, 77.5946]
};

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  bindEvents();

  // Initial load with current controls
  const city = document.getElementById("city-select").value;
  const date = document.getElementById("date-select").value;
  loadPrediction(city, date);
  loadAllCitiesMap(date);
});

// ----------------------------------------------------------------------------------------
// 1. Leaflet.js Map Initialization
// ----------------------------------------------------------------------------------------
function initMap() {
  const mapElement = document.getElementById("leaflet-map");
  if (!mapElement) return;

  // Center on India
  map = L.map("leaflet-map", {
    center: [21.5, 78.5],
    zoom: 5,
    minZoom: 4,
    maxZoom: 9,
    zoomControl: true,
  });

  // High-Resolution Dark Gray Canvas Tiles (Free, Zero API Key Required, No Watermark)
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: '&copy; <a href="https://www.esri.com/" target="_blank">Esri</a> | HeatGuard AI',
      maxZoom: 16,
    }
  ).addTo(map);

  // Administrative Boundaries & City Labels Layer
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
    {
      attribution: "",
      maxZoom: 16,
    }
  ).addTo(map);

  markersLayer = L.layerGroup().addTo(map);
}

// ----------------------------------------------------------------------------------------
// 2. Event Listeners & Preset Handlers
// ----------------------------------------------------------------------------------------
function bindEvents() {
  const citySelect = document.getElementById("city-select");
  const dateSelect = document.getElementById("date-select");
  const runBtn = document.getElementById("run-predict-btn");
  const refreshAllBtn = document.getElementById("refresh-all-btn");

  if (runBtn) {
    runBtn.addEventListener("click", () => {
      loadPrediction(citySelect.value, dateSelect.value);
      loadAllCitiesMap(dateSelect.value);
    });
  }

  if (citySelect) {
    citySelect.addEventListener("change", () => {
      loadPrediction(citySelect.value, dateSelect.value);
    });
  }

  if (dateSelect) {
    dateSelect.addEventListener("change", () => {
      loadPrediction(citySelect.value, dateSelect.value);
      loadAllCitiesMap(dateSelect.value);
    });
  }

  if (refreshAllBtn) {
    refreshAllBtn.addEventListener("click", () => {
      loadAllCitiesMap(dateSelect.value);
    });
  }

  // Preset Chips
  document.querySelectorAll(".chip-tag[data-color]").forEach(tag => {
    const c = tag.getAttribute("data-color");
    if (c) {
      tag.style.backgroundColor = c + "33";
      tag.style.color = c;
      tag.style.borderColor = c + "66";
    }
  });

  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".preset-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");

      const city = chip.getAttribute("data-city");
      const date = chip.getAttribute("data-date");

      if (citySelect) citySelect.value = city;
      if (dateSelect) dateSelect.value = date;

      loadPrediction(city, date);
      loadAllCitiesMap(date);
    });
  });

  // Layer Switcher Buttons
  const markersBtn = document.getElementById("layer-markers-btn");
  const heatmapBtn = document.getElementById("layer-heatmap-btn");

  if (markersBtn && heatmapBtn) {
    markersBtn.addEventListener("click", () => {
      activeLayer = 'markers';
      markersBtn.classList.add("active");
      heatmapBtn.classList.remove("active");
      if (heatLayer) map.removeLayer(heatLayer);
      if (markersLayer) map.addLayer(markersLayer);
    });

    heatmapBtn.addEventListener("click", () => {
      activeLayer = 'heatmap';
      heatmapBtn.classList.add("active");
      markersBtn.classList.remove("active");
      if (markersLayer) map.removeLayer(markersLayer);
      loadHeatmapLayer(dateSelect.value);
    });
  }

  // Climate Stress Simulator Sliders
  const tempSlider = document.getElementById("sim-temp-offset");
  const nightSlider = document.getElementById("sim-night-offset");
  const rainSlider = document.getElementById("sim-rain-offset");

  if (tempSlider) {
    tempSlider.addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      const sign = v >= 0 ? "+" : "";
      document.getElementById("slider-temp-badge").innerText = `${sign}${v.toFixed(1)}°C`;
    });
  }

  if (nightSlider) {
    nightSlider.addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      const sign = v >= 0 ? "+" : "";
      document.getElementById("slider-night-badge").innerText = `${sign}${v.toFixed(1)}°C`;
    });
  }

  if (rainSlider) {
    rainSlider.addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      const sign = v >= 0 ? "+" : "";
      document.getElementById("slider-rain-badge").innerText = `${sign}${v.toFixed(1)} mm`;
    });
  }

  const runSimBtn = document.getElementById("run-simulation-btn");
  if (runSimBtn) {
    runSimBtn.addEventListener("click", runClimateSimulation);
  }

  const dispatchBtn = document.getElementById("dispatch-alert-btn");
  if (dispatchBtn) {
    dispatchBtn.addEventListener("click", dispatchEmergencyAlert);
  }
}

// ----------------------------------------------------------------------------------------
// 3. API Sync: Predict Single City & Update KPIs
// ----------------------------------------------------------------------------------------
async function loadPrediction(city, date) {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ city, date }),
    });

    const data = await res.json();
    if (data.status === "success") {
      currentCityData = data.data;
      updateKPICards(currentCityData);
      updateRiskGauge(currentCityData);
      updateTrajectoryChart(city, date);

      // Pan map smoothly to the selected city
      if (map && CITY_COORDS[city]) {
        map.panTo(CITY_COORDS[city], { animate: true, duration: 1 });
      }
    } else {
      console.error("Prediction API returned error:", data.message);
    }
  } catch (err) {
    console.error("Failed to load prediction:", err);
  }
}

// ----------------------------------------------------------------------------------------
// 4. Update KPI Cards & Confidence Gauge
// ----------------------------------------------------------------------------------------
function updateKPICards(data) {
  const pred = data.prediction;

  // KPI 1: Observed Max Temp & Departure
  document.getElementById("kpi-current-temp").innerText = data.current_temp_max.toFixed(1);
  const depBadge = document.getElementById("kpi-departure-badge");
  const depSign = data.current_departure >= 0 ? "+" : "";
  depBadge.innerText = `${depSign}${data.current_departure.toFixed(1)}°C`;
  document.getElementById("kpi-threshold").innerText = `${data.current_threshold.toFixed(1)}°C`;

  if (data.current_departure >= 2.0) {
    depBadge.className = "badge badge-severe";
  } else if (data.current_departure >= 0.0) {
    depBadge.className = "badge badge-warning";
  } else {
    depBadge.className = "badge badge-normal";
  }

  // KPI 2: 3-Day Moving Average & Trend
  document.getElementById("kpi-3d-avg").innerText = data.temp_max_3d_avg.toFixed(1);
  document.getElementById("kpi-trend-text").innerText = pred.trend;

  // KPI 3: Heatwave Streak & Precipitation
  document.getElementById("kpi-streak").innerText = data.heatwave_streak_days;
  const statusBadge = document.getElementById("kpi-status-badge");
  if (data.is_heatwave_today === 1) {
    statusBadge.innerText = "Heatwave";
    statusBadge.className = "badge badge-severe";
  } else {
    statusBadge.innerText = "Normal";
    statusBadge.className = "badge badge-normal";
  }
  document.getElementById("kpi-rain").innerText = `${data.rain.toFixed(1)} mm`;

  // Map subtext
  const mapDateEl = document.getElementById("map-target-date");
  if (mapDateEl) mapDateEl.innerText = data.target_date;

  // Advisory link pre-fill
  const advLink = document.getElementById("generate-advisory-link");
  if (advLink) {
    advLink.href = `/advisory?city=${encodeURIComponent(data.city)}&date=${encodeURIComponent(data.observation_date)}`;
  }

  // Refresh Feather icons in updated DOM elements
  feather.replace();
}

function updateRiskGauge(data) {
  const pred = data.prediction;
  const prob = pred.probability_pct; // 0 to 100
  const color = pred.alert_color;

  // Animated SVG dashoffset (radius = 40, circumference ≈ 251.32)
  const circle = document.getElementById("gauge-circle");
  const percentText = document.getElementById("gauge-percent-text");
  const circumference = 2 * Math.PI * 40;

  if (circle) {
    const offset = circumference - (prob / 100) * circumference;
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = color;
  }

  if (percentText) {
    animateValue(percentText, parseFloat(percentText.innerText) || 0, prob, 600);
  }

  // Severity Badge
  const sevBadge = document.getElementById("severity-badge-pill");
  if (sevBadge) {
    sevBadge.innerText = pred.severity;
    sevBadge.className = `badge badge-${pred.severity.toLowerCase()}`;
  }

  // Risk Banner
  const banner = document.getElementById("risk-banner");
  const bannerText = document.getElementById("risk-banner-text");
  if (banner && bannerText) {
    bannerText.innerText = `${pred.risk_level} THERMAL RISK`;
    banner.style.background = `${color}22`;
    banner.style.color = color;
    banner.style.borderColor = `${color}55`;
  }

  // Details List
  document.getElementById("risk-target-date").innerText = `Forecast for Tomorrow (${data.target_date})`;
  document.getElementById("detail-pred-temp").innerText = `${pred.predicted_temp_max.toFixed(1)}°C`;
  document.getElementById("detail-uncertainty").innerText = `[${pred.temp_lower_p10.toFixed(1)}°C to ${pred.temp_upper_p90.toFixed(1)}°C]`;
  const depSign = pred.predicted_departure >= 0 ? "+" : "";
  document.getElementById("detail-departure").innerText = `${depSign}${pred.predicted_departure.toFixed(2)}°C`;
}

// ----------------------------------------------------------------------------------------
// 5. Multi-City Map Synchronization (Pins & Tooltips)
// ----------------------------------------------------------------------------------------
async function loadAllCitiesMap(date) {
  try {
    const res = await fetch("/api/predict-all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date }),
    });

    const data = await res.json();
    if (data.status === "success" && markersLayer) {
      markersLayer.clearLayers();

      data.cities.forEach(item => {
        const city = item.city;
        const lat = item.latitude;
        const lon = item.longitude;
        const pred = item.prediction;
        const color = pred.alert_color;

        // Custom pulsing HTML marker
        const pulseIcon = L.divIcon({
          className: "city-pulse-marker",
          iconSize: [36, 36],
          iconAnchor: [18, 18],
          html: `
            <div class="pulse-ring" style="background:${color}44;"></div>
            <div class="pulse-core" style="background:${color};">${pred.predicted_temp_max.toFixed(0)}°</div>
          `,
        });

        const marker = L.marker([lat, lon], { icon: pulseIcon }).addTo(markersLayer);

        // Interactive Cyber Tooltip
        marker.bindPopup(`
          <div style="font-family:var(--font-sans); color:#f8fafc; min-width:190px; padding:2px;">
            <div style="font-weight:800; font-size:1.05rem; margin-bottom:0.25rem; color:#00f0ff;">${city}</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.6rem;">Forecast: ${item.target_date}</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem; font-size:0.85rem;">
              <span style="color:#94a3b8;">Predicted Max:</span>
              <strong style="color:#fff;">${pred.predicted_temp_max.toFixed(1)}°C</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem; font-size:0.85rem;">
              <span style="color:#94a3b8;">Departure:</span>
              <strong style="color:${color};">${pred.predicted_departure >= 0 ? '+' : ''}${pred.predicted_departure.toFixed(1)}°C</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.65rem; font-size:0.85rem;">
              <span style="color:#94a3b8;">Heatwave Prob:</span>
              <strong style="color:#00f0ff;">${pred.probability_pct}%</strong>
            </div>
            <button onclick="selectCityFromMap('${city}')" style="width:100%; padding:0.45rem; background:linear-gradient(135deg, #00f0ff 0%, #0284c7 100%); color:#030712; border:none; border-radius:6px; font-weight:700; cursor:pointer; font-size:0.8rem; box-shadow:0 0 12px rgba(0,240,255,0.4); transition:transform 0.15s ease;">
              Inspect ${city} View
            </button>
          </div>
        `);
      });

      if (activeLayer === 'heatmap') {
        loadHeatmapLayer(date);
      }
    }
  } catch (err) {
    console.error("Failed to load map markers:", err);
  }
}

// Global hook for popup buttons
window.selectCityFromMap = function(city) {
  const citySelect = document.getElementById("city-select");
  const dateSelect = document.getElementById("date-select");
  if (citySelect) {
    citySelect.value = city;
    loadPrediction(city, dateSelect.value);
  }
};

// ----------------------------------------------------------------------------------------
// 6. Leaflet Thermal Heatmap Layer
// ----------------------------------------------------------------------------------------
async function loadHeatmapLayer(date) {
  if (typeof L.heatLayer === 'undefined') return;

  try {
    const res = await fetch(`/api/heatmap-data?date=${encodeURIComponent(date)}`);
    const json = await res.json();

    if (json.status === "success") {
      if (heatLayer) map.removeLayer(heatLayer);

      // Points format: [lat, lng, intensity]
      const points = json.points.map(p => [p.lat, p.lng, p.intensity]);

      heatLayer = L.heatLayer(points, {
        radius: 45,
        blur: 28,
        maxZoom: 8,
        gradient: {
          0.2: '#06b6d4',
          0.4: '#10b981',
          0.6: '#f59e0b',
          0.8: '#f97316',
          1.0: '#ef4444'
        }
      }).addTo(map);
    }
  } catch (err) {
    console.error("Failed to load thermal heatmap:", err);
  }
}

// ----------------------------------------------------------------------------------------
// 7. Chart.js 14-Day Trajectory Visualization
// ----------------------------------------------------------------------------------------
async function updateTrajectoryChart(city, date) {
  const canvas = document.getElementById("trajectory-chart");
  if (!canvas) return;

  try {
    const res = await fetch(`/api/history/${encodeURIComponent(city)}?date=${encodeURIComponent(date)}&days=14`);
    const json = await res.json();

    if (json.status === "success") {
      const traj = json.trajectory;
      const titleEl = document.getElementById("trajectory-title");
      if (titleEl) {
        titleEl.innerText = `${city} — 14-Day Observed Trajectory & Tomorrow's ML Projection`;
      }

      const labels = traj.dates.map(d => d.replace(/^\d{4}-/, "")); // MM-DD format
      const tMaxData = traj.full_temp_max;
      const tMinData = traj.temp_min.concat([null]); // No min temp forecast for tomorrow
      const thrData = traj.full_thresholds;

      if (trajectoryChart) {
        trajectoryChart.destroy();
      }

      const ctx = canvas.getContext("2d");
      trajectoryChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Observed & Forecast T_max (°C)",
              data: tMaxData,
              borderColor: "#f97316",
              backgroundColor: "rgba(249, 115, 22, 0.12)",
              borderWidth: 2.5,
              tension: 0.3,
              fill: true,
              pointBackgroundColor: (context) => {
                const index = context.dataIndex;
                return index === tMaxData.length - 1 ? "#ec4899" : "#f97316";
              },
              pointRadius: (context) => {
                const index = context.dataIndex;
                return index === tMaxData.length - 1 ? 7 : 4;
              },
            },
            {
              label: "Observed T_min (°C)",
              data: tMinData,
              borderColor: "#00f0ff",
              borderWidth: 2.0,
              borderDash: [4, 3],
              tension: 0.3,
              fill: false,
              pointBackgroundColor: "#00f0ff",
              pointRadius: 3.5,
            },
            {
              label: "IMD Threshold (°C)",
              data: thrData,
              borderColor: "#ef4444",
              borderWidth: 2,
              borderDash: [6, 4],
              fill: false,
              pointRadius: 0,
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: "rgba(3, 7, 18, 0.95)",
              titleColor: "#00f0ff",
              bodyColor: "#f8fafc",
              borderColor: "rgba(0, 240, 255, 0.4)",
              borderWidth: 1,
              padding: 11,
              cornerRadius: 8,
            }
          },
          scales: {
            x: {
              grid: { color: "rgba(0, 240, 255, 0.06)" },
              ticks: { color: "#94a3b8", font: { size: 11 } }
            },
            y: {
              grid: { color: "rgba(0, 240, 255, 0.06)" },
              ticks: {
                color: "#94a3b8",
                callback: (val) => `${val}°C`
              }
            }
          }
        }
      });
    }
  } catch (err) {
    console.error("Failed to load trajectory chart:", err);
  }
}

// Smooth Number Counter Animation
function animateValue(obj, start, end, duration) {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    const current = Math.floor(progress * (end - start) + start);
    obj.innerText = `${current}%`;
    if (progress < 1) {
      window.requestAnimationFrame(step);
    } else {
      obj.innerText = `${end.toFixed(1)}%`;
    }
  };
  window.requestAnimationFrame(step);
}

// ----------------------------------------------------------------------------------------
// 8. Climate Stress Simulator Handler
// ----------------------------------------------------------------------------------------
async function runClimateSimulation() {
  const citySelect = document.getElementById("city-select");
  const dateSelect = document.getElementById("date-select");
  const statusMsg = document.getElementById("sim-status-msg");
  const outputPanel = document.getElementById("sim-output-panel");
  const runBtn = document.getElementById("run-simulation-btn");

  const city = citySelect ? citySelect.value : "Delhi";
  const date = dateSelect ? dateSelect.value : "2024-05-28";
  const tempOffset = parseFloat(document.getElementById("sim-temp-offset").value) || 0;
  const nightOffset = parseFloat(document.getElementById("sim-night-offset").value) || 0;
  const rainOffset = parseFloat(document.getElementById("sim-rain-offset").value) || 0;

  if (runBtn) {
    runBtn.disabled = true;
    runBtn.innerHTML = `<span style="display:inline-block;width:16px;height:16px;border:2px solid #030712;border-top-color:transparent;border-radius:50%;animation:spin 0.8s linear infinite;"></span> Simulating...`;
  }
  if (statusMsg) {
    statusMsg.innerText = `Executing in-silico atmospheric perturbation for ${city}...`;
  }

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: city,
        date: date,
        temp_offset: tempOffset,
        night_temp_offset: nightOffset,
        rain_offset: rainOffset,
      }),
    });

    const json = await res.json();
    if (json.status === "success") {
      const sim = json.simulation;
      const base = sim.baseline;
      const shock = sim.simulated;
      const deltas = sim.deltas;

      // Populate Baseline
      document.getElementById("sim-base-temp").innerText = `${base.temp_max.toFixed(1)}°C`;
      document.getElementById("sim-base-prob").innerText = `${base.probability_pct.toFixed(1)}%`;
      const baseBadge = document.getElementById("sim-base-badge");
      baseBadge.innerText = base.severity;
      baseBadge.className = `badge badge-${base.severity.toLowerCase()}`;

      // Populate Simulated
      document.getElementById("sim-shock-temp").innerText = `${shock.temp_max.toFixed(1)}°C`;
      const probColor = shock.probability_pct > 60 ? "#ef4444" : (shock.probability_pct > 23 ? "#f97316" : "#00f0ff");
      document.getElementById("sim-shock-prob").innerText = `${shock.probability_pct.toFixed(1)}%`;
      document.getElementById("sim-shock-prob").style.color = probColor;

      const shockBadge = document.getElementById("sim-shock-badge");
      shockBadge.innerText = shock.severity;
      shockBadge.className = `badge badge-${shock.severity.toLowerCase()}`;

      // Takeaway
      document.getElementById("sim-takeaway-text").innerHTML = `
        <strong>Simulation Result:</strong> ${sim.scientific_takeaway}
        ${deltas.risk_shifted ? '<span style="color:#f87171; font-weight:700; margin-left:6px;">⚠️ RISK TIER ESCALATION DETECTED</span>' : ''}
      `;

      if (outputPanel) outputPanel.style.display = "grid";
      if (statusMsg) {
        statusMsg.innerText = `Simulation complete: Probability delta ${deltas.prob_delta_pct > 0 ? '+' : ''}${deltas.prob_delta_pct}%.`;
      }
      feather.replace();
    } else {
      if (statusMsg) statusMsg.innerText = `Simulation error: ${json.message}`;
    }
  } catch (err) {
    console.error("Simulation failed:", err);
    if (statusMsg) statusMsg.innerText = "Network error during simulation.";
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.innerHTML = `<i data-feather="activity"></i> <span>Run Stress Simulation</span>`;
      feather.replace();
    }
  }
}

// ----------------------------------------------------------------------------------------
// 9. Emergency Dispatch Alert Broadcast Handler
// ----------------------------------------------------------------------------------------
async function dispatchEmergencyAlert() {
  const citySelect = document.getElementById("city-select");
  const dateSelect = document.getElementById("date-select");
  const dispatchBtn = document.getElementById("dispatch-alert-btn");

  const city = citySelect ? citySelect.value : "Delhi";
  const date = dateSelect ? dateSelect.value : "2024-05-28";

  if (dispatchBtn) {
    dispatchBtn.disabled = true;
    dispatchBtn.innerText = "Broadcasting...";
  }

  try {
    const res = await fetch("/api/alerts/dispatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: city,
        date: date,
        channels: ["Municipal_Disaster_Cell", "SMS_Public_Warning", "Hospital_Cooling_Wards", "Transit_Hub_Sirens"]
      }),
    });

    const json = await res.json();
    if (json.status === "success") {
      const r = json.receipt;
      alert(
        `🚨 EMERGENCY DISPATCH ISSUED!\n\n` +
        `Dispatch ID: ${r.dispatch_id}\n` +
        `City: ${r.target_city} | Forecast Date: ${r.forecast_date}\n` +
        `Assessed Severity: ${r.severity} (${r.risk_level})\n` +
        `Forecast Temperature: ${r.predicted_temp}°C\n` +
        `Channels Broadcast: ${r.channels_broadcast.join(', ')}\n` +
        `Status: ${r.status}`
      );
    }
  } catch (err) {
    console.error("Dispatch failed:", err);
    alert("Network error: Could not reach alert dispatch service.");
  } finally {
    if (dispatchBtn) {
      dispatchBtn.disabled = false;
      dispatchBtn.innerHTML = `<i data-feather="send" style="width:13px;height:13px; color:var(--neon-blue);"></i> <span>Broadcast Emergency Alert</span>`;
      feather.replace();
    }
  }
}

