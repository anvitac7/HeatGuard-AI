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

  // Dark-Matter Tiles
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> | HeatGuard AI',
    subdomains: "abcd",
    maxZoom: 19,
  }).addTo(map);

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

        // Interactive Tooltip
        marker.bindPopup(`
          <div style="font-family:var(--font-sans); color:#111; min-width:180px;">
            <div style="font-weight:700; font-size:1rem; margin-bottom:0.25rem;">${city}</div>
            <div style="font-size:0.75rem; color:#666; margin-bottom:0.5rem;">Target: ${item.target_date}</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
              <span>Predicted Max:</span>
              <strong>${pred.predicted_temp_max.toFixed(1)}°C</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
              <span>Departure:</span>
              <strong style="color:${color};">${pred.predicted_departure >= 0 ? '+' : ''}${pred.predicted_departure.toFixed(1)}°C</strong>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
              <span>Heatwave Prob:</span>
              <strong>${pred.probability_pct}%</strong>
            </div>
            <button onclick="selectCityFromMap('${city}')" style="width:100%; padding:0.35rem; background:#f97316; color:#fff; border:none; border-radius:4px; font-weight:600; cursor:pointer;">
              Inspect ${city}
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
              borderColor: "#06b6d4",
              borderWidth: 1.8,
              borderDash: [3, 3],
              tension: 0.3,
              fill: false,
              pointRadius: 3,
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
              backgroundColor: "rgba(13, 19, 34, 0.95)",
              titleColor: "#f8fafc",
              bodyColor: "#cbd5e1",
              borderColor: "rgba(255, 255, 255, 0.15)",
              borderWidth: 1,
              padding: 10,
              cornerRadius: 8,
            }
          },
          scales: {
            x: {
              grid: { color: "rgba(255, 255, 255, 0.05)" },
              ticks: { color: "#94a3b8", font: { size: 11 } }
            },
            y: {
              grid: { color: "rgba(255, 255, 255, 0.05)" },
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
