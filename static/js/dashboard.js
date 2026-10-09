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

// City coordinates and IMD meteorological metadata
const CITY_COORDS = {
  "Delhi": { lat: 28.6139, lon: 77.2090, state: "NCT of Delhi", threshold: "40.0°C (Plains Threshold)" },
  "Ahmedabad": { lat: 23.0225, lon: 72.5714, state: "Gujarat", threshold: "40.0°C (Plains Threshold)" },
  "Chennai": { lat: 13.0827, lon: 80.2707, state: "Tamil Nadu", threshold: "37.0°C (Coastal Threshold)" },
  "Kolkata": { lat: 22.5726, lon: 88.3639, state: "West Bengal", threshold: "37.0°C (Coastal Threshold)" },
  "Pune": { lat: 18.5204, lon: 73.8567, state: "Maharashtra", threshold: "40.0°C (Plains Threshold)" },
  "Mumbai": { lat: 19.0760, lon: 72.8777, state: "Maharashtra", threshold: "37.0°C (Coastal Threshold)" },
  "Bengaluru": { lat: 12.9716, lon: 77.5946, state: "Karnataka", threshold: "38.0°C (Plateau Threshold)" }
};

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  bindEvents();

  // Initial load with current controls
  const city = document.getElementById("city-select").value;
  const date = document.getElementById("date-select").value;
  loadPrediction(city, date);
});

let currentTileLayer = null;

function getTileConfig() {
  return {
    url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    options: {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | HeatGuard AI'
    }
  };
}

function updateMapTheme(theme) {
  if (!map) return;
  if (!currentTileLayer) {
    const cfg = getTileConfig();
    currentTileLayer = L.tileLayer(cfg.url, cfg.options).addTo(map);
  }
}

// ----------------------------------------------------------------------------------------
// 1. Leaflet.js Map Initialization (Locked, Focused Station Monitor)
// ----------------------------------------------------------------------------------------
function initMap() {
  const mapElement = document.getElementById("leaflet-map");
  if (!mapElement) return;

  // Initialize locked station map (no user dragging, zoom controls or accidental drifting)
  map = L.map("leaflet-map", {
    center: [28.6139, 77.2090],
    zoom: 11,
    minZoom: 4,
    maxZoom: 14,
    dragging: false,
    touchZoom: false,
    scrollWheelZoom: false,
    doubleClickZoom: false,
    boxZoom: false,
    keyboard: false,
    zoomControl: false,
    attributionControl: false
  });

  const activeTheme = document.documentElement.getAttribute("data-theme") || "light";
  updateMapTheme(activeTheme);

  markersLayer = L.layerGroup().addTo(map);

  setTimeout(() => {
    if (map) map.invalidateSize();
  }, 200);
}

// ----------------------------------------------------------------------------------------
// 2. Event Listeners & Preset Handlers
// ----------------------------------------------------------------------------------------
function advanceSelectedDateByOneDay() {
  const citySelect = document.getElementById("city-select");
  const dateSelect = document.getElementById("date-select");
  if (!citySelect || !dateSelect || !dateSelect.value) return;

  const currentDate = new Date(dateSelect.value + "T00:00:00");
  if (Number.isNaN(currentDate.getTime())) return;

  const maxDate = dateSelect.max ? new Date(dateSelect.max + "T00:00:00") : null;
  const nextDate = new Date(currentDate);
  nextDate.setDate(nextDate.getDate() + 1);

  if (maxDate && nextDate > maxDate) return;

  const year = nextDate.getFullYear();
  const month = String(nextDate.getMonth() + 1).padStart(2, "0");
  const day = String(nextDate.getDate()).padStart(2, "0");
  const nextDateStr = `${year}-${month}-${day}`;

  dateSelect.value = nextDateStr;
  loadPrediction(citySelect.value, nextDateStr);
}

async function synchronizeAllMetros(date, selectedCity, button) {
  if (button.disabled) return;

  const originalHtml = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `<span>Synchronizing 7 metros...</span>`;

  try {
    const response = await fetch("/api/predict-all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date })
    });
    const result = await response.json();

    if (!response.ok || result.status !== "success") {
      throw new Error(result.message || "The all-city forecast request failed.");
    }

    const cities = result.cities || [];
    if (cities.length !== 7) {
      throw new Error(`Only ${cities.length} of 7 metro forecasts were returned. Please retry.`);
    }

    const selectedData = cities.find((item) => item.city === selectedCity);
    if (!selectedData) {
      throw new Error(`No forecast was returned for the selected city, ${selectedCity}.`);
    }

    currentCityData = selectedData;
    updateKPICards(selectedData);
    updateRiskGauge(selectedData);
    updateTrajectoryChart(selectedCity, date);
    updateAllMetroMap(cities, date, selectedCity);

    button.innerHTML = `<i data-feather="check" style="width:13px;height:13px;"></i><span>7 Metros Synced</span>`;
    feather.replace();
    window.setTimeout(() => {
      button.innerHTML = originalHtml;
      feather.replace();
    }, 2500);
  } catch (error) {
    console.error("Failed to synchronize forecasts for all metros:", error);
    alert(`Could not synchronize all 7 metros: ${error.message}`);
    button.innerHTML = originalHtml;
    feather.replace();
  } finally {
    button.disabled = false;
  }
}

function bindEvents() {
  const citySelect = document.getElementById("city-select");
  const dateSelect = document.getElementById("date-select");
  const runBtn = document.getElementById("run-predict-btn");
  const refreshAllBtn = document.getElementById("refresh-all-btn");
  const generateAdvisoryLink = document.getElementById("generate-advisory-link");

  // Listen for dark/light theme switch
  window.addEventListener("themechanged", (e) => {
    updateMapTheme(e.detail.theme);
    if (citySelect && dateSelect) {
      updateTrajectoryChart(citySelect.value, dateSelect.value);
    }
  });

  if (runBtn) {
    runBtn.addEventListener("click", () => {
      advanceSelectedDateByOneDay();
    });
  }

  if (generateAdvisoryLink) {
    generateAdvisoryLink.addEventListener("click", (event) => {
      event.preventDefault();
      generateAllPersonaAdvisoryPdf(citySelect.value, dateSelect.value, generateAdvisoryLink);
    });
  }

  if (citySelect) {
    citySelect.addEventListener("change", () => {
      loadPrediction(citySelect.value, dateSelect.value);
    });
  }

  async function generateAllPersonaAdvisoryPdf(city, date, button) {
    if (button.dataset.generating === "true") return;

    const JsPDF = window.jspdf && window.jspdf.jsPDF;
    if (!JsPDF) {
      alert("PDF generation is unavailable because the PDF library did not load. Please check your connection and try again.");
      return;
    }

    const originalHtml = button.innerHTML;
    const originalPointerEvents = button.style.pointerEvents;
    button.dataset.generating = "true";
    button.disabled = true;
    button.setAttribute("aria-disabled", "true");
    button.style.pointerEvents = "none";
    button.innerHTML = `<span>Generating 4 stakeholder advisories...</span>`;

    try {
      const audiences = ["Citizen", "Farmer", "Health Agency", "Municipal Authority"];
      const advisories = [];

      for (const audience of audiences) {
        const response = await fetch("/api/advisory", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ city, date, audience })
        });
        const result = await response.json();
        if (!response.ok || result.status !== "success" || !result.advisory) {
          throw new Error(result.message || `Could not generate the ${audience} advisory.`);
        }
        advisories.push(result.advisory);
        button.innerHTML = `<span>Generated ${advisories.length} of ${audiences.length} advisories...</span>`;
      }

      const doc = new JsPDF({ unit: "mm", format: "a4" });
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 18;
      const contentWidth = pageWidth - margin * 2;
      let y = 20;

      const addPage = () => {
        doc.addPage();
        y = 20;
      };

      const ensureSpace = (height) => {
        if (y + height > pageHeight - 18) addPage();
      };

      const writeText = (text, options = {}) => {
        const fontSize = options.fontSize || 10;
        const lineHeight = options.lineHeight || 5;
        const indent = options.indent || 0;
        doc.setFont("helvetica", options.bold ? "bold" : "normal");
        doc.setFontSize(fontSize);
        doc.setTextColor(...(options.color || [31, 41, 55]));
        const lines = doc.splitTextToSize(String(text), contentWidth - indent);
        lines.forEach((line) => {
          ensureSpace(lineHeight);
          doc.text(line, margin + indent, y);
          y += lineHeight;
        });
      };

      const cleanMarkdown = (text) => text
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/\*(.*?)\*/g, "$1")
        .replace(/`([^`]+)`/g, "$1")
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)");

      advisories.forEach((advisory, index) => {
        if (index > 0) addPage();

        doc.setFillColor(15, 118, 110);
        doc.rect(0, 0, pageWidth, 5, "F");
        writeText("HeatGuard AI", { fontSize: 19, bold: true, color: [15, 76, 73], lineHeight: 9 });
        writeText("Combined Stakeholder Advisory Report", {
          fontSize: 10, color: [71, 85, 105], lineHeight: 7
        });
        y += 3;
        writeText(`Advisory for ${advisory.persona_info?.title || advisory.audience}`, {
          fontSize: 15, bold: true, color: [15, 23, 42], lineHeight: 8
        });
        if (advisory.persona_info?.tagline) {
          writeText(advisory.persona_info.tagline, { fontSize: 9, color: [71, 85, 105], lineHeight: 6 });
        }
        y += 3;

        doc.setDrawColor(203, 213, 225);
        doc.setFillColor(248, 250, 252);
        doc.roundedRect(margin, y, contentWidth, 38, 2, 2, "FD");
        const contextTop = y + 7;
        y = contextTop;
        writeText(`City: ${advisory.city}    Forecast date: ${advisory.target_date}`, {
          fontSize: 10, bold: true, indent: 5, lineHeight: 6
        });
        y = contextTop + 7;
        const departure = Number(advisory.predicted_departure);
        writeText(
          `Predicted maximum: ${Number(advisory.predicted_temp_max).toFixed(1)} C    Departure: ${departure >= 0 ? "+" : ""}${departure.toFixed(1)} C`,
          { fontSize: 10, indent: 5, lineHeight: 6 }
        );
        y += 1;
        writeText(
          `Heatwave probability: ${Number(advisory.probability_pct).toFixed(1)}%    Risk tier: ${advisory.risk_level} (${advisory.severity})`,
          { fontSize: 10, indent: 5, lineHeight: 6 }
        );
        y += 8;

        writeText("Advisory Details", { fontSize: 12, bold: true, color: [15, 76, 73], lineHeight: 7 });
        y += 1;
        String(advisory.advisory_markdown || "").split(/\r?\n/).forEach((rawLine) => {
          const line = rawLine.trim();
          if (!line) {
            y += 2;
            return;
          }
          const heading = line.match(/^#{1,4}\s+(.*)$/);
          if (heading) {
            y += 2;
            writeText(cleanMarkdown(heading[1]), {
              fontSize: heading[0].startsWith("## ") ? 12 : 11,
              bold: true,
              color: [15, 23, 42],
              lineHeight: 6
            });
            y += 1;
            return;
          }
          const listItem = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
          writeText(listItem ? `- ${cleanMarkdown(listItem[1])}` : cleanMarkdown(line), {
            indent: listItem ? 3 : 0,
            lineHeight: 5
          });
          y += 1;
        });

        ensureSpace(16);
        y += 3;
        writeText("Priority Action Checklist", { fontSize: 12, bold: true, color: [15, 76, 73], lineHeight: 7 });
        const checklist = advisory.priority_checklist || [];
        if (checklist.length) {
          checklist.forEach((item) => {
            writeText(`[ ] ${item}`, { indent: 2, lineHeight: 5 });
            y += 1;
          });
        } else {
          writeText("No critical action items for this risk level.", { lineHeight: 5 });
        }

        y += 4;
        writeText(`Grounding status: ${advisory.guardrail_status || "Verified prediction context"}`, {
          fontSize: 9, color: [71, 85, 105], lineHeight: 5
        });
        if (index === 0) {
          writeText(`Report generated: ${new Date().toLocaleString()}`, {
            fontSize: 9, color: [71, 85, 105], lineHeight: 5
          });
        }
      });

      const pageCount = doc.internal.getNumberOfPages();
      for (let page = 1; page <= pageCount; page += 1) {
        doc.setPage(page);
        doc.setDrawColor(226, 232, 240);
        doc.line(margin, pageHeight - 13, pageWidth - margin, pageHeight - 13);
        doc.setFont("helvetica", "normal");
        doc.setFontSize(8);
        doc.setTextColor(100, 116, 139);
        doc.text("HeatGuard AI - Stakeholder Advisory Report", margin, pageHeight - 8);
        doc.text(`Page ${page} of ${pageCount}`, pageWidth - margin, pageHeight - 8, { align: "right" });
      }

      const safeCity = city.replace(/[^a-z0-9_-]+/gi, "_");
      doc.save(`HeatGuard_Stakeholder_Advisories_${safeCity}_${date}.pdf`);
    } catch (error) {
      console.error("Combined stakeholder advisory PDF generation failed:", error);
      alert(`Could not generate the combined stakeholder PDF: ${error.message}`);
    } finally {
      button.innerHTML = originalHtml;
      button.disabled = false;
      button.removeAttribute("aria-disabled");
      button.style.pointerEvents = originalPointerEvents;
      delete button.dataset.generating;
      feather.replace();
    }
  }

  if (dateSelect) {
    dateSelect.addEventListener("change", () => {
      loadPrediction(citySelect.value, dateSelect.value);
    });
  }

  if (refreshAllBtn) {
    refreshAllBtn.addEventListener("click", () => {
      synchronizeAllMetros(dateSelect.value, citySelect.value, refreshAllBtn);
    });
  }

  // Preset Chips
  document.querySelectorAll(".chip-tag[data-color]").forEach(tag => {
    const c = tag.getAttribute("data-color");
    if (c) {
      tag.style.backgroundColor = c + "22";
      tag.style.color = c;
      tag.style.borderColor = c + "44";
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
    });
  });

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
      updateCityStationMap(city, currentCityData);
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

  // KPI 4: 74-Year Historical Benchmark
  if (data.historical_peak != null) {
    document.getElementById("kpi-historical-peak").innerText = data.historical_peak.toFixed(1);
  }
  if (data.total_historical_hw_days != null) {
    document.getElementById("kpi-total-hw").innerText = `Total historical heatwave days: ${data.total_historical_hw_days}`;
  }

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
  const decisionThreshold = document.getElementById("detail-decision-threshold");
  if (decisionThreshold && pred.threshold_applied != null) {
    decisionThreshold.innerHTML = `τ* = ${Number(pred.threshold_applied).toFixed(2)} <span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">(model score cutoff; probabilities are not calibrated)</span>`;
  }
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
    banner.style.background = `${color}18`;
    banner.style.color = color;
    banner.style.borderColor = `${color}44`;
  }

  // Details List
  document.getElementById("risk-target-date").innerText = `Forecast for Tomorrow (${data.target_date})`;
  document.getElementById("detail-pred-temp").innerText = `${pred.predicted_temp_max.toFixed(1)}°C`;
  document.getElementById("detail-uncertainty").innerText = `[${pred.temp_lower_p10.toFixed(1)}°C to ${pred.temp_upper_p90.toFixed(1)}°C]`;
  const depSign = pred.predicted_departure >= 0 ? "+" : "";
  document.getElementById("detail-departure").innerText = `${depSign}${pred.predicted_departure.toFixed(2)}°C`;
}

// ----------------------------------------------------------------------------------------
// 5. Single-City Geospatial Station Monitor (Locked & Centered)
// ----------------------------------------------------------------------------------------
function updateAllMetroMap(cities, date, selectedCity) {
  if (!map || !markersLayer) return;

  markersLayer.clearLayers();
  const bounds = [];
  const forecastDate = cities[0]?.target_date || date;

  cities.forEach((data) => {
    const meta = CITY_COORDS[data.city];
    const pred = data.prediction;
    if (!meta || !pred) return;

    bounds.push([meta.lat, meta.lon]);
    const color = pred.alert_color || "#16a34a";
    const marker = L.marker([meta.lat, meta.lon], {
      icon: L.divIcon({
        className: "city-station-marker",
        iconSize: [38, 38],
        iconAnchor: [19, 19],
        html: `<div class="station-badge" style="background:${color};"><span>${pred.predicted_temp_max.toFixed(0)}°</span></div>`
      })
    }).addTo(markersLayer);

    marker.bindPopup(`
      <div style="font-family:var(--font-sans); color:#0f172a; min-width:200px; padding:4px;">
        <div style="font-weight:700; font-size:1rem; margin-bottom:0.15rem;">${data.city} Forecast</div>
        <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.5rem;">${meta.state} · ${meta.lat.toFixed(4)}°N, ${meta.lon.toFixed(4)}°E</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem; font-size:0.8rem;">
          <span style="color:#64748b;">Predicted max:</span>
          <strong>${pred.predicted_temp_max.toFixed(1)}°C</strong>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
          <span style="color:#64748b;">Heatwave risk:</span>
          <strong style="color:${color};">${pred.risk_level} (${pred.probability_pct.toFixed(1)}%)</strong>
        </div>
      </div>
    `);
  });

  if (bounds.length) {
    map.invalidateSize();
    map.fitBounds(bounds, { padding: [24, 24], maxZoom: 5, animate: true });
  }

  const mapCityName = document.getElementById("map-city-name");
  if (mapCityName) mapCityName.innerText = "All 7 Metros";
  const overlayCity = document.getElementById("overlay-city");
  if (overlayCity) overlayCity.innerText = "All 7 Metro Forecasts";
  const overlayCoords = document.getElementById("overlay-coords");
  if (overlayCoords) overlayCoords.innerText = "Select a marker to inspect a city";
  const overlayThreshold = document.getElementById("overlay-threshold");
  if (overlayThreshold) overlayThreshold.innerText = `Observation date: ${date}`;
  const mapDateEl = document.getElementById("map-target-date");
  if (mapDateEl) mapDateEl.innerText = forecastDate;
  const stationStatus = document.getElementById("station-status-badge");
  if (stationStatus) {
    stationStatus.innerText = `7 Metros Synced · ${selectedCity} selected`;
    stationStatus.className = "badge badge-normal";
  }
}

function updateCityStationMap(city, data) {
  if (!map || !CITY_COORDS[city]) return;
  const meta = CITY_COORDS[city];

  // Update card header and overlay telemetry texts
  const mapCityName = document.getElementById("map-city-name");
  if (mapCityName) mapCityName.innerText = city;
  const stationStatus = document.getElementById("station-status-badge");
  if (stationStatus) {
    stationStatus.innerHTML = '<i data-feather="radio" style="width:12px;height:12px;"></i> Station Active';
    stationStatus.className = "badge badge-normal";
  }

  const overlayCity = document.getElementById("overlay-city");
  if (overlayCity) overlayCity.innerText = `${city} (${meta.state})`;

  const overlayCoords = document.getElementById("overlay-coords");
  if (overlayCoords) overlayCoords.innerText = `${meta.lat.toFixed(4)}° N, ${meta.lon.toFixed(4)}° E`;

  const overlayThreshold = document.getElementById("overlay-threshold");
  if (overlayThreshold) overlayThreshold.innerText = meta.threshold;

  const mapDateEl = document.getElementById("map-target-date");
  if (mapDateEl && data && data.target_date) {
    mapDateEl.innerText = data.target_date;
  }

  // Smoothly center and lock map on the selected target city at high-precision zoom
  map.invalidateSize();
  map.setView([meta.lat, meta.lon], 11, { animate: true });

  // Place clean, glowing station radar pulse marker
  if (markersLayer) {
    markersLayer.clearLayers();

    const pred = (data && data.prediction) ? data.prediction : {};
    const color = pred.alert_color || "#16a34a";
    const tempVal = pred.predicted_temp_max != null
      ? `${pred.predicted_temp_max.toFixed(0)}°`
      : (data && data.current_temp_max != null ? `${data.current_temp_max.toFixed(0)}°` : "--");

    const stationIcon = L.divIcon({
      className: "city-station-marker",
      iconSize: [34, 34],
      iconAnchor: [17, 17],
      html: `
        <div class="station-badge" style="background:${color};">
          <span>${tempVal}</span>
        </div>
      `,
    });

    const marker = L.marker([meta.lat, meta.lon], { icon: stationIcon }).addTo(markersLayer);

    marker.bindPopup(`
      <div style="font-family:var(--font-sans); color:#0f172a; min-width:200px; padding:4px;">
        <div style="font-weight:700; font-size:1rem; margin-bottom:0.15rem; color:#0f172a;">${city} IMD Station</div>
        <div style="font-size:0.75rem; color:#64748b; margin-bottom:0.5rem;">${meta.state} • ${meta.lat.toFixed(4)}°N, ${meta.lon.toFixed(4)}°E</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem; font-size:0.8rem;">
          <span style="color:#64748b;">Predicted Max:</span>
          <strong style="color:#0f172a;">${pred.predicted_temp_max ? pred.predicted_temp_max.toFixed(1) + '°C' : '--'}</strong>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem; font-size:0.8rem;">
          <span style="color:#64748b;">Thermal Risk:</span>
          <strong style="color:${color};">${pred.severity || 'Normal'} (${pred.probability_pct || 0}%)</strong>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; border-top:1px solid #e2e8f0; padding-top:4px; margin-top:4px;">
          <span style="color:#64748b;">IMD Threshold:</span>
          <strong style="color:#334155;">${meta.threshold}</strong>
        </div>
      </div>
    `);
  }
}

// ----------------------------------------------------------------------------------------
// 7. Chart.js 14-Day Trajectory Visualization (Enterprise Clean Styling)
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

      const isDark = document.documentElement.getAttribute("data-theme") === "dark";
      const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(15, 23, 42, 0.05)";
      const tickColor = isDark ? "#94a3b8" : "#64748b";
      const tooltipBg = isDark ? "#0f172a" : "#ffffff";
      const tooltipTitle = isDark ? "#f8fafc" : "#0f172a";
      const tooltipBody = isDark ? "#cbd5e1" : "#334155";
      const tooltipBorder = isDark ? "#334155" : "#e2e8f0";

      // Refined Scientific Color Palette (Teal Emerald / Royal Violet / Crimson)
      const maxColor = isDark ? "#2dd4bf" : "#0d9488";
      const maxBg = isDark ? "rgba(45, 212, 191, 0.16)" : "rgba(13, 148, 136, 0.09)";
      const minColor = isDark ? "#a78bfa" : "#7c3aed";
      const thrColor = isDark ? "#fb7185" : "#e11d48";
      const projPointColor = isDark ? "#14b8a6" : "#0f766e";

      const ctx = canvas.getContext("2d");
      trajectoryChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Observed & Forecast T_max (°C)",
              data: tMaxData,
              borderColor: maxColor,
              backgroundColor: maxBg,
              borderWidth: 2.4,
              tension: 0.3,
              fill: true,
              pointBackgroundColor: (context) => {
                const index = context.dataIndex;
                return index === tMaxData.length - 1 ? projPointColor : maxColor;
              },
              pointBorderColor: (context) => {
                const index = context.dataIndex;
                return index === tMaxData.length - 1 ? "#ffffff" : maxColor;
              },
              pointBorderWidth: (context) => {
                const index = context.dataIndex;
                return index === tMaxData.length - 1 ? 2 : 1;
              },
              pointRadius: (context) => {
                const index = context.dataIndex;
                return index === tMaxData.length - 1 ? 6.5 : 3.5;
              },
            },
            {
              label: "Observed T_min (°C)",
              data: tMinData,
              borderColor: minColor,
              borderWidth: 2,
              borderDash: [4, 3],
              tension: 0.3,
              fill: false,
              pointBackgroundColor: minColor,
              pointRadius: 3,
            },
            {
              label: "IMD Threshold (°C)",
              data: thrData,
              borderColor: thrColor,
              borderWidth: 1.8,
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
              backgroundColor: tooltipBg,
              titleColor: tooltipTitle,
              bodyColor: tooltipBody,
              borderColor: tooltipBorder,
              borderWidth: 1,
              padding: 10,
              cornerRadius: 6,
              boxShadow: "0 4px 6px -1px rgba(0,0,0,0.2)",
            }
          },
          scales: {
            x: {
              grid: { color: gridColor },
              ticks: { color: tickColor, font: { size: 11 } }
            },
            y: {
              grid: { color: gridColor },
              ticks: {
                color: tickColor,
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
    runBtn.innerHTML = `<span style="display:inline-block;width:14px;height:14px;border:2px solid #ffffff;border-top-color:transparent;border-radius:50%;animation:spin 0.8s linear infinite;"></span> Simulating...`;
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
      const probColor = shock.probability_pct > 60 ? "#dc2626" : (shock.probability_pct > 23 ? "#ea580c" : "#16a34a");
      document.getElementById("sim-shock-prob").innerText = `${shock.probability_pct.toFixed(1)}%`;
      document.getElementById("sim-shock-prob").style.color = probColor;

      const shockBadge = document.getElementById("sim-shock-badge");
      shockBadge.innerText = shock.severity;
      shockBadge.className = `badge badge-${shock.severity.toLowerCase()}`;

      // Takeaway
      document.getElementById("sim-takeaway-text").innerHTML = `
        <strong>Simulation Result:</strong> ${sim.scientific_takeaway}
        ${deltas.risk_shifted ? '<span style="color:#dc2626; font-weight:700; margin-left:6px;">⚠️ RISK TIER ESCALATION DETECTED</span>' : ''}
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
      runBtn.innerHTML = `<i data-feather="activity" style="width:15px;height:15px;"></i> <span>Run Stress Simulation</span>`;
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
      dispatchBtn.innerHTML = `<i data-feather="send" style="width:13px;height:13px; color:var(--brand-600);"></i> <span>Broadcast Emergency Alert</span>`;
      feather.replace();
    }
  }
}
