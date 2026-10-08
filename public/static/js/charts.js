/**
 * HeatGuard AI - 74-Year Climate Analytics Charts
 * Visualizes decadal heatwave acceleration, city spatial disparity,
 * monthly seasonality distributions, and historical climate records using Chart.js.
 */

let decadalChartInstance = null;
let citiesChartInstance = null;
let monthlyChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  renderAllCharts();
  loadRecordsTable();

  // Re-render charts when dark/light theme is toggled
  window.addEventListener("themechanged", () => {
    renderAllCharts();
  });
});

function renderAllCharts() {
  loadDecadalChart();
  loadCitiesChart();
  loadMonthlyChart();
}

function getThemeColors() {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  return {
    isDark,
    gridColor: isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(15, 23, 42, 0.05)",
    tickColor: isDark ? "#94a3b8" : "#64748b",
    headingColor: isDark ? "#f8fafc" : "#0f172a",
    tooltipBg: isDark ? "#0f172a" : "#ffffff",
    tooltipTitle: isDark ? "#f8fafc" : "#0f172a",
    tooltipBody: isDark ? "#cbd5e1" : "#334155",
    tooltipBorder: isDark ? "#334155" : "#e2e8f0",
  };
}

// ----------------------------------------------------------------------------------------
// 1. Decadal Heatwave Acceleration (Grouped Bar Chart)
// ----------------------------------------------------------------------------------------
async function loadDecadalChart() {
  const canvas = document.getElementById("decades-chart");
  if (!canvas) return;

  try {
    const res = await fetch("/api/analytics/decades");
    const json = await res.json();
    if (json.status !== "success") return;

    const data = json.data;
    const ctx = canvas.getContext("2d");
    const colors = getThemeColors();

    if (decadalChartInstance) {
      decadalChartInstance.destroy();
    }

    decadalChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "Total Heatwave Days",
            data: data.total_heatwave_days,
            backgroundColor: data.labels.map(l => l.includes("2020") ? "#ea580c" : "#0d9488"),
            borderColor: data.labels.map(l => l.includes("2020") ? "#c2410c" : "#0f766e"),
            borderWidth: 1.2,
            borderRadius: 4,
          },
          {
            label: "Severe Heatwave Days",
            data: data.severe_heatwave_days,
            backgroundColor: "#dc2626",
            borderColor: "#b91c1c",
            borderWidth: 1.2,
            borderRadius: 4,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            titleColor: colors.tooltipTitle,
            bodyColor: colors.tooltipBody,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            padding: 10,
            cornerRadius: 6,
            boxShadow: "0 4px 6px -1px rgba(0,0,0,0.2)",
          }
        },
        scales: {
          x: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.tickColor, font: { size: 11 } }
          },
          y: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.tickColor },
            title: { display: true, text: "Number of Days", color: colors.tickColor, font: { size: 11 } }
          }
        }
      }
    });
  } catch (err) {
    console.error("Failed to load decadal chart:", err);
  }
}

// ----------------------------------------------------------------------------------------
// 2. City Spatial Disparity Chart (Horizontal Bar Chart)
// ----------------------------------------------------------------------------------------
async function loadCitiesChart() {
  const canvas = document.getElementById("cities-chart");
  if (!canvas) return;

  try {
    const res = await fetch("/api/analytics/city-comparison");
    const json = await res.json();
    if (json.status !== "success") return;

    const items = json.data.data;
    const cities = items.map(i => i.city);
    const counts = items.map(i => i.heatwave_days);

    const barColors = [
      "#dc2626", // Delhi
      "#ea580c", // Ahmedabad
      "#f59e0b", // Chennai
      "#d97706", // Kolkata
      "#0d9488", // Pune
      "#0284c7", // Mumbai
      "#16a34a", // Bengaluru
    ];

    const ctx = canvas.getContext("2d");
    const colors = getThemeColors();

    if (citiesChartInstance) {
      citiesChartInstance.destroy();
    }

    citiesChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: cities,
        datasets: [{
          label: "Total Heatwave Days (1951–2024)",
          data: counts,
          backgroundColor: barColors,
          borderWidth: 0,
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            titleColor: colors.tooltipTitle,
            bodyColor: colors.tooltipBody,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            padding: 10,
            cornerRadius: 6,
          }
        },
        scales: {
          x: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.tickColor }
          },
          y: {
            grid: { display: false },
            ticks: { color: colors.headingColor, font: { weight: "600", size: 12 } }
          }
        }
      }
    });
  } catch (err) {
    console.error("Failed to load city comparison chart:", err);
  }
}

// ----------------------------------------------------------------------------------------
// 3. Monthly Seasonality Breakdown (Bar / Line Combo)
// ----------------------------------------------------------------------------------------
async function loadMonthlyChart() {
  const canvas = document.getElementById("monthly-chart");
  if (!canvas) return;

  try {
    const res = await fetch("/api/analytics/monthly");
    const json = await res.json();
    if (json.status !== "success") return;

    const data = json.data;
    const ctx = canvas.getContext("2d");
    const colors = getThemeColors();

    if (monthlyChartInstance) {
      monthlyChartInstance.destroy();
    }

    monthlyChartInstance = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.months,
        datasets: [
          {
            type: "bar",
            label: "Heatwave Days Count",
            data: data.total_heatwave_days,
            backgroundColor: "#ea580c",
            borderRadius: 4,
            yAxisID: "y",
          },
          {
            type: "line",
            label: "Average Max Temp (°C)",
            data: data.avg_max_temp,
            borderColor: colors.isDark ? "#2dd4bf" : "#0f766e",
            backgroundColor: colors.isDark ? "rgba(45, 212, 191, 0.15)" : "rgba(15, 118, 110, 0.08)",
            borderWidth: 2.2,
            tension: 0.35,
            pointBackgroundColor: colors.isDark ? "#2dd4bf" : "#0f766e",
            pointBorderColor: colors.isDark ? "#0f172a" : "#ffffff",
            pointBorderWidth: 1.5,
            pointRadius: 4,
            pointHoverRadius: 6,
            yAxisID: "y1",
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: { boxWidth: 12, font: { size: 11 }, color: colors.tickColor }
          },
          tooltip: {
            backgroundColor: colors.tooltipBg,
            titleColor: colors.tooltipTitle,
            bodyColor: colors.tooltipBody,
            borderColor: colors.tooltipBorder,
            borderWidth: 1,
            padding: 10,
            cornerRadius: 6,
          }
        },
        scales: {
          x: {
            grid: { color: colors.gridColor },
            ticks: { color: colors.tickColor }
          },
          y: {
            type: "linear",
            display: true,
            position: "left",
            grid: { color: colors.gridColor },
            ticks: { color: "#ea580c" },
            title: { display: true, text: "Heatwave Days", color: "#ea580c", font: { size: 11 } }
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            grid: { drawOnChartArea: false },
            ticks: {
              color: colors.tickColor,
              callback: (val) => `${val}°C`
            },
            title: { display: true, text: "Avg Temp (°C)", color: colors.tickColor, font: { size: 11 } }
          }
        }
      }
    });
  } catch (err) {
    console.error("Failed to load monthly seasonality chart:", err);
  }
}

// ----------------------------------------------------------------------------------------
// 4. All-Time Historical Record Extremes Table
// ----------------------------------------------------------------------------------------
async function loadRecordsTable() {
  const tbody = document.getElementById("records-table-body");
  if (!tbody) return;

  try {
    const res = await fetch("/api/analytics/records");
    const json = await res.json();
    if (json.status !== "success") throw new Error(json.message || "Failed to fetch records");

    const records = Array.isArray(json.data) ? json.data : (json.data?.records || json.records || []);
    tbody.innerHTML = "";

    if (!records || records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="color:var(--text-muted); padding:1.5rem; text-align:center;">No records found.</td></tr>`;
      return;
    }

    records.forEach((rec, idx) => {
      const tr = document.createElement("tr");

      let badgeClass = "badge-normal";
      if (rec.severity === "Extreme") badgeClass = "badge-extreme";
      else if (rec.severity === "Severe") badgeClass = "badge-severe";
      else if (rec.severity === "Warning") badgeClass = "badge-warning";

      tr.innerHTML = `
        <td style="font-weight:700; color:var(--text-muted);">#${idx + 1}</td>
        <td style="font-weight:600; color:var(--text-primary);">${rec.city}</td>
        <td style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary);">${rec.date}</td>
        <td><strong style="color:var(--risk-warning); font-size:0.95rem;">${rec.temp_max.toFixed(2)}°C</strong></td>
        <td style="color:var(--text-secondary);">${rec.threshold.toFixed(2)}°C</td>
        <td><strong style="color:${rec.departure >= 4.0 ? 'var(--risk-severe)' : (rec.departure >= 2.0 ? 'var(--risk-warning)' : 'var(--risk-moderate)')};">+${rec.departure.toFixed(2)}°C</strong></td>
        <td><span class="badge ${badgeClass}">${rec.severity}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load records table:", err);
    tbody.innerHTML = `<tr><td colspan="7" style="color:#ef4444; padding:1.5rem; text-align:center;">Failed to load historical records.</td></tr>`;
  }
}
