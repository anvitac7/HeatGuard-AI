/**
 * HeatGuard AI - 74-Year Climate Analytics Charts
 * Visualizes decadal heatwave acceleration, city spatial disparity,
 * monthly seasonality distributions, and historical climate records using Chart.js.
 */

document.addEventListener("DOMContentLoaded", () => {
  loadDecadalChart();
  loadCitiesChart();
  loadMonthlyChart();
  loadRecordsTable();
});

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

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "Total Heatwave Days",
            data: data.total_heatwave_days,
            backgroundColor: data.labels.map(l => l.includes("2020") ? "rgba(249, 115, 22, 0.85)" : "rgba(0, 240, 255, 0.65)"),
            borderColor: data.labels.map(l => l.includes("2020") ? "#f97316" : "#00f0ff"),
            borderWidth: 1.8,
            borderRadius: 6,
          },
          {
            label: "Severe Heatwave Days",
            data: data.severe_heatwave_days,
            backgroundColor: data.labels.map(l => l.includes("2020") ? "rgba(239, 68, 68, 0.85)" : "rgba(239, 68, 68, 0.5)"),
            borderColor: "#ef4444",
            borderWidth: 1.8,
            borderRadius: 6,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
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
            ticks: { color: "#94a3b8" }
          },
          y: {
            grid: { color: "rgba(0, 240, 255, 0.06)" },
            ticks: { color: "#94a3b8" },
            title: { display: true, text: "Number of Days", color: "#64748b" }
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

    const colors = [
      "#ef4444", // Delhi
      "#f97316", // Ahmedabad
      "#fb923c", // Chennai
      "#f59e0b", // Kolkata
      "#eab308", // Pune
      "#06b6d4", // Mumbai
      "#10b981", // Bengaluru
    ];

    const ctx = canvas.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: cities,
        datasets: [{
          label: "Total Heatwave Days (1951–2024)",
          data: counts,
          backgroundColor: colors.map(c => `${c}cc`),
          borderColor: colors,
          borderWidth: 1.5,
          borderRadius: 6,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(13, 19, 34, 0.95)",
            padding: 10,
            cornerRadius: 8,
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.05)" },
            ticks: { color: "#94a3b8" }
          },
          y: {
            grid: { display: false },
            ticks: { color: "#f8fafc", font: { weight: "600" } }
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

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.months,
        datasets: [
          {
            type: "bar",
            label: "Heatwave Days Count",
            data: data.total_heatwave_days,
            backgroundColor: "rgba(249, 115, 22, 0.7)",
            borderColor: "#f97316",
            borderWidth: 1.5,
            borderRadius: 6,
            yAxisID: "y",
          },
          {
            type: "line",
            label: "Average Max Temp (°C)",
            data: data.avg_max_temp,
            borderColor: "#00f0ff",
            backgroundColor: "rgba(0, 240, 255, 0.15)",
            borderWidth: 2.8,
            tension: 0.35,
            pointBackgroundColor: "#00f0ff",
            pointBorderColor: "#fff",
            pointBorderWidth: 1.5,
            pointRadius: 5,
            pointHoverRadius: 8,
            yAxisID: "y1",
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: "#e0f2fe", font: { size: 11, weight: "bold" } }
          },
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
            ticks: { color: "#94a3b8" }
          },
          y: {
            type: "linear",
            display: true,
            position: "left",
            grid: { color: "rgba(0, 240, 255, 0.06)" },
            ticks: { color: "#94a3b8" },
            title: { display: true, text: "Heatwave Days", color: "#f97316" }
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            grid: { drawOnChartArea: false },
            ticks: {
              color: "#00f0ff",
              callback: (val) => `${val}°C`
            },
            title: { display: true, text: "Avg Temp (°C)", color: "#00f0ff" }
          }
        }
      }
    });
  } catch (err) {
    console.error("Failed to load monthly chart:", err);
  }
}

// ----------------------------------------------------------------------------------------
// 4. All-Time Historical Records Table
// ----------------------------------------------------------------------------------------
async function loadRecordsTable() {
  const tbody = document.getElementById("records-table-body");
  if (!tbody) return;

  try {
    const res = await fetch("/api/analytics/records");
    const json = await res.json();
    if (json.status !== "success") return;

    const records = json.data;
    tbody.innerHTML = "";

    records.forEach((r, idx) => {
      const tr = document.createElement("tr");

      let badgeClass = "badge-warning";
      if (r.severity === "Severe") badgeClass = "badge-severe";
      if (r.severity === "Extreme") badgeClass = "badge-extreme";

      tr.innerHTML = `
        <td><strong style="color:var(--accent-orange); font-size:0.95rem;">#${idx + 1}</strong></td>
        <td><strong style="color:var(--text-primary); font-size:0.95rem;">${r.city}</strong></td>
        <td>${r.date}</td>
        <td><strong style="font-size:1.05rem; color:#fff;">${r.temp_max.toFixed(2)}°C</strong></td>
        <td>${r.threshold.toFixed(2)}°C</td>
        <td><strong style="color:#f87171;">+${r.departure.toFixed(2)}°C</strong></td>
        <td><span class="badge ${badgeClass}">${r.severity}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load records table:", err);
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:#ef4444;">Failed to load records.</td></tr>`;
  }
}
