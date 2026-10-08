/**
 * HeatGuard AI - Advisory Studio Controller
 * Handles persona switching, grounded advisory generation via REST API,
 * interactive checklist management, and clipboard export.
 */

let currentAudience = "Citizen";
let rawAdvisoryText = "";

document.addEventListener("DOMContentLoaded", () => {
  // Check URL parameters for pre-selected city & date from Dashboard
  const urlParams = new URLSearchParams(window.location.search);
  const paramCity = urlParams.get("city");
  const paramDate = urlParams.get("date");

  const citySelect = document.getElementById("adv-city-select");
  const dateSelect = document.getElementById("adv-date-select");

  if (paramCity && citySelect) citySelect.value = paramCity;
  if (paramDate && dateSelect) dateSelect.value = paramDate;

  bindEvents();
  fetchAdvisory();
});

function bindEvents() {
  const genBtn = document.getElementById("generate-adv-btn");
  const copyBtn = document.getElementById("copy-adv-btn");
  const downloadBtn = document.getElementById("download-adv-btn");

  if (genBtn) {
    genBtn.addEventListener("click", () => fetchAdvisory());
  }

  // Persona Tabs
  document.querySelectorAll(".persona-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".persona-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentAudience = tab.getAttribute("data-audience");
      fetchAdvisory();
    });
  });

  // Presets
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const city = chip.getAttribute("data-city");
      const date = chip.getAttribute("data-date");
      const citySelect = document.getElementById("adv-city-select");
      const dateSelect = document.getElementById("adv-date-select");

      if (citySelect) citySelect.value = city;
      if (dateSelect) dateSelect.value = date;
      fetchAdvisory();
    });
  });

  // Clipboard copy
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      if (!rawAdvisoryText) return;
      navigator.clipboard.writeText(rawAdvisoryText).then(() => {
        const origHtml = copyBtn.innerHTML;
        copyBtn.innerHTML = `<i data-feather="check" style="width:13px;height:13px;color:#16a34a;"></i> Copied!`;
        feather.replace();
        setTimeout(() => {
          copyBtn.innerHTML = origHtml;
          feather.replace();
        }, 2000);
      });
    });
  }

  // Download .txt
  if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {
      if (!rawAdvisoryText) return;
      const city = document.getElementById("adv-city-select").value;
      const date = document.getElementById("adv-date-select").value;
      const blob = new Blob([rawAdvisoryText], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `HeatGuard_Advisory_${city}_${date}_${currentAudience.replace(/\s+/g, '_')}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }
}

async function fetchAdvisory() {
  const city = document.getElementById("adv-city-select").value;
  const date = document.getElementById("adv-date-select").value;
  const contentEl = document.getElementById("advisory-content");

  contentEl.innerHTML = `
    <div style="text-align:center; padding:3rem 1rem; color:var(--text-muted);">
      <div style="display:inline-block; width:28px; height:28px; border:3px solid var(--brand-600); border-top-color:transparent; border-radius:50%; animation:spin 0.8s linear infinite;"></div>
      <div style="margin-top:0.75rem; font-size:0.88rem; font-weight:500;">Compiling verified ML context & generating grounded advisory...</div>
    </div>
  `;

  try {
    const res = await fetch("/api/advisory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: city,
        date: date,
        audience: currentAudience,
      }),
    });

    const json = await res.json();
    if (json.status === "success") {
      renderAdvisory(json.advisory);
    } else {
      contentEl.innerHTML = `<div style="color:#dc2626; padding:1.5rem;">Failed to generate advisory: ${json.message}</div>`;
    }
  } catch (err) {
    console.error("Advisory error:", err);
    contentEl.innerHTML = `<div style="color:#dc2626; padding:1.5rem;">Network connection error.</div>`;
  }
}

function renderAdvisory(adv) {
  rawAdvisoryText = adv.advisory_markdown;

  // Update Context Chips
  document.getElementById("ctx-city").innerText = adv.city;
  document.getElementById("ctx-date").innerText = adv.target_date;
  document.getElementById("ctx-temp").innerText = `${adv.predicted_temp_max.toFixed(1)}°C`;
  const depSign = adv.predicted_departure >= 0 ? "+" : "";
  document.getElementById("ctx-departure").innerText = `${depSign}${adv.predicted_departure.toFixed(1)}°C`;

  const riskBadge = document.getElementById("ctx-risk-badge");
  riskBadge.innerText = `${adv.risk_level} (${adv.severity})`;
  riskBadge.className = `badge badge-${adv.severity.toLowerCase()}`;

  document.getElementById("advisory-title").innerHTML = `
    <i data-feather="file-text"></i>
    <span>Advisory for ${adv.persona_info.title}</span>
  `;
  document.getElementById("advisory-subtitle").innerText = adv.persona_info.tagline;

  // Format Markdown to Clean HTML
  const contentEl = document.getElementById("advisory-content");
  contentEl.innerHTML = markdownToHtml(adv.advisory_markdown);

  // Render Priority Checklist
  renderChecklist(adv.priority_checklist);

  feather.replace();
}

function renderChecklist(items) {
  const container = document.getElementById("checklist-items");
  const countBadge = document.getElementById("checklist-count-badge");
  if (!container) return;

  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = `<div style="color:var(--text-muted); font-size:0.85rem;">No critical items for this risk level.</div>`;
    if (countBadge) countBadge.innerText = "0 items";
    return;
  }

  let completed = 0;
  const updateCount = () => {
    if (countBadge) countBadge.innerText = `${completed} / ${items.length} Complete`;
  };

  items.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "checklist-item";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = `chk-${idx}`;

    const label = document.createElement("label");
    label.htmlFor = `chk-${idx}`;
    label.innerText = item;
    label.style.cursor = "pointer";
    label.style.flex = "1";

    cb.addEventListener("change", () => {
      if (cb.checked) {
        div.classList.add("checked");
        completed++;
      } else {
        div.classList.remove("checked");
        completed--;
      }
      updateCount();
    });

    div.appendChild(cb);
    div.appendChild(label);
    container.appendChild(div);
  });

  updateCount();
}

// Lightweight Markdown Converter
function markdownToHtml(md) {
  if (!md) return "";
  let html = md
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/gim, '<em>$1</em>')
    .replace(/^\d+\.\s+(.*$)/gim, '<li>$1</li>')
    .replace(/^\-\s+(.*$)/gim, '<li>$1</li>')
    .replace(/\n\n+/g, '<br><br>');

  // Wrap lists in <ol> if containing <li>
  if (html.includes('<li>')) {
    html = html.replace(/(<li>.*?<\/li>)/gims, '<ol>$1</ol>');
  }
  return html;
}
