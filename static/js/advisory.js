/**
 * HeatGuard AI - Advisory Studio Controller
 * Handles persona switching, grounded advisory generation via REST API,
 * interactive checklist management, and clipboard export.
 */

let currentAudience = "Citizen";
let rawAdvisoryText = "";
let currentAdvisory = null;

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

  // Download the complete advisory report as a PDF.
  if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {
      downloadAdvisoryPdf();
    });
  }
}

function downloadAdvisoryPdf() {
  const JsPDF = window.jspdf && window.jspdf.jsPDF;
  if (!JsPDF) {
    console.error("PDF export is unavailable because the PDF library did not load.");
    alert("PDF export is unavailable right now. Please check your connection and try again.");
    return;
  }
  if (!currentAdvisory || !rawAdvisoryText) {
    alert("Generate an advisory before exporting it.");
    return;
  }

  const doc = new JsPDF({ unit: "mm", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 18;
  const textWidth = pageWidth - margin * 2;
  let y = 20;

  const ensureSpace = (height) => {
    if (y + height > pageHeight - 18) {
      doc.addPage();
      y = 20;
    }
  };

  const writeText = (text, options = {}) => {
    const fontSize = options.fontSize || 10;
    const indent = options.indent || 0;
    const lineHeight = options.lineHeight || fontSize * 0.48;
    doc.setFont("helvetica", options.bold ? "bold" : "normal");
    doc.setFontSize(fontSize);
    doc.setTextColor(...(options.color || [31, 41, 55]));
    const lines = doc.splitTextToSize(String(text), textWidth - indent);
    lines.forEach((line) => {
      ensureSpace(lineHeight);
      doc.text(line, margin + indent, y);
      y += lineHeight;
    });
  };

  const personaTitle = currentAdvisory.persona_info?.title || currentAdvisory.audience || currentAudience;
  const city = currentAdvisory.city || document.getElementById("adv-city-select").value;
  const targetDate = currentAdvisory.target_date || document.getElementById("adv-date-select").value;
  const departure = Number(currentAdvisory.predicted_departure);
  const departureText = `${departure >= 0 ? "+" : ""}${departure.toFixed(1)} C`;

  doc.setFillColor(15, 118, 110);
  doc.rect(0, 0, pageWidth, 5, "F");
  writeText("HeatGuard AI", { fontSize: 19, bold: true, color: [15, 76, 73], lineHeight: 9 });
  writeText("Grounded Climate Early-Warning Advisory Report", { fontSize: 10, color: [71, 85, 105], lineHeight: 7 });
  y += 3;
  writeText(`Advisory for ${personaTitle}`, { fontSize: 15, bold: true, color: [15, 23, 42], lineHeight: 8 });
  if (currentAdvisory.persona_info?.tagline) {
    writeText(currentAdvisory.persona_info.tagline, { fontSize: 9, color: [71, 85, 105], lineHeight: 6 });
  }
  y += 3;

  doc.setDrawColor(203, 213, 225);
  doc.setFillColor(248, 250, 252);
  doc.roundedRect(margin, y, textWidth, 35, 2, 2, "FD");
  const contextTop = y + 7;
  const riskTier = `${currentAdvisory.risk_level} (${currentAdvisory.severity})`;
  y = contextTop;
  writeText(`City: ${city}    Forecast date: ${targetDate}`, {
    fontSize: 10, bold: true, indent: 5, lineHeight: 6
  });
  y = contextTop + 7;
  writeText(
    `Predicted maximum: ${Number(currentAdvisory.predicted_temp_max).toFixed(1)} C    Departure: ${departureText}`,
    { fontSize: 10, indent: 5, lineHeight: 6 }
  );
  y += 1;
  writeText(
    `Heatwave probability: ${Number(currentAdvisory.probability_pct).toFixed(1)}%    Risk tier: ${riskTier}`,
    { fontSize: 10, indent: 5, lineHeight: 6 }
  );
  y += 7;

  writeText("Advisory Details", { fontSize: 12, bold: true, color: [15, 76, 73], lineHeight: 7 });
  y += 1;
  rawAdvisoryText.split(/\r?\n/).forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      y += 2;
      return;
    }

    const heading = line.match(/^#{1,4}\s+(.*)$/);
    if (heading) {
      y += 2;
      writeText(heading[1].replace(/\*\*/g, ""), {
        fontSize: heading[0].startsWith("## ") ? 12 : 11,
        bold: true,
        color: [15, 23, 42],
        lineHeight: 6
      });
      y += 1;
      return;
    }

    const listItem = line.match(/^(\s*)(?:[-*]|\d+\.)\s+(.*)$/);
    const content = (listItem ? listItem[2] : line)
      .replace(/\*\*(.*?)\*\*/g, "$1")
      .replace(/\*(.*?)\*/g, "$1")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1 ($2)");
    writeText(listItem ? `- ${content}` : content, {
      indent: listItem ? 3 : 0,
      lineHeight: 5
    });
    y += 1;
  });

  ensureSpace(16);
  y += 3;
  writeText("Priority Action Checklist", { fontSize: 12, bold: true, color: [15, 76, 73], lineHeight: 7 });
  const checklist = currentAdvisory.priority_checklist || [];
  if (checklist.length) {
    checklist.forEach((item, index) => {
      const checkbox = document.querySelector(`#checklist-items input[id="chk-${index}"]`);
      writeText(`${checkbox?.checked ? "[x]" : "[ ]"} ${item}`, { indent: 2, lineHeight: 5 });
      y += 1;
    });
  } else {
    writeText("No critical action items for this risk level.", { lineHeight: 5 });
  }

  y += 4;
  writeText(`Grounding status: ${currentAdvisory.guardrail_status || "Verified prediction context"}`, {
    fontSize: 9, color: [71, 85, 105], lineHeight: 5
  });
  writeText(`Report generated: ${new Date().toLocaleString()}`, {
    fontSize: 9, color: [71, 85, 105], lineHeight: 5
  });

  const pageCount = doc.internal.getNumberOfPages();
  for (let page = 1; page <= pageCount; page += 1) {
    doc.setPage(page);
    doc.setDrawColor(226, 232, 240);
    doc.line(margin, pageHeight - 13, pageWidth - margin, pageHeight - 13);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(100, 116, 139);
    doc.text("HeatGuard AI - Operational Climate Early-Warning", margin, pageHeight - 8);
    doc.text(`Page ${page} of ${pageCount}`, pageWidth - margin, pageHeight - 8, { align: "right" });
  }

  const safeCity = city.replace(/[^a-z0-9_-]+/gi, "_");
  const safeAudience = String(currentAdvisory.audience || currentAudience).replace(/[^a-z0-9_-]+/gi, "_");
  doc.save(`HeatGuard_Advisory_${safeCity}_${targetDate}_${safeAudience}.pdf`);
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
  currentAdvisory = adv;
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
