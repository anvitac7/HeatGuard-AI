/**
 * HeatGuard AI — Responsible AI & Algorithmic Governance Hub
 * Controls 3-pillar tabs, interactive city deep-dive, Chart.js XAI feature rankings,
 * local prediction factor attribution inspector, and live advisory compliance auditor.
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initFairnessDeepDive();
  initXAI();
  initSafetyAuditor();
});

// ======================================================================================
// Tab Switching Controller
// ======================================================================================
function initTabs() {
  const tabs = document.querySelectorAll('.ethics-tab');
  const panes = {
    fairness: document.getElementById('pane-fairness'),
    xai: document.getElementById('pane-xai'),
    safety: document.getElementById('pane-safety')
  };

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      if (!panes[target]) return;

      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      Object.values(panes).forEach(p => p && p.classList.remove('active'));
      panes[target].classList.add('active');

      if (window.feather) feather.replace();

      // Trigger Chart.js resize if switching to XAI
      if (target === 'xai' && window._globalImportanceChart) {
        window._globalImportanceChart.resize();
      }
    });
  });
}

// ======================================================================================
// Pillar 1: Interactive Single-City Deep Dive
// ======================================================================================
function initFairnessDeepDive() {
  const select = document.getElementById('deep-dive-city-select');
  if (!select) return;

  const updateDeepDive = async (city) => {
    try {
      const res = await fetch(`/api/ethics/fairness/${encodeURIComponent(city)}`);
      const data = await res.json();
      if (data.status !== 'success') return;

      const m = data.city_audit;
      const el = (id) => document.getElementById(id);

      el('dd-city-name').textContent = m.city;
      el('dd-tp').textContent = m.true_positives;
      el('dd-fp').textContent = m.false_positives;
      el('dd-fn').textContent = m.false_negatives;
      el('dd-tn').textContent = m.true_negatives;

      el('dd-recall').textContent = m.recall_pct !== null ? `${m.recall_pct}%` : 'N/A';
      el('dd-precision').textContent = m.precision_pct !== null ? `${m.precision_pct}%` : 'N/A';
      el('dd-specificity').textContent = `${m.specificity_pct}%`;
      el('dd-far').textContent = `${m.false_alarm_rate_pct}%`;

      const badge = el('dd-status-badge');
      badge.textContent = m.bias_tier;
      badge.className = 'badge';
      if (m.status_color === 'success') {
        badge.classList.add('badge-normal');
      } else if (m.status_color === 'warning') {
        badge.classList.add('badge-warning');
      } else if (m.status_color === 'info') {
        badge.style.background = 'var(--brand-50)';
        badge.style.color = 'var(--brand-700)';
      } else {
        badge.style.background = 'var(--bg-muted)';
        badge.style.color = 'var(--text-muted)';
      }

      el('dd-insight').textContent = m.ethical_insight;
      if (window.feather) feather.replace();
    } catch (err) {
      console.error('Error fetching city deep dive:', err);
    }
  };

  select.addEventListener('change', () => updateDeepDive(select.value));
  // Initial load
  updateDeepDive(select.value || 'Delhi');
}

// ======================================================================================
// Pillar 2: Explainable AI (XAI)
// ======================================================================================
function initXAI() {
  loadGlobalImportance();
  initLocalInspector();
}

async function loadGlobalImportance() {
  const chartCanvas = document.getElementById('global-importance-chart');
  const roleList = document.getElementById('feature-role-list');
  if (!chartCanvas) return;

  try {
    const res = await fetch('/api/ethics/explain');
    const data = await res.json();
    if (data.status !== 'success') return;

    const features = data.global_importance.features;

    // Render Chart.js Horizontal Bar Chart
    const labels = features.map(f => f.display_name);
    const values = features.map(f => f.importance_pct);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';
    const textColor = isDark ? '#cbd5e1' : '#475569';

    if (window._globalImportanceChart) {
      window._globalImportanceChart.destroy();
    }

    const ctx = chartCanvas.getContext('2d');
    window._globalImportanceChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Feature Importance (%)',
          data: values,
          backgroundColor: features.map(f => {
            if (f.category.includes('Threshold')) return 'rgba(13, 148, 136, 0.85)';
            if (f.category.includes('Conditions')) return 'rgba(20, 184, 166, 0.75)';
            if (f.category.includes('Persistence')) return 'rgba(234, 88, 12, 0.8)';
            if (f.category.includes('Moisture')) return 'rgba(34, 197, 94, 0.75)';
            return 'rgba(100, 116, 139, 0.75)';
          }),
          borderColor: 'transparent',
          borderRadius: 5,
          borderSkipped: false,
          barThickness: 18
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => features[items[0].dataIndex].display_name,
              label: (item) => ` ${item.raw}% Importance`,
              afterLabel: (item) => {
                const f = features[item.dataIndex];
                return `Category: ${f.category}\nRole: ${f.role}`;
              }
            }
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            max: 40,
            grid: { color: gridColor },
            ticks: {
              color: textColor,
              callback: (v) => `${v}%`
            }
          },
          y: {
            grid: { display: false },
            ticks: {
              color: textColor,
              font: { size: 11, weight: '500' }
            }
          }
        }
      }
    });

    // Populate Feature Role Summary List
    if (roleList) {
      roleList.innerHTML = features.map((f, i) => `
        <div style="padding:0.6rem 0.8rem; background:var(--bg-subtle); border-radius:var(--radius-md); border:1px solid var(--border-subtle); display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem;">
          <div>
            <div style="font-weight:700; font-size:0.83rem; color:var(--text-primary); display:flex; align-items:center; gap:0.35rem;">
              <span>${i + 1}. ${f.display_name}</span>
              <span class="badge" style="background:var(--bg-muted); color:var(--text-muted); font-size:0.68rem; padding:0.15rem 0.4rem;">${f.category}</span>
            </div>
            <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:0.2rem;">${f.description}</div>
            <div style="font-size:0.73rem; color:var(--brand-700); font-weight:600; margin-top:0.15rem;">Role: ${f.role}</div>
          </div>
          <div style="font-weight:800; font-size:0.95rem; color:var(--brand-600); font-family:var(--font-mono); white-space:nowrap;">
            ${f.importance_pct}%
          </div>
        </div>
      `).join('');
    }

  } catch (err) {
    console.error('Error loading global feature importance:', err);
  }
}

function initLocalInspector() {
  const citySelect = document.getElementById('xai-city-select');
  const dateInput = document.getElementById('xai-date-select');
  const inspectBtn = document.getElementById('inspect-xai-btn');
  const presets = document.querySelectorAll('[data-xai-city]');

  if (!inspectBtn) return;

  const runInspection = async (city, date) => {
    inspectBtn.disabled = true;
    inspectBtn.innerHTML = '<i data-feather="loader" style="width:14px;height:14px;animation:spin 1s infinite linear;"></i> Computing...';
    if (window.feather) feather.replace();

    try {
      const res = await fetch(`/api/ethics/explain?city=${encodeURIComponent(city)}&date=${encodeURIComponent(date)}`);
      const data = await res.json();

      if (data.status !== 'success') {
        alert(data.message || 'Inspection failed');
        return;
      }

      const exp = data.explanation;
      const el = (id) => document.getElementById(id);

      el('xai-ctx-city').textContent = exp.city;
      el('xai-ctx-date').textContent = exp.target_date;
      el('xai-ctx-temp').textContent = `${exp.predicted_temp_max}°C`;
      el('xai-ctx-dep').textContent = `${exp.predicted_departure >= 0 ? '+' : ''}${exp.predicted_departure}°C`;
      el('xai-ctx-prob').textContent = `${exp.probability_pct}%`;

      const badge = el('xai-ctx-badge');
      badge.textContent = exp.risk_level;
      badge.className = 'badge';
      if (exp.risk_level === 'EXTREME') badge.classList.add('badge-severe');
      else if (exp.risk_level === 'HIGH') badge.classList.add('badge-warning');
      else if (exp.risk_level === 'MODERATE') badge.classList.add('badge-moderate');
      else badge.classList.add('badge-normal');

      el('xai-synthesis-text').textContent = exp.plain_language_synthesis;

      // Render 5 Factor Cards
      const factorsList = el('xai-factors-list');
      factorsList.innerHTML = exp.top_contributing_factors.map(f => {
        let iconName = 'arrow-up-right';
        let iconColor = 'var(--risk-warning)';
        let impactBadgeClass = 'badge-warning';

        if (f.direction === 'down') {
          iconName = 'arrow-down-right';
          iconColor = 'var(--risk-normal)';
          impactBadgeClass = 'badge-normal';
        } else if (f.direction === 'neutral') {
          iconName = 'minus';
          iconColor = 'var(--text-muted)';
          impactBadgeClass = '';
        }

        return `
          <div class="factor-card">
            <div class="factor-header">
              <div class="factor-name">
                <i data-feather="${iconName}" style="color:${iconColor}; width:16px;height:16px;"></i>
                <span>${f.factor}</span>
              </div>
              <div style="display:flex; align-items:center; gap:0.5rem;">
                <span class="factor-value-tag">${f.value}</span>
                <span class="badge ${impactBadgeClass}" style="font-size:0.7rem;">${f.impact}</span>
              </div>
            </div>
            <div class="factor-explanation">${f.explanation}</div>
          </div>
        `;
      }).join('');

      if (window.feather) feather.replace();

    } catch (err) {
      console.error('Error running factor inspection:', err);
    } finally {
      inspectBtn.disabled = false;
      inspectBtn.innerHTML = '<i data-feather="zap" style="width:15px;height:15px;"></i> Inspect Factor Drivers';
      if (window.feather) feather.replace();
    }
  };

  inspectBtn.addEventListener('click', () => {
    runInspection(citySelect.value, dateInput.value);
  });

  presets.forEach(p => {
    p.addEventListener('click', () => {
      citySelect.value = p.dataset.xaiCity;
      dateInput.value = p.dataset.xaiDate;
      runInspection(citySelect.value, dateInput.value);
    });
  });

  // Initial inspection for Delhi
  runInspection(citySelect.value || 'Delhi', dateInput.value || '2024-05-28');
}

// ======================================================================================
// Pillar 3: GenAI Safety & Advisory Compliance Auditor
// ======================================================================================
function initSafetyAuditor() {
  const runBtn = document.getElementById('run-audit-btn');
  const sampleBtn = document.getElementById('load-sample-adv-btn');
  const citySelect = document.getElementById('audit-city-select');
  const riskSelect = document.getElementById('audit-risk-select');
  const personaSelect = document.getElementById('audit-persona-select');
  const textInput = document.getElementById('audit-text-input');

  if (!runBtn) return;

  const samples = [
    {
      city: 'Delhi',
      risk: 'HIGH',
      persona: 'Citizen',
      text: 'HeatGuard AI Operational Advisory for Delhi: Severe heat conditions predicted tomorrow with peak temperature of 44.7°C. Stay indoors during peak afternoon hours (12 PM to 4 PM), maintain regular hydration with oral rehydration solutions, and avoid direct sun exposure. Municipal cooling shelters are active. Consult local health facilities if heat exhaustion symptoms arise.'
    },
    {
      city: 'Ahmedabad',
      risk: 'EXTREME',
      persona: 'Municipal Authority',
      text: 'HeatGuard AI Disaster Command Directive for Ahmedabad: Forecast temperature of 46.6°C triggers Red Alert protocols under the Ahmedabad Heat Action Plan. Activate urban water distribution points across 48 ward hubs. Shift outdoor construction and municipal sanitation work to twilight hours. Ensure government hospital emergency wards maintain adequate ice packs and intravenous saline supplies.'
    },
    {
      city: 'Bengaluru',
      risk: 'NORMAL',
      persona: 'Farmer',
      text: 'HeatGuard AI Agricultural Bulletin for Bengaluru: Moderate temperatures of 32.5°C remain within seasonal historical baselines. Standard morning irrigation cycles recommended. No extraordinary thermal stress countermeasures required for standing crops.'
    }
  ];

  let sampleIdx = 0;
  if (sampleBtn) {
    sampleBtn.addEventListener('click', () => {
      sampleIdx = (sampleIdx + 1) % samples.length;
      const s = samples[sampleIdx];
      citySelect.value = s.city;
      riskSelect.value = s.risk;
      personaSelect.value = s.persona;
      textInput.value = s.text;
      runAudit();
    });
  }

  const runAudit = async () => {
    const text = textInput.value.trim();
    if (!text) {
      alert('Please enter advisory text to audit');
      return;
    }

    runBtn.disabled = true;
    runBtn.innerHTML = '<i data-feather="loader" style="width:14px;height:14px;animation:spin 1s infinite linear;"></i> Running 5-Point Safety Audit...';
    if (window.feather) feather.replace();

    try {
      const res = await fetch('/api/ethics/audit-advisory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          city: citySelect.value,
          risk_level: riskSelect.value,
          persona: personaSelect.value,
          advisory_text: text
        })
      });

      const data = await res.json();
      if (data.status !== 'success') {
        alert(data.message || 'Audit request failed');
        return;
      }

      const audit = data.audit;
      const el = (id) => document.getElementById(id);

      const statusBadge = el('audit-status-badge');
      statusBadge.textContent = audit.overall_status;
      statusBadge.className = 'badge';
      if (audit.overall_status === 'VERIFIED_SAFE') {
        statusBadge.classList.add('badge-normal');
      } else {
        statusBadge.classList.add('badge-warning');
      }

      el('audit-score-num').textContent = `${audit.safety_score_pct}%`;

      const checksContainer = el('audit-checks-container');
      checksContainer.innerHTML = audit.checks.map(c => {
        let badgeClass = 'badge-normal';
        let iconName = 'check-circle';
        if (c.status === 'WARN') {
          badgeClass = 'badge-warning';
          iconName = 'alert-triangle';
        } else if (c.status === 'FAIL') {
          badgeClass = 'badge-severe';
          iconName = 'x-circle';
        }

        return `
          <div class="audit-check-item">
            <div>
              <div class="audit-check-title">${c.name}</div>
              <div class="audit-check-desc">${c.description}</div>
              <div class="audit-check-detail">${c.details.join(' • ')}</div>
            </div>
            <span class="badge ${badgeClass}" style="font-size:0.75rem; white-space:nowrap;">
              <i data-feather="${iconName}" style="width:12px;height:12px;"></i> ${c.status}
            </span>
          </div>
        `;
      }).join('');

      if (window.feather) feather.replace();

    } catch (err) {
      console.error('Error running advisory audit:', err);
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = '<i data-feather="shield" style="width:15px;height:15px;"></i> Run Safety Compliance Audit';
      if (window.feather) feather.replace();
    }
  };

  runBtn.addEventListener('click', runAudit);
  // Initial run
  runAudit();
}
