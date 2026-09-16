// SCOGS & MedGemma Results Dashboard Application Logic

let currentRunData = null;
let currentAuditData = [];
let selectedPatientUid = null;
let activeHighlightQuote = null;
let filteredPatientUids = [];
let charts = {};

const OUTCOME_NAMES = {
  "28": "Acute Sickle Cell Pain Episode",
  "48": "Acute Chest Syndrome (ACS)",
  "36": "Fever",
  "19": "Acute Kidney Injury (AKI)"
};

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupEventListeners();
  setupKeyboardShortcuts();
  
  if (window.DEFAULT_RUN_DATA) {
    loadRunData(window.DEFAULT_RUN_DATA);
  } else {
    fetchRunFromAPI("a100_27b_ollama.json");
  }

  if (window.DEFAULT_AUDIT_DATA && window.DEFAULT_AUDIT_DATA.length > 0) {
    currentAuditData = window.DEFAULT_AUDIT_DATA;
    renderAuditView();
  }
});

function setupTabs() {
  const triggers = document.querySelectorAll(".tab-trigger");
  triggers.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      
      triggers.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      
      document.querySelectorAll(".tab-pane").forEach(tp => {
        tp.classList.remove("active");
      });
      
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
      
      if (targetId === "tab-overview") {
        setTimeout(resizeCharts, 50);
      }
    });
  });
}

function setupEventListeners() {
  document.getElementById("case-search")?.addEventListener("input", filterAndRenderCases);
  document.getElementById("filter-selection")?.addEventListener("change", filterAndRenderCases);
  document.getElementById("filter-outcome")?.addEventListener("change", filterAndRenderCases);
  document.getElementById("filter-status")?.addEventListener("change", filterAndRenderCases);

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", handleFileSelect);

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });
  }

  document.querySelectorAll(".modal-close-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.getElementById("raw-modal")?.classList.remove("active");
    });
  });

  const modalOverlay = document.getElementById("raw-modal");
  if (modalOverlay) {
    modalOverlay.addEventListener("click", (e) => {
      if (e.target === modalOverlay) {
        modalOverlay.classList.remove("active");
      }
    });
  }
}

function setupKeyboardShortcuts() {
  window.addEventListener("keydown", (e) => {
    // Focus search with '/'
    if (e.key === "/" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
      e.preventDefault();
      document.getElementById("case-search")?.focus();
      return;
    }

    // Escape closes modal
    if (e.key === "Escape") {
      document.getElementById("raw-modal")?.classList.remove("active");
      return;
    }

    // Arrow navigation in Case Explorer
    if ((e.key === "ArrowDown" || e.key === "ArrowUp") && document.activeElement.tagName !== "INPUT") {
      if (!filteredPatientUids || filteredPatientUids.length === 0) return;
      e.preventDefault();
      const curIndex = filteredPatientUids.indexOf(selectedPatientUid);
      let nextIndex = 0;
      if (e.key === "ArrowDown") {
        nextIndex = curIndex < filteredPatientUids.length - 1 ? curIndex + 1 : 0;
      } else {
        nextIndex = curIndex > 0 ? curIndex - 1 : filteredPatientUids.length - 1;
      }
      selectCase(filteredPatientUids[nextIndex]);
      
      // Scroll list item into view
      const activeEl = document.querySelector(".case-row-card.active");
      activeEl?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });
}

async function fetchRunFromAPI(filename) {
  try {
    const res = await fetch(`/api/run/${filename}`);
    if (res.ok) {
      const data = await res.json();
      loadRunData(data);
    }
  } catch (err) {
    console.warn("Could not fetch from API", err);
  }
}

function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    processUploadedFile(e.target.files[0]);
  }
}

function processUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      if (file.name.endsWith(".json")) {
        const data = JSON.parse(e.target.result);
        loadRunData(data);
        showToast(`Loaded ${file.name}`);
      } else if (file.name.endsWith(".csv")) {
        parseAuditCSV(e.target.result);
        showToast(`Loaded ${file.name}`);
      }
    } catch (err) {
      alert("Error parsing file: " + err.message);
    }
  };
  reader.readAsText(file);
}

function loadRunData(data) {
  currentRunData = data;
  renderOverview(data);
  populateFilterOptions(data);
  filterAndRenderCases();
  
  if (data.detailed_records && data.detailed_records.length > 0) {
    selectCase(data.detailed_records[0].patient_uid);
  }

  if (!currentAuditData || currentAuditData.length === 0) {
    deriveAuditDataFromRun(data);
  }
  renderAuditView();
}

function renderOverview(data) {
  const prov = data.provenance || {};
  const prof = data.profiling || {};
  const auto = data.automated_metrics || {};

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl("kpi-model", prov.served_as || prov.weights || "MedGemma 27B");
  setEl("kpi-notes", prov.notes_count ?? (data.detailed_records?.length || 0));
  setEl("kpi-sec-per-note", prof.sec_per_note ? `${prof.sec_per_note}s / note` : "N/A");
  setEl("kpi-throughput", prof.completion_tokens_per_sec ? `${prof.completion_tokens_per_sec} tok/s` : "N/A");
  
  const quotePct = auto.quote_verified_pct_of_quoted ?? 100;
  setEl("kpi-quote-verified", `${quotePct}%`);
  setEl("kpi-hallucinations", `${auto.hallucinated_pct_of_quoted ?? 0}%`);
  setEl("kpi-consistency", `${auto.run_to_run_consistency_pct ?? 100}%`);
  setEl("kpi-invalid-vals", `${auto.invalid_value_count ?? 0}`);

  const provContainer = document.getElementById("provenance-container");
  if (provContainer) {
    provContainer.innerHTML = `
      <div class="spec-cell"><span class="spec-key">Tier:</span><span class="spec-val">${prov.tier || "full"}</span></div>
      <div class="spec-cell"><span class="spec-key">Base Weights:</span><span class="spec-val">${prov.weights || "google/medgemma-27b-text-it"}</span></div>
      <div class="spec-cell"><span class="spec-key">Backend:</span><span class="spec-val">${prov.backend || "ollama"}</span></div>
      <div class="spec-cell"><span class="spec-key">Quantization:</span><span class="spec-val">${prov.quant || "file_type_32"}</span></div>
      <div class="spec-cell"><span class="spec-key">Cohort:</span><span class="spec-val">${prov.cohort || "loose"}</span></div>
      <div class="spec-cell"><span class="spec-key">Stratified Holdout:</span><span class="spec-val">${prov.stratified ? `${(prov.holdout_frac || 0.25)*100}%` : "No"}</span></div>
      <div class="spec-cell"><span class="spec-key">Repeats / Concurrency:</span><span class="spec-val">${prov.repeat || 1} / ${prov.concurrency || 1}</span></div>
      <div class="spec-cell"><span class="spec-key">Run ID:</span><span class="spec-val">${prov.run_id || "N/A"}</span></div>
      <div class="spec-cell"><span class="spec-key">Timestamp:</span><span class="spec-val">${prov.timestamp || "N/A"}</span></div>
      <div class="spec-cell"><span class="spec-key">Total Wall Clock:</span><span class="spec-val">${prof.total_wall_clock_sec ? `${prof.total_wall_clock_sec}s (${(prof.total_wall_clock_sec/60).toFixed(1)}m)` : "N/A"}</span></div>
      <div class="spec-cell"><span class="spec-key">Prompt Tokens:</span><span class="spec-val">${prof.total_prompt_tokens?.toLocaleString() || "N/A"}</span></div>
      <div class="spec-cell"><span class="spec-key">Completion Tokens:</span><span class="spec-val">${prof.total_completion_tokens?.toLocaleString() || "N/A"}</span></div>
    `;
  }

  renderCharts(data);
}

function renderCharts(data) {
  if (typeof Chart === "undefined") return;

  const outcomeCanvas = document.getElementById("chart-outcome-status");
  if (outcomeCanvas) {
    if (charts.outcomeStatus) charts.outcomeStatus.destroy();
    
    const byOutcome = data.grade_status_by_outcome || {};
    const outcomeIds = Object.keys(byOutcome);
    const labels = outcomeIds.map(id => `${id}: ${OUTCOME_NAMES[id] || id}`);

    const statusKeys = ["graded", "grade_set", "cannot_grade", "refuted", "absent"];
    const statusColors = {
      graded: "#10b981",
      grade_set: "#0ea5e9",
      cannot_grade: "#f59e0b",
      refuted: "#ef4444",
      absent: "#64748b"
    };

    const datasets = statusKeys.map(status => ({
      label: status.replace("_", " ").toUpperCase(),
      backgroundColor: statusColors[status],
      data: outcomeIds.map(id => byOutcome[id]?.[status] || 0)
    }));

    charts.outcomeStatus = new Chart(outcomeCanvas, {
      type: "bar",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { stacked: true, grid: { color: "#1e293b" }, ticks: { color: "#94a3b8", font: { size: 10 } } },
          y: { stacked: true, grid: { color: "#1e293b" }, ticks: { color: "#94a3b8" } }
        },
        plugins: {
          legend: { position: "top", labels: { color: "#f1f5f9", boxWidth: 10, font: { size: 11 } } }
        }
      }
    });
  }

  const featuresCanvas = document.getElementById("chart-features");
  if (featuresCanvas) {
    if (charts.features) charts.features.destroy();

    const featObj = data.features_extracted || {};
    const sortedFeats = Object.entries(featObj).sort((a, b) => b[1] - a[1]);
    const featLabels = sortedFeats.map(s => s[0]);
    const featCounts = sortedFeats.map(s => s[1]);

    charts.features = new Chart(featuresCanvas, {
      type: "bar",
      data: {
        labels: featLabels,
        datasets: [{
          label: "Mentions Extracted",
          data: featCounts,
          backgroundColor: "#0284c7",
          borderRadius: 2
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { grid: { color: "#1e293b" }, ticks: { color: "#94a3b8", stepSize: 1 } },
          y: { grid: { display: false }, ticks: { color: "#f1f5f9", font: { family: "monospace", size: 11 } } }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  const funnelCanvas = document.getElementById("chart-funnel");
  if (funnelCanvas) {
    if (charts.funnel) charts.funnel.destroy();

    const auto = data.automated_metrics || {};
    const funnelLabels = ["Proposed", "Quoted", "Quote Verified", "Null Placeholder", "Invalid Values"];
    const funnelValues = [
      auto.proposed || 0,
      auto.quoted || 0,
      auto.quote_verified || 0,
      auto.null_placeholder || 0,
      auto.invalid_value_count || 0
    ];

    charts.funnel = new Chart(funnelCanvas, {
      type: "doughnut",
      data: {
        labels: funnelLabels,
        datasets: [{
          data: funnelValues,
          backgroundColor: ["#6366f1", "#0284c7", "#10b981", "#64748b", "#ef4444"],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { position: "right", labels: { color: "#f1f5f9", font: { size: 11 }, boxWidth: 10 } }
        }
      }
    });
  }
}

function resizeCharts() {
  Object.values(charts).forEach(c => c && c.resize());
}

function populateFilterOptions(data) {
  const selSelect = document.getElementById("filter-selection");
  if (!selSelect || !data.detailed_records) return;

  const selections = new Set();
  data.detailed_records.forEach(r => {
    if (r.selection) selections.add(r.selection);
  });

  selSelect.innerHTML = '<option value="all">All Cohorts &amp; Seeds</option>';
  selections.forEach(s => {
    selSelect.innerHTML += `<option value="${s}">${s}</option>`;
  });
}

function filterAndRenderCases() {
  if (!currentRunData || !currentRunData.detailed_records) return;

  const searchText = (document.getElementById("case-search")?.value || "").toLowerCase();
  const filterSelection = document.getElementById("filter-selection")?.value || "all";
  const filterOutcome = document.getElementById("filter-outcome")?.value || "all";
  const filterStatus = document.getElementById("filter-status")?.value || "all";

  const listContainer = document.getElementById("cases-list");
  if (!listContainer) return;

  listContainer.innerHTML = "";

  const filtered = currentRunData.detailed_records.filter(rec => {
    if (filterSelection !== "all" && rec.selection !== filterSelection) return false;

    if (filterOutcome !== "all" || filterStatus !== "all") {
      let matchesOutcomeStatus = false;
      for (const [outId, outData] of Object.entries(rec.outcomes || {})) {
        if (filterOutcome !== "all" && outId !== filterOutcome) continue;
        
        const status = outData.grade_result?.status || "absent";
        if (filterStatus !== "all" && status !== filterStatus) continue;

        matchesOutcomeStatus = true;
        break;
      }
      if (!matchesOutcomeStatus) return false;
    }

    if (searchText) {
      const matchUid = rec.patient_uid?.toLowerCase().includes(searchText);
      const matchTitle = rec.title?.toLowerCase().includes(searchText);
      const matchNote = rec.patient_note?.toLowerCase().includes(searchText);
      if (!matchUid && !matchTitle && !matchNote) return false;
    }

    return true;
  });

  filteredPatientUids = filtered.map(r => r.patient_uid);
  document.getElementById("cases-count-badge").textContent = `${filtered.length} of ${currentRunData.detailed_records.length}`;

  if (filtered.length === 0) {
    listContainer.innerHTML = `<div style="padding: 1.25rem; text-align: center; color: var(--text-dim); font-size: 0.8rem;">No cases match criteria</div>`;
    return;
  }

  filtered.forEach(rec => {
    const item = document.createElement("div");
    item.className = `case-row-card ${rec.patient_uid === selectedPatientUid ? "active" : ""}`;
    item.onclick = () => selectCase(rec.patient_uid);

    const badges = [];
    const isHoldout = rec.selection === "holdout";
    badges.push(`<span class="status-badge ${isHoldout ? "badge-holdout" : "badge-seeded"}">${rec.selection}</span>`);

    for (const [outId, outData] of Object.entries(rec.outcomes || {})) {
      if (outData.present) {
        const st = outData.grade_result?.status || "present";
        badges.push(`<span class="status-badge badge-${st}">${outId}:${st}</span>`);
      }
    }

    item.innerHTML = `
      <div class="case-row-header">
        <span class="case-row-uid">#${rec.patient_uid}</span>
      </div>
      <div class="case-row-title" title="${escapeHtml(rec.title || "")}">${escapeHtml(rec.title || "No Title")}</div>
      <div class="case-row-badges">${badges.join(" ")}</div>
    `;

    listContainer.appendChild(item);
  });
}

function selectCase(uid) {
  selectedPatientUid = uid;
  activeHighlightQuote = null;

  document.querySelectorAll(".case-row-card").forEach(el => {
    el.classList.remove("active");
  });
  
  const rec = currentRunData?.detailed_records?.find(r => r.patient_uid === uid);
  if (!rec) return;

  // Highlight active row in sidebar
  const currentCard = Array.from(document.querySelectorAll(".case-row-card")).find(el => el.textContent.includes(`#${uid}`));
  if (currentCard) currentCard.classList.add("active");

  document.getElementById("detail-uid").textContent = `Patient #${rec.patient_uid}`;
  document.getElementById("detail-title").textContent = rec.title || "Untitled Note";
  
  const ageDisplay = Array.isArray(rec.age) && rec.age[0] ? `${rec.age[0][0]} ${rec.age[0][1]}s` : (rec.age || "Unknown");
  document.getElementById("detail-meta").innerHTML = `
    <span><strong>Cohort:</strong> ${rec.selection || "N/A"}</span>
    <span><strong>Gender:</strong> ${rec.gender || "Unknown"}</span>
    <span><strong>Age:</strong> ${ageDisplay}</span>
    <span><strong>SCD Primary:</strong> ${rec.scd_primary ? "Yes" : "No"}</span>
  `;

  renderCaseOutcomes(rec);
  renderPatientNoteWithHighlights(rec);
}

function renderPatientNoteWithHighlights(rec) {
  const container = document.getElementById("note-text-display");
  if (!container) return;

  let rawText = rec.patient_note || "No clinical note text available.";
  const allQuotes = [];

  for (const [outId, outData] of Object.entries(rec.outcomes || {})) {
    if (outData.accepted_findings && Array.isArray(outData.accepted_findings)) {
      outData.accepted_findings.forEach(f => {
        if (f.quote && f.quote.trim().length > 1) {
          allQuotes.push({
            quote: f.quote.trim(),
            feature: f.feature,
            value: f.value,
            outcome: outId
          });
        }
      });
    }
  }

  if (allQuotes.length === 0) {
    container.textContent = rawText;
    return;
  }

  allQuotes.sort((a, b) => b.quote.length - a.quote.length);
  let highlighted = escapeHtml(rawText);

  allQuotes.forEach((q, idx) => {
    const escapedQuote = escapeHtml(q.quote);
    const regex = new RegExp(escapeRegExp(escapedQuote), "gi");
    const isActive = activeHighlightQuote === q.quote;
    const highlightClass = isActive ? "quote-highlight active-quote" : "quote-highlight";
    
    highlighted = highlighted.replace(regex, `<mark class="${highlightClass}" data-quote="${escapeHtml(q.quote)}" title="Feature: ${q.feature} = ${q.value}" id="quote-mark-${idx}">$&</mark>`);
  });

  container.innerHTML = highlighted;

  container.querySelectorAll(".quote-highlight").forEach(mark => {
    mark.addEventListener("click", () => {
      const qText = mark.getAttribute("data-quote");
      showToast(`Selected quote: "${qText}"`);
    });
  });
}

function renderCaseOutcomes(rec) {
  const container = document.getElementById("outcomes-eval-container");
  if (!container) return;

  container.innerHTML = "";
  const outcomes = rec.outcomes || {};
  const outcomeEntries = Object.entries(outcomes);

  if (outcomeEntries.length === 0) {
    container.innerHTML = `<div style="color: var(--text-dim); font-size: 0.8rem;">No evaluated outcomes found.</div>`;
    return;
  }

  outcomeEntries.forEach(([outId, outData]) => {
    const gr = outData.grade_result || {};
    const status = gr.status || (outData.present ? "present" : "absent");
    const outName = outData.outcome_name || OUTCOME_NAMES[outId] || `Outcome ${outId}`;

    const card = document.createElement("div");
    card.className = "outcome-box";

    let findingsTableHtml = "";
    if (outData.accepted_findings && outData.accepted_findings.length > 0) {
      findingsTableHtml = `
        <table class="findings-grid-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Value</th>
              <th>Verbatim Quote</th>
              <th>Unit</th>
            </tr>
          </thead>
          <tbody>
            ${outData.accepted_findings.map(f => `
              <tr class="finding-row">
                <td><code>${escapeHtml(f.feature || '')}</code></td>
                <td><strong>${escapeHtml(String(f.value ?? ''))}</strong></td>
                <td><span class="quote-pill" onclick="jumpToQuote('${escapeJsString(f.quote || '')}')">"${escapeHtml(f.quote || '')}"</span></td>
                <td>${escapeHtml(f.unit || '-')}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    } else {
      findingsTableHtml = `<div style="font-size: 0.75rem; color: var(--text-dim); margin: 0.35rem 0;">No accepted findings extracted for this outcome.</div>`;
    }

    const reasonHtml = gr.reason ? `<div style="font-size: 0.775rem; margin-top: 0.35rem; color: var(--text-muted);"><strong>Rule Rationale:</strong> ${escapeHtml(gr.reason)}</div>` : "";
    const gradeHtml = gr.grade ? `<div style="font-size: 0.8rem; margin-top: 0.15rem; color: var(--status-graded);"><strong>Computed Grade:</strong> Level ${gr.grade}</div>` : "";

    card.innerHTML = `
      <div class="outcome-box-header">
        <div>
          <span class="outcome-box-title">${outId}: ${escapeHtml(outName)}</span>
          <span style="font-size: 0.725rem; color: var(--text-dim); margin-left: 0.4rem;">Model Present: <strong>${outData.present ? "Yes" : "No"}</strong></span>
        </div>
        <div>
          <span class="status-badge badge-${status}">${status}</span>
        </div>
      </div>
      ${findingsTableHtml}
      ${gradeHtml}
      ${reasonHtml}
      <div style="margin-top: 0.4rem; display: flex; justify-content: flex-end;">
        <button class="btn btn-sm" onclick='showRawModal(${JSON.stringify(outData.raw_reply || "{}")})'>Raw JSON</button>
      </div>
    `;

    container.appendChild(card);
  });
}

window.jumpToQuote = function(quoteText) {
  if (!quoteText) return;
  activeHighlightQuote = quoteText;
  
  const rec = currentRunData?.detailed_records?.find(r => r.patient_uid === selectedPatientUid);
  if (rec) {
    renderPatientNoteWithHighlights(rec);
    setTimeout(() => {
      const activeMark = document.querySelector(".quote-highlight.active-quote");
      if (activeMark) {
        activeMark.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 50);
  }
};

window.showRawModal = function(rawReply) {
  const modal = document.getElementById("raw-modal");
  const codeEl = document.getElementById("raw-json-content");
  
  let formatted = rawReply;
  try {
    if (typeof rawReply === "string") {
      formatted = JSON.stringify(JSON.parse(rawReply), null, 2);
    } else {
      formatted = JSON.stringify(rawReply, null, 2);
    }
  } catch (e) {
    formatted = String(rawReply);
  }

  if (codeEl) codeEl.textContent = formatted;
  if (modal) modal.classList.add("active");
};

function renderAuditView() {
  const tableBody = document.getElementById("audit-table-body");
  if (!tableBody) return;

  tableBody.innerHTML = "";

  if (!currentAuditData || currentAuditData.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 1.5rem;">No audit records available.</td></tr>`;
    return;
  }

  currentAuditData.forEach((row, idx) => {
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td><strong>${escapeHtml(row.uid || "")}</strong><br><span style="font-size: 0.725rem; color: var(--text-dim);">${escapeHtml(row.selection || "")}</span></td>
      <td><strong>${escapeHtml(row.outcome || "")}</strong>: ${escapeHtml(row.outcome_name || "")}</td>
      <td><code>${escapeHtml(row.extracted_features || "")}</code></td>
      <td><span style="color: var(--status-refuted); font-size: 0.775rem;">${escapeHtml(row.rule_reason || row.look_for || "Refuted by rubric")}</span></td>
      <td><input type="checkbox" id="audit-truly-absent-${idx}" ${row.truly_absent === "True" || row.truly_absent === true ? "checked" : ""} onchange="updateAuditRecord(${idx}, 'truly_absent', this.checked)"></td>
      <td><input type="text" class="input-text" style="width: 100%; font-size: 0.775rem;" value="${escapeHtml(row.reviewer_note || "")}" placeholder="Reviewer comments..." onchange="updateAuditRecord(${idx}, 'reviewer_note', this.value)"></td>
      <td><button class="btn btn-sm" onclick="openCaseInExplorer('${row.uid}')">Inspect</button></td>
    `;

    tableBody.appendChild(tr);
  });
}

function deriveAuditDataFromRun(data) {
  const derived = [];
  if (!data.detailed_records) return;

  data.detailed_records.forEach(rec => {
    for (const [outId, outData] of Object.entries(rec.outcomes || {})) {
      const gr = outData.grade_result || {};
      if (gr.status === "refuted" || (outData.present && gr.present === false)) {
        derived.push({
          run_id: data.provenance?.run_id || "",
          uid: rec.patient_uid,
          selection: rec.selection,
          outcome: outId,
          outcome_name: outData.outcome_name || OUTCOME_NAMES[outId] || outId,
          model_said_present: "True",
          truly_absent: "",
          reviewer_note: "",
          look_for: gr.reason || "",
          title: rec.title || "",
          note_text: rec.patient_note || "",
          extracted_features: JSON.stringify(outData.accepted_findings?.reduce((acc, f) => ({ ...acc, [f.feature]: f.value }), {}) || {}),
          rule_reason: gr.reason || "Outcome conditions not met"
        });
      }
    }
  });

  currentAuditData = derived;
}

window.updateAuditRecord = function(idx, field, value) {
  if (currentAuditData[idx]) {
    currentAuditData[idx][field] = value;
  }
};

window.openCaseInExplorer = function(uid) {
  const tabExplorerBtn = document.querySelector('[data-tab="tab-explorer"]');
  if (tabExplorerBtn) tabExplorerBtn.click();
  selectCase(uid);
};

window.exportAuditCSV = function() {
  if (!currentAuditData || currentAuditData.length === 0) {
    alert("No audit data to export.");
    return;
  }

  const headers = ["run_id", "uid", "selection", "outcome", "outcome_name", "model_said_present", "truly_absent", "reviewer_note", "look_for", "title", "extracted_features", "rule_reason"];
  const csvRows = [headers.join(",")];

  currentAuditData.forEach(row => {
    const values = headers.map(h => {
      const val = row[h] ?? "";
      const escaped = String(val).replace(/"/g, '""');
      return `"${escaped}"`;
    });
    csvRows.push(values.join(","));
  });

  const csvBlob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(csvBlob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `refuted_audit_reviewed_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("Exported audit CSV");
};

function parseAuditCSV(csvText) {
  const lines = csvText.trim().split("\n");
  if (lines.length < 2) return;

  const parseCSVLine = (text) => {
    const result = [];
    let cur = "";
    let inQuote = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (ch === '"' && text[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') {
        inQuote = !inQuote;
      } else if (ch === ',' && !inQuote) {
        result.push(cur);
        cur = "";
      } else {
        cur += ch;
      }
    }
    result.push(cur);
    return result;
  };

  const headers = parseCSVLine(lines[0]).map(h => h.trim());
  const records = [];

  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const vals = parseCSVLine(lines[i]);
    const rec = {};
    headers.forEach((h, idx) => {
      rec[h] = vals[idx] ?? "";
    });
    records.push(rec);
  }

  currentAuditData = records;
  renderAuditView();
}

function showToast(msg) {
  const toast = document.getElementById("toast-bar");
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2500);
}

function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeRegExp(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function escapeJsString(str) {
  return String(str).replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\n/g, "\\n");
}
