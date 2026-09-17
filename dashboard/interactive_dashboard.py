"""SCOGS Interactive Clinical Evaluation & Extraction Dashboard.

Provides a unified Shiny Core dashboard operating in Dual Mode:
  - Mode 1: Pre-computed Run Explorer (case-by-case inspection of batch runs)
  - Mode 2: Live Note Evaluator (on-demand extraction and deterministic rule grading)

Designed with anti-slop clinical engineering principles:
  - Dual-mode Light / Dark theme system with CSS variables and instant toggle
  - High-trust scientific instrument aesthetic (Carbon / Primer clinical density)
  - Zero em-dashes throughout the interface
  - WCAG AA compliant contrast ratios and consistent border-radius tokens
"""
from __future__ import annotations

import glob
import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from htmltools import HTML
from shiny import App, reactive, render, ui

# `shiny run dashboard/interactive_dashboard.py` imports this file by path, so
# only `dashboard/` is on sys.path. Add `scripts/` (for `scogs.…` and
# `experiments.…`) and the repository root (for `dashboard.…`): the same two
# roots pyproject.toml gives the tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.evaluation import (
    FOCUS_OUTCOMES,
    extract_and_grade_note,
    grade_badge,
    is_derived,
    is_ollama_available,
)
from dashboard.view_state import explore_view_state, grade_details, live_view_state


# ==============================================================================
# 1. Backend Bridges
# ==============================================================================

def load_run_file(filepath: str | Path) -> dict[str, Any]:
    """Reads structured run JSON and formats records indexed by patient_uid."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Run file not found: {filepath}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    detailed = data.get("detailed_records", [])
    records_by_uid: dict[str, dict] = {}
    for r in detailed:
        if isinstance(r, dict) and "patient_uid" in r:
            records_by_uid[str(r["patient_uid"])] = r

    return {
        "provenance": data.get("provenance", {}),
        "profiling": data.get("profiling", {}),
        "automated_metrics": data.get("automated_metrics", {}),
        "grade_status": data.get("grade_status", {}),
        "grade_status_by_outcome": data.get("grade_status_by_outcome", {}),
        "grade_status_by_selection": data.get("grade_status_by_selection", {}),
        "features_extracted": data.get("features_extracted", {}),
        "records_by_uid": records_by_uid,
        "detailed_records": detailed,
    }


def load_csv_notes(filepath: str | Path = "data/clinical_notes.csv") -> pd.DataFrame:
    """Loads and standardizes clinical notes CSV into expected schema."""
    path = Path(filepath)
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "patient_id",
                "visit_datetime",
                "facility_type",
                "clinical_note",
                "note_text",
                "gender",
                "patient_sex",
                "patient_age",
            ]
        )

    df = pd.read_csv(path)

    # Note text mapping
    if "clinical_note" in df.columns:
        df["note_text"] = df["clinical_note"].fillna("")
    elif "note_text" in df.columns:
        df["clinical_note"] = df["note_text"].fillna("")
    else:
        df["note_text"] = ""
        df["clinical_note"] = ""

    # Gender mapping
    if "gender" in df.columns:
        df["patient_sex"] = df["gender"].map({"Female": "female", "Male": "male"}).fillna("unknown")
    else:
        df["patient_sex"] = "unknown"

    # Age calculation from DOB and visit datetime (handling 2-digit century rollover)
    if "visit_datetime" in df.columns and "dob" in df.columns:
        try:
            visit_dt = pd.to_datetime(df["visit_datetime"], format="mixed", errors="coerce")
            dob_dt = pd.to_datetime(df["dob"], format="mixed", errors="coerce")
            future_dob = dob_dt > visit_dt
            dob_dt.loc[future_dob] = dob_dt.loc[future_dob] - pd.DateOffset(years=100)
            df["patient_age"] = (visit_dt - dob_dt).dt.days / 365.25
        except Exception:
            df["patient_age"] = None
    else:
        df["patient_age"] = None

    return df


def find_quote_spans(note_text: str, quotes: list[str]) -> list[tuple[int, int, str]]:
    """Finds non-overlapping match intervals for quotes in note_text, prioritizing longer quotes."""
    if not note_text:
        return []

    clean_quotes = sorted(
        {q.strip() for q in quotes if isinstance(q, str) and q.strip()},
        key=len,
        reverse=True,
    )
    lower_text = note_text.lower()
    occupied = [False] * len(note_text)
    spans: list[tuple[int, int, str]] = []

    for q in clean_quotes:
        q_lower = q.lower()
        q_len = len(q)
        start = 0
        while True:
            idx = lower_text.find(q_lower, start)
            if idx == -1:
                break
            end = idx + q_len
            if not any(occupied[idx:end]):
                spans.append((idx, end, q))
                for i in range(idx, end):
                    occupied[i] = True
            start = idx + 1

    spans.sort(key=lambda x: x[0])
    return spans


def highlight_note_quotes(note_text: str, findings: list[dict[str, Any]]) -> str:
    """Wraps matched verified quote substrings in <mark> tags with tooltip metadata."""
    if not note_text:
        return "<p class='text-muted'><em>No clinical note text available.</em></p>"

    quotes = [f.get("quote") for f in findings if isinstance(f, dict) and f.get("quote")]
    spans = find_quote_spans(note_text, quotes)

    if not spans:
        return (
            f"<div class='note-text-body' style='white-space: pre-wrap; font-family: ui-monospace, "
            f"\"SF Mono\", Menlo, monospace; font-size: 0.84rem; line-height: 1.65; "
            f"padding: 18px; border-radius: 6px; max-height: 480px; overflow-y: auto;'>"
            f"{html.escape(note_text)}</div>"
        )

    chunks: list[str] = []
    last_idx = 0
    for start, end, quote in spans:
        if start > last_idx:
            chunks.append(html.escape(note_text[last_idx:start]))
        matched_text = html.escape(note_text[start:end])
        safe_title = html.escape(f"Verified Quote: {quote}")
        chunks.append(
            f'<mark class="quote-highlight" title="{safe_title}">{matched_text}</mark>'
        )
        last_idx = end

    if last_idx < len(note_text):
        chunks.append(html.escape(note_text[last_idx:]))

    highlighted_body = "".join(chunks)
    return (
        f"<div class='note-text-body' style='white-space: pre-wrap; font-family: ui-monospace, "
        f"\"SF Mono\", Menlo, monospace; font-size: 0.84rem; line-height: 1.65; "
        f"padding: 18px; border-radius: 6px; max-height: 480px; overflow-y: auto;'>"
        f"{highlighted_body}</div>"
    )





# ==============================================================================
# 2. UI Components & Dual-Mode Styling
# ==============================================================================

def get_available_run_files() -> list[tuple[str, str]]:
    """Discovers available run JSON files or defaults to bundled test fixtures."""
    files: list[tuple[str, str]] = []
    for p in sorted(glob.glob("results/*.json")):
        files.append((p, f"results/{Path(p).name}"))
    for p in sorted(glob.glob("tests/fixtures/*.json")):
        files.append((p, f"fixture: {Path(p).name}"))
    if not files:
        files.append(("tests/fixtures/sample_run.json", "tests/fixtures/sample_run.json (Sample)"))
    return files


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.span("WORKSPACE CONTROL", class_="sidebar-section-label"),
            ui.input_radio_buttons(
                "app_mode",
                label="",
                choices={
                    "explore": "Explore Saved Runs",
                    "live": "Evaluate Live Note",
                },
                selected="explore",
            ),
            class_="mb-3",
        ),
        ui.hr(class_="sidebar-divider"),
        ui.output_ui("sidebar_controls"),
        width=340,
        class_="clinical-sidebar",
    ),
    ui.tags.head(
        ui.tags.style(
            """
            /* Clinical Design Tokens - Light Theme (Default) */
            :root, [data-bs-theme="light"] {
                --scogs-canvas: #f8fafc;
                --scogs-surface: #ffffff;
                --scogs-surface-subtle: #f1f5f9;
                --scogs-border: #e2e8f0;
                --scogs-border-subtle: #f1f5f9;
                --scogs-text: #0f172a;
                --scogs-text-muted: #64748b;
                --scogs-text-secondary: #475569;
                --scogs-primary: #0284c7;
                --scogs-primary-hover: #0369a1;
                --scogs-code-bg: #f8fafc;
                --scogs-code-text: #0f172a;
                --scogs-note-bg: #f8fafc;
                --scogs-note-text: #334155;
                --scogs-highlight-bg: #fef3c7;
                --scogs-highlight-text: #92400e;
                --scogs-highlight-border: #f59e0b;
                --radius-sm: 4px;
                --radius-md: 6px;
                --radius-lg: 8px;
            }

            /* Clinical Design Tokens - Dark Theme */
            [data-bs-theme="dark"] {
                --scogs-canvas: #090d16;
                --scogs-surface: #111827;
                --scogs-surface-subtle: #1f293d;
                --scogs-border: #243048;
                --scogs-border-subtle: #1a233a;
                --scogs-text: #f8fafc;
                --scogs-text-muted: #94a3b8;
                --scogs-text-secondary: #cbd5e1;
                --scogs-primary: #38bdf8;
                --scogs-primary-hover: #0ea5e9;
                --scogs-code-bg: #090d16;
                --scogs-code-text: #38bdf8;
                --scogs-note-bg: #090d16;
                --scogs-note-text: #cbd5e1;
                --scogs-highlight-bg: #78350f;
                --scogs-highlight-text: #fef3c7;
                --scogs-highlight-border: #f59e0b;
                --radius-sm: 4px;
                --radius-md: 6px;
                --radius-lg: 8px;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", "Helvetica Neue", sans-serif;
                background-color: var(--scogs-canvas);
                color: var(--scogs-text);
                -webkit-font-smoothing: antialiased;
                transition: background-color 0.2s ease, color 0.2s ease;
            }

            /* Header Title */
            .app-title-main {
                font-size: 1.35rem;
                font-weight: 700;
                color: var(--scogs-text);
                letter-spacing: -0.025em;
            }
            .app-title-sub {
                font-size: 0.85rem;
                color: var(--scogs-text-muted);
                letter-spacing: -0.01em;
            }

            /* Brand Badge */
            .badge-brand {
                background-color: var(--scogs-primary);
                color: #ffffff;
                font-weight: 800;
                font-size: 0.74rem;
                letter-spacing: 0.08em;
                padding: 4px 8px;
                border-radius: var(--radius-sm);
                text-transform: uppercase;
            }
            [data-bs-theme="dark"] .badge-brand {
                color: #090d16;
            }

            .badge-engine {
                background: var(--scogs-surface-subtle);
                border: 1px solid var(--scogs-border);
                color: var(--scogs-text-secondary);
                font-weight: 600;
                font-size: 0.76rem;
                padding: 4px 11px;
                border-radius: 9999px;
                display: inline-flex;
                align-items: center;
                gap: 5px;
            }

            /* Theme Switcher */
            .theme-toggle-container {
                display: inline-flex;
                background: var(--scogs-surface-subtle);
                border: 1px solid var(--scogs-border);
                border-radius: var(--radius-sm);
                padding: 2px;
                gap: 2px;
            }
            .theme-btn {
                border: none;
                background: transparent;
                color: var(--scogs-text-muted);
                font-size: 0.74rem;
                font-weight: 600;
                padding: 3px 8px;
                border-radius: 3px;
                cursor: pointer;
                transition: background 0.15s ease, color 0.15s ease;
            }
            .theme-btn:hover {
                color: var(--scogs-text);
            }
            .theme-btn.active {
                background: var(--scogs-surface);
                color: var(--scogs-text);
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
            }

            /* Sidebar Styling */
            .clinical-sidebar {
                background: var(--scogs-surface) !important;
                border-right: 1px solid var(--scogs-border) !important;
            }
            .sidebar-section-label {
                font-size: 0.72rem;
                font-weight: 700;
                color: var(--scogs-text-muted);
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 8px;
                display: block;
            }
            .sidebar-divider {
                border-top: 1px solid var(--scogs-border);
                margin: 14px 0;
            }

            /* Cards */
            .card {
                background: var(--scogs-surface) !important;
                border: 1px solid var(--scogs-border) !important;
                border-radius: var(--radius-lg) !important;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            }
            .card-header {
                background: var(--scogs-surface) !important;
                border-bottom: 1px solid var(--scogs-border) !important;
                padding: 12px 18px;
            }
            .card-header-title {
                font-size: 0.95rem;
                font-weight: 600;
                color: var(--scogs-text);
                letter-spacing: -0.01em;
            }

            /* Metric Boxes */
            .metric-box {
                border-radius: var(--radius-md);
                padding: 14px 16px;
                background: var(--scogs-surface);
                border: 1px solid var(--scogs-border);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                height: 100%;
            }
            .metric-val-tabular {
                font-variant-numeric: tabular-nums;
                letter-spacing: -0.02em;
            }

            /* Severity & Status Badges (Light and Dark compliant) */
            .badge-grade-pill {
                font-size: 1.05rem;
                font-weight: 700;
                padding: 7px 18px;
                border-radius: 9999px;
                display: inline-flex;
                align-items: center;
                letter-spacing: 0.02em;
                border: 1px solid transparent;
            }

            .badge-grade-1 { background-color: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
            .badge-grade-2 { background-color: #fffbeb; color: #92400e; border-color: #fde68a; }
            .badge-grade-3, .badge-grade-4, .badge-grade-5 { background-color: #fef2f2; color: #991b1b; border-color: #fecaca; }
            .badge-grade-set { background-color: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
            .badge-cannot-grade { background-color: #fff7ed; color: #9a3412; border-color: #fed7aa; }
            .badge-not-applicable { background-color: #f1f5f9; color: #475569; border-color: #cbd5e1; }
            .badge-absent { background-color: #f8fafc; color: #64748b; border-color: #e2e8f0; }
            .badge-refuted { background-color: #faf5ff; color: #6b21a8; border-color: #e9d5ff; }

            [data-bs-theme="dark"] .badge-grade-1 { background-color: rgba(6, 78, 59, 0.45); color: #6ee7b7; border-color: #059669; }
            [data-bs-theme="dark"] .badge-grade-2 { background-color: rgba(120, 53, 15, 0.45); color: #fcd34d; border-color: #d97706; }
            [data-bs-theme="dark"] .badge-grade-3, [data-bs-theme="dark"] .badge-grade-4, [data-bs-theme="dark"] .badge-grade-5 { background-color: rgba(127, 29, 29, 0.45); color: #fca5a5; border-color: #dc2626; }
            [data-bs-theme="dark"] .badge-grade-set { background-color: rgba(30, 58, 138, 0.45); color: #93c5fd; border-color: #2563eb; }
            [data-bs-theme="dark"] .badge-cannot-grade { background-color: rgba(124, 45, 18, 0.45); color: #fdba74; border-color: #ea580c; }
            [data-bs-theme="dark"] .badge-not-applicable { background-color: rgba(30, 41, 59, 0.5); color: #94a3b8; border-color: #334155; }
            [data-bs-theme="dark"] .badge-absent { background-color: rgba(15, 23, 42, 0.5); color: #94a3b8; border-color: #334155; }
            [data-bs-theme="dark"] .badge-refuted { background-color: rgba(88, 28, 135, 0.45); color: #d8b4fe; border-color: #9333ea; }

            /* Clean Findings Table */
            .findings-table th {
                background: var(--scogs-surface-subtle);
                color: var(--scogs-text-muted);
                font-weight: 600;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                border-bottom: 1px solid var(--scogs-border);
                padding: 10px 14px;
            }
            .findings-table td {
                font-size: 0.84rem;
                vertical-align: middle;
                border-bottom: 1px solid var(--scogs-border-subtle);
                padding: 9px 14px;
                color: var(--scogs-text);
            }
            .findings-table tr:hover td {
                background-color: var(--scogs-surface-subtle);
            }

            /* Micro Status Badges */
            .badge-status {
                font-size: 0.76rem;
                font-weight: 600;
                padding: 3px 9px;
                border-radius: 9999px;
                display: inline-flex;
                align-items: center;
                gap: 4px;
            }
            .badge-status-verified { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
            .badge-status-unfound { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
            .badge-status-derived { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }

            [data-bs-theme="dark"] .badge-status-verified { background: rgba(6, 78, 59, 0.35); color: #6ee7b7; border: 1px solid #059669; }
            [data-bs-theme="dark"] .badge-status-unfound { background: rgba(127, 29, 29, 0.35); color: #fca5a5; border: 1px solid #dc2626; }
            [data-bs-theme="dark"] .badge-status-derived { background: rgba(120, 53, 15, 0.35); color: #fcd34d; border: 1px solid #d97706; }

            /* DSL Predicate Code Box */
            .dsl-code-block {
                font-family: ui-monospace, "SF Mono", Menlo, monospace;
                font-size: 0.82rem;
                color: var(--scogs-code-text);
                background: var(--scogs-code-bg);
                border: 1px solid var(--scogs-border);
                border-radius: var(--radius-sm);
                padding: 6px 10px;
                display: inline-block;
                max-width: 100%;
                overflow-x: auto;
            }

            /* Clinical Note Inspector Body */
            .note-text-body {
                background: var(--scogs-note-bg);
                color: var(--scogs-note-text);
                border: 1px solid var(--scogs-border);
            }
            .quote-highlight {
                background-color: var(--scogs-highlight-bg);
                color: var(--scogs-highlight-text);
                border-bottom: 2px solid var(--scogs-highlight-border);
                padding: 1px 4px;
                border-radius: 3px;
                cursor: help;
                font-weight: 500;
            }

            /* Form Elements in Dark Mode */
            [data-bs-theme="dark"] .form-control,
            [data-bs-theme="dark"] .form-select {
                background-color: #0b0f19;
                color: #f8fafc;
                border-color: #243048;
            }
            [data-bs-theme="dark"] .form-control:focus,
            [data-bs-theme="dark"] .form-select:focus {
                border-color: var(--scogs-primary);
                box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25);
            }

            /* Primary Action Button */
            .btn-clinical-primary {
                background-color: var(--scogs-primary);
                border-color: var(--scogs-primary);
                color: #ffffff;
                font-weight: 600;
                font-size: 0.88rem;
                border-radius: var(--radius-md);
                padding: 9px 14px;
                transition: background-color 0.15s ease, transform 0.05s ease;
            }
            .btn-clinical-primary:hover {
                background-color: var(--scogs-primary-hover);
                border-color: var(--scogs-primary-hover);
                color: #ffffff;
            }
            .btn-clinical-primary:active {
                transform: translateY(1px);
            }
            [data-bs-theme="dark"] .btn-clinical-primary {
                color: #090d16;
            }

            /* Custom Scrollbar */
            .note-text-body::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            .note-text-body::-webkit-scrollbar-thumb {
                background: #94a3b8;
                border-radius: 3px;
            }
            """
        ),
        ui.tags.script(
            """
            function applyTheme(theme) {
                let effective = theme;
                if (theme === 'auto') {
                    effective = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
                }
                document.documentElement.setAttribute('data-bs-theme', effective);
                document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
                const btn = document.getElementById('theme-btn-' + theme);
                if (btn) btn.classList.add('active');
            }

            function setTheme(theme) {
                localStorage.setItem('scogs-theme', theme);
                applyTheme(theme);
            }

            document.addEventListener('DOMContentLoaded', function() {
                const saved = localStorage.getItem('scogs-theme') || 'auto';
                applyTheme(saved);
            });

            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
                if ((localStorage.getItem('scogs-theme') || 'auto') === 'auto') {
                    applyTheme('auto');
                }
            });
            """
        ),
    ),
    ui.div(
        ui.div(
            ui.div(
                ui.div(
                    ui.span("SCOGS", class_="badge-brand me-2"),
                    ui.span("Clinical Evaluation & Extraction Dashboard", class_="app-title-main"),
                    class_="d-flex align-items-center flex-wrap gap-1",
                ),
                ui.p(
                    "Deterministic Sickle Cell Severity Grading & MedGemma Grounding Verification",
                    class_="app-title-sub mb-0 mt-1",
                ),
                class_="d-flex flex-column",
            ),
            ui.div(
                ui.span("53 SCOGS Outcome Tables Active", class_="badge-engine me-2"),
                ui.output_ui("header_status_badge"),
                ui.div(
                    ui.tags.button("Light", type="button", class_="theme-btn", id="theme-btn-light", onclick="setTheme('light')", title="Light Theme"),
                    ui.tags.button("Dark", type="button", class_="theme-btn", id="theme-btn-dark", onclick="setTheme('dark')", title="Dark Theme"),
                    ui.tags.button("Auto", type="button", class_="theme-btn active", id="theme-btn-auto", onclick="setTheme('auto')", title="System Theme"),
                    class_="theme-toggle-container ms-2",
                ),
                class_="d-flex align-items-center flex-wrap gap-2 mt-2 mt-md-0",
            ),
            class_="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center mb-3 pb-3 border-bottom",
        ),
        ui.output_ui("executive_grade_card"),
        ui.div(class_="my-3"),
        ui.layout_columns(
            ui.card(
                ui.card_header(ui.span("Clinical Findings & Grounding Verification", class_="card-header-title")),
                ui.output_ui("findings_table_ui"),
            ),
            ui.card(
                ui.card_header(ui.span("Clinical Note Context & Verified Spans", class_="card-header-title")),
                ui.output_ui("note_inspector_ui"),
            ),
            col_widths=[6, 6],
        ),
        ui.div(class_="my-3"),
        ui.output_ui("profiling_card"),
        class_="container-fluid py-2",
    ),
    title="SCOGS Clinical Dashboard",
)


# ==============================================================================
# 3. Server Logic
# ==============================================================================

def server(input, output, session):
    # Reactive state
    current_run_cache = reactive.value(None)
    csv_data_cache = reactive.value(None)
    live_eval_result = reactive.value(None)

    # Initial loading of CSV dataset
    @reactive.effect
    def _load_csv():
        csv_df = load_csv_notes("data/clinical_notes.csv")
        csv_data_cache.set(csv_df)

    @output
    @render.ui
    def header_status_badge():
        ollama_up = is_ollama_available()
        if ollama_up:
            return ui.span("Ollama Connected (MedGemma 27B)", class_="badge bg-success", style="font-size: 0.76rem;")
        return ui.span("Ollama Offline (Deterministic Mode)", class_="badge bg-secondary", style="font-size: 0.76rem;")

    # Dynamic Sidebar Controls
    @output
    @render.ui
    def sidebar_controls():
        mode = input.app_mode()
        if mode == "explore":
            run_files = get_available_run_files()
            file_choices = {path: label for path, label in run_files}
            default_file = run_files[0][0] if run_files else "tests/fixtures/sample_run.json"

            return ui.div(
                ui.span("RUN REPOSITORY", class_="sidebar-section-label"),
                ui.input_select("selected_run_file", "Run JSON Output:", choices=file_choices, selected=default_file),
                ui.output_ui("run_case_selector_ui"),
                ui.output_ui("run_outcome_selector_ui"),
            )
        else:
            # Mode 2: Live Note Evaluator
            ollama_up = is_ollama_available()
            badge = (
                ui.span("Ollama Online (MedGemma 27B)", class_="badge bg-success mb-2")
                if ollama_up
                else ui.span("Ollama Offline (Deterministic Mode)", class_="badge bg-secondary mb-2")
            )

            df = csv_data_cache.get()
            csv_choices = {}
            if df is not None and not df.empty:
                for idx, row in df.head(50).iterrows():
                    pid = str(row.get("patient_id", f"Row {idx}"))
                    vdate = str(row.get("visit_datetime", ""))
                    ftype = str(row.get("facility_type", ""))
                    csv_choices[str(idx)] = f"{pid} - {vdate} ({ftype})"

            outcome_choices = {k: f"#{k} {v['name']}" for k, v in FOCUS_OUTCOMES.items()}

            return ui.div(
                ui.span("NOTE SOURCE & INFERENCE", class_="sidebar-section-label"),
                badge,
                ui.input_radio_buttons(
                    "live_source",
                    "Input Source:",
                    choices={"csv": "From CSV Dataset", "custom": "Paste Custom Note"},
                    selected="csv" if csv_choices else "custom",
                ),
                ui.panel_conditional(
                    "input.live_source === 'csv'",
                    ui.input_select(
                        "csv_patient_idx",
                        "Select Patient Encounter:",
                        choices=csv_choices,
                        selected=next(iter(csv_choices.keys())) if csv_choices else None,
                    ),
                ),
                ui.input_select(
                    "live_outcome",
                    "Target Focus Outcome:",
                    choices=outcome_choices,
                    selected="28",
                ),
                ui.layout_columns(
                    ui.input_numeric("patient_age_input", "Age (years):", value=25.0, min=0.0, max=120.0, step=0.5),
                    ui.input_select("patient_sex_input", "Sex:", choices={"male": "Male", "female": "Female", "unknown": "Unknown"}, selected="male"),
                    col_widths=[6, 6],
                ),
                ui.input_checkbox("outcome_present_input", "Outcome explicitly present in note", value=True),
                ui.input_text_area(
                    "live_note_text",
                    "Clinical Narrative:",
                    rows=7,
                    placeholder="Enter or review clinical note text...",
                ),
                ui.input_text_area(
                    "manual_features_json",
                    "Feature Values (JSON format for grading):",
                    rows=4,
                    value='{"care_setting": "inpatient", "pain_co_complication": false, "death_attributed": false, "life_support": false}',
                ),
                ui.input_action_button("btn_analyze", "Analyze & Grade Note", class_="btn btn-clinical-primary w-100 mt-2"),
            )

    # Sync CSV fields when patient is selected
    @reactive.effect
    @reactive.event(input.csv_patient_idx, input.live_source)
    def _sync_csv_selection():
        if input.app_mode() != "live" or input.live_source() != "csv":
            return
        df = csv_data_cache.get()
        if df is None or df.empty or not input.csv_patient_idx():
            return
        try:
            row_idx = int(input.csv_patient_idx())
            row = df.iloc[row_idx]
            ui.update_text_area("live_note_text", value=str(row.get("clinical_note", "")))
            sex_val = str(row.get("patient_sex", "unknown"))
            if sex_val in ("male", "female", "unknown"):
                ui.update_select("patient_sex_input", selected=sex_val)
            age_val = row.get("patient_age")
            if pd.notna(age_val):
                ui.update_numeric("patient_age_input", value=round(float(age_val), 1))
        except Exception:
            pass

    # Load Run File Reactively
    @reactive.effect
    def _load_selected_run():
        if input.app_mode() != "explore":
            return
        filepath = input.selected_run_file()
        if not filepath:
            return
        try:
            data = load_run_file(filepath)
            current_run_cache.set(data)
        except Exception as e:
            ui.notification_show(f"Failed to load run file: {e}", type="error")

    # Mode 1 Case Selector UI
    @output
    @render.ui
    def run_case_selector_ui():
        data = current_run_cache.get()
        if not data or not data.get("records_by_uid"):
            return ui.p("No records found in this run file.", class_="text-muted")

        records = data["records_by_uid"]
        choices = {}
        for uid, rec in records.items():
            title = rec.get("title") or "Case Report"
            choices[uid] = f"{uid} - {title[:40]}..."

        first_uid = next(iter(choices.keys())) if choices else None
        return ui.input_select("selected_patient_uid", "Select Patient Case:", choices=choices, selected=first_uid)

    # Mode 1 Outcome Selector UI
    @output
    @render.ui
    def run_outcome_selector_ui():
        data = current_run_cache.get()
        uid = input.selected_patient_uid()
        if not data or not uid or uid not in data["records_by_uid"]:
            return ui.div()

        rec = data["records_by_uid"][uid]
        outcomes = rec.get("outcomes", {})
        choices = {}
        for k, o_data in outcomes.items():
            name = o_data.get("outcome_name") or FOCUS_OUTCOMES.get(k, {}).get("name") or f"Outcome {k}"
            choices[k] = f"#{k} {name}"

        if not choices:
            choices = {k: f"#{k} {v['name']}" for k, v in FOCUS_OUTCOMES.items()}

        first_k = next(iter(choices.keys())) if choices else None
        return ui.input_select("selected_outcome_num", "Select Outcome:", choices=choices, selected=first_k)

    # Trigger Live Note Evaluation
    @reactive.effect
    @reactive.event(input.btn_analyze)
    def _perform_live_eval():
        note_text = input.live_note_text() or ""
        outcome_id = input.live_outcome() or "28"
        patient_sex = input.patient_sex_input() or "unknown"
        patient_age = input.patient_age_input()
        present = bool(input.outcome_present_input())

        manual_dict: dict[str, Any] = {}
        raw_manual = input.manual_features_json()
        if raw_manual and raw_manual.strip():
            try:
                manual_dict = json.loads(raw_manual.strip())
            except Exception as e:
                ui.notification_show(f"JSON Parse Error in manual features: {e}", type="warning")

        manual_dict["present"] = present

        res = extract_and_grade_note(
            note_text=note_text,
            outcomes=[outcome_id],
            patient_context={"patient_sex": patient_sex, "patient_age": patient_age},
            use_ollama=is_ollama_available(),
            manual_features={outcome_id: manual_dict},
        )
        live_eval_result.set(res)
        ui.notification_show("Analysis and grading complete", type="message")

    # Helper: resolve active state data
    def get_current_view_state() -> dict[str, Any]:
        if input.app_mode() == "explore":
            run_data = current_run_cache.get()
            if not run_data:
                # Before a file loads, the case selectors do not exist yet; reading
                # them would cancel the render.
                return {}
            return explore_view_state(run_data, input.selected_patient_uid(),
                                      input.selected_outcome_num())
        return live_view_state(live_eval_result.get(), input.live_outcome() or "28",
                               input.live_note_text() or "", input.patient_age_input(),
                               input.patient_sex_input())

    # ==========================================================================
    # 4. Renderers: Cards, Tables, Inspector
    # ==========================================================================

    @output
    @render.ui
    def executive_grade_card():
        state = get_current_view_state()
        if not state:
            return ui.card(ui.p("Select a run file and case in the sidebar to begin inspection.", class_="text-muted p-3"))

        # The harness status (experiments/grading.py) decides the badge. The card
        # never re-derives it from `present` or the features.
        status = state.get("status") or "pending"
        details = grade_details(state.get("grade_result"))
        grade_val, grades = details["grade"], details["grades"]
        matched, reason = details["matched"], details["reason"]
        missing, undecided = details["missing"], details["undecided"]
        needs_review = details["needs_review"]
        badge_class, badge_label = grade_badge(status, grade_val, grades)

        review_alert = None
        if needs_review:
            review_alert = ui.div(
                ui.span("Review Advisory: ", class_="fw-bold"),
                ui.span(
                    f"Ambiguous features or missing data ({', '.join(missing) if missing else 'Review criteria'})."
                ),
                class_="alert alert-warning py-2 px-3 mb-3 rounded-2 border-warning",
                style="font-size: 0.88rem;",
            )

        details_row = []
        if matched:
            details_row.append(
                ui.div(
                    ui.span("Matched Rule Predicate: ", class_="fw-semibold text-secondary fs-7"),
                    ui.div(ui.code(str(matched), class_="dsl-code-block")),
                )
            )
        if reason:
            details_row.append(
                ui.div(
                    ui.span("Decision Rationale: ", class_="fw-semibold text-secondary fs-7"),
                    ui.span(str(reason), class_="fs-7"),
                )
            )
        if missing:
            details_row.append(
                ui.div(
                    ui.span("Missing Required Features: ", class_="fw-semibold text-danger fs-7"),
                    ui.span(", ".join(str(m) for m in missing), class_="text-danger fs-7"),
                )
            )
        if undecided:
            details_row.append(
                ui.div(
                    ui.span("Undecided Clauses: ", class_="fw-semibold text-muted fs-7"),
                    ui.span(str(undecided), class_="text-muted fs-7"),
                )
            )

        meta = FOCUS_OUTCOMES.get(str(state.get("outcome_num")), {})
        meta_info = f"Organ System: {meta.get('organ_system', 'General')} | Acuity: {meta.get('acuity', 'Standard')}"

        return ui.card(
            ui.card_header(
                ui.div(
                    ui.span(f"Outcome #{state.get('outcome_num')}: {state.get('outcome_name')}", class_="card-header-title"),
                    ui.span(meta_info, class_="badge bg-light text-secondary border fw-normal", style="font-size: 0.78rem;"),
                    class_="d-flex justify-content-between align-items-center flex-wrap gap-2",
                )
            ),
            ui.div(
                review_alert,
                ui.div(
                    ui.div(
                        ui.span("SCOGS Deterministic Grade:", class_="text-muted text-uppercase fw-semibold fs-7 mb-1"),
                        ui.div(
                            ui.span(badge_label, class_=f"badge-grade-pill shadow-xs {badge_class}"),
                            class_="mb-2",
                        ),
                        class_="d-flex flex-column",
                    ),
                    ui.div(
                        *details_row,
                        class_="d-flex flex-column gap-2 ms-md-4 flex-grow-1",
                    ),
                    class_="d-flex flex-column flex-md-row align-items-start align-items-md-center justify-content-between gap-3",
                ),
                class_="p-3",
            ),
        )

    @output
    @render.ui
    def findings_table_ui():
        state = get_current_view_state()
        if not state:
            return ui.p("No case selected.", class_="text-muted p-3")

        findings = state.get("accepted_findings", [])

        if not findings:
            return ui.p("No extracted findings or features recorded for this outcome.", class_="text-muted p-3")

        rows = []
        for item in findings:
            feat = item.get("feature", "N/A")
            val = item.get("value", "N/A")
            unit = item.get("unit") or "N/A"
            quote = item.get("quote")

            # Every finding in a results file or a live result already passed quote
            # grounding in experiments/verification.py. Re-checking it here with a
            # looser test is how this table and the grade used to disagree.
            if is_derived(feat):
                status_icon = ui.span("Derived", class_="badge-status badge-status-derived")
            elif item.get("source") == "manual":
                status_icon = ui.span("Manual entry", class_="badge-status badge-status-derived")
            else:
                status_icon = ui.span("Verified", class_="badge-status badge-status-verified")

            quote_display = (
                ui.span(f'"{quote}"', class_="fst-italic text-secondary", style="font-size: 0.82rem;")
                if quote
                else ui.span("N/A", class_="text-muted")
            )

            rows.append(
                ui.tags.tr(
                    ui.tags.td(ui.code(str(feat), class_="text-primary fw-medium")),
                    ui.tags.td(ui.span(str(val), class_="fw-bold metric-val-tabular")),
                    ui.tags.td(ui.span(str(unit), class_="text-muted fs-7")),
                    ui.tags.td(quote_display),
                    ui.tags.td(status_icon),
                )
            )

        return ui.div(
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Feature Name"),
                        ui.tags.th("Extracted Value"),
                        ui.tags.th("Unit"),
                        ui.tags.th("Verbatim Quote Grounding"),
                        ui.tags.th("Verification"),
                    )
                ),
                ui.tags.tbody(*rows),
                class_="table table-sm findings-table mb-0",
            ),
            class_="table-responsive p-0",
        )

    @output
    @render.ui
    def note_inspector_ui():
        state = get_current_view_state()
        if not state or not state.get("note_text"):
            return ui.p("No note content to display.", class_="text-muted p-3")

        note_html = highlight_note_quotes(state["note_text"], state.get("accepted_findings", []))
        return ui.div(
            ui.div(
                ui.span(f"Patient ID: {state.get('patient_uid')}", class_="badge bg-light text-dark border me-2"),
                ui.span(f"Sex: {state.get('patient_sex') or 'unknown'}", class_="badge bg-light text-dark border me-2"),
                ui.span(
                    f"Age: {state.get('patient_age'):.1f} yrs" if state.get("patient_age") is not None else "Age: N/A",
                    class_="badge bg-light text-dark border",
                ),
                class_="mb-2 d-flex align-items-center flex-wrap gap-1",
            ),
            HTML(note_html),
            class_="p-3",
        )

    @output
    @render.ui
    def profiling_card():
        # Read for its reactive dependencies only: re-render when the selected case changes.
        _ = get_current_view_state()
        mode = input.app_mode()

        if mode == "explore":
            run_data = current_run_cache.get()
            if not run_data:
                return ui.div()

            prov = run_data.get("provenance", {})
            prof = run_data.get("profiling", {})
            auto = run_data.get("automated_metrics", {})
            statuses = run_data.get("grade_status", {})

            return ui.card(
                ui.card_header(ui.span("Model Profiling & Provenance Telemetry", class_="card-header-title")),
                ui.div(
                    ui.layout_columns(
                        ui.div(
                            ui.span("Model Architecture", class_="sidebar-section-label"),
                            ui.h5(prov.get("weights", "MedGemma 27B"), class_="fw-bold mb-0 text-dark"),
                            ui.span(f"Prompt Stage: {prov.get('prompt_stage', 'N/A')} ({prov.get('quant', 'F16')})", class_="fs-7 text-secondary mt-1"),
                            class_="metric-box",
                        ),
                        ui.div(
                            ui.span("Generation Speed", class_="sidebar-section-label"),
                            ui.h5(f"{prof.get('completion_tokens_per_sec', 0.0)} tok/s", class_="fw-bold mb-0 text-primary metric-val-tabular"),
                            ui.span(f"Latency: {prof.get('sec_per_note', 0.0)}s / note", class_="fs-7 text-secondary mt-1 metric-val-tabular"),
                            class_="metric-box",
                        ),
                        ui.div(
                            ui.span("Grounding Accuracy", class_="sidebar-section-label"),
                            ui.h5(f"{auto.get('quote_verified_pct', 0.0):.1f}% Verified", class_="fw-bold mb-0 text-success metric-val-tabular"),
                            ui.span(f"Hallucination: {auto.get('hallucinated_quote_pct', 0.0):.1f}%", class_="fs-7 text-danger mt-1 metric-val-tabular"),
                            class_="metric-box",
                        ),
                        ui.div(
                            ui.span("Grade Distribution", class_="sidebar-section-label"),
                            ui.div(
                                ui.span(f"Graded: {statuses.get('graded', 0)}", class_="badge bg-success me-1"),
                                ui.span(f"Absent: {statuses.get('absent', 0)}", class_="badge bg-secondary me-1"),
                                ui.span(f"Cannot: {statuses.get('cannot_grade', 0)}", class_="badge bg-warning text-dark me-1"),
                                ui.span(f"Refuted: {statuses.get('refuted', 0)}", class_="badge bg-purple text-white"),
                                class_="d-flex flex-wrap gap-1 mt-1",
                            ),
                            class_="metric-box",
                        ),
                        col_widths=[3, 3, 3, 3],
                    ),
                    class_="p-3",
                ),
            )
        else:
            return ui.card(
                ui.card_header(ui.span("Live Execution Telemetry", class_="card-header-title")),
                ui.div(
                    ui.layout_columns(
                        ui.div(
                            ui.span("Execution Target", class_="sidebar-section-label"),
                            ui.h5("Local Deterministic Rule Engine", class_="fw-bold mb-0 text-primary"),
                            ui.span("53 SCOGS decision tables loaded", class_="fs-7 text-secondary mt-1"),
                            class_="metric-box",
                        ),
                        ui.div(
                            ui.span("Inference Engine", class_="sidebar-section-label"),
                            ui.h5("Ollama / MedGemma 27B", class_="fw-bold mb-0"),
                            ui.span("Real-time clinical entity extraction", class_="fs-7 text-secondary mt-1"),
                            class_="metric-box",
                        ),
                        col_widths=[6, 6],
                    ),
                    class_="p-3",
                ),
            )


app = App(app_ui, server)
