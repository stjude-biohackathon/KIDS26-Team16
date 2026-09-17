"""Reactive server logic for the SCOGS Shiny dashboard."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from htmltools import HTML
from shiny import reactive, render, ui

from dashboard.data import get_available_run_files, load_csv_notes, load_run_file
from dashboard.evaluation import (
    FOCUS_OUTCOMES,
    extract_and_grade_note,
    grade_badge,
    is_derived,
    is_ollama_available,
)
from dashboard.highlight import highlight_note_quotes
from dashboard.view_state import explore_view_state, grade_details, live_view_state


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
