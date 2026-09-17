"""Reactive server logic for the SCOGS Shiny dashboard."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from htmltools import HTML
from shiny import reactive, render, ui
from shiny.types import SilentException

from dashboard.data import get_available_run_files, load_csv_notes, load_run_file
from dashboard.evaluation import (
    FOCUS_OUTCOMES,
    LIVE_CONCURRENCY,
    extract_and_grade_note,
    grade_badge,
    is_derived,
    is_ollama_available,
)
from dashboard.highlight import highlight_note_quotes
from dashboard.view_state import (
    ABSENT,
    BUCKET_LABELS,
    CANNOT_GRADE,
    OUTCOME_BUCKETS,
    PRESENT,
    explore_view_state,
    grade_details,
    grade_result_dict,
    live_view_state,
    outcome_bucket,
    outcome_rank,
    outcome_status,
)

#: Colour of each group heading in the overview, matching its pills.
BUCKET_HEADING_CLASS = {
    PRESENT: "text-success",
    CANNOT_GRADE: "text-warning",
    ABSENT: "text-muted",
}
#: Pill style per group.
BUCKET_PILL_CLASS = {
    PRESENT: "outcome-btn-graded",
    CANNOT_GRADE: "outcome-btn-cannot",
    ABSENT: "outcome-btn-absent",
}


# ==============================================================================
# 3. Server Logic
# ==============================================================================

def server(input, output, session):
    # Reactive state
    current_run_cache = reactive.value(None)
    live_eval_result = reactive.value(None)

    def current(name: str, default: Any = None) -> Any:
        """-> input `name`'s value, or `default` until the client has sent one.

        `reactive.Value.get()` registers the dependency *before* raising
        SilentException for an unset value, so the fallback keeps the caller
        reactive. The selectors below render the very inputs they read: without
        this they cancel their own render, the `<select>` is never sent, the
        input is therefore never set, and explore mode stays blank forever.
        """
        try:
            return input[name]()
        except SilentException:
            return default

    # The clinical-notes CSV is tens of megabytes and only the live evaluator
    # reads it, so it loads on first use rather than at session start: an
    # explore-only session never touches it.
    @reactive.calc
    def csv_notes():
        return load_csv_notes("data/clinical_notes.csv")

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

            df = csv_notes()
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
                # All 14 focus outcomes by default: the clinical question is
                # "what has this patient got?", which one outcome cannot answer.
                # Narrowing the list is the way to a faster run, not the way in.
                ui.input_selectize(
                    "live_outcome",
                    "Target Focus Outcomes:",
                    choices=outcome_choices,
                    selected=list(FOCUS_OUTCOMES),
                    multiple=True,
                    options={"plugins": ["remove_button"]},
                ),
                ui.layout_columns(
                    ui.input_numeric("patient_age_input", "Age (years):", value=25.0, min=0.0, max=120.0, step=0.5),
                    ui.input_select("patient_sex_input", "Sex:", choices={"male": "Male", "female": "Female", "unknown": "Unknown"}, selected="male"),
                    col_widths=[6, 6],
                ),
                ui.input_checkbox("outcome_present_input",
                                  "Outcomes explicitly present in note (deterministic mode only)",
                                  value=True),
                ui.input_text_area(
                    "live_note_text",
                    "Clinical Narrative:",
                    rows=7,
                    placeholder="Enter or review clinical note text...",
                ),
                ui.input_text_area(
                    "manual_features_json",
                    "Feature Values (JSON, graded against every selected outcome):",
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
        df = csv_notes()
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
            title = rec.get("title")
            if not title:
                present_names = [
                    o.get("outcome_name") or f"Outcome #{k}"
                    for k, o in rec.get("outcomes", {}).items()
                    if o.get("present")
                ]
                title = ", ".join(present_names[:2]) if present_names else "Case Report"
            choices[uid] = f"{uid} - {title[:40]}"

        current_uid = current("selected_patient_uid")
        selected_uid = current_uid if (current_uid in choices) else next(iter(choices.keys()))
        return ui.input_select("selected_patient_uid", "Select Patient Case:", choices=choices, selected=selected_uid)

    # Mode 1 Outcome Selector UI
    @output
    @render.ui
    def run_outcome_selector_ui():
        data = current_run_cache.get()
        records = (data or {}).get("records_by_uid", {})
        if not records:
            return ui.div()

        uid = current("selected_patient_uid")
        if uid not in records:
            uid = next(iter(records.keys()))

        rec = records[uid]
        outcomes = rec.get("outcomes", {})
        choices = {}

        best_default = None
        if outcomes:
            best_default = sorted(outcomes.items(), key=outcome_rank)[0][0]

        for k, o_data in outcomes.items():
            name = o_data.get("outcome_name") or FOCUS_OUTCOMES.get(k, {}).get("name") or f"Outcome {k}"
            gr = o_data.get("grade_result") or {}
            st = outcome_status(o_data)
            g = gr.get("grade")
            if st in ("graded", "grade_set") and g is not None:
                tag = f"[Grade {g}]"
            elif st == "cannot_grade":
                tag = "[Cannot Grade]"
            elif o_data.get("present"):
                tag = "[Present]"
            else:
                tag = "[Absent]"
            choices[k] = f"#{k} {name} {tag}"

        if not choices:
            choices = {k: f"#{k} {v['name']}" for k, v in FOCUS_OUTCOMES.items()}

        current_sel = current("selected_outcome_num")
        selected_k = current_sel if (current_sel in choices) else (best_default or next(iter(choices.keys()), None))
        return ui.input_select("selected_outcome_num", "Select Outcome:", choices=choices, selected=selected_k)

    def selected_live_outcomes() -> list[str]:
        """-> the outcomes the live evaluator grades: the sidebar's picks, or all
        14 focus outcomes while it has not reported them or the clinician has
        cleared the box."""
        picked = current("live_outcome") or ()
        if isinstance(picked, str):
            picked = (picked,)
        chosen = [str(num) for num in picked if str(num) in FOCUS_OUTCOMES]
        return chosen or list(FOCUS_OUTCOMES)

    def live_outcome_num(results: dict | None) -> str:
        """-> the outcome the live cards below the overview show: the pill the
        clinician clicked when it was graded, else the most informative result."""
        clicked = str(current("selected_outcome_num") or "")
        if results:
            return clicked if clicked in results else sorted(results.items(), key=outcome_rank)[0][0]
        chosen = selected_live_outcomes()
        return clicked if clicked in chosen else chosen[0]

    # Trigger Live Note Evaluation
    @reactive.effect
    @reactive.event(input.btn_analyze)
    def _perform_live_eval():
        note_text = input.live_note_text() or ""
        outcome_ids = selected_live_outcomes()
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
            outcomes=outcome_ids,
            patient_context={"patient_sex": patient_sex, "patient_age": patient_age},
            use_ollama=is_ollama_available(),
            # Deterministic mode has one set of typed values; every selected
            # outcome is graded against it.
            manual_features={num: dict(manual_dict) for num in outcome_ids},
            concurrency=LIVE_CONCURRENCY,
        )
        live_eval_result.set(res)
        ui.notification_show(f"Analysis and grading complete: {len(res)} outcomes", type="message")

    # Helper: resolve active state data
    def get_current_view_state() -> dict[str, Any]:
        if input.app_mode() == "explore":
            run_data = current_run_cache.get()
            if not run_data:
                # Before a file loads, the case selectors do not exist yet; reading
                # them would cancel the render.
                return {}
            records = run_data.get("records_by_uid", {})
            if not records:
                return {}
            uid = current("selected_patient_uid")
            if uid not in records:
                uid = next(iter(records.keys()))
            rec = records.get(uid, {})
            outcomes = rec.get("outcomes", {})
            outcome_num = current("selected_outcome_num")
            if outcome_num not in outcomes:
                if outcomes:
                    outcome_num = sorted(outcomes.items(), key=outcome_rank)[0][0]
                else:
                    outcome_num = "28"
            return explore_view_state(run_data, uid, str(outcome_num))
        results = live_eval_result.get()
        return live_view_state(results, live_outcome_num(results),
                               input.live_note_text() or "", input.patient_age_input(),
                               input.patient_sex_input())

    # ==========================================================================
    # 4. Renderers: Cards, Tables, Inspector
    # ==========================================================================

    @output
    @render.ui
    def outcomes_card_title():
        """The overview card's heading; the card shell and its filter are static."""
        if input.app_mode() == "live":
            title = ui.span("Live Note Outcomes & Severity Overview", class_="card-header-title")
            graded = len(live_eval_result.get() or {})
            if not graded:
                return title
            return ui.div(
                title,
                ui.span(f"{graded} outcomes graded" if graded != 1 else "1 outcome graded",
                        class_="badge bg-light text-secondary border fw-normal ms-2",
                        style="font-size: 0.78rem;"),
                class_="d-flex align-items-center flex-wrap gap-1",
            )

        state = get_current_view_state()
        records = (current_run_cache.get() or {}).get("records_by_uid", {})
        uid = state.get("patient_uid")
        rec = records.get(uid) if uid else None
        if not rec:
            return ui.span("Encounter Outcomes & Severity Overview", class_="card-header-title")
        return ui.div(
            ui.span(f"Encounter Outcomes & Severity Overview: {uid}", class_="card-header-title"),
            ui.span(rec.get("title") or "Clinical Encounter",
                    class_="badge bg-light text-secondary border fw-normal ms-2",
                    style="font-size: 0.78rem;"),
            class_="d-flex align-items-center flex-wrap gap-1",
        )

    @output
    @render.ui
    def patient_outcomes_summary_ui():
        """Every outcome of the selected case, grouped by state and filterable.

        This is the answer to "what has this patient got?". Picking one outcome
        is the follow-up question, not the way in, so the overview never depends
        on that selection - only on the case, saved or live.
        """
        state = get_current_view_state()
        if input.app_mode() == "live":
            outcomes = live_eval_result.get() or {}
            empty_hint = ("Click 'Analyze & Grade Note' to grade this note against every "
                          "focus outcome selected in the sidebar.")
        else:
            records = (current_run_cache.get() or {}).get("records_by_uid", {})
            uid = state.get("patient_uid")
            outcomes = (records.get(uid) or {}).get("outcomes", {}) if uid else {}
            empty_hint = ("Choose a saved run and a patient case in the sidebar to list "
                          "this patient's outcomes.")
        if not outcomes:
            return ui.p(empty_hint, class_="text-muted p-3 mb-0")

        # Unset (the page has not reported the boxes yet) means all three; the
        # client sends None once the clinician unticks the last one.
        shown = set(current("outcome_filter", OUTCOME_BUCKETS) or ())
        selected_outcome = str(state.get("outcome_num"))

        groups: dict[str, list] = {bucket: [] for bucket in OUTCOME_BUCKETS}
        for k, o_data in sorted(outcomes.items(), key=outcome_rank):
            groups[outcome_bucket(o_data)].append(_outcome_pill(k, o_data, selected_outcome))

        sections = []
        for bucket in OUTCOME_BUCKETS:
            pills = groups[bucket]
            if not pills or bucket not in shown:
                continue
            sections.append(
                ui.div(
                    ui.span(f"{BUCKET_LABELS[bucket]} ({len(pills)}):",
                            class_=f"sidebar-section-label mb-2 d-block {BUCKET_HEADING_CLASS[bucket]}"),
                    ui.div(*pills, class_="d-flex flex-wrap gap-2 mb-3"),
                )
            )

        if not sections:
            hint = ("Tick a category above to list this patient's outcomes."
                    if not shown
                    else "This patient has no outcomes in the ticked categories.")
            return ui.p(hint, class_="text-muted p-3 mb-0")

        return ui.div(*sections, class_="p-3 pb-1")

    def _outcome_pill(num: str, outcome: dict, selected_outcome: str):
        """One clickable outcome chip; clicking it drives the rest of the page."""
        bucket = outcome_bucket(outcome)
        name = (outcome.get("outcome_name")
                or FOCUS_OUTCOMES.get(num, {}).get("name")
                or f"Outcome {num}")
        grade_result = grade_result_dict(outcome)
        active = "outcome-pill-active" if str(num) == selected_outcome else ""

        label = [ui.span(f"#{num} {name}",
                         class_="text-muted" if bucket == ABSENT else "fw-semibold me-2")]
        if bucket != ABSENT:
            badge_css, badge_txt = grade_badge(
                outcome_status(outcome), grade_result.get("grade"),
                tuple(grade_result.get("grades") or ()),
            )
            label.append(ui.span(badge_txt, class_=f"badge {badge_css} px-2 py-1"))

        return ui.tags.button(
            *label,
            type="button",
            class_=f"outcome-summary-btn {BUCKET_PILL_CLASS[bucket]} {active}",
            onclick=f"Shiny.setInputValue('selected_outcome_num', '{num}', {{priority: 'event'}})",
            title=f"Click to inspect Outcome #{num}",
        )


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
                    ),
                    class_="sticky-top",
                ),
                ui.tags.tbody(*rows),
                class_="table table-sm findings-table mb-0",
            ),
            class_="table-responsive p-0 findings-scroll-container",
            style="max-height: 420px; overflow-y: auto;",
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
                            ui.h5(prov.get("weights") or "MedGemma 27B", class_="fw-bold mb-0 text-dark"),
                            ui.span(f"Prompt Stage: {prov.get('prompt_stage', 'N/A')} ({prov.get('quant') or 'F16'})", class_="fs-7 text-secondary mt-1"),
                            class_="metric-box",
                        ),
                        ui.div(
                            ui.span("Generation Speed", class_="sidebar-section-label"),
                            ui.h5(f"{(prof.get('completion_tokens_per_sec') or 0.0)} tok/s", class_="fw-bold mb-0 text-primary metric-val-tabular"),
                            ui.span(f"Latency: {(prof.get('sec_per_note') or 0.0)}s / note", class_="fs-7 text-secondary mt-1 metric-val-tabular"),
                            class_="metric-box",
                        ),
                        ui.div(
                            ui.span("Grounding Accuracy", class_="sidebar-section-label"),
                            ui.h5(f"{(auto.get('quote_verified_pct') or 0.0):.1f}% Verified", class_="fw-bold mb-0 text-success metric-val-tabular"),
                            ui.span(f"Hallucination: {(auto.get('hallucinated_quote_pct') or 0.0):.1f}%", class_="fs-7 text-danger mt-1 metric-val-tabular"),
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
