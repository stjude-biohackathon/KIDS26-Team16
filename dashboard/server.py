"""Reactive server logic for the SCOGS Shiny dashboard."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
from htmltools import HTML
from shiny import reactive, render, ui
from shiny.types import SilentException

from dashboard.data import get_available_run_files, load_csv_notes, load_run_file
from dashboard.evaluation import (
    DEFAULT_LIVE_MODEL,
    FOCUS_OUTCOMES,
    OLLAMA_NUM_CTX,
    TABLES,
    check_ollama_status,
    detect_system_hardware,
    extract_and_grade_note,
    get_all_scogs_outcomes,
    get_concurrency_assessment,
    get_installed_models,
    get_live_concurrency,
    get_model_choices,
    get_outcome_choices,
    grade_badge,
    is_derived,
    is_ollama_available,
    normalize_outcome_id,
    outcome_meta,
    save_live_patient_results,
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


#: Where live evaluations are kept: one JSON file per patient
#: (evaluation.save_live_patient_results). Tests point it at a temporary folder.
LIVE_RESULTS_DIR = "results/live"

#: One line under the verdict's grade saying what the harness status means.
VERDICT_STATUS_TEXT = {
    "graded": "Deterministic grade from the matched rubric row.",
    "grade_set": "More than one grade fits the evidence.",
    "cannot_grade": "The rules could not decide a grade from the features found.",
    "missed_presence": "Model said absent, but objective criteria are met.",
    "absent": "Outcome not present in this note.",
    "not_applicable": "Outcome does not apply to this patient.",
    "refuted": "The note contradicts this outcome.",
    "pending": "Run the analysis to grade this outcome.",
}
#: The severity scale drawn under the grade.
SEVERITY_GRADES = range(1, 6)


def _severity_scale(status: str, grade: Any, grades: tuple) -> Any:
    """Grades 1-5 as segments: lit up to a grade, or just the grades still in play."""
    if status == "graded" and grade is not None:
        current = {int(grade)}
        lit = set(range(1, int(grade) + 1))
    elif status == "grade_set":
        current = lit = {int(g) for g in grades or ()}
    else:
        current = lit = set()
    return ui.div(
        ui.div(*[
            ui.span(class_=" ".join(["verdict-scale-seg"]
                                    + (["is-lit"] if g in lit else [])
                                    + (["is-current"] if g in current else [])))
            for g in SEVERITY_GRADES
        ], class_="verdict-scale-track"),
        ui.div(*[ui.span(str(g), class_="is-current" if g in current else None) for g in SEVERITY_GRADES],
               class_="verdict-scale-ticks"),
        class_="verdict-scale",
        aria_hidden="true",
    )


def _verdict_field(label: str, body: Any, *, danger: bool = False) -> Any:
    """One labelled block in the verdict's detail column."""
    return ui.div(
        ui.div(label, class_="verdict-field-label" + (" is-danger" if danger else "")),
        body,
        class_="verdict-field",
    )


# ==============================================================================
# 3. Server Logic
# ==============================================================================

def server(input, output, session):
    # Reactive state
    current_run_cache = reactive.value(None)
    live_eval_result = reactive.value(None)
    # The outcome pill last clicked in live mode. Explore mode writes the same
    # `selected_outcome_num` input, so reading the input directly would open a
    # fresh live analysis on whatever outcome was last inspected in a saved run.
    live_clicked_outcome = reactive.value("")
    session_case_outcomes: dict[tuple[str, str], str] = {}

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
        except (SilentException, KeyError, AttributeError):
            return default

    # The clinical-notes CSV is tens of megabytes and only the live evaluator
    # reads it, so it loads on first use rather than at session start: an
    # explore-only session never touches it.
    @reactive.calc
    def csv_notes():
        return load_csv_notes("data/clinical_notes.csv")

    def get_effective_model() -> str:
        try:
            sel = input.live_model_select() if hasattr(input, "live_model_select") else None
        except Exception:
            sel = None
        sel = sel or DEFAULT_LIVE_MODEL
        if sel == "__custom__":
            try:
                custom = input.custom_model_input() if hasattr(input, "custom_model_input") else None
            except Exception:
                custom = None
            if custom and custom.strip():
                return custom.strip()
            return DEFAULT_LIVE_MODEL
        return sel

    @output
    @render.ui
    def header_status_badge():
        selected_model = get_effective_model()
        status, _ = check_ollama_status(model=selected_model)

        if status == "ready":
            return ui.span(f"Ollama Online ({selected_model} • {OLLAMA_NUM_CTX:,} ctx)", class_="badge badge-grade-1", style="font-size: 0.76rem;")
        elif status == "not_installed":
            return ui.span(f"Model Not Installed ({selected_model})", class_="badge badge-cannot-grade", style="font-size: 0.76rem;")
        return ui.span("Ollama Offline", class_="badge bg-secondary", style="font-size: 0.76rem;")

    @output
    @render.ui
    def live_model_status_badge():
        selected_model = get_effective_model()
        status, _ = check_ollama_status(model=selected_model)

        if status == "ready":
            return ui.div(
                ui.div(
                    ui.span("● Ollama Online", class_="badge badge-grade-1 me-1"),
                    ui.span("Ready", class_="badge bg-success-subtle text-success-emphasis"),
                    class_="d-flex align-items-center flex-wrap gap-1 mb-1",
                ),
                ui.div(
                    f"{selected_model} • {OLLAMA_NUM_CTX:,} ctx",
                    class_="small text-muted font-monospace",
                    style="font-size: 0.72rem; word-break: break-all;",
                ),
                class_="sidebar-status-card p-2 rounded mb-2",
            )
        elif status == "not_installed":
            return ui.div(
                ui.span(f"▲ Model Not Installed: {selected_model}", class_="badge badge-cannot-grade mb-1 text-wrap text-start"),
                ui.div(
                    f"Ollama is running, but '{selected_model}' is not installed. Run 'ollama pull {selected_model}' in terminal to download it, or analyze below in deterministic mode.",
                    class_="small text-muted mb-1",
                    style="font-size: 0.74rem; line-height: 1.35;",
                ),
                class_="sidebar-status-card p-2 rounded mb-2",
            )
        return ui.div(
            ui.div(
                ui.span("● Ollama Offline", class_="badge bg-secondary me-1"),
                ui.span("Deterministic Mode", class_="badge bg-secondary-subtle text-secondary-emphasis"),
                class_="d-flex align-items-center flex-wrap gap-1 mb-1",
            ),
            ui.div(
                f"Model target: {selected_model}",
                class_="small text-muted font-monospace",
                style="font-size: 0.72rem; word-break: break-all;",
            ),
            class_="sidebar-status-card p-2 rounded mb-2",
        )

    @output
    @render.ui
    def live_concurrency_advisory():
        selected_model = get_effective_model()
        try:
            val = input.live_concurrency_input()
            conc = int(val) if val else get_live_concurrency(selected_model)
        except (ValueError, TypeError):
            conc = get_live_concurrency(selected_model)

        assessment = get_concurrency_assessment(selected_model, concurrency=conc)
        status = assessment["status"]
        if status == "safe":
            color_class = "text-success border-success-subtle bg-success-subtle"
        elif status == "caution":
            color_class = "text-warning-emphasis border-warning-subtle bg-warning-subtle"
        else:
            color_class = "text-danger border-danger-subtle bg-danger-subtle"

        return ui.div(
            ui.div(assessment["message"], class_=f"p-1 px-2 rounded border {color_class}"),
            class_="mb-2",
            style="font-size: 0.74rem; line-height: 1.25;",
        )

    @reactive.effect
    @reactive.event(input.live_model_select, input.custom_model_input)
    def _update_concurrency_for_selected_model():
        model = get_effective_model()
        if model:
            rec = get_concurrency_assessment(model)["recommended"]
            ui.update_numeric("live_concurrency_input", value=rec)

    @reactive.effect
    @reactive.event(input.refresh_ollama_models)
    def _refresh_ollama_models():
        choices = get_model_choices(grouped=True, include_custom=True)
        flat = get_model_choices(grouped=False, include_custom=True)
        cur = input.live_model_select() if hasattr(input, "live_model_select") else None
        sel = cur if cur in flat else DEFAULT_LIVE_MODEL
        ui.update_select("live_model_select", choices=choices, selected=sel)
        installed = get_installed_models()
        if installed:
            ui.notification_show(f"Refreshed: {len(installed)} model(s) served by Ollama.", type="message")
        else:
            ui.notification_show("Ollama is offline or reports 0 installed models.", type="warning")

    @reactive.effect
    @reactive.event(input.btn_select_14)
    def _select_14_outcomes():
        ui.update_selectize("live_outcome", selected=list(FOCUS_OUTCOMES))

    @reactive.effect
    @reactive.event(input.btn_select_53)
    def _select_53_outcomes():
        choices = get_outcome_choices(grouped=False)
        ui.update_selectize("live_outcome", selected=list(choices.keys()))

    @reactive.effect
    @reactive.event(input.btn_clear_outcomes)
    def _clear_outcomes():
        ui.update_selectize("live_outcome", selected=[])



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
                ui.div(
                    ui.div(
                        ui.span("RUN REPOSITORY", class_="sidebar-section-label mb-0"),
                        ui.span("Dataset", class_="sidebar-badge-subtle"),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    ui.div(
                        ui.input_select("selected_run_file", "Run JSON Output:", choices=file_choices, selected=default_file),
                        class_="mb-2",
                    ),
                    ui.output_ui("run_case_selector_ui"),
                    ui.output_ui("run_outcome_selector_ui"),
                    class_="sidebar-panel-card mb-2",
                ),
                ui.div(
                    ui.div(
                        ui.span("VERIFICATION COHORT", class_="sidebar-section-label mb-0"),
                        ui.span("CTCAE v5.0", class_="sidebar-badge-subtle"),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    ui.p(
                        "Inspect deterministic severity grades, rule predicates, and MedGemma grounded quotes for each patient case.",
                        class_="sidebar-help-text mb-0",
                    ),
                    class_="sidebar-panel-card sidebar-panel-info",
                ),
            )
        else:
            # Mode 2: Live Note Evaluator
            model_choices = get_model_choices(grouped=True)
            flat_choices = get_model_choices(grouped=False)
            default_model = DEFAULT_LIVE_MODEL if DEFAULT_LIVE_MODEL in flat_choices else next(iter(flat_choices.keys()))

            df = csv_notes()
            csv_choices = {}
            if df is not None and not df.empty:
                for idx, row in df.head(50).iterrows():
                    pid = str(row.get("patient_id", f"Row {idx}"))
                    vdate = str(row.get("visit_datetime", ""))
                    ftype = str(row.get("facility_type", ""))
                    csv_choices[str(idx)] = f"{pid} - {vdate} ({ftype})"

            outcome_choices = get_outcome_choices(grouped=True)

            return ui.div(
                ui.div(
                    ui.div(
                        ui.span("INFERENCE RUNTIME", class_="sidebar-section-label mb-0"),
                        ui.span("Ollama", class_="sidebar-badge-subtle"),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    ui.input_select(
                        "live_model_select",
                        "Inference Model:",
                        choices=model_choices,
                        selected=default_model,
                    ),
                    ui.panel_conditional(
                        "input.live_model_select === '__custom__'",
                        ui.input_text(
                            "custom_model_input",
                            "Custom Ollama Model Name / Tag:",
                            placeholder="e.g. llama3.2:3b, mistral:7b, qwen2.5:7b",
                        ),
                    ),
                    ui.input_action_button(
                        "refresh_ollama_models",
                        "↻ Refresh Ollama Models",
                        class_="btn-sm btn-outline-secondary w-100 my-2",
                    ),
                    ui.output_ui("live_model_status_badge"),
                    ui.div(
                        ui.input_numeric(
                            "live_concurrency_input",
                            "Concurrency (Parallel Workers):",
                            value=get_concurrency_assessment(default_model)["recommended"],
                            min=1,
                            max=14,
                            step=1,
                        ),
                        class_="mt-2",
                    ),
                    ui.output_ui("live_concurrency_advisory"),
                    class_="sidebar-panel-card mb-2",
                ),
                ui.div(
                    ui.div(
                        ui.span("ENCOUNTER & NARRATIVE", class_="sidebar-section-label mb-0"),
                        ui.span("Clinical Input", class_="sidebar-badge-subtle"),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    ui.div(
                        ui.input_radio_buttons(
                            "live_source",
                            "Input Source:",
                            choices={"csv": "From CSV Dataset", "custom": "Paste Custom Note"},
                            selected="custom",
                        ),
                        class_="sidebar-segmented-radios mb-2",
                    ),
                    ui.panel_conditional(
                        "input.live_source === 'csv'",
                        ui.div(
                            ui.input_select(
                                "csv_patient_idx",
                                "Select Patient Encounter:",
                                choices=csv_choices,
                                selected=next(iter(csv_choices.keys())) if csv_choices else None,
                            ),
                            class_="mb-2",
                        ),
                    ),
                    ui.layout_columns(
                        ui.input_numeric("patient_age_input", "Age (years):", value=None, min=0.0, max=120.0, step=0.5),
                        ui.input_select(
                            "patient_sex_input",
                            "Sex:",
                            choices={"": "Select sex (optional)", "male": "Male", "female": "Female", "unknown": "Unknown"},
                            selected="",
                        ),
                        col_widths=[6, 6],
                        class_="mb-2",
                    ),
                    ui.input_text_area(
                        "live_note_text",
                        "Clinical Narrative:",
                        rows=6,
                        placeholder="Enter or review clinical note text...",
                    ),
                    class_="sidebar-panel-card mb-2",
                ),
                ui.div(
                    ui.div(
                        ui.span("EVALUATION SCOPE", class_="sidebar-section-label mb-0"),
                        ui.span("Consensus Tables", class_="sidebar-badge-subtle"),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    ui.div(
                        ui.input_radio_buttons(
                            "live_outcome_mode",
                            "Target Outcomes Scope:",
                            choices={
                                "14_focus": "#14 Focus Outcomes (Default)",
                                "all_53": "All 53 SCOGS Outcomes",
                                "custom": "Custom Selection (Pick & Choose)",
                            },
                            selected="14_focus",
                        ),
                        class_="sidebar-segmented-radios mb-2",
                    ),
                    ui.panel_conditional(
                        "input.live_outcome_mode === '14_focus'",
                        ui.div(
                            ui.div(
                                ui.span("🎯 14 Core Consensus Outcomes Active", class_="fw-bold d-block text-primary"),
                                ui.span(
                                    "Evaluates the 14 Delphi focus outcomes: VOC (#28), Stroke (#15), ACS (#48), Priapism (#24), "
                                    "Splenic Sequestration (#29), CKD (#21), Retinopathy (#17), Chronic Pain (#10), "
                                    "Cognitive Dysfunction (#11), TCD (#12), Depression (#47), Asthma (#49), "
                                    "AVN (#39), and Leg Ulcer (#40).",
                                    class_="small text-muted",
                                ),
                                class_="p-2 rounded bg-body-tertiary border mb-2",
                                style="font-size: 0.75rem; line-height: 1.35;",
                            ),
                        ),
                    ),
                    ui.panel_conditional(
                        "input.live_outcome_mode === 'all_53'",
                        ui.div(
                            ui.div(
                                ui.span("🌐 All 53 SCOGS Decision Tables Active", class_="fw-bold d-block text-success"),
                                ui.span(
                                    "Comprehensive evaluation across all 53 CTCAE v5.0 and Delphi consensus tables (#01 to #53).",
                                    class_="small text-muted",
                                ),
                                class_="p-2 rounded bg-body-tertiary border mb-2",
                                style="font-size: 0.75rem; line-height: 1.35;",
                            ),
                        ),
                    ),
                    ui.panel_conditional(
                        "input.live_outcome_mode === 'custom'",
                        ui.div(
                            ui.layout_columns(
                                ui.input_action_button("btn_select_14", "Select 14 Focus", class_="btn-sm btn-outline-primary w-100"),
                                ui.input_action_button("btn_select_53", "Select All 53", class_="btn-sm btn-outline-secondary w-100"),
                                ui.input_action_button("btn_clear_outcomes", "Clear", class_="btn-sm btn-outline-danger w-100"),
                                col_widths=[5, 5, 2],
                                class_="mb-1",
                            ),
                            ui.input_selectize(
                                "live_outcome",
                                "Choose Target Outcomes:",
                                choices=outcome_choices,
                                selected=list(FOCUS_OUTCOMES),
                                multiple=True,
                                options={"plugins": ["remove_button"], "placeholder": "Search by outcome name or #number..."},
                            ),
                        ),
                    ),
                    ui.tags.details(
                        ui.tags.summary(
                            ui.span("📖 Directory of All 53 Possible Outcomes", class_="fw-semibold text-primary", style="font-size: 0.78rem;"),
                            class_="d-flex align-items-center justify-content-between",
                        ),
                        ui.div(
                            ui.div(
                                ui.tags.table(
                                    ui.tags.thead(
                                        ui.tags.tr(
                                            ui.tags.th("#", style="width: 15%;"),
                                            ui.tags.th("Outcome", style="width: 55%;"),
                                            ui.tags.th("Organ System", style="width: 30%;"),
                                        ),
                                        style="font-size: 0.74rem;",
                                    ),
                                    ui.tags.tbody(
                                        *[
                                            ui.tags.tr(
                                                ui.tags.td(
                                                    ui.span(
                                                        f"#{norm}",
                                                        class_="badge badge-grade-1" if meta["is_focus"] else "badge bg-secondary-subtle text-secondary-emphasis",
                                                        style="font-size: 0.7rem;",
                                                    )
                                                ),
                                                ui.tags.td(
                                                    ui.span(meta["name"], class_="fw-medium" if meta["is_focus"] else ""),
                                                    ui.span(" ★", class_="text-primary small fw-bold") if meta["is_focus"] else "",
                                                ),
                                                ui.tags.td(ui.span(meta["organ_system"], class_="text-muted small")),
                                                style="font-size: 0.73rem;",
                                            )
                                            for norm, meta in get_all_scogs_outcomes().items()
                                        ]
                                    ),
                                    class_="table table-sm table-hover mb-0",
                                ),
                                class_="sidebar-directory-table-wrap mt-2",
                            ),
                            ui.p("★ Green badge indicates a core Focus Outcome.", class_="text-muted mt-1 mb-0", style="font-size: 0.7rem;"),
                        ),
                        class_="sidebar-directory-details mt-2",
                    ),
                    class_="sidebar-panel-card mb-2",
                ),
                ui.div(
                    ui.div(
                        ui.span("DETERMINISTIC SIMULATION", class_="sidebar-section-label mb-0"),
                        ui.span("Offline Mode", class_="sidebar-badge-subtle"),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    ui.p(
                        "Active when Ollama is offline, or to test CTCAE rules directly without LLM extraction:",
                        class_="small text-muted mb-2",
                    ),
                    ui.div(
                        ui.input_checkbox(
                            "outcome_present_input",
                            "Assume outcomes are clinically present",
                            value=True,
                        ),
                        class_="mb-1",
                    ),
                    ui.div(
                        "When unchecked, outcomes evaluate as Absent (Grade 0). In online mode, the LLM extracts presence from text.",
                        class_="small text-muted mb-2 ms-4",
                        style="font-size: 0.75rem;",
                    ),
                    ui.div(
                        ui.input_text_area(
                            "manual_features_json",
                            "Simulated Feature Values (JSON):",
                            rows=3,
                            value='{"care_setting": "inpatient", "pain_co_complication": false, "death_attributed": false, "life_support": false}',
                        ),
                        class_="simulation-json-wrapper",
                    ),
                    ui.div(
                        "Key-value features evaluated directly by CTCAE rules across all selected outcomes when Ollama is offline.",
                        class_="small text-muted mt-1 mb-0",
                        style="font-size: 0.74rem;",
                    ),
                    class_="sidebar-panel-card sidebar-simulation-box mb-2",
                ),
                ui.input_action_button("btn_analyze", "Analyze & Grade Note", class_="btn btn-clinical-primary w-100 mt-1 mb-2"),
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
            else:
                ui.update_select("patient_sex_input", selected="")
            age_val = row.get("patient_age")
            if pd.notna(age_val):
                ui.update_numeric("patient_age_input", value=round(float(age_val), 1))
            else:
                ui.update_numeric("patient_age_input", value=None)
        except Exception:
            pass

    @reactive.effect
    @reactive.event(input.live_source)
    def _handle_custom_source():
        if input.app_mode() == "live" and input.live_source() == "custom":
            ui.update_text_area("live_note_text", value="")
            ui.update_numeric("patient_age_input", value=None)
            ui.update_select("patient_sex_input", selected="")

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
        return ui.div(
            ui.input_select("selected_patient_uid", "Select Patient Case:", choices=choices, selected=selected_uid),
            class_="mb-2",
        )

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

        filepath = current("selected_run_file", "")
        case_key = (filepath, uid)
        current_sel = current("selected_outcome_num")
        if case_key not in session_case_outcomes:
            selected_k = best_default or next(iter(choices.keys()), None)
            session_case_outcomes[case_key] = selected_k
        elif current_sel and current_sel in choices:
            selected_k = current_sel
            session_case_outcomes[case_key] = selected_k
        else:
            selected_k = session_case_outcomes.get(case_key) or best_default or next(iter(choices.keys()), None)
        return ui.div(
            ui.input_select("selected_outcome_num", "Focus SCOGS Outcome:", choices=choices, selected=selected_k),
            class_="mb-0",
        )

    def selected_live_outcomes() -> list[str]:
        """-> the outcomes the live evaluator grades: 14 focus (default), all 53, or custom picks."""
        mode = current("live_outcome_mode", "14_focus")
        if mode == "all_53":
            return [normalize_outcome_id(k) for k in sorted(TABLES.keys())]
        if mode == "custom":
            # The picker stays in the page (hidden) in the other modes, so its
            # picks count only here: "14 Focus" must mean the 14.
            picked = current("live_outcome")
            if picked is None:
                return [normalize_outcome_id(k) for k in FOCUS_OUTCOMES]
            if isinstance(picked, str):
                picked = (picked,)
            return [normalize_outcome_id(num) for num in picked if normalize_outcome_id(num) in TABLES]
        return [normalize_outcome_id(k) for k in FOCUS_OUTCOMES]

    @reactive.effect
    @reactive.event(input.selected_outcome_num)
    def _remember_live_click():
        if current("app_mode") == "live":
            live_clicked_outcome.set(str(input.selected_outcome_num() or ""))

    def selected_csv_encounter() -> dict[str, str] | None:
        """-> the patient id, visit and a title for the CSV row being graded;
        None for a pasted note, which has no patient id."""
        if current("live_source") != "csv":
            return None
        try:
            row = csv_notes().iloc[int(current("csv_patient_idx"))]
        except (TypeError, ValueError, IndexError):
            return None
        patient_id = str(row.get("patient_id") or "").strip()
        if not patient_id:
            return None
        visit = str(row.get("visit_datetime") or "").strip()
        facility = str(row.get("facility_type") or "").strip()
        return {
            "patient_id": patient_id,
            "visit": visit,
            "title": " ".join(part for part in (facility, "visit", visit) if part),
        }

    def live_outcome_num(results: dict | None) -> str:
        """-> the outcome the live cards below the overview show: the pill the
        clinician clicked when it was graded, else the most informative result."""
        clicked = live_clicked_outcome.get()
        if results:
            return clicked if clicked in results else sorted(results.items(), key=outcome_rank)[0][0]
        chosen = selected_live_outcomes()
        return clicked if clicked in chosen else (chosen[0] if chosen else "")

    # Trigger Live Note Evaluation
    @reactive.effect
    @reactive.event(input.btn_analyze)
    async def _perform_live_eval():
        note_text = current("live_note_text", "") or ""
        outcome_ids = selected_live_outcomes()
        # Checked before the Ollama preflight, which can take up to a minute.
        if not outcome_ids:
            ui.notification_show("No outcomes selected. Choose at least one outcome in the sidebar.", type="warning")
            return
        patient_sex = current("patient_sex_input", "") or "unknown"
        patient_age = current("patient_age_input", None)
        present = bool(current("outcome_present_input", True))

        manual_dict: dict[str, Any] = {}
        raw_manual = current("manual_features_json", "")
        if raw_manual and raw_manual.strip():
            try:
                manual_dict = json.loads(raw_manual.strip())
            except Exception as e:
                ui.notification_show(f"JSON Parse Error in manual features: {e}", type="warning")
                return

        manual_dict["present"] = present
        context = {"patient_sex": patient_sex, "patient_age": patient_age}

        # Both calls below run in a worker thread. Grading 14 outcomes is minutes
        # of blocking work, and on the event loop it starves the session's
        # websocket: the browser gives up on the ping, and the finished results
        # are then written to a closed socket ("socket.send() raised exception").
        selected_model = get_effective_model()
        online = await asyncio.to_thread(is_ollama_available, model=selected_model)
        if online and not note_text.strip():
            # Deterministic mode grades the typed features and needs no note;
            # the model has nothing to extract from an empty one.
            ui.notification_show("The clinical narrative is empty. Paste or select a note to extract from.", type="warning")
            return
        results: dict[str, Any] = {}
        raw_concurrency = current("live_concurrency_input", None)
        try:
            user_concurrency = int(raw_concurrency) if raw_concurrency is not None else get_live_concurrency(selected_model)
        except (ValueError, TypeError):
            user_concurrency = get_live_concurrency(selected_model)
        model_concurrency = max(1, min(user_concurrency, len(outcome_ids)))

        with ui.Progress(min=0, max=len(outcome_ids)) as progress:
            progress.set(0, message="Grading outcomes", detail=f"0 of {len(outcome_ids)}")
            # A chunk at a time, so the overview fills in as outcomes land instead
            # of staying empty until the last one is graded.
            for start in range(0, len(outcome_ids), model_concurrency):
                chunk = outcome_ids[start:start + model_concurrency]
                results.update(await asyncio.to_thread(
                    extract_and_grade_note,
                    note_text=note_text,
                    outcomes=chunk,
                    patient_context=context,
                    use_ollama=online,
                    model=selected_model,
                    # Deterministic mode has one set of typed values; every
                    # selected outcome is graded against it.
                    manual_features={num: dict(manual_dict) for num in chunk},
                    concurrency=model_concurrency,
                    num_ctx=OLLAMA_NUM_CTX,
                ))
                live_eval_result.set(dict(results))
                progress.set(len(results), detail=f"{len(results)} of {len(outcome_ids)}")

        backend = "ollama" if online else "deterministic"
        summary = f"Grading complete: {len(results)} outcome{'s' if len(results) != 1 else ''} ({backend})."
        encounter = selected_csv_encounter() or {}
        try:
            saved = await asyncio.to_thread(
                save_live_patient_results,
                note_text,
                results,
                patient_id=encounter.get("patient_id"),
                visit=encounter.get("visit"),
                title=encounter.get("title"),
                model=selected_model,
                backend=backend,
                patient_age=patient_age,
                patient_sex=patient_sex,
                output_dir=LIVE_RESULTS_DIR,
                num_ctx=OLLAMA_NUM_CTX,
            )
            ui.notification_show(f"{summary} Saved to {saved.as_posix()}", type="message")
        except Exception as e:
            ui.notification_show(f"{summary} Saving the patient's results failed: {e}", type="warning")

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
            filepath = current("selected_run_file", "")
            case_key = (filepath, uid)
            current_sel = current("selected_outcome_num")
            best_default = sorted(outcomes.items(), key=outcome_rank)[0][0] if outcomes else "28"
            if case_key not in session_case_outcomes:
                outcome_num = best_default
                session_case_outcomes[case_key] = str(outcome_num)
            elif current_sel and str(current_sel) in outcomes:
                outcome_num = str(current_sel)
                session_case_outcomes[case_key] = outcome_num
            elif session_case_outcomes.get(case_key) in outcomes:
                outcome_num = session_case_outcomes[case_key]
            else:
                outcome_num = best_default
                session_case_outcomes[case_key] = str(outcome_num)
            return explore_view_state(run_data, uid, str(outcome_num))
        results = live_eval_result.get()
        return live_view_state(results, live_outcome_num(results),
                               current("live_note_text", "") or "",
                               current("patient_age_input", None),
                               current("patient_sex_input", None))

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
                        class_="badge-engine fw-normal ms-2",
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
                    class_="badge-engine fw-normal ms-2",
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
            class_=f"outcome-summary-btn {BUCKET_PILL_CLASS[bucket]} {active}".strip(),
            data_outcome_num=str(num),
            onclick=f"Shiny.setInputValue('selected_outcome_num', '{num}', {{priority: 'event'}})",
            title=f"Click to inspect Outcome #{num}",
        )


    @output
    @render.ui
    def executive_grade_card():
        state = get_current_view_state()
        if not state:
            return ui.card(
                ui.div(
                    ui.span("No case selected", class_="verdict-empty-title"),
                    ui.span("Select a run file and case in the sidebar to begin inspection."),
                    class_="verdict-empty m-3",
                ),
                class_="verdict-card",
            )

        # The harness status (experiments/grading.py) decides the verdict. The
        # card never re-derives it from `present` or the features.
        status = state.get("status") or "pending"
        pending = status == "pending"
        details = grade_details(state.get("grade_result"))
        grade_val, grades = details["grade"], details["grades"]
        badge_class, badge_label = grade_badge(status, grade_val, grades)
        tone = "pending" if pending else badge_class.removeprefix("badge-")

        num = str(state.get("outcome_num") or "")
        meta = outcome_meta(num) if num else {}
        name = state.get("outcome_name") or meta.get("name") or f"Outcome {num}"
        meta_chips = [
            ui.span(ui.span(key, class_="verdict-meta-key"), value, class_="verdict-meta-chip")
            for key, value in (("System", meta.get("organ_system")), ("Acuity", meta.get("acuity")))
            if value
        ]

        grade_tile = ui.div(
            ui.span("SCOGS grade", class_="verdict-field-label"),
            ui.div("NOT GRADED" if pending else badge_label, class_="verdict-grade-value"),
            ui.p(VERDICT_STATUS_TEXT.get(status, ""), class_="verdict-grade-caption mb-0"),
            _severity_scale(status, grade_val, grades),
            class_=f"verdict-grade-tile verdict-tone-{tone}",
            role="status",
        )

        fields = []
        if details["needs_review"]:
            fields.append(ui.div(
                ui.span("Needs clinician review: ", class_="fw-semibold"),
                "the evidence does not settle a single grade.",
                class_="verdict-review",
            ))
        if pending:
            fields.append(ui.div(
                ui.span("Awaiting analysis", class_="verdict-empty-title"),
                ui.span(str(details["reason"])),
                class_="verdict-empty",
            ))
        else:
            if details["reason"]:
                fields.append(_verdict_field(
                    "Decision rationale", ui.p(str(details["reason"]), class_="verdict-rationale mb-0")))
            if details["matched"]:
                fields.append(_verdict_field(
                    "Matched rule predicate", ui.code(str(details["matched"]), class_="dsl-code-block")))
            if details["missing"]:
                fields.append(_verdict_field(
                    "Missing required features",
                    ui.div(*[ui.code(str(m), class_="verdict-missing-chip") for m in details["missing"]],
                           class_="d-flex flex-wrap gap-1"),
                    danger=True,
                ))
            if details["undecided"]:
                fields.append(_verdict_field(
                    "Undecided clauses",
                    ui.div(*[
                        ui.div(ui.span(f"Grade {grade_num}", class_="verdict-clause-grade"),
                               ui.code(str(clause), class_="dsl-code-block"),
                               class_="verdict-clause")
                        for grade_num, clause in details["undecided"]
                    ], class_="d-flex flex-column gap-2"),
                ))
            if len(fields) == int(details["needs_review"]):
                fields.append(ui.p("No rule details were recorded for this outcome.",
                                   class_="text-muted small mb-0"))

        return ui.card(
            ui.card_header(
                ui.div(
                    ui.div(
                        ui.span("DETERMINISTIC VERDICT", class_="eyebrow-tag"),
                        ui.span(f"#{num}", class_="verdict-outcome-num") if num else None,
                        ui.span(name, class_="card-header-title"),
                        class_="d-flex align-items-center flex-wrap gap-2",
                    ),
                    ui.div(*meta_chips, class_="d-flex flex-wrap gap-1") if meta_chips else None,
                    class_="d-flex justify-content-between align-items-center flex-wrap gap-2 w-100",
                )
            ),
            ui.div(
                ui.div(grade_tile, ui.div(*fields, class_="verdict-details"), class_="verdict-body"),
                class_="verdict-shell",
            ),
            class_="verdict-card",
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
                ui.span(f"Patient ID: {state.get('patient_uid')}", class_="badge-engine me-2"),
                ui.span(f"Sex: {state.get('patient_sex') or 'unknown'}", class_="badge-engine me-2"),
                ui.span(
                    f"Age: {state.get('patient_age'):.1f} yrs" if state.get("patient_age") is not None else "Age: N/A",
                    class_="badge-engine",
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
                ui.card_header(
                    ui.div(
                        ui.span("INFERENCE TELEMETRY", class_="eyebrow-tag me-2"),
                        ui.span("Model Profiling & Provenance Telemetry", class_="card-header-title"),
                        class_="d-flex align-items-center flex-wrap gap-1",
                    )
                ),
                ui.div(
                    ui.layout_columns(
                        ui.div(
                            ui.span("Model Architecture", class_="sidebar-section-label"),
                            ui.h5(prov.get("weights") or "MedGemma 27B", class_="fw-bold mb-0"),
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
                                ui.span(f"Graded: {statuses.get('graded', 0)}", class_="badge badge-grade-1 me-1"),
                                ui.span(f"Absent: {statuses.get('absent', 0)}", class_="badge badge-absent me-1"),
                                ui.span(f"Cannot: {statuses.get('cannot_grade', 0)}", class_="badge badge-cannot-grade me-1"),
                                ui.span(f"Refuted: {statuses.get('refuted', 0)}", class_="badge badge-refuted"),
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
            live_model = get_effective_model()
            status, _ = check_ollama_status(model=live_model)

            try:
                val = input.live_concurrency_input()
                conc = int(val) if val else get_live_concurrency(live_model)
            except (ValueError, TypeError):
                conc = get_live_concurrency(live_model)

            hw = detect_system_hardware()
            hw_str = f"{hw['total_ram_gb']:.0f} GB RAM • {hw['cpu_count']} Cores"
            if status == "ready":
                engine_title = f"Ollama / {live_model}"
                engine_desc = f"Real-time extraction (Installed & Ready • {conc}x Concurrency)"
                engine_color = "text-success"
            elif status == "not_installed":
                engine_title = f"Ollama / {live_model}"
                engine_desc = "Model not installed (Deterministic fallback active)"
                engine_color = "text-warning"
            else:
                engine_title = "Deterministic Engine"
                engine_desc = "Ollama offline (Deterministic fallback active)"
                engine_color = "text-secondary"

            return ui.card(
                ui.card_header(
                    ui.div(
                        ui.span("LIVE METRICS", class_="eyebrow-tag me-2"),
                        ui.span("Live Execution Telemetry", class_="card-header-title"),
                        class_="d-flex align-items-center flex-wrap gap-1",
                    )
                ),
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
                            ui.h5(engine_title, class_="fw-bold mb-0"),
                            ui.span(engine_desc, class_=f"fs-7 {engine_color} mt-1 d-block"),
                            ui.span(f"Host: {hw_str} • Context: {OLLAMA_NUM_CTX:,} tokens • Concurrency: {conc} workers", class_="fs-7 text-secondary mt-1"),
                            class_="metric-box",
                        ),
                        col_widths=[6, 6],
                    ),
                    class_="p-3",
                ),
            )
