"""Reactive server logic for the SCOGS Shiny dashboard."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
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
    save_live_run_results,
    screen_outcomes_pcai,
    screened_out_item,
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


# Official SCOGS grading tables displayed in the UI for the 14 finalized outcomes.
_DISPLAY_TABLE_PATH = Path(__file__).resolve().parent / "scogs_14_display_tables.json"
try:
    SCOGS_14_DISPLAY_TABLES = json.loads(_DISPLAY_TABLE_PATH.read_text(encoding="utf-8"))
except Exception:
    SCOGS_14_DISPLAY_TABLES = {}

FOCUS_OUTCOME_LIST = [
    ("28", "Acute Sickle Cell Pain Episode"),
    ("15", "Stroke (Hemorrhagic or Ischemic)"),
    ("29", "Acute Splenic Sequestration"),
    ("48", "Acute Chest Syndrome (ACS)"),
    ("24", "Priapism"),
    ("10", "Chronic Pain"),
    ("21", "Chronic Kidney Disease (CKD)"),
    ("17", "Sickle Cell Retinopathy (SCR)"),
    ("11", "Cognitive Dysfunction"),
    ("47", "Depression"),
    ("12", "Elevated TCD Ultrasonography Velocity"),
    ("49", "Asthma Exacerbation"),
    ("39", "Avascular Necrosis of Joints (AVN)"),
    ("40", "Leg Ulcer"),
]


# ==============================================================================
# 3. Server Logic
# ==============================================================================

def server(input, output, session):
    # Reactive state
    current_run_cache = reactive.value(None)
    live_eval_result = reactive.value(None)
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
        return load_csv_notes("dashboard/SCD_summaries.csv")

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
            return ui.span(f"PCAI Online ({selected_model} • 16,384 ctx)", class_="badge badge-grade-1", style="font-size: 0.76rem;")
        elif status == "not_installed":
            return ui.span(f"Model Not Installed ({selected_model})", class_="badge badge-cannot-grade", style="font-size: 0.76rem;")
        return ui.span("PCAI Not Configured", class_="badge bg-secondary", style="font-size: 0.76rem;")

    @output
    @render.ui
    def live_model_status_badge():
        selected_model = get_effective_model()
        status, _ = check_ollama_status(model=selected_model)

        if status == "ready":
            return ui.div(
                ui.div(
                    ui.span("● PCAI Online", class_="badge badge-grade-1 me-1"),
                    ui.span("Ready", class_="badge bg-success-subtle text-success-emphasis"),
                    class_="d-flex align-items-center flex-wrap gap-1 mb-1",
                ),
                ui.div(
                    f"{selected_model} • remote PCAI inference",
                    class_="small text-muted font-monospace",
                    style="font-size: 0.72rem; word-break: break-all;",
                ),
                class_="sidebar-status-card p-2 rounded mb-2",
            )
        elif status == "not_installed":
            return ui.div(
                ui.span(f"▲ Model Not Installed: {selected_model}", class_="badge badge-cannot-grade mb-1 text-wrap text-start"),
                ui.div(
                    f"PCAI is configured, but '{selected_model}' is not installed. The selected model is not available through this PCAI configuration.",
                    class_="small text-muted mb-1",
                    style="font-size: 0.74rem; line-height: 1.35;",
                ),
                class_="sidebar-status-card p-2 rounded mb-2",
            )
        return ui.div(
            ui.div(
                ui.span("● PCAI Not Configured", class_="badge bg-secondary me-1"),
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
            ui.notification_show(f"PCAI credential detected. Selected model: {get_effective_model()}", type="message")
        else:
            ui.notification_show("PCAI_API_KEY is not set in this Shiny process.", type="warning")

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
            #
            # Presentation is intentionally minimal. The controls below preserve
            # the existing reactive/server contract but keep fixed runtime choices
            # out of the clinician-facing UI.
            outcome_choices = get_outcome_choices(grouped=True)

            return ui.div(
                # Hidden runtime inputs retained for compatibility.
                ui.div(
                    ui.input_select(
                        "live_model_select",
                        "PCAI Grading Model:",
                        choices={DEFAULT_LIVE_MODEL: "GPT-OSS 120B"},
                        selected=DEFAULT_LIVE_MODEL,
                    ),
                    ui.input_numeric(
                        "live_concurrency_input",
                        "Requests in Parallel:",
                        value=1,
                        min=1,
                        max=1,
                        step=1,
                    ),
                    ui.input_radio_buttons(
                        "live_source",
                        "Input Source:",
                        choices={"custom": "Paste Custom Note"},
                        selected="custom",
                    ),
                    ui.input_numeric(
                        "patient_age_input",
                        "Age (years):",
                        value=None,
                        min=0.0,
                        max=120.0,
                        step=0.5,
                    ),
                    ui.input_select(
                        "patient_sex_input",
                        "Sex:",
                        choices={"": "Not supplied", "unknown": "Unknown"},
                        selected="",
                    ),
                    ui.input_checkbox(
                        "outcome_present_input",
                        "Assume outcomes are clinically present",
                        value=True,
                    ),
                    ui.input_text_area(
                        "manual_features_json",
                        "Simulated Feature Values (JSON):",
                        rows=1,
                        value='{}',
                    ),
                    style="display:none;",
                ),

                # Primary clinical input.
                ui.div(
                    ui.div(
                        ui.span("CLINICAL NOTE", class_="sidebar-section-label mb-0"),
                        ui.span("GPT-OSS 120B", class_="sidebar-badge-subtle"),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    ui.input_text_area(
                        "live_note_text",
                        "",
                        rows=18,
                        placeholder="Paste the clinical note here...",
                    ),
                    ui.div(
                        ui.span("● PCAI Online", class_="badge badge-grade-1 me-1"),
                        ui.span("GPT-OSS 120B • one outcome at a time", class_="small text-muted"),
                        class_="d-flex align-items-center flex-wrap gap-1 mt-2",
                    ),
                    class_="sidebar-panel-card mb-2",
                ),

                # Fixed 14-outcome scope for the clinical dashboard.
                ui.div(
                    ui.input_radio_buttons(
                        "live_outcome_mode",
                        "Outcomes:",
                        choices={"14_focus": "14 Outcomes"},
                        selected="14_focus",
                    ),
                    ui.input_selectize(
                        "live_outcome",
                        "Choose Target Outcomes:",
                        choices={k: v for k, v in FOCUS_OUTCOME_LIST},
                        selected=[k for k, _ in FOCUS_OUTCOME_LIST],
                        multiple=True,
                    ),
                    ui.input_checkbox(
                        "fast_screen_53",
                        "Experimental Fast 53 pre-screen",
                        value=False,
                    ),
                    style="display:none;",
                ),

                ui.div(
                    ui.div(
                        ui.span("14 SCOGS OUTCOMES", class_="sidebar-section-label mb-0"),
                        ui.span("Fixed Scope", class_="sidebar-badge-subtle"),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    ui.div(
                        *[
                            ui.div(
                                ui.span(f"{i}.", class_="focus-outcome-num"),
                                ui.span(label, class_="focus-outcome-name"),
                                class_="focus-outcome-item",
                            )
                            for i, (_, label) in enumerate(FOCUS_OUTCOME_LIST, start=1)
                        ],
                        class_="focus-outcome-list",
                    ),
                    class_="sidebar-panel-card mb-2",
                ),

                ui.input_action_button(
                    "btn_analyze",
                    "Analyze & Grade Note",
                    class_="btn btn-clinical-primary w-100 mt-1 mb-2",
                ),
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
        picked = current("live_outcome")

        if mode == "all_53":
            return [normalize_outcome_id(k) for k in sorted(TABLES.keys())]
        elif mode == "custom":
            if picked is not None:
                if isinstance(picked, str):
                    picked = (picked,)
                chosen = [normalize_outcome_id(num) for num in picked if normalize_outcome_id(num) in TABLES]
                return chosen
            return [normalize_outcome_id(k) for k in FOCUS_OUTCOMES]
        else:
            # Mode "14_focus" (default)
            # If specifically overridden in a test or caller passing a custom subset in live_outcome:
            if picked is not None:
                if isinstance(picked, str):
                    picked = (picked,)
                picked_norm = [normalize_outcome_id(p) for p in picked if normalize_outcome_id(p) in TABLES]
                focus_norm = [normalize_outcome_id(f) for f in FOCUS_OUTCOMES]
                if set(picked_norm) != set(focus_norm) and len(picked_norm) > 0:
                    return picked_norm
            return [normalize_outcome_id(k) for k in FOCUS_OUTCOMES]

    def live_outcome_num(results: dict | None) -> str:
        """-> the outcome the live cards below the overview show: the pill the
        clinician clicked when it was graded, else the most informative result."""
        clicked = str(current("selected_outcome_num") or "")
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
        if not online:
            ui.notification_show(
                "PCAI is not configured. Set PCAI_API_KEY in the terminal that launched Shiny and restart the app.",
                type="error",
                duration=8,
            )
            return
        results: dict[str, Any] = {}
        # Fixed sequential grading for stability and predictable progress:
        # exactly one SCOGS outcome per PCAI request.
        user_concurrency = 1
        model_concurrency = 1
        if not outcome_ids:
            ui.notification_show("No outcomes selected. Choose at least one outcome in the sidebar.", type="warning")
            return

        # Optional high-recall pre-screen for the 53-outcome mode. This can cut
        # the number of expensive detailed GPT-OSS calls substantially, but we
        # do not call screened-out outcomes absent: they remain CANNOT_GRADE
        # unless they receive a full outcome-level evaluation.
        outcome_ids_to_grade = list(outcome_ids)
        if current("live_outcome_mode", "14_focus") == "all_53" and bool(current("fast_screen_53", False)):
            ui.notification_show(
                "Running high-recall GPT-OSS screen across 53 outcomes…",
                type="message",
                duration=4,
            )
            try:
                screened = await asyncio.to_thread(
                    screen_outcomes_pcai,
                    note_text,
                    outcome_ids,
                    selected_model,
                )
                candidates = [num for num in outcome_ids if num in set(screened)]
                skipped = [num for num in outcome_ids if num not in set(candidates)]
                results.update({num: screened_out_item(num) for num in skipped})
                outcome_ids_to_grade = candidates
                live_eval_result.set(dict(results))
                ui.notification_show(
                    f"Fast 53 screen selected {len(candidates)} of {len(outcome_ids)} outcomes for detailed grading.",
                    type="message",
                    duration=6,
                )
            except Exception as exc:
                outcome_ids_to_grade = list(outcome_ids)
                ui.notification_show(
                    f"Fast screen failed ({exc}); falling back to exhaustive 53-outcome grading.",
                    type="warning",
                    duration=7,
                )

        with ui.Progress(min=0, max=len(outcome_ids)) as progress:
            progress.set(0, message="Grading outcomes", detail=f"0 of {len(outcome_ids)}")
            # A chunk at a time, so the overview fills in as outcomes land instead
            # of staying empty until the last one is graded.
            for start in range(0, len(outcome_ids_to_grade), model_concurrency):
                chunk = outcome_ids_to_grade[start:start + model_concurrency]
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

        # Save live run outputs into results/live
        try:
            saved_path = await asyncio.to_thread(
                save_live_run_results,
                note_text=note_text,
                outcomes=outcome_ids,
                results=results,
                model=selected_model,
                backend="pcai",
                patient_age=patient_age,
                patient_sex=patient_sex,
                num_ctx=OLLAMA_NUM_CTX,
            )
            ui.notification_show(
                f"Analysis and grading complete ({len(results)} outcomes). Saved to {saved_path}",
                type="message",
            )
        except Exception as e:
            ui.notification_show(
                f"Analysis complete ({len(results)} outcomes), but saving to results/live failed: {e}",
                type="warning",
            )

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
                ui.span(f"{graded} outcomes evaluated" if graded != 1 else "1 outcome evaluated",
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
        shown = set(current("outcome_filter", (PRESENT,)) or ())
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

        # PCAI/GPT-OSS metadata travels in extracted_features so it remains
        # compatible with saved-run JSON and the existing view-state contract.
        model_meta = state.get("extracted_features") or {}
        model_conf = model_meta.get("model_confidence_pct")
        if model_conf is not None:
            details_row.append(
                ui.div(
                    ui.span("Selected-Model Confidence", class_="sidebar-section-label mb-1"),
                    ui.span(f"{float(model_conf):.1f}% (uncalibrated)", class_="fs-7 text-secondary"),
                    class_="mb-2",
                )
            )

        conformal_set = model_meta.get("conformal_prediction_set") or []
        conformal_target = model_meta.get("conformal_target_coverage_pct")
        conformal_p = model_meta.get("conformal_predicted_class_p_value_pct")
        if conformal_set:
            c_text = f"{conformal_set}"
            if conformal_target is not None:
                c_text += f" at {conformal_target}% target coverage"
            if conformal_p is not None:
                c_text += f" • selected-class conformal p-value {conformal_p}%"
            details_row.append(
                ui.div(
                    ui.span("Conformal Prediction Set", class_="sidebar-section-label mb-1"),
                    ui.span(c_text, class_="fs-7 text-secondary"),
                    class_="mb-2",
                )
            )

        if reason:
            details_row.append(
                ui.div(
                    ui.span("Decision Rationale", class_="sidebar-section-label mb-1"),
                    ui.span(str(reason), class_="fs-7 text-secondary"),
                    class_="mb-2"
                )
            )
        if matched:
            details_row.append(
                ui.div(
                    ui.span("Matched Rule Predicate", class_="sidebar-section-label mb-1"),
                    ui.div(ui.code(str(matched), class_="dsl-code-block")),
                    class_="mb-2"
                )
            )
        if missing:
            details_row.append(
                ui.div(
                    ui.span("Missing Required Features", class_="sidebar-section-label text-danger mb-1"),
                    ui.span(", ".join(str(m) for m in missing), class_="text-danger fs-7"),
                    class_="mb-2"
                )
            )
        if undecided:
            clause_elements = []
            for grade_num, clause in undecided:
                clause_elements.append(
                    ui.div(
                        ui.span(f"Grade {grade_num}", class_="badge badge-not-applicable px-2 py-1 me-2 fw-semibold", style="font-size: 0.72rem;"),
                        ui.code(str(clause), class_="dsl-code-block fs-7"),
                        class_="mb-2 d-flex align-items-center"
                    )
                )
            details_row.append(
                ui.div(
                    ui.span("Undecided Clauses", class_="sidebar-section-label mb-2"),
                    ui.div(*clause_elements, class_="ps-2 border-start border-2 border-secondary"),
                    class_="mb-2 w-100"
                )
            )

        meta = FOCUS_OUTCOMES.get(str(state.get("outcome_num")), {})
        meta_info = f"Organ System: {meta.get('organ_system', 'General')} | Acuity: {meta.get('acuity', 'Standard')}"

        grading_table_ui = None
        table_info = SCOGS_14_DISPLAY_TABLES.get(str(state.get("outcome_num")))
        if grade_val is not None and table_info:
            grade_rows = []
            for grade_num, definition in table_info.get("grades", {}).items():
                is_selected = str(grade_num) == str(grade_val)
                row_class = "scogs-grade-row scogs-grade-row-selected" if is_selected else "scogs-grade-row"
                grade_rows.append(
                    ui.tags.tr(
                        ui.tags.td(
                            ui.span(f"Grade {grade_num}", class_="scogs-grade-label"),
                            class_="scogs-grade-col",
                        ),
                        ui.tags.td(str(definition), class_="scogs-grade-definition"),
                        class_=row_class,
                    )
                )

            grading_table_ui = ui.div(
                ui.div(
                    ui.div(
                        ui.span("OFFICIAL SCOGS GRADING TABLE", class_="sidebar-section-label mb-0"),
                        ui.span(
                            f"Selected: Grade {grade_val}",
                            class_="badge badge-grade-1",
                        ),
                        class_="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-2",
                    ),
                    ui.tags.table(
                        ui.tags.thead(
                            ui.tags.tr(
                                ui.tags.th("Grade", class_="scogs-grade-col"),
                                ui.tags.th("Criteria"),
                            )
                        ),
                        ui.tags.tbody(*grade_rows),
                        class_="table table-sm scogs-grading-table mb-1",
                    ),
                    ui.div(
                        f"SCOGS booklet • {table_info.get('title', state.get('outcome_name'))} • page {table_info.get('page')}",
                        class_="small text-muted mt-2",
                    ),
                    class_="scogs-grading-table-wrap",
                ),
                class_="mt-3",
            )

        return ui.card(
            ui.card_header(
                ui.div(
                    ui.div(
                        ui.span("PCAI MODEL SCOGS VERDICT", class_="eyebrow-tag me-2"),
                        ui.span(f"Outcome #{state.get('outcome_num')}: {state.get('outcome_name')}", class_="card-header-title"),
                        class_="d-flex align-items-center flex-wrap gap-1",
                    ),
                    ui.span(meta_info, class_="badge-engine fw-normal", style="font-size: 0.78rem;"),
                    class_="d-flex justify-content-between align-items-center flex-wrap gap-2",
                )
            ),
            ui.div(
                review_alert,
                ui.div(
                    ui.div(
                        ui.span("SCOGS Grade:", class_="text-muted text-uppercase fw-semibold fs-7 mb-1"),
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
                grading_table_ui,
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
                ui.span(f"Sex: {state.get('patient_sex') or 'unknown'}", class_="badge-engine me-2", style="display:none;"),
                ui.span(
                    f"Age: {state.get('patient_age'):.1f} yrs" if state.get("patient_age") is not None else "Age: N/A",
                    class_="badge-engine",
                    style="display:none;",
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

        # Keep live telemetry implementation below for future use, but hide it
        # from the live clinical workflow for a cleaner presentation.
        if mode == "live":
            return ui.div(style="display:none;")

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
                            ui.span(f"Host: {hw_str} • Context: 16,384 tokens • Concurrency: {conc} workers", class_="fs-7 text-secondary mt-1"),
                            class_="metric-box",
                        ),
                        col_widths=[6, 6],
                    ),
                    class_="p-3",
                ),
            )
