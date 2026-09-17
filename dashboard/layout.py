"""Defines the Shiny dashboard UI layout and inlined static assets."""
from __future__ import annotations

from pathlib import Path
from shiny import ui

from dashboard.view_state import FILTER_LABELS, OUTCOME_BUCKETS

STATIC_DIR = Path(__file__).resolve().parent / "static"

# The overview's card shell lives here rather than in the renderer so its filter
# is a static control: a checkbox group re-created on every render would lose the
# clinician's ticks, and could not be read by the render that creates it.
# Both modes fill it: a saved case's outcomes, or the focus outcomes a live note
# was just graded against.
outcomes_overview_card = ui.card(
    ui.card_header(
        ui.div(
            ui.output_ui("outcomes_card_title"),
            ui.div(
                ui.span("SHOW", class_="sidebar-section-label me-2"),
                ui.input_checkbox_group(
                    "outcome_filter",
                    label="",
                    choices={b: FILTER_LABELS[b] for b in OUTCOME_BUCKETS},
                    selected=list(OUTCOME_BUCKETS),
                    inline=True,
                ),
                class_="d-flex align-items-center outcome-filter",
            ),
            class_="d-flex justify-content-between align-items-center flex-wrap gap-2",
        ),
    ),
    ui.output_ui("patient_outcomes_summary_ui"),
    class_="mb-3 shadow-xs",
)

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
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
        ui.tags.link(href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap", rel="stylesheet"),
        # Inlined into the page rather than served as files, so the app has no
        # static-route configuration to get wrong.
        ui.include_css(STATIC_DIR / "dashboard.css", method="inline"),
        ui.include_js(STATIC_DIR / "theme.js", method="inline"),
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
        ui.div(
            outcomes_overview_card,
            ui.output_ui("executive_grade_card"),
            ui.card(
                ui.card_header(ui.span("Clinical Note Context & Verified Spans", class_="card-header-title")),
                ui.output_ui("note_inspector_ui"),
                class_="shadow-xs",
            ),
            ui.card(
                ui.card_header(ui.span("Clinical Findings & Grounding Verification", class_="card-header-title")),
                ui.output_ui("findings_table_ui"),
                class_="shadow-xs",
            ),
            ui.output_ui("profiling_card"),
            class_="d-flex flex-column gap-4"
        ),
        class_="container-fluid py-2",
    ),
    title="SCOGS Clinical Dashboard",
)
