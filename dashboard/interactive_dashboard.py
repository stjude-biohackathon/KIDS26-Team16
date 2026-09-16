from datetime import date
import re

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui


# ============================================================
# UI - User interface
# ============================================================

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Patient Input"),
        ui.input_text(
            "patient_id",
            "Patient ID:",
            placeholder="e.g., SYN001",
        ),
        ui.input_text(
            "encounter_id",
            "Encounter ID:",
            placeholder="e.g., ENC001",
        ),
        ui.input_date(
            "encounter_date",
            "Encounter Date:",
            value=date.today(),
        ),
        ui.input_select(
            "note_type",
            "Note Type:",
            choices=[
                "",
                "admission_note",
                "progress_note",
                "discharge_note",
                "consultation_note",
            ],
            selected="",
        ),
        ui.input_text_area(
            "evidence",
            "Clinical Evidence (one per line):",
            height="200px",
            placeholder="Enter clinical evidence, one item per line...",
        ),
        ui.input_action_button(
            "analyze",
            "Analyze Patient",
            class_="btn-primary w-100",
        ),
    ),
    ui.layout_columns(
        # Diagnosis Result Card
        ui.card(
            ui.card_header("Diagnosis Result"),
            ui.card_body(
                ui.div(
                    ui.output_ui("outcome_display"),
                    style="text-align: center;",
                )
            ),
        ),
        # SCOGS Grade Card
        ui.card(
            ui.card_header("SCOGS Grade Assessment"),
            ui.card_body(
                ui.div(
                    ui.output_ui("grade_display"),
                    style="text-align: center;",
                )
            ),
        ),
        # Clinical Summary Card
        ui.card(
            ui.card_header("Clinical Summary"),
            ui.card_body(ui.output_table("clinical_summary")),
        ),
        col_widths=[6, 6, 12],
    ),
    title="Patient Diagnosis Dashboard",
)


# ============================================================
# Analysis function
# ============================================================

def analyze_patient(
    patient_id: str,
    encounter_id: str,
    encounter_date,
    note_type: str,
    evidence_list: str,
) -> dict:
    """Python equivalent of the R analyze_patient() function."""

    # Split the textarea into one evidence item per line and remove empty strings.
    evidence = evidence_list.splitlines()
    evidence = [item for item in evidence if item != ""]

    # Initial/default values.
    outcome = "Unknown"
    organ_system = "Unknown"
    frequency_classification = "Unknown"
    scogs_grade = 0
    rule_triggered = ""

    # Same behavior as:
    # tolower(paste(evidence, collapse = " "))
    evidence_lower = " ".join(evidence).lower()

    # Same branch order as the R code. Because these are if/elif branches,
    # only the first matching category is assigned.
    if re.search(r"pulmonary|infiltrate|chest|respiratory|oxygen", evidence_lower):
        outcome = "Acute Chest Syndrome"
        organ_system = "Pulmonary"
        frequency_classification = "Acute"

        if re.search(r"high-flow|ventilat|intubat", evidence_lower):
            # Preserved exactly from the R source: scogs_grade <- 30.
            # This appears likely to be a typo for Grade 3 because the rule text
            # says "Grade 3" and the color mapping below recognizes 3, not 30.
            scogs_grade = 30
            rule_triggered = "High-flow oxygen meets SCOGS Grade 3 criteria"
        elif re.search(r"oxygen", evidence_lower):
            scogs_grade = 2
            rule_triggered = "Supplemental oxygen meets SCOGS Grade 2 criteria"
        else:
            scogs_grade = 1
            rule_triggered = "Clinical symptoms meet SCOGS Grade 1 criteria"

    elif re.search(r"pain|crisis|analgesic|opioid", evidence_lower):
        outcome = "Vaso-occlusive Crisis"
        organ_system = "Vascular"
        frequency_classification = "Acute"

        if re.search(r"parenteral|iv|intravenous", evidence_lower):
            scogs_grade = 2
            rule_triggered = "Parenteral analgesics meet SCOGS Grade 2 criteria"
        else:
            scogs_grade = 1
            rule_triggered = "Pain management required - SCOGS Grade 1"

    elif re.search(r"stroke|neurologic|seizure", evidence_lower):
        outcome = "Neurological Event"
        organ_system = "Neurological"
        frequency_classification = "Acute"
        scogs_grade = 4
        rule_triggered = "Neurological event meets SCOGS Grade 4 criteria"

    return {
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "encounter_date": str(encounter_date),
        "note_type": note_type,
        "outcome": outcome,
        "organ_system": organ_system,
        "frequency_classification": frequency_classification,
        "scogs_grade": scogs_grade,
        "evidence": evidence,
        "rule_triggered": rule_triggered,
    }


# ============================================================
# Server - behavior and calculations
# ============================================================

def server(input: Inputs, output: Outputs, session: Session):
    # Equivalent of R's reactiveVal(NULL).
    analysis_results = reactive.value(None)

    # Equivalent of observeEvent(input$analyze, {...}).
    @reactive.effect
    @reactive.event(input.analyze)
    def _analyze_patient_event():
        # Equivalent of req(...): do not continue unless required fields exist.
        req(
            input.patient_id(),
            input.encounter_id(),
            input.note_type(),
            input.evidence(),
        )

        results = analyze_patient(
            input.patient_id(),
            input.encounter_id(),
            input.encounter_date(),
            input.note_type(),
            input.evidence(),
        )

        analysis_results.set(results)
        ui.notification_show("Analysis complete!", type="message")

    # Diagnosis Result display.
    @render.ui
    def outcome_display():
        results = analysis_results()
        req(results)

        return ui.div(
            ui.h3(results["outcome"], class_="text-primary"),
            ui.hr(),
            ui.h5("Organ System: ", results["organ_system"]),
            ui.h5("Classification: ", results["frequency_classification"]),
        )

    # SCOGS Grade display with Bootstrap color coding.
    @render.ui
    def grade_display():
        results = analysis_results()
        req(results)

        grade_color = {
            "1": "success",
            "2": "warning",
            "3": "danger",
            "4": "danger",
            "5": "danger",
        }.get(str(results["scogs_grade"]), "secondary")

        return ui.div(
            ui.h1(
                f'Grade {results["scogs_grade"]}',
                class_=f"text-{grade_color}",
                style="font-size: 3rem; font-weight: bold;",
            ),
            ui.hr(),
            ui.p(
                results["rule_triggered"],
                class_="text-muted",
                style="font-size: 1.1rem;",
            ),
        )

    # Clinical Summary table.
    @render.table(
        classes="table table-striped table-hover table-bordered shiny-table w-auto"
    )
    def clinical_summary():
        results = analysis_results()
        req(results)

        summary_df = pd.DataFrame(
            {
                "Item": [
                    "Patient ID",
                    "Encounter ID",
                    "Date",
                    "Note Type",
                    "Evidence Count",
                ],
                "Details": [
                    results["patient_id"],
                    results["encounter_id"],
                    results["encounter_date"],
                    results["note_type"],
                    len(results["evidence"]),
                ],
            }
        )

        evidence_df = pd.DataFrame(
            {
                "Item": [
                    f"Evidence {i}"
                    for i in range(1, len(results["evidence"]) + 1)
                ],
                "Details": results["evidence"],
            }
        )

        # Equivalent of rbind(summary_df, evidence_df).
        return pd.concat([summary_df, evidence_df], ignore_index=True)


app = App(app_ui, server)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, launch_browser=True)