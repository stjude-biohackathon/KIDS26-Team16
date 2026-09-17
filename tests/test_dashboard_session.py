"""Explore Saved Runs, driven end to end through a real reactive session.

These tests exist because the unit tests could not see the failure that mattered:
the case and outcome selectors read the very inputs they create, which cancels
their own render, so the selector never appears and every card below it goes
blank. Only a session that echoes inputs back the way a browser does shows it.
"""
import json
from html import escape

from dashboard.view_state import PENDING_REASON
from tests.shiny_session import explore, live

FIXTURE = "tests/fixtures/sample_run.json"


def mixed_status_run(tmp_path) -> str:
    """-> a run file whose one case covers all three states a clinician filters by."""
    run = {
        "detailed_records": [
            {
                "patient_uid": "MIXED-1",
                "title": "Mixed status case",
                "patient_note": "Patient presented with severe chest pain.",
                "gender": "female",
                "age": [[31.0, "year"]],
                "outcomes": {
                    "28": {
                        "outcome_name": "Acute Sickle Cell Pain Episode (VOC)",
                        "present": True,
                        "accepted_findings": [
                            {"feature": "care_setting", "value": "inpatient", "quote": "severe chest pain"}
                        ],
                        "grade_result": {"status": "graded", "grade": 3.0, "reason": "inpatient admission"},
                    },
                    "48": {
                        "outcome_name": "Acute Chest Syndrome (ACS)",
                        "present": True,
                        "accepted_findings": [],
                        "grade_result": {"status": "cannot_grade", "grade": None, "missing": ["hypoxia"]},
                    },
                    "40": {
                        "outcome_name": "Chronic Leg Ulcer",
                        "present": False,
                        "accepted_findings": [],
                        "grade_result": {"status": "absent", "grade": None},
                    },
                },
            }
        ]
    }
    path = tmp_path / "mixed_run.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    return str(path)


def test_explore_renders_the_case_selector():
    """The case dropdown must appear; it reads the input it creates."""
    out = explore(FIXTURE)
    assert 'id="selected_patient_uid"' in out["run_case_selector_ui"]
    assert "5793438-1" in out["run_case_selector_ui"]


def test_explore_renders_the_outcome_selector():
    """The outcome dropdown must appear; it reads the input it creates."""
    out = explore(FIXTURE)
    assert 'id="selected_outcome_num"' in out["run_outcome_selector_ui"]


def test_explore_grades_the_first_case_without_any_clicking():
    """Landing on the page is enough to see a grade - no selection required."""
    out = explore(FIXTURE)
    assert "GRADE" in out["executive_grade_card"]
    assert "Select a run file" not in out["executive_grade_card"]


def test_explore_shows_the_note_and_its_findings():
    out = explore(FIXTURE)
    assert "5793438-1" in out["note_inspector_ui"]
    assert "No case selected" not in out["findings_table_ui"]


def test_explore_lists_every_outcome_of_the_case_by_state(tmp_path):
    """The overview answers "what has this patient got?" before any drilling in."""
    out = explore(mixed_status_run(tmp_path))
    summary = out["patient_outcomes_summary_ui"]
    assert "Acute Sickle Cell Pain Episode (VOC)" in summary
    assert "Acute Chest Syndrome (ACS)" in summary
    assert "Chronic Leg Ulcer" in summary


def test_filter_can_hide_every_state_but_present(tmp_path):
    out = explore(mixed_status_run(tmp_path), outcome_filter=["present"])
    summary = out["patient_outcomes_summary_ui"]
    assert "Acute Sickle Cell Pain Episode (VOC)" in summary
    assert "Acute Chest Syndrome (ACS)" not in summary
    assert "Chronic Leg Ulcer" not in summary


def test_filter_can_show_only_what_could_not_be_graded(tmp_path):
    out = explore(mixed_status_run(tmp_path), outcome_filter=["cannot_grade"])
    summary = out["patient_outcomes_summary_ui"]
    assert "Acute Chest Syndrome (ACS)" in summary
    assert "Acute Sickle Cell Pain Episode (VOC)" not in summary
    assert "Chronic Leg Ulcer" not in summary


def test_filter_can_show_only_the_outcomes_ruled_out(tmp_path):
    out = explore(mixed_status_run(tmp_path), outcome_filter=["absent"])
    summary = out["patient_outcomes_summary_ui"]
    assert "Chronic Leg Ulcer" in summary
    assert "Acute Sickle Cell Pain Episode (VOC)" not in summary


def test_clearing_every_filter_explains_the_empty_overview(tmp_path):
    out = explore(mixed_status_run(tmp_path), outcome_filter=[])
    assert "Tick a category" in out["patient_outcomes_summary_ui"]


def test_filtering_the_overview_leaves_the_selected_outcome_on_screen(tmp_path):
    """Hiding a category is a view filter, not a change of case."""
    out = explore(mixed_status_run(tmp_path), outcome_filter=["absent"])
    assert "GRADE 3" in out["executive_grade_card"]


def test_explore_never_reads_the_clinical_notes_csv(monkeypatch):
    """data/clinical_notes.csv is 26 MB and only the live evaluator needs it.

    Loading it eagerly delayed every session start, including sessions that only
    ever look at saved runs.
    """
    import dashboard.server

    calls = []
    real = dashboard.server.load_csv_notes
    monkeypatch.setattr(dashboard.server, "load_csv_notes",
                        lambda *a, **k: calls.append(a) or real("tests/fixtures/no_such.csv"))
    explore(FIXTURE)
    assert calls == []


def test_live_mode_renders_its_cards_once_the_sidebar_reports_its_inputs():
    """The grade card reads inputs the sidebar creates, so it must survive the gap."""
    out = live()
    assert escape(PENDING_REASON, quote=False) in out["executive_grade_card"]


def test_live_mode_hides_the_saved_run_overview():
    """The overview describes a saved case; a live note has no case to overview."""
    assert live()["patient_outcomes_summary_ui"].strip() in ("", "<div></div>")
