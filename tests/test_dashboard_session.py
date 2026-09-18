"""Explore Saved Runs, driven end to end through a real reactive session.

These tests exist because the unit tests could not see the failure that mattered:
the case and outcome selectors read the very inputs they create, which cancels
their own render, so the selector never appears and every card below it goes
blank. Only a session that echoes inputs back the way a browser does shows it.
"""
import json
import re
from html import escape

from dashboard.evaluation import FOCUS_OUTCOMES
from dashboard.view_state import PENDING_REASON
from tests.shiny_session import analyze_live_note, explore, live, loop_ticks_during_analysis

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


def test_selecting_different_outcome_updates_active_pill_highlight(tmp_path):
    """Clicking different outcomes moves the outcome-pill-active highlight."""
    import asyncio
    from tests.shiny_session import FakeBrowser

    async def run_flow():
        browser = FakeBrowser()
        await browser.start(app_mode="explore")
        await browser.send_inputs(selected_run_file=mixed_status_run(tmp_path))

        # Default on outcome 28
        s28 = browser.html("patient_outcomes_summary_ui")
        assert re.search(r'class="[^"]*outcome-pill-active[^"]*"[^>]*data-outcome-num="28"', s28)

        # Clinician clicks outcome 48 (Cannot grade)
        await browser.send_inputs(selected_outcome_num="48")
        s48 = browser.html("patient_outcomes_summary_ui")
        assert re.search(r'class="[^"]*outcome-pill-active[^"]*"[^>]*data-outcome-num="48"', s48)
        assert not re.search(r'class="[^"]*outcome-pill-active[^"]*"[^>]*data-outcome-num="28"', s48)

        # Clinician clicks outcome 40 (Absent)
        await browser.send_inputs(selected_outcome_num="40")
        s40 = browser.html("patient_outcomes_summary_ui")
        assert re.search(r'class="[^"]*outcome-pill-active[^"]*"[^>]*data-outcome-num="40"', s40)
        assert not re.search(r'class="[^"]*outcome-pill-active[^"]*"[^>]*data-outcome-num="48"', s40)

        await browser.stop()

    asyncio.run(run_flow())


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


def test_live_mode_defaults_to_every_focus_outcome():
    """A custom note is graded against all 14 focus outcomes without any picking."""
    sidebar = live()["sidebar_controls"]
    picker = re.search(r'<select[^>]*id="live_outcome".*?</select>', sidebar, re.S).group(0)
    assert set(re.findall(r'<option value="(\d+)" selected=""', picker)) == set(FOCUS_OUTCOMES)


def test_live_mode_overview_waits_for_the_analysis():
    """Before the button is clicked there is nothing graded to list."""
    assert escape("Analyze & Grade Note") in live()["patient_outcomes_summary_ui"]


def test_live_analysis_grades_every_focus_outcome():
    """Clicking Analyze grades all 14 and lists them, without narrowing first."""
    out = analyze_live_note()
    summary = out["patient_outcomes_summary_ui"]
    for num, meta in FOCUS_OUTCOMES.items():
        assert meta["name"] in summary, num
    assert "14 outcomes graded" in out["outcomes_card_title"]


def test_live_analysis_lands_on_a_graded_outcome():
    """The cards below the overview open on a result, not on a pending card."""
    out = analyze_live_note()
    assert escape(PENDING_REASON, quote=False) not in out["executive_grade_card"]
    assert "GRADE" in out["executive_grade_card"]


def test_live_analysis_leaves_the_event_loop_free():
    """Grading must not run on the event loop.

    14 outcomes is minutes of blocking work. On the event loop it starves the
    session's websocket: the browser drops the connection and the finished
    results go to a closed socket ("socket.send() raised exception"), so the
    clinician waits out the whole run and sees nothing.
    """
    assert loop_ticks_during_analysis(0.3) > 0


def test_live_analysis_can_be_narrowed_to_one_outcome():
    """Deselecting is still how a clinician runs a single outcome."""
    out = analyze_live_note(live_outcome=["48"])
    summary = out["patient_outcomes_summary_ui"]
    assert "Acute Chest Syndrome (ACS)" in summary
    assert "Chronic Leg Ulcer" not in summary


def test_live_model_status_badge_installed_and_uninstalled(monkeypatch):
    from tests.shiny_session import _render
    from dashboard import evaluation

    monkeypatch.setattr(evaluation, "get_installed_models", lambda host: ["medgemma-1.5-4b-it:latest"])

    # When medgemma-1.5-4b-it is selected (default, installed)
    out_installed = _render(mode="live", then=None, inputs={})
    assert "Ollama Online" in out_installed["live_model_status_badge"]
    assert "Ready" in out_installed["live_model_status_badge"]

    # When gemma4:12b is selected (not installed)
    out_uninstalled = _render(mode="live", then={"live_model_select": "gemma4:12b"}, inputs={})
    assert "Model Not Installed" in out_uninstalled["live_model_status_badge"]
    assert "gemma4:12b" in out_uninstalled["live_model_status_badge"]
    assert "ollama pull" in out_uninstalled["live_model_status_badge"]

    # When custom model is entered
    out_custom = _render(
        mode="live",
        then={"live_model_select": "__custom__", "custom_model_input": "custom-biomed-model"},
        inputs={},
    )
    assert "custom-biomed-model" in out_custom["live_model_status_badge"]


def test_live_outcome_modes_and_directory():
    from tests.shiny_session import live

    sidebar = live()["sidebar_controls"]
    # Radio buttons for scope
    assert "live_outcome_mode" in sidebar
    assert "14 Focus Outcomes" in sidebar
    assert "All 53 SCOGS Outcomes" in sidebar
    assert "Custom Selection" in sidebar

    # Directory of all 53 tables
    assert "Directory of All 53 Possible Outcomes" in sidebar
    assert "#01" in sidebar
    assert "Arrhythmia" in sidebar
    assert "#53" in sidebar
    assert "Sleep Apnea" in sidebar


def test_live_analysis_all_53_outcomes():
    from tests.shiny_session import analyze_live_note

    out = analyze_live_note(live_outcome_mode="all_53")
    assert "53 outcomes graded" in out["outcomes_card_title"]
    summary = out["patient_outcomes_summary_ui"]
    assert "Arrhythmia" in summary
    assert "Sleep Apnea" in summary
    assert "Stroke" in summary



