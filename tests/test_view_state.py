"""dashboard/view_state.py: what the cards render, for saved runs and live notes."""
import json
import pathlib

from dashboard.evaluation import evaluate_clinical_features
from dashboard.view_state import PENDING_REASON, explore_view_state, grade_details, live_view_state

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "sample_run.json"
VOC_INPATIENT = {"care_setting": "inpatient", "pain_co_complication": False,
                 "death_attributed": False, "life_support": False}


def saved_run():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {"records_by_uid": {r["patient_uid"]: r for r in data["detailed_records"]}}


def test_explore_state_reads_one_saved_case():
    state = explore_view_state(saved_run(), "5793438-1", "28")
    assert (state["status"], state["present"], state["patient_age"], state["patient_sex"]) == \
        ("graded", True, 33.0, "M")
    assert state["grade_result"]["grade"] == 3.0
    assert len(state["accepted_findings"]) == 2


def test_explore_state_is_empty_before_a_file_is_loaded():
    assert explore_view_state(None, "5793438-1", "28") == {}


def test_live_state_is_pending_until_analysed():
    state = live_view_state(None, "28", "a note", 30.0, "male")
    assert (state["status"], state["grade_result"], state["title"]) == \
        ("pending", None, "Interactive Live Case")
    assert grade_details(state["grade_result"])["reason"] == PENDING_REASON


def test_grade_details_from_a_live_result_and_a_saved_record():
    live = grade_details(evaluate_clinical_features("28", VOC_INPATIENT).result)
    assert live["grade"] == 3 and "care_setting >= inpatient" in live["matched"]
    saved = grade_details({"status": "cannot_grade", "grade": None, "reason": "no rule decided"})
    assert saved["needs_review"] is True and saved["missing"] == ()
