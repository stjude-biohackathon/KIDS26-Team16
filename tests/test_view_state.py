"""dashboard/view_state.py: what the cards render, for saved runs and live notes."""
import json
import pathlib

from dashboard.evaluation import evaluate_clinical_features
from dashboard.view_state import (
    ABSENT,
    CANNOT_GRADE,
    PENDING_REASON,
    PRESENT,
    explore_view_state,
    grade_details,
    live_view_state,
    outcome_bucket,
    outcome_rank,
    outcome_status,
)

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


# --- The three states the outcome overview groups and filters by -------------

def outcome(status=None, present=False, findings=0, grade=None):
    result = {"status": status, "grade": grade} if status else {}
    return {"present": present, "grade_result": result,
            "accepted_findings": [{"feature": f"f{i}"} for i in range(findings)]}


def test_a_graded_outcome_is_present():
    assert outcome_bucket(outcome("graded", present=True, grade=3.0)) == PRESENT
    assert outcome_bucket(outcome("grade_set", present=True)) == PRESENT


def test_a_detected_outcome_the_rules_could_not_grade_is_its_own_category():
    assert outcome_bucket(outcome("cannot_grade", present=True)) == CANNOT_GRADE


def test_an_outcome_flagged_present_without_a_status_counts_as_present():
    """Older results files carry `present` but no grade_result."""
    assert outcome_bucket(outcome(present=True)) == PRESENT


def test_everything_else_is_ruled_out():
    assert outcome_bucket(outcome("absent")) == ABSENT
    assert outcome_bucket(outcome()) == ABSENT
    assert outcome_bucket(outcome("refuted")) == ABSENT


def test_outcome_status_falls_back_to_the_presence_flag():
    assert outcome_status(outcome(present=True)) == "graded"
    assert outcome_status(outcome(present=False)) == "absent"
    assert outcome_status(outcome("cannot_grade", present=True)) == "cannot_grade"


def test_the_most_informative_outcome_sorts_first():
    """Landing on a case should show a grade, not the first key in the file."""
    outcomes = {
        "40": outcome("absent"),
        "48": outcome("cannot_grade", present=True, findings=1),
        "28": outcome("graded", present=True, findings=2, grade=3.0),
        "15": outcome(present=True),
    }
    assert [k for k, _ in sorted(outcomes.items(), key=outcome_rank)] == ["28", "48", "15", "40"]
