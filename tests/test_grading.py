"""grade_outcome: where a verified feature set becomes a grade status.

The CLI's results files and the dashboard both call it, so behaviour pinned here
holds in both places.
"""
from experiments.grading import OutcomeGrade, grade_outcome
from scogs.evaluate import GradeResult
from scogs.predicates import UNKNOWN

VOC_INPATIENT = {"care_setting": "inpatient", "pain_co_complication": False,
                 "death_attributed": False, "life_support": False}
AKI_TRIPLED_CREATININE = {"creatinine": 3.0, "creatinine_baseline": 1.0,
                          "creatinine_increase_mg_dl": 2.0, "death_attributed": False,
                          "renal_replacement_therapy": False, "renal_replacement": False,
                          "esrd": False, "esrd_progression": False, "patient_age": 25.0}


def test_a_decided_rule_is_graded():
    graded = grade_outcome("28", VOC_INPATIENT, True)
    assert isinstance(graded, OutcomeGrade) and isinstance(graded.result, GradeResult)
    assert (graded.status, graded.result.grade) == ("graded", 3)
    assert graded.grade_if_present is None


def test_a_sex_restriction_needs_the_schema_value():
    # rules.md §24: priapism applies to male patients only.
    assert grade_outcome("24", {"patient_sex": "female"}, True).status == "not_applicable"
    assert grade_outcome("24", {"patient_sex": "male"}, True).status == "cannot_grade"


def test_unstated_sex_leaves_applicability_unknown_not_false():
    graded = grade_outcome("24", {}, True)
    assert graded.applicability is UNKNOWN
    assert graded.status == "cannot_grade"


def test_met_criteria_called_absent_are_flagged_with_the_grade_they_would_get():
    graded = grade_outcome("19", AKI_TRIPLED_CREATININE, False)
    assert (graded.status, graded.result.status) == ("missed_presence", "absent")
    assert graded.grade_if_present == 3


def test_the_record_keeps_the_results_file_layout():
    record = grade_outcome("28", VOC_INPATIENT, True).to_record(VOC_INPATIENT, True)
    assert list(record) == ["status", "rule_status", "grade", "features", "present",
                            "applicability", "criteria_met", "grade_if_present", "reason"]
    assert record["applicability"] is True
    assert record["criteria_met"] == "unknown"
