"""dashboard/evaluation.py: the dashboard grades through the CLI's own extraction and grading."""
import json

import pytest

from dashboard import evaluation
from dashboard.evaluation import (
    FOCUS_OUTCOMES, clinician_context, evaluate_clinical_features, extract_and_grade_note,
    extract_with_harness, grade_badge, is_derived, is_ollama_available,
)
from experiments.medgemma_extraction import DEFAULT_OUTCOMES
from scogs import GradeResult

NOTE = ("A 14-year-old with HbSS presented with chest pain. FiO2 was escalated to 60%. "
        "He received a simple transfusion of 2 units and was started on norepinephrine.")
VOC_INPATIENT = {"care_setting": "inpatient", "pain_co_complication": False,
                 "death_attributed": False, "life_support": False}
AKI_TRIPLED_CREATININE = {"creatinine": 3.0, "creatinine_baseline": 1.0,
                          "creatinine_increase_mg_dl": 2.0, "death_attributed": False,
                          "renal_replacement_therapy": False, "renal_replacement": False,
                          "esrd": False, "esrd_progression": False, "patient_age": 25.0}


# ------------------------------------------------------------ live extraction

def test_live_extraction_grades_only_verified_findings():
    # The mock backend returns two findings for one feature: value False quoted
    # from the note ("presented") and value True quoted from a sentence the note
    # does not contain. The old dashboard graded the invented True.
    item = extract_with_harness(NOTE, ["48"], backend="mock", model="mock", host="")["48"]
    assert len(json.loads(item["raw_reply"])["findings"]) == 2
    assert item["accepted_findings"] == [
        {"feature": "death_attributed", "value": False, "quote": "presented", "unit": None}]
    assert item["extracted_features"] == {"death_attributed": False}
    assert item["status"] == "cannot_grade"


def test_a_backend_failure_is_shown_on_the_card_not_raised(monkeypatch):
    def unreachable(*args, **kwargs):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(evaluation, "extract_with_harness", unreachable)
    item = extract_and_grade_note(NOTE, ["28"], use_ollama=True)["28"]
    assert item["status"] == "cannot_grade"
    assert "connection refused" in item["error"]


# ------------------------------------------------------ clinician-entered values

def test_clinician_sex_codes_reach_the_rules_as_schema_values():
    # rules.md §24: priapism applies to male patients only. "M" used to be passed
    # through unchanged, and the rules do not know "M".
    def status(sex):
        return extract_and_grade_note("", ["24"], patient_context={"patient_sex": sex},
                                      manual_features={"24": {}})["24"]["status"]
    assert status("M") == "cannot_grade"
    assert status("female") == "not_applicable"


@pytest.mark.parametrize("sex, age, expected", [
    ("M", 33, {"patient_sex": "male", "patient_age": 33.0}),
    ("female", "12.5", {"patient_sex": "female", "patient_age": 12.5}),
    ("unknown", None, {}),
    ("", "not a number", {}),
])
def test_clinician_context_uses_the_schema_vocabulary(sex, age, expected):
    assert clinician_context(sex, age) == expected


def test_manual_voc_is_graded_with_the_matched_rule():
    graded = evaluate_clinical_features("28", VOC_INPATIENT, patient_sex="M", patient_age=33.0)
    assert isinstance(graded.result, GradeResult)
    assert (graded.status, graded.result.grade) == ("graded", 3)
    assert "care_setting >= inpatient" in graded.result.matched


def test_manual_derived_features_are_resolved():
    graded = evaluate_clinical_features("19", AKI_TRIPLED_CREATININE, patient_sex="female",
                                        patient_age=25.0)
    assert (graded.status, graded.result.grade) == ("graded", 3)


def test_an_unknown_outcome_cannot_be_graded():
    graded = evaluate_clinical_features("999", {}, patient_sex="unknown")
    assert graded.status == "cannot_grade"
    assert "Unknown outcome" in graded.result.reason


# ------------------------------------------------------------------ display helpers

def test_derived_features_come_from_the_schema():
    assert is_derived("bp_stage") and is_derived("life_support")
    assert not is_derived("creatinine")
    assert not is_derived("bmi")          # not a SCOGS feature at all


@pytest.mark.parametrize("status, grade, grades, expected", [
    ("graded", 3, (), ("badge-grade-3", "GRADE 3")),
    ("graded", 1, (), ("badge-grade-1", "GRADE 1")),
    ("grade_set", None, (2, 3), ("badge-grade-set", "GRADE SET: (2, 3)")),
    ("missed_presence", None, (), ("badge-cannot-grade", "MISSED PRESENCE (REVIEW)")),
    ("refuted", None, (), ("badge-refuted", "REFUTED (CONTRADICTED)")),
    ("pending", None, (), ("badge-not-applicable", "PENDING")),
])
def test_grade_badge(status, grade, grades, expected):
    assert grade_badge(status, grade, grades) == expected


def test_focus_outcomes_are_the_cli_default_and_described():
    assert set(FOCUS_OUTCOMES) == set(DEFAULT_OUTCOMES.split(","))
    for meta in FOCUS_OUTCOMES.values():
        assert {"name", "organ_system", "acuity"} <= set(meta)


def test_is_ollama_available_offline():
    assert is_ollama_available(host="http://localhost:59999") is False
