"""Tests for objective presence criteria."""
from scogs.criteria import criteria_met
from scogs.predicates import UNKNOWN
from scogs.evaluate import grade, GRADED, ABSENT
from experiments.medgemma_extraction import harness_status


def test_criteria_met_evaluates_numeric_definitions():
    # 08: trv >= 2.5
    assert criteria_met("08", {"trv": 2.6}) is True
    assert criteria_met("08", {"trv": 2.4}) is False
    assert criteria_met("08", {}) is UNKNOWN

    # 12: tcd_velocity >= 170
    assert criteria_met("12", {"tcd_velocity": 175}) is True
    assert criteria_met("12", {"tcd_velocity": 160}) is False

    # 19: creatinine_x_baseline >= 1.5 or creatinine_increase_mg_dl >= 0.3
    assert criteria_met("19", {"creatinine_x_baseline": 1.6}) is True
    assert criteria_met("19", {"creatinine_increase_mg_dl": 0.4}) is True
    assert criteria_met("19", {"creatinine_x_baseline": 1.2, "creatinine_increase_mg_dl": 0.1}) is False

    # 34: liver_iron_conc > 2.5 or ferritin > 1000 or organ_dysfunction_iron
    assert criteria_met("34", {"liver_iron_conc": 3.0}) is True
    assert criteria_met("34", {"ferritin": 1500}) is True
    assert criteria_met("34", {"organ_dysfunction_iron": True}) is True
    assert criteria_met("34", {"liver_iron_conc": 2.0, "ferritin": 500, "organ_dysfunction_iron": False}) is False

    # 36: temperature >= 38.0
    assert criteria_met("36", {"temperature": 38.5}) is True
    assert criteria_met("36", {"temperature": 37.0}) is False


def test_outcome_not_in_criteria_returns_unknown():
    assert criteria_met("01", {}) is UNKNOWN
    assert criteria_met("99", {}) is UNKNOWN


def test_end_to_end_missed_presence():
    # trv=3.1, present=False -> missed_presence, grade_if_present == 4
    feats = {"trv": 3.1}
    crit = criteria_met("08", feats)
    assert crit is True
    res = grade("08", feats, present=False)
    assert res.status == ABSENT
    status = harness_status(res.status, present=False, criteria=crit is True)
    assert status == "missed_presence"
    grade_if_present = grade("08", feats, present=True).grade
    assert grade_if_present == 4

    # trv=3.1, present=True -> graded
    res_true = grade("08", feats, present=True)
    assert res_true.status == GRADED and res_true.grade == 4
    status_true = harness_status(res_true.status, present=True, criteria=crit is True)
    assert status_true == "graded"

    # temperature=37.0, present=False -> absent
    temp_feats = {"temperature": 37.0}
    temp_crit = criteria_met("36", temp_feats)
    assert temp_crit is False
    temp_res = grade("36", temp_feats, present=False)
    assert temp_res.status == ABSENT
    temp_status = harness_status(temp_res.status, present=False, criteria=temp_crit is True)
    assert temp_status == "absent"
