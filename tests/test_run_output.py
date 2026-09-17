"""run_output: the pure parts of a run's report (grading summary, consistency, quality bands)."""
from collections import Counter

from experiments.run_output import measure_consistency, quality_band, summarize_grades
from experiments.verification import Tally

VOC_INPATIENT = {"care_setting": "inpatient", "pain_co_complication": False,
                 "death_attributed": False, "life_support": False}
AKI_TRIPLED_CREATININE = {"creatinine": 3.0, "creatinine_baseline": 1.0,
                          "creatinine_increase_mg_dl": 2.0, "death_attributed": False,
                          "renal_replacement_therapy": False, "renal_replacement": False,
                          "esrd": False, "esrd_progression": False, "patient_age": 25.0}


def test_quality_band_edges():
    assert quality_band(95, 95, 85) == "GOOD (≥95%)"
    assert quality_band(85, 95, 85) == "WORKABLE (85-95%)"
    assert quality_band(84.9, 95, 85) == "CONCERNING (<85%)"


def test_consistency_compares_features_and_presence_not_reply_text():
    first = {"u1": {"28": ({"care_setting": "inpatient"}, True, "reply A", {}),
                    "48": ({}, False, "reply A", {})}}
    second = {"u1": {"28": ({"care_setting": "inpatient"}, True, "reply B", {}),
                     "48": ({}, True, "reply B", {})}}
    assert measure_consistency([first, second], ["28", "48"]) == (1, 2)


def test_each_pair_is_counted_and_recorded():
    tally = Tally()
    summary = summarize_grades({"u1": {"28": (VOC_INPATIENT, True, "", {})}},
                               ["28"], {"u1": "seeded:28"}, tally)
    assert summary.statuses == Counter({"graded": 1})
    assert summary.by_selection["seeded"] == Counter({"graded": 1})
    assert summary.records["u1"]["28"]["grade"] == 3
    assert tally.presence_contradicted == 0


def test_a_missed_presence_is_added_to_the_tally():
    tally = Tally()
    summary = summarize_grades({"u1": {"19": (AKI_TRIPLED_CREATININE, False, "", {})}},
                               ["19"], {"u1": "holdout"}, tally)
    assert summary.statuses == Counter({"missed_presence": 1})
    assert summary.records["u1"]["19"]["grade_if_present"] == 3
    assert tally.presence_contradicted == 1
