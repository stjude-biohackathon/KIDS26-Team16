"""Tests for clinical outcome applicability rules (rules.md)."""

from scogs.applicability import applicability
from scogs.evaluate import NOT_APPLICABLE, grade
from scogs.predicates import UNKNOWN


def test_fever_age_exclusion_0_to_59_days():
    # 59 days / 365.25 = 0.161533... years -> not applicable
    age_59d = 59.0 / 365.25
    assert applicability("36", {"patient_age": age_59d}) is False
    res_59d = grade("36", {"temperature": 39.0, "patient_age": age_59d}, applicable=False)
    assert res_59d.status == NOT_APPLICABLE

    # 60 days / 365.25 = 0.164271... years -> applicable
    age_60d = 60.0 / 365.25
    assert applicability("36", {"patient_age": age_60d}) is True
    res_60d = grade("36", {"temperature": 39.0, "patient_age": age_60d}, applicable=True)
    assert res_60d.grade == 2


def test_sex_restricted_outcomes():
    # Male + outcome 44 (pregnancy/postpartum cardiomyopathy) -> not applicable
    assert applicability("44", {"patient_sex": "male"}) is False
    res_male_44 = grade("44", {}, applicable=False)
    assert res_male_44.status == NOT_APPLICABLE

    # Female + outcome 44 -> applicable
    assert applicability("44", {"patient_sex": "female"}) is True

    # Male + outcome 24 (priapism) -> applicable
    assert applicability("24", {"patient_sex": "male"}) is True
    # Female + outcome 24 -> not applicable
    assert applicability("24", {"patient_sex": "female"}) is False


def test_sex_unknown_grades_normally():
    # Outcome 22 (female ovarian dysfunction) with sex unknown
    ok = applicability("22", {})
    assert ok is UNKNOWN
    # Per D5: unknown applicability grades normally (applicable=True)
    res = grade("22", {"ovarian_reserve_state": "diminished"}, applicable=(ok is not False))
    assert res.grade == 1


def test_unrestricted_outcomes_default_to_applicable():
    assert applicability("10", {}) is True
    assert applicability("48", {"patient_sex": "female"}) is True
