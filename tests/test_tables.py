"""Every declared grade is reachable, and the traps behave.

Reachability is proved by search rather than by 217 hand-written fixtures: for
each outcome the candidate space is built from its own predicates - every enum
value, and every numeric literal plus the points just either side of it, which is
where band boundaries fail. If a grade never appears across that space, the row
is masked by an earlier one and the table is wrong.
"""
import itertools
import random

import pytest

from scogs.evaluate import ABSENT, CANNOT_GRADE, GRADE_SET, GRADED, grade
from scogs.features import FEATURES
from scogs.predicates import Between, Cmp, In, parse
from scogs.tables import TABLES

SEED = 20260828
MAX_COMBOS = 60_000


def _nodes(node):
    yield node
    if hasattr(node, "child"): yield from _nodes(node.child)
    if hasattr(node, "children"):
        for c in node.children: yield from _nodes(c)


def candidate_space(num):
    """feature -> tuple of candidate values, drawn from the outcome's own rows."""
    table = TABLES[num]
    literals: dict[str, set] = {}
    names: set[str] = set()
    for _, pred in table.all_rows():
        for n in _nodes(parse(pred)):
            if not hasattr(n, "name"): continue
            names.add(n.name)
            bucket = literals.setdefault(n.name, set())
            if isinstance(n, Cmp) and isinstance(n.operand, float): bucket.add(n.operand)
            elif isinstance(n, Between): bucket.update({n.lo, n.hi})
            elif isinstance(n, In):
                bucket.update(o for o in n.options if isinstance(o, float))

    space = {}
    for name in sorted(names):
        spec = FEATURES[name]
        if spec["type"] == "bool":
            space[name] = (True, False)
        elif spec["type"] in {"ord", "cat"}:
            space[name] = tuple(spec["values"])
        else:
            pts = sorted(literals.get(name) or {0.0})
            vals = set()
            for p in pts:
                vals.update({p - 0.1, p, p + 0.1})
            vals.add(min(pts) - 10)
            vals.add(max(pts) + 10)
            space[name] = tuple(sorted(vals))
    return space


def assignments(num):
    space = candidate_space(num)
    keys = sorted(space)
    sizes = [len(space[k]) for k in keys]
    total = 1
    for s in sizes: total *= s
    if total <= MAX_COMBOS:
        for combo in itertools.product(*(space[k] for k in keys)):
            yield dict(zip(keys, combo))
    else:
        rng = random.Random(SEED)
        for _ in range(MAX_COMBOS):
            yield {k: rng.choice(space[k]) for k in keys}


def reachable(num, definite_only):
    table = TABLES[num]
    seen = set()
    strata = ([None] if table.eval != "stratified"
              else [{table.on: s} for s in table.strata])
    for extra in strata:
        for env in assignments(num):
            if extra: env = {**env, **extra}
            r = grade(num, env)
            if r.status == GRADED: seen.add(r.grade)
            elif r.status == GRADE_SET and not definite_only: seen.update(r.grades)
    return seen


# A grade the rubric declares but that no fully-documented patient can ever be
# assigned, because a higher cell subsumes it. Not an implementation choice -
# these are defects in the rubric, tracked here so they cannot regress silently
# and so the list can go to a clinical reviewer.
UNREACHABLE_BY_RUBRIC = {
    ("30", 1): "Grade 1 is 'not requiring intervention' and Grade 2 is 'decline "
               "< 20% AND not requiring intervention', so every Grade 1 patient "
               "also satisfies Grade 2. Grade 1 survives only as a member of a "
               "grade set when the haemoglobin decline is undocumented.",
}


@pytest.mark.parametrize("num", sorted(TABLES))
def test_every_declared_grade_is_reachable(num):
    expected_dead = {g for (n, g) in UNREACHABLE_BY_RUBRIC if n == num}
    missing = TABLES[num].grades() - reachable(num, definite_only=True)
    assert missing == expected_dead, (
        f"outcome {num} {TABLES[num].name}: grade(s) {sorted(missing)} cannot be "
        f"produced for any fully-documented patient; expected {sorted(expected_dead)}")


@pytest.mark.parametrize("num", sorted(TABLES))
def test_no_grade_outside_the_table_is_ever_produced(num):
    extra = sorted(reachable(num, definite_only=False) - TABLES[num].grades())
    assert not extra, f"outcome {num}: produced N/A grade(s) {extra}"


def test_the_unreachable_row_is_not_dead_code():
    """Outcome 30 Grade 1 is still emitted - as a candidate, when the decline is
    not documented. If it were removed the grade set would lose a real option."""
    r = grade("30", dict(hemolysis_intervention=False, death_attributed=False,
                         vasopressors=False, renal_replacement=False,
                         life_support_other=False, resp_support="room_air"))
    assert r.status == GRADE_SET and 1 in r.grades
    assert "hb_decline_pct" in r.missing


# ------------------------------------------------------------- the ordering traps

def test_19_aki_grade4_needs_esrd_and_is_not_shadowed_by_grade3():
    base = dict(creatinine_x_baseline=3.5, creatinine=2.0, renal_replacement=False,
                patient_age=30, egfr=50, death_attributed=False,
                creatinine_increase_mg_dl=0.0)
    assert grade("19", {**base, "esrd_progression": True}).grade == 4
    assert grade("19", {**base, "esrd_progression": False}).grade == 3


def test_34_iron_overload_grade4_is_tested_before_grade3_on_shared_lic():
    """LIC >= 15 triggers both cells; ferritin > 10,000 is what separates them."""
    base = dict(liver_iron_conc=20, mri_t2star=25, organ_dysfunction_iron=False,
                death_attributed=False)
    assert grade("34", {**base, "ferritin": 12000}).grade == 4
    assert grade("34", {**base, "ferritin": 8000}).grade == 3


def test_48_acs_grade4_is_grade3_plus_critical_support():
    base = dict(fio2_pct=60, resp_support="low_flow_o2", transfusion_type="simple",
                death_attributed=False, acs_other_support=False,
                renal_replacement=False, life_support_other=False)
    assert grade("48", {**base, "vasopressors": True}).grade == 4
    assert grade("48", {**base, "vasopressors": False}).grade == 3


@pytest.mark.parametrize("num,high,low", [("19", 4, 3), ("34", 4, 3), ("48", 4, 3)])
def test_shared_trigger_cells_would_under_grade_if_evaluated_upward(num, high, low):
    rows = TABLES[num].rows
    order = [g for g, _ in rows]
    assert order.index(high) < order.index(low)


# --------------------------------------------------------------------- max_of (52)

def test_52_takes_the_highest_grade_across_disagreeing_axes():
    """The rubric's own worked example: mPAP says Grade 2, NYHA says Grade 3."""
    r = grade("52", dict(mpap=28, nyha_class="3", right_heart_failure=False,
                         low_cardiac_output=False, resp_support="room_air",
                         death_attributed=False))
    assert r.status == GRADED and r.grade == 3


def test_52_would_return_nothing_under_first_match():
    """mPAP 28 with NYHA IV matches no single conjoined cell; max_of still grades."""
    r = grade("52", dict(mpap=28, nyha_class="4", right_heart_failure=False,
                         low_cardiac_output=False, resp_support="room_air",
                         death_attributed=False))
    assert r.grade == 4


def test_52_death_axis_wins():
    r = grade("52", dict(mpap=22, nyha_class="1", right_heart_failure=False,
                         low_cardiac_output=False, resp_support="room_air",
                         death_attributed=True))
    assert r.grade == 5


# ----------------------------------------------------------- stratified outcomes

def test_53_same_ahi_grades_differently_by_age():
    ped = grade("53", dict(ahi=20, spo2_desat_over_3min=True, patient_age=9))
    adult = grade("53", dict(ahi=20, spo2_desat_over_3min=True, patient_age=40,
                             sleep_apnea_treatment_indicated=True))
    assert (ped.grade, adult.grade) == (4, 3)


def test_42_uses_t_score_in_adults_and_z_score_in_children():
    adult = grade("42", dict(patient_age=40, bmd_t_score=-1.5, low_bmd_on_imaging=False,
                             significant_fracture_history=False, height_loss_cm=0,
                             care_setting="home", adl_limitation="none",
                             bmd_therapy_indicated=False))
    ped = grade("42", dict(patient_age=10, bmd_z_score=-2.5, low_bmd_on_imaging=True,
                           significant_fracture_history=False,
                           adl_limitation="none", bmd_therapy_indicated=False))
    assert adult.grade == 1 and ped.grade == 1


def test_42_fracture_history_separates_grade1_from_grade2():
    """The conjunct the earlier draft dropped: without it the two are identical."""
    base = dict(patient_age=10, bmd_z_score=-2.5, low_bmd_on_imaging=True,
                adl_limitation="none", bmd_therapy_indicated=False)
    assert grade("42", {**base, "significant_fracture_history": False}).grade == 1
    assert grade("42", {**base, "significant_fracture_history": True}).grade == 2


def test_26_stratum_is_data_availability_not_age():
    single = grade("26", dict(has_serial_height=False, height_for_age_z=-3.5,
                              death_attributed=False))
    serial = grade("26", dict(has_serial_height=True, height_z_decline=2,
                              death_attributed=False))
    assert (single.grade, serial.grade) == (4, 3)


@pytest.mark.parametrize("num", ["42", "53"])
def test_age_stratified_outcomes_say_cannot_grade_rather_than_null(num):
    r = grade(num, dict(ahi=20, spo2_desat_over_3min=True, bmd_t_score=-3,
                        significant_fracture_history=False))
    assert r.status == CANNOT_GRADE
    assert r.reason == "cannot grade: age unknown"
    assert r.grade is None


def test_26_missing_stratum_names_its_own_prerequisite():
    r = grade("26", dict(height_for_age_z=-3.5))
    assert r.status == CANNOT_GRADE and "has_serial_height" in r.reason


# ------------------------------------------------ absent / cannot-determine / set

def test_absent_when_every_rule_is_decided_and_none_match():
    r = grade("36", dict(temperature=37.0))
    assert r.status == ABSENT and r.grade is None


def test_missing_evidence_yields_a_grade_set_not_a_silent_downgrade():
    r = grade("48", dict(fio2_pct=60, resp_support="low_flow_o2",
                         transfusion_type="simple"))
    assert r.status == GRADE_SET
    assert r.grades == (5, 4, 3)
    assert "death_attributed" in r.missing and "life_support" in r.missing


def test_grade_set_carries_the_clause_that_could_not_be_decided():
    r = grade("48", dict(fio2_pct=60, resp_support="low_flow_o2",
                         transfusion_type="simple", death_attributed=False))
    assert r.grades == (4, 3)
    assert any(g == 4 for g, _ in r.undecided)
    assert r.needs_review


def test_an_undocumented_finding_is_not_read_as_negative():
    """`not treated` must not become true just because nobody wrote 'treated'."""
    r = grade("35", dict(hb_nadir=6))
    assert r.status != GRADED
    assert "treated" in r.missing


def test_a_definite_grade_reports_the_rubric_clause_that_fired():
    r = grade("12", dict(tcd_velocity=205))
    assert r.status == GRADED and r.grade == 3
    assert r.matched == "tcd_velocity >= 200"
    assert not r.needs_review


def test_not_applicable_is_distinct_from_absent():
    r = grade("45", {}, applicable=False)
    assert r.status == "not_applicable"
    assert grade("45", dict(fetal_loss=False)).status == ABSENT


# ----------------------------------------- booklet audit fixes (2026-08-28, P2/P3)

def test_31_grade1_requires_the_isolated_line_to_be_platelets():
    """Booklet Grade 1 is 'isolated THROMBOCYTOPENIA', not any single cytopenia."""
    base = dict(cytopenia_count="1", splenectomy=False)
    assert grade("31", {**base, "thrombocytopenia": True}).grade == 1
    assert grade("31", {**base, "thrombocytopenia": False}).status == ABSENT


def test_23_grade1_needs_documented_infertility_not_just_normal_labs():
    base = dict(semen_category="normal", fsh=5.0, fertility_intervention_indicated=False)
    assert grade("23", {**base, "infertility": True}).grade == 1
    assert grade("23", {**base, "infertility": False}).status == ABSENT
    r = grade("23", base)                 # infertility undocumented -> not definite
    assert r.status == CANNOT_GRADE and 1 in r.grades and "infertility" in r.missing


def test_39_joint_replacement_with_self_care_limitation_is_grade4():
    """`== instrumental_adl` would leave the worst-limited patient matching no cell."""
    r = grade("39", dict(avn_on_imaging=True, adl_limitation="self_care_adl",
                         avn_intervention="joint_replacement", symptomatic=True))
    assert r.grade == 4


def test_39_adl_clause_is_conjoined_in_grade3_as_in_grade2():
    """Intervention documented but ADL status not: Grade 3 stays a candidate rather
    than firing definitively without its ADL conjunct."""
    r = grade("39", dict(avn_on_imaging=True, avn_intervention="conservative",
                         symptomatic=True))
    assert r.status == CANNOT_GRADE and 3 in r.grades and "adl_limitation" in r.missing


def test_42_pediatric_grade1_reads_with_as_a_conjunction():
    """'Radiologic evidence of low BMD WITH z-score <= -2.0' - both, not either."""
    base = dict(patient_age=10, significant_fracture_history=False,
                adl_limitation="none", bmd_therapy_indicated=False)
    assert grade("42", {**base, "low_bmd_on_imaging": True, "bmd_z_score": -2.5}).grade == 1
    assert grade("42", {**base, "low_bmd_on_imaging": False, "bmd_z_score": -2.5}).status == ABSENT
    r = grade("42", {**base, "low_bmd_on_imaging": True})   # z-score undocumented
    assert r.status == CANNOT_GRADE and 1 in r.grades


def test_51_grade4_requires_documented_symptoms():
    base = dict(care_setting="icu", cardiac_compromise=True, hemodynamic_instability=True,
                vasopressors=True, renal_replacement=False, life_support_other=False,
                resp_support="invasive_ventilation", death_attributed=False,
                incidental_finding=False)
    assert grade("51", {**base, "symptomatic": True}).grade == 4
    r = grade("51", base)                 # symptoms undocumented -> not definite
    assert r.status == CANNOT_GRADE and 4 in r.grades and "symptomatic" in r.missing


def test_40_wound_of_exactly_8_cm2_falls_in_the_documented_gap():
    base = dict(wound_depth="partial_thickness", exudate="minimal",
                periwound_status="intact")
    assert grade("40", {**base, "wound_area_cm2": 7.9}).grade == 2
    assert grade("40", {**base, "wound_area_cm2": 8.0}).status == ABSENT


# --------------------------------------------------------- rubric gaps, recorded

def test_36_temperature_385_is_a_fever_the_rubric_gap_is_closed():
    """The rubric leaves (38.4, 38.5] matching no grade, so 38.5 degC - a fever -
    graded as absent. Closed to '>= 38.5': the rubric annotates that bound as
    "(101.3 degrees Fahrenheit)", which is exactly 38.5 C, and a strict bound is not
    annotated with the value it excludes. Two pairs in the 2026-09-01 A100 run landed
    on exactly 38.5 and were reported as refuted, which is how this surfaced."""
    assert grade("36", dict(temperature=38.4)).grade == 1
    assert grade("36", dict(temperature=38.5)).grade == 2
    assert grade("36", dict(temperature=38.6)).grade == 2
    assert grade("36", dict(temperature=38.45)).grade == 1   # nothing falls through now
    assert grade("36", dict(temperature=37.9)).status == ABSENT


def test_36_the_rubrics_own_fahrenheit_bands_grade_correctly():
    """The Celsius text is a rounding of the Fahrenheit bands, which are contiguous:
    Grade 1 "(100.4 - 101.2 F)" is 38.0000-38.4444 C, Grade 2 "(101.3 F)" is exactly
    38.5000 C. Under the literal "<= 38.4" a note reporting 101.2 F - the rubric's own
    Grade 1 endpoint - converts to 38.4444 and graded as absent. The extraction harness
    converts Fahrenheit quotes, so this is reachable, not theoretical."""
    f_to_c = lambda f: round((f - 32.0) * 5.0 / 9.0, 4)
    assert grade("36", dict(temperature=f_to_c(100.4))).grade == 1   # 38.0
    assert grade("36", dict(temperature=f_to_c(101.2))).grade == 1   # 38.4444
    assert grade("36", dict(temperature=f_to_c(101.3))).grade == 2   # 38.5
    assert grade("36", dict(temperature=f_to_c(100.3))).status == ABSENT


def test_10_pain_hurt_score_60_falls_in_the_documented_gap():
    # PROMIS scores held in their Grade 1 bands so the Pain-and-Hurt axis decides
    common = dict(unplanned_visits_12mo=0, promis_interference_t=40,
                  promis_behavior_t=30, pro_severe_count=0)
    assert grade("10", {**common, "pain_hurt_score": 59}).grade == 3
    assert grade("10", {**common, "pain_hurt_score": 61}).grade == 2
    # 60 matches neither band, so only the visit-count route decides it
    r = grade("10", {**common, "pain_hurt_score": 60})
    assert r.grade == 1


@pytest.mark.parametrize("num", ["10", "17", "23", "29", "31", "36", "39", "40", "42", "47", "52"])
def test_documented_ambiguities_are_recorded_on_the_table(num):
    assert TABLES[num].notes, f"outcome {num} resolves a rubric ambiguity silently"


# ------------------------------------------- presence is the caller's to supply

def test_presence_gate_prevents_cross_outcome_bleed():
    """Most Grade 4 cells encode only 'resulting in a life-threatening
    complication...', so a patient on pressors for ACS would otherwise be graded
    4 for papillary necrosis as well. Layer 4 grades what it is told is present."""
    on_pressors = dict(vasopressors=True, renal_replacement=False,
                       life_support_other=False, resp_support="low_flow_o2",
                       death_attributed=False)
    assert grade("20", on_pressors).grade == 4
    assert grade("20", on_pressors, present=False).status == ABSENT


def test_absent_and_not_applicable_carry_different_reasons():
    a = grade("45", {}, present=False)
    n = grade("45", {}, applicable=False)
    assert (a.status, n.status) == (ABSENT, "not_applicable")
    assert a.reason != n.reason
