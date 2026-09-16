"""The MedGemma extraction check's verification layer - the §2 rule that turns a hallucinated quote
into a counted rejection rather than a silent wrong answer."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts" / "experiments"))
from medgemma_extraction import (  # noqa: E402
    Tally, build_prompt, coerce, harness_status, is_scd_primary, normalize,
    outcome_seed, call_mock, reconcile, reduce_policy, scd_mentions,
    select_notes, unit_guard, verify, GGUF_FILE_TYPES,
)

NOTE = ("A 14-year-old with HbSS presented with chest pain. FiO2 was escalated to 60%. "
        "He received a simple transfusion of 2 units and was started on norepinephrine.")


def run_verify(findings, present=True, note=NOTE):
    t = Tally()
    feats, p, _ = verify(json.dumps({"present": present, "findings": findings}), note, t)
    return feats, p, t


# ------------------------------------------------------------ quote enforcement

def test_a_verbatim_quote_is_accepted():
    feats, _, t = run_verify([{"feature": "fio2_pct", "value": 60,
                               "quote": "FiO2 was escalated to 60%"}])
    assert feats == {"fio2_pct": 60.0} and t.accepted == 1 and t.quote_ok == 1


def test_a_quote_not_in_the_note_is_rejected_not_downweighted():
    feats, _, t = run_verify([{"feature": "fio2_pct", "value": 90,
                               "quote": "FiO2 was escalated to 90%"}])
    assert feats == {} and t.quote_unfound == 1 and t.accepted == 0


def test_a_finding_with_no_quote_is_rejected():
    feats, _, t = run_verify([{"feature": "fio2_pct", "value": 60}])
    assert feats == {} and t.quote_missing == 1


def test_quote_matching_tolerates_whitespace_but_not_paraphrase():
    ok, _, _ = run_verify([{"feature": "fio2_pct", "value": 60,
                            "quote": "FiO2   was\n escalated to 60%"}])
    assert ok == {"fio2_pct": 60.0}
    bad, _, t = run_verify([{"feature": "fio2_pct", "value": 60,
                             "quote": "the FiO2 was raised to 60 percent"}])
    assert bad == {} and t.quote_unfound == 1


# --------------------------------------------------------------- value coercion

@pytest.mark.parametrize("value,expected", [
    (True, True), ("true", True), ("yes", True), (False, False), ("no", False),
])
def test_bool_coercion(value, expected):
    assert coerce("death_attributed", value) == (True, expected)


def test_bool_rejects_a_non_boolean():
    assert coerce("death_attributed", "probably") == (False, None)


def test_numeric_coercion_and_rejection():
    assert coerce("fio2_pct", "60") == (True, 60.0)
    assert coerce("fio2_pct", "sixty") == (False, None)


def test_enum_value_must_be_declared():
    assert coerce("transfusion_type", "exchange") == (True, "exchange")
    assert coerce("transfusion_type", "EXCHANGE") == (True, "exchange")
    assert coerce("transfusion_type", "double volume") == (False, None)


def test_an_invented_feature_name_is_rejected():
    feats, _, t = run_verify([{"feature": "vibes_score", "value": 3, "quote": "chest pain"}])
    assert feats == {} and t.unknown_feature == 1


def test_a_valid_quote_with_an_invalid_value_is_still_rejected():
    feats, _, t = run_verify([{"feature": "transfusion_type", "value": "megatransfusion",
                               "quote": "received a simple transfusion"}])
    assert feats == {} and t.quote_ok == 1 and t.value_bad == 1


def test_unparseable_reply_is_counted_not_raised():
    t = Tally()
    feats, present, _ = verify("here is my answer, thanks!", NOTE, t)
    assert feats == {} and present is None and t.bad_json == 1


# ------------------------------------------------------------------ prompt shape

def test_prompt_asks_only_for_features_the_outcome_grades_on():
    prompt = build_prompt(NOTE, "36")            # Fever: temperature only
    assert '"temperature"' in prompt
    assert '"fio2_pct"' not in prompt


def test_prompt_omits_derived_features_the_model_cannot_observe():
    prompt = build_prompt(NOTE, "48")
    assert '"life_support"' not in prompt        # derived from its components
    assert '"vasopressors"' in prompt or "vasopressor" in prompt


def test_prompt_carries_the_outcome_specific_definition_of_treated():
    prompt = build_prompt(NOTE, "29")
    assert "For this outcome specifically" in prompt
    assert "splenectomy" in prompt


def test_prompt_forbids_grading():
    assert "NOT assigning a severity grade" in build_prompt(NOTE, "48")


# ------------------------------------------------------- reporting denominators

def _rec(title, body):
    return {"title": title, "patient": body}


def test_null_placeholders_are_kept_out_of_the_grounding_denominator():
    """A finding with no quote is a prompt-compliance failure, not a grounding
    result. Leaving it in the denominator lets a clean extractor look like a
    hallucinating one - which is exactly how the A100 run read as 7.4%."""
    feats, _, t = run_verify([
        {"feature": "fio2_pct", "value": 60, "quote": "FiO2 was escalated to 60%"},
        {"feature": "vasopressors", "value": None, "quote": None},
        {"feature": "renal_replacement", "value": None, "quote": None},
        {"feature": "temperature", "value": None, "quote": None},
    ])
    rep = t.report()
    assert rep["proposed"] == 4 and rep["quoted"] == 1
    assert rep["null_placeholder"] == 3 and rep["null_placeholder_pct"] == 75.0
    assert rep["quote_verified_pct_of_quoted"] == 100.0   # the model grounded what it quoted
    assert rep["quote_verified_pct"] == 25.0              # ...and ignored the omission rule


def test_both_grounding_denominators_are_always_reported():
    _, _, t = run_verify([{"feature": "fio2_pct", "value": 90,
                           "quote": "FiO2 was escalated to 90%"}])
    rep = t.report()
    assert rep["hallucinated_pct_of_quoted"] == 100.0 and rep["hallucinated_quote_pct"] == 100.0


def test_corrupt_gguf_byte_tokens_are_counted_not_silently_passed():
    _, _, t = run_verify([{"feature": "fio2_pct", "value": 60,
                           "quote": "FiO2[UNK_BYTE_0xe29681\u2581was]was escalated to 60%"}])
    assert t.tokenizer_artifacts == 1


# --------------------------------------------------------------- cohort gate

def test_a_negated_mention_is_not_a_case():
    assert not is_scd_primary(_rec("Cardiac arrest in an athlete",
                                   "The family denied a family history of SCD."))


def test_sickle_cell_trait_is_not_sickle_cell_disease():
    assert not is_scd_primary(_rec("Cerebellar edema after opioid ingestion",
                                   "A 25-month-old male with sickle cell trait was noted "
                                   "to be breathing heavily. Sickle cell trait is benign."))


def test_scd_alone_does_not_qualify_because_cardiology_means_sudden_cardiac_death():
    assert not is_scd_primary(_rec("Electrical storm in hypertrophic cardiomyopathy",
                                   "The ESC HCM Risk-SCD score was low. Risk of SCD at 5 "
                                   "years was 3%. An ICD prevents SCD."))


def test_a_mention_belonging_to_the_mother_is_not_the_patients_diagnosis():
    assert not is_scd_primary(_rec("Novel dystrophin variant",
                                   "The pregnancy was complicated by maternal sickle cell "
                                   "disease. Family history was significant for sickle cell disease."))


def test_a_title_match_qualifies():
    assert is_scd_primary(_rec("Intracardiac Thrombosis in Sickle Cell Disease", "..."))


def test_a_without_title_does_not_qualify():
    assert not is_scd_primary(_rec("Salmonella Osteomyelitis in a Child without Sickle Cell",
                                   "The child had osteomyelitis."))


def test_the_patients_own_repeated_diagnosis_qualifies_without_a_title_match():
    assert is_scd_primary(_rec("Rapid development of seizures and PRES in a COVID-19 patient",
                               "A 43-year-old female with sickle cell disease (SCD) and chronic "
                               "opioid use became gradually unresponsive. Her SCD was managed with "
                               "hydroxyurea."))


def test_trait_mentions_do_not_count_toward_the_threshold():
    assert scd_mentions("sickle cell trait was noted; sickle cell trait is benign") == 0


# ---------------------------------------------- note selection (present AND absent)

def _pool(n=200):
    """A synthetic pool where outcome prevalence is known by construction."""
    out = []
    for i in range(n):
        body = "A patient with sickle cell disease was admitted. "
        if i % 4 == 0:  body += "He was treated for a vaso-occlusive pain crisis. "
        if i % 5 == 0:  body += "Acute chest syndrome was diagnosed. "
        if i % 3 == 0:  body += "The patient was febrile on admission. "
        out.append({"patient_uid": f"u{i}", "title": "Case report", "patient": body})
    return out


def test_the_seed_falls_back_to_the_outcome_name():
    assert outcome_seed("24").search("presented with priapism")          # Priapism
    assert outcome_seed("40").search("a chronic leg ulcer")              # Leg Ulcer


def test_a_slash_separated_name_matches_either_side():
    pat = outcome_seed("18")   # Cholecystitis/Cholelithiasis (gallstones)
    assert pat.search("acute cholecystitis") and pat.search("cholelithiasis on ultrasound")


def test_an_abbreviation_in_the_name_is_usable_on_its_own():
    assert outcome_seed("13").search("developed PRES")   # ...Encephalopathy Syndrome (PRES)


def test_selection_reaches_every_target_outcome():
    """Random sampling yields 40 pain crises and zero leg ulcers (plan §7)."""
    _, sel = select_notes(_pool(), 20, ["28", "48", "36"])
    seeded = {v.split(":")[1] for v in sel.values() if v.startswith("seeded:")}
    assert seeded == {"28", "48", "36"}


def test_selection_keeps_an_unstratified_holdout_so_the_bias_is_measurable():
    _, sel = select_notes(_pool(), 20, ["28", "48"], holdout_frac=0.25)
    assert sum(1 for v in sel.values() if v == "holdout") == 5


def test_a_note_is_never_selected_twice():
    notes, sel = select_notes(_pool(), 24, ["28", "48", "36"])
    uids = [r["patient_uid"] for r in notes]
    assert len(uids) == len(set(uids)) == len(sel)


def test_stratification_can_be_turned_off_for_an_unbiased_sample():
    _, sel = select_notes(_pool(), 10, ["28"], stratify=False)
    assert set(sel.values()) == {"random"}


def test_the_pool_keeps_notes_where_outcomes_are_genuinely_absent():
    """`absent` is a first-class answer (plan §1) and the absence audit needs
    negatives, so selection must not quietly drop notes that lack an outcome."""
    pool = _pool()
    notes, _ = select_notes(pool, 20, ["28"])
    assert any("vaso-occlusive" not in r["patient"] for r in notes)


# ------------------------------------------------------------------ the mock

def test_the_mock_still_reports_exactly_half_its_quotes_verified():
    """The P11 protocol uses this as the harness's own smoke test."""
    t = Tally()
    verify(call_mock(build_prompt(NOTE, "48"), "", ""), NOTE, t)
    assert t.quote_ok == 1 and t.quote_unfound == 1


def test_the_mock_exercises_the_absent_path_too():
    """Without an absent branch the absence audit has nothing to run on, and a
    broken absent path ships unnoticed."""
    seen = set()
    for i in range(40):
        note = f"Case {i}: a patient with sickle cell disease was admitted."
        for num in ("28", "48", "36", "19"):
            seen.add(json.loads(call_mock(build_prompt(note, num), "", ""))["present"])
    assert seen == {True, False}


# ------------------------------------------------------------------ unit guard
#
# `coerce` only ever asked whether a value parses as a float. The schema declares
# the unit; the verified quote says which unit the number was written in. When
# those disagree the number is wrong by a factor, and nothing downstream can see it.

CREAT_NOTE = ("Bicarbonate reserves were 15.69 mmol/L. Serum creatinine was at 7 mg/L, "
              "calcemia at 90 mg/L. On readmission creatinine 1.8 mg dl-1 was recorded, "
              "and a repeat creatinine of 250 umol/L followed.")
FEVER_NOTE = ("She developed fever (102.6 Fahrenheit) overnight. "
              "Her initial vital signs showed a temperature of 38.1 \u00b0C.")


def test_a_value_written_in_another_unit_is_converted_not_taken_at_face_value():
    """The real failure: a note using mg/L throughout put 7.0 into an mg/dL field,
    a 10x overstatement that grades an AKI at its ceiling."""
    feats, _, t = run_verify([{"feature": "creatinine", "value": 7,
                               "quote": "Serum creatinine was at 7 mg/L"}],
                             note=CREAT_NOTE)
    assert feats == {"creatinine": 0.7} and t.unit_converted == 1


def test_a_value_already_in_the_declared_unit_is_untouched():
    feats, _, t = run_verify([{"feature": "creatinine", "value": 1.8,
                               "quote": "creatinine 1.8 mg dl-1"}], note=CREAT_NOTE)
    assert feats == {"creatinine": 1.8} and t.unit_converted == 0


def test_si_creatinine_is_converted_by_its_molar_mass():
    feats, _, t = run_verify([{"feature": "creatinine", "value": 250,
                               "quote": "a repeat creatinine of 250 umol/L"}],
                             note=CREAT_NOTE)
    assert feats["creatinine"] == round(250 / 88.4, 4) and t.unit_converted == 1


def test_fahrenheit_is_converted_from_the_quote_not_from_the_model_arithmetic():
    """The model copies the number across without the unit: 102.6 into a degC field."""
    feats, _, t = run_verify([{"feature": "temperature", "value": 102.6,
                               "quote": "fever (102.6 Fahrenheit)"}], note=FEVER_NOTE)
    assert feats == {"temperature": 39.2222} and t.unit_converted == 1


def test_a_botched_conversion_is_rejected_rather_than_re_converted():
    """The model read '102.6 Fahrenheit' and wrote 38.9. It is 39.2, and 38.9 is on
    the far side of a grade boundary. The value matches neither the quote's number
    nor its conversion, so there is nothing here to trust and nothing to repair."""
    feats, _, t = run_verify([{"feature": "temperature", "value": 38.9,
                               "quote": "fever (102.6 Fahrenheit)"}], note=FEVER_NOTE)
    assert feats == {} and t.quote_value_mismatch == 1 and t.unit_converted == 0


def test_a_correct_conversion_is_accepted_and_canonicalised():
    feats, _, t = run_verify([{"feature": "temperature", "value": 39.2,
                               "quote": "fever (102.6 Fahrenheit)"}], note=FEVER_NOTE)
    assert feats == {"temperature": 39.2222} and t.quote_value_mismatch == 0


def test_celsius_in_a_celsius_field_is_left_alone():
    feats, _, t = run_verify([{"feature": "temperature", "value": 38.1,
                               "quote": "a temperature of 38.1 \u00b0C"}], note=FEVER_NOTE)
    assert feats == {"temperature": 38.1} and t.unit_converted == 0


def test_a_number_that_is_not_the_one_in_its_own_quote_is_rejected():
    """The quote verifies - the words are all in the note - and still does not
    support the value. Where a unit anchors a number, that much is checkable."""
    feats, _, t = run_verify([{"feature": "creatinine", "value": 1.9,
                               "quote": "creatinine 1.8 mg dl-1"}], note=CREAT_NOTE)
    assert feats == {} and t.quote_value_mismatch == 1


def test_a_quote_carrying_several_units_is_left_alone_not_guessed_at():
    note = "Creatinine was 1.8 mg/dL, having been 250 umol/L on admission."
    feats, _, t = run_verify([{"feature": "creatinine", "value": 1.8,
                               "quote": "Creatinine was 1.8 mg/dL, having been 250 umol/L"}],
                             note=note)
    # two different unit tokens: which one belongs to this number is not decidable
    assert feats == {"creatinine": 1.8} and t.unit_ambiguous == 1 and t.unit_converted == 0


def test_an_unconvertible_unit_is_rejected_rather_than_accepted_at_face_value():
    feats, _, t = run_verify([{"feature": "creatinine", "value": 15.69,
                               "quote": "Bicarbonate reserves were 15.69 mmol/L"}],
                             note=CREAT_NOTE)
    assert feats == {} and t.unit_mismatch == 1


def test_a_feature_with_no_convertible_unit_family_is_never_touched():
    """fio2_pct is a percentage and the schema sanctions 'room air' -> 21, which no
    quote spells out as a number. The guard has no business here."""
    assert unit_guard("fio2_pct", 60.0, "FiO2 was escalated to 60%") == ("ok", 60.0, None)
    assert unit_guard("fio2_pct", 21.0, "on room air") == ("ok", 21.0, None)
    assert unit_guard("patient_age", 30.0, "A 30-year-old female") == ("ok", 30.0, None)


# ------------------------------------------------------- multi-value reconciliation
#
# One (note, outcome) routinely yields several verified values for one feature.
# `out[name] = val` in a loop picked whichever the model emitted last.

def test_an_ordinal_collapses_to_its_schema_declared_extreme():
    """care_setting is 'Highest level of care this event actually reached', and its
    values are declared low-to-high. A note with five settings has one answer."""
    assert reconcile("care_setting", ["inpatient", "home", "ed_treat_release"]) \
        == ("inpatient", None)
    assert reconcile("resp_support", ["room_air", "low_flow_o2"]) == ("low_flow_o2", None)
    assert reconcile("transfusion_type", ["exchange", "simple"]) == ("exchange", None)


def test_emission_order_no_longer_decides_the_value():
    """The same three settings in the order that used to produce the wrong answer."""
    assert reconcile("care_setting", ["inpatient", "inpatient", "ed_treat_release"]) \
        == ("inpatient", None)


def test_a_numeric_feature_that_declares_an_extreme_collapses_to_it():
    assert reconcile("temperature", [37.0, 38.5, 36.9]) == (38.5, None)
    assert reconcile("fio2_pct", [21.0, 80.0]) == (80.0, None)


def test_a_feature_with_no_aggregation_rule_reports_a_conflict_instead_of_guessing():
    """Five creatinines across twelve years, one of them the transplant donor's.
    There is no rule that says which is 'the' creatinine, so none is invented."""
    value, clash = reconcile("creatinine", [0.9, 1.0, 1.6, 1.8])
    assert value is None and clash == [0.9, 1.0, 1.6, 1.8]


def test_repeated_identical_values_are_not_a_conflict():
    assert reconcile("creatinine", [1.8, 1.8, 1.8]) == (1.8, None)
    assert reconcile("patient_age", [30.0]) == (30.0, None)


def test_a_conflicted_feature_is_withheld_from_grading_and_reported():
    note = ("Creatinine at the time of explant was 0.9 mg/dl. "
            "Her kidney function was stable, with creatinine values of 1.6 mg/dl.")
    feats, _, t = run_verify([
        {"feature": "creatinine", "value": 0.9,
         "quote": "Creatinine at the time of explant was 0.9 mg/dl"},
        {"feature": "creatinine", "value": 1.6,
         "quote": "creatinine values of 1.6 mg/dl"},
    ], note=note)
    assert feats == {}                      # never silently 1.6
    assert t.value_conflicts == 1
    assert t.accepted == 2                  # both cleared §2; the disagreement is downstream


def test_the_detail_payload_carries_what_the_review_sheets_are_built_from():
    """The sheets read this instead of re-deriving acceptance, which is how a
    hand-check sheet ends up listing 42 rows for a run that accepted 46."""
    t = Tally()
    _, _, detail = verify(json.dumps({"present": True, "findings": [
        {"feature": "fio2_pct", "value": 60, "quote": "FiO2 was escalated to 60%"},
        {"feature": "fio2_pct", "value": 90, "quote": "FiO2 was escalated to 90%"},
    ]}), NOTE, t)
    assert [f["feature"] for f in detail["accepted"]] == ["fio2_pct"]
    assert detail["accepted"][0]["value"] == 60.0 and detail["conflicts"] == {}


def test_reduce_policy_is_read_off_the_schema_not_hardcoded():
    assert reduce_policy("care_setting") == "max"       # "Highest level of care..."
    assert reduce_policy("resp_support") == "max"       # "Maximum respiratory support..."
    assert reduce_policy("transfusion_type") == "max"   # "Most intensive..."
    assert reduce_policy("temperature") == "max"        # "Highest documented..."
    assert reduce_policy("creatinine") is None          # "Serum creatinine." - no rule
    assert reduce_policy("patient_age") is None


# ------------------------------------------------- absent vs rules-refuted

def test_an_absence_the_rules_produced_is_not_an_absence_the_model_produced():
    """A 36.5 degC 'fever': the model said present, the tables overruled it. Pooled
    into `absent` it sends a reviewer to confirm an absence the model never asserted."""
    assert harness_status("absent", True) == "refuted"
    assert harness_status("absent", False) == "absent"
    assert harness_status("absent", None) == "absent"


def test_every_other_status_passes_through_untouched():
    for st in ("graded", "grade_set", "cannot_grade", "not_applicable"):
        assert harness_status(st, True) == st and harness_status(st, False) == st


# ------------------------------------------------------- provenance labelling

def test_bf16_is_file_type_32_not_30():
    """The 2026-09-01 A100 run served BF16 weights and recorded `file_type_32`,
    because 30 was mapped to BF16 and 32 was missing. 30 is IQ4_XS."""
    assert GGUF_FILE_TYPES[32] == "BF16"
    assert GGUF_FILE_TYPES[30] == "IQ4_XS"
    assert GGUF_FILE_TYPES[1] == "F16" and GGUF_FILE_TYPES[7] == "Q8_0"


def test_an_unknown_file_type_is_recorded_raw_never_guessed():
    """A quantisation the map does not know must not be labelled as one it does.
    Recording the number keeps the provenance honest and debuggable."""
    assert GGUF_FILE_TYPES.get(999) is None


# ------------------------------------------- prompt rules that unblock grading

def test_the_survival_bullet_appears_only_where_a_table_grades_on_death():
    """Every outcome with a `death_attributed` row is capped at grade_set while that
    value is unknown, because the engine cannot rule out the top grade. The 2026-09-01
    A100 run extracted it zero times in 80 pairs, and every graded pair it produced was
    Fever - the one outcome of the four with no death row. Asking for it elsewhere would
    be noise, so the bullet is conditional on the table needing it."""
    for num in ("28", "48", "19"):                      # tables with a grade-5 death row
        assert '"death_attributed": false' in build_prompt(NOTE, num), num
    assert "death_attributed" not in build_prompt(NOTE, "36")   # Fever grades on temp only


def test_the_prompt_tells_the_model_not_to_convert_units():
    """The pipeline converts from the quote, which is the only trustworthy source. A model
    that converts first produces a number matching neither its quote nor the conversion,
    and the unit guard has to reject it - losing a real value."""
    p = build_prompt(NOTE, "36")
    assert "OWN units" in p and "Do not convert" in p


def test_the_prompt_forbids_null_findings_rather_than_merely_discouraging_them():
    p = build_prompt(NOTE, "48")
    assert "NEVER emit a finding whose value or quote is null" in p
    assert "An empty\n  list is a valid" in p


def test_the_prompt_says_which_of_several_values_to_report():
    """'At most ONCE' alone does not say WHICH one, and the schema already answers it."""
    p = build_prompt(NOTE, "28")
    assert "at most ONCE" in p and '"highest"' in p
