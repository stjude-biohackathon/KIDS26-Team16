"""The MedGemma extraction check's verification layer - the §2 rule that turns a hallucinated quote
into a counted rejection rather than a silent wrong answer."""
import json

import pytest

from experiments.grading import harness_status
from experiments.cohort import (
    is_scd_primary, outcome_seed, scd_mentions, select_notes,
)
from experiments.medgemma_extraction import (
    DEFAULT_OUTCOMES, build_prompt, call_mock,
)
from experiments.ollama_backend import GGUF_FILE_TYPES
from experiments.verification import (
    Tally, coerce, reconcile, reduce_policy, unit_guard, verify,
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


def test_seeds_for_tcd_and_retinopathy():
    assert outcome_seed("12").search("elevated TCD velocities")
    assert outcome_seed("12").search("transcranial doppler showed abnormal velocity")
    assert outcome_seed("17").search("proliferative sickle retinopathy on exam")
    assert outcome_seed("17").search("fundoscopic examination revealed retinopathy")


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


def test_tcd_velocity_converts_from_meters_per_second():
    # Model extracted raw 2.1 from quote without converting -> guard converts to 210.0 cm/s
    assert unit_guard("tcd_velocity", 2.1, "TCD velocity was 2.1 m/s") == (
        "converted", 210.0, "2.1 m/s -> 210.0 cm/s")
    # Model already converted or quote was in cm/s -> ok
    assert unit_guard("tcd_velocity", 210.0, "TCD showed 210 cm/s") == ("ok", 210.0, None)
    assert unit_guard("tcd_velocity", 195.0, "measured 195 cm/sec") == ("ok", 195.0, None)


def test_albuminuria_converts_from_mg_per_mmol_and_accepts_ug_per_mg():
    # 10 mg/mmol * 8.84 = 88.4 mg/g
    assert unit_guard("albuminuria", 10.0, "UACR was 10 mg/mmol") == (
        "converted", 88.4, "10.0 mg/mmol -> 88.4 mg/g")
    # 50 ug/mg is 1:1 with mg/g
    assert unit_guard("albuminuria", 50.0, "urine albumin 50 ug/mg") == ("ok", 50.0, None)
    assert unit_guard("albuminuria", 45.0, "urine albumin 45 mg/g") == ("ok", 45.0, None)


def test_wound_area_converts_from_sq_mm():
    # 800 mm2 / 100 = 8.0 cm2
    assert unit_guard("wound_area_cm2", 800.0, "ulcer area was 800 mm2") == (
        "converted", 8.0, "800.0 mm2 -> 8.0 cm2")
    assert unit_guard("wound_area_cm2", 12.0, "wound area was 12 cm2") == ("ok", 12.0, None)
    assert unit_guard("wound_area_cm2", 15.5, "wound of 15.5 cm²") == ("ok", 15.5, None)


# ------------------------------------------------------------------ micrograms
#
# The Hb `g/l` token only refused a preceding a-z letter, so the `g/L` inside
# "µg/L" - micro sign or Greek mu - read as grams per litre, a millionfold apart.

@pytest.mark.parametrize("mu", ["µ", "μ"])
def test_a_microgram_quote_is_never_read_as_grams(mu):
    """A ferritin sentence offered as haemoglobin was 'converted' 900 -> 90 g/dL."""
    status, value, _ = unit_guard("hb_nadir", 900.0, f"ferritin 900 {mu}g/L")
    assert (status, value) == ("bad", None)


def test_grams_per_litre_still_converts_to_grams_per_decilitre():
    assert unit_guard("hb_nadir", 70.0, "haemoglobin 70 g/L") == (
        "converted", 7.0, "70.0 g/l -> 7.0 g/dL")


@pytest.mark.parametrize("quote", [
    "ferritin 900 ng/mL", "ferritin 900 µg/L", "ferritin 900 μg/L",
    "ferritin 900 ug/L", "ferritin 900 mcg/L",
])
def test_ferritin_accepts_micrograms_per_litre_as_nanograms_per_millilitre(quote):
    assert unit_guard("ferritin", 900.0, quote) == ("ok", 900.0, None)


def test_ferritin_in_milligrams_per_litre_is_rejected_not_accepted_1000x_low():
    status, value, _ = unit_guard("ferritin", 900.0, "ferritin 900 mg/L")
    assert (status, value) == ("bad", None)


@pytest.mark.parametrize("value,quote", [
    (27469.0, "ferritin was 27,469 ng/mL"),
    (47000.0, "ferritin from 47,000 μg/L"),
    (1200.0, "urine albumin 1,200 mg/g"),
])
def test_a_thousands_separator_is_one_number(value, quote):
    status, got, _ = unit_guard("ferritin" if "ferritin" in quote else "albuminuria",
                                value, quote)
    assert (status, got) == ("ok", value)


def test_trv_converts_from_centimeters_per_second():
    assert unit_guard("trv", 280.0, "TRV was 280 cm/s") == (
        "converted", 2.8, "280.0 cm/s -> 2.8 m/s"
    )
    assert unit_guard("trv", 2.8, "TRV 2.8 m/s") == ("ok", 2.8, None)


def test_scoped_units_do_not_collide_with_other_measurements():
    assert unit_guard("wound_area_cm2", 12.0, "wound area 3 cm x 4 cm") == ("ok", 12.0, None)
    assert unit_guard("creatinine", 2.1, "rose to 2.1 mg/dL over 3 days") == ("ok", 2.1, None)
    assert unit_guard("hb_nadir", 7.5, "Hb 7.5 g/dL") == ("ok", 7.5, None)


def test_hearing_lowest_affected_khz_converts_from_hz():
    assert unit_guard("hearing_lowest_affected_khz", 4000.0, "loss at 4000 Hz") == (
        "converted", 4.0, "4000.0 hz -> 4.0 kHz"
    )
    assert unit_guard("hearing_lowest_affected_khz", 4.0, "loss at 4 kHz") == ("ok", 4.0, None)


def test_fsh_accepts_iu_per_l():
    assert unit_guard("fsh", 7.6, "FSH 7.6 IU/L") == ("ok", 7.6, None)
    assert unit_guard("fsh", 7.6, "FSH was 7.6 mIU/mL") == ("ok", 7.6, None)


def test_height_loss_cm_converts_from_inches_and_mm():
    assert unit_guard("height_loss_cm", 2.0, "lost 2 inches in height") == (
        "converted", 5.08, "2.0 inch -> 5.08 cm"
    )
    assert unit_guard("height_loss_cm", 20.0, "height loss 20 mm") == (
        "converted", 2.0, "20.0 mm -> 2.0 cm"
    )
    assert unit_guard("height_loss_cm", 5.0, "height loss 5 cm") == ("ok", 5.0, None)


def test_ferritin_converts_from_pmol_per_l():
    assert unit_guard("ferritin", 2247.0, "ferritin was 2247 pmol/L") == (
        "converted", 1000.0, "2247.0 pmol/l -> 1000.0 ng/mL"
    )


def test_tlc_pct_pred_guards_against_lung_volume_in_liters():
    # Model extracted raw volume in Liters from a volume-only quote -> rejected as bad
    assert unit_guard("tlc_pct_pred", 3.2, "TLC 3.2 L") == (
        "bad", None, "3.2 L is a lung volume, not percent predicted"
    )
    assert unit_guard("tlc_pct_pred", 3.5, "TLC was 3.5 liters") == (
        "bad", None, "3.5 liters is a lung volume, not percent predicted"
    )
    assert unit_guard("tlc_pct_pred", 3200.0, "TLC 3200 mL") == (
        "bad", None, "3200.0 mL is a lung volume, not percent predicted"
    )

    # Model extracted raw volume in Liters when % is also in quote -> rescued to percent predicted
    assert unit_guard("tlc_pct_pred", 3.2, "TLC 3.2 L (78% predicted)") == (
        "converted", 78.0, "3.2 L -> 78.0% predicted"
    )
    assert unit_guard("tlc_pct_pred", 3.2, "FEV1 2.1 L, FVC 2.6 L, TLC 3.2 L (78% predicted)") == (
        "converted", 78.0, "3.2 L -> 78.0% predicted"
    )

    # Model correctly extracted percent predicted -> ok
    assert unit_guard("tlc_pct_pred", 78.0, "TLC 3.2 L (78% predicted)") == ("ok", 78.0, None)
    assert unit_guard("tlc_pct_pred", 78.0, "TLC 78% predicted") == ("ok", 78.0, None)
    assert unit_guard("tlc_pct_pred", 78.0, "TLC 78%") == ("ok", 78.0, None)
    assert unit_guard("tlc_pct_pred", 75.0, "TLC was 75") == ("ok", 75.0, None)


def test_tlc_pct_pred_run_verify_prevents_catastrophic_grade4():
    pft_note = "Pulmonary function test showed TLC 3.2 L (78% predicted)."
    # If the model extracts 3.2, run_verify converts to 78.0 rather than letting 3.2 grade as < 50
    feats, _, t = run_verify([{"feature": "tlc_pct_pred", "value": 3.2,
                               "quote": "TLC 3.2 L (78% predicted)"}],
                             note=pft_note)
    assert feats == {"tlc_pct_pred": 78.0} and t.unit_converted == 1

    # If the quote only contains the raw volume, it is rejected
    feats_raw, _, t_raw = run_verify([{"feature": "tlc_pct_pred", "value": 3.2,
                                       "quote": "TLC 3.2 L"}],
                                     note=pft_note)
    assert feats_raw == {} and t_raw.unit_mismatch == 1


# ------------------------------------------------------------------ patient age
#
# The schema holds age in years; notes write "10-day-old", "18-month-old" and
# "2 years 10-month-old". Unconverted, an 18-month-old was graded as an adult.

D, W = 1 / 365.25, 7 / 365.25

@pytest.mark.parametrize("value,quote,status,years", [
    # one unit
    (30.0, "A 30-year-old female", "ok", 30.0),
    (13.0, "a 13-yr-old boy", "ok", 13.0),
    (1.5, "1.5 years-old", "ok", 1.5),
    (10.0, "a 10-day-old infant", "converted", 10 * D),
    (5.0, "5 days old", "converted", 5 * D),
    (3.0, "day of life 3", "converted", 3 * D),
    (8.0, "8 weeks old", "converted", 8 * W),
    (10.0, "a 10-week-old", "converted", 10 * W),
    (18.0, "an 18-month-old boy", "converted", 1.5),
    (3.0, "3 months of age", "converted", 0.25),
    # any combination, largest unit first
    (2.0, "2 years 10-month-old", "converted", 2 + 10 / 12),
    (3.0, "3 years and 4 months old", "converted", 3 + 4 / 12),
    (1.0, "1 year 20 days old", "converted", 1 + 20 * D),
    (2.0, "2 years 3 weeks old", "converted", 2 + 3 * W),
    (4.0, "4 years, 2 months and 10 days old", "converted", 4 + 2 / 12 + 10 * D),
    (2.0, "2 months and 5 days old", "converted", 2 / 12 + 5 * D),
    (1.0, "1 month 2 weeks old", "converted", 1 / 12 + 2 * W),
    (6.0, "6 weeks and 3 days old", "converted", 6 * W + 3 * D),
    (1.0, "aged 1 year, 1 month, 1 week and 1 day", "converted", 1 + 1 / 12 + W + D),
    (2.0, "2y 3m old", "converted", 2.25),
    # any number in the phrase identifies it; the whole phrase is the age
    (10.0, "2 years 10-month-old", "converted", 2 + 10 / 12),
    # the model already did the arithmetic
    (0.0274, "a 10-day-old infant", "ok", 10 * D),
    (2.83, "2 years 10-month-old", "ok", 2 + 10 / 12),
])
def test_patient_age_is_naturalised_to_years(value, quote, status, years):
    got_status, got, _ = unit_guard("patient_age", value, quote)
    assert got_status == status
    assert got == pytest.approx(years, abs=1e-4)


def test_an_age_is_told_apart_from_a_duration():
    assert unit_guard("patient_age", 12.0, "diagnosed 3 years ago, now 12 years old")[:2] \
        == ("ok", 12.0)
    assert unit_guard("patient_age", 5.0, "a 5-year-old with pain for 3 days")[:2] \
        == ("ok", 5.0)
    # the duration is not the age once the quote carries a marked one
    assert unit_guard("patient_age", 3.0, "a 5-year-old with pain for 3 days")[:2] \
        == ("value_mismatch", None)


def test_gestational_age_is_not_the_patients_age():
    q = "a 7-month-old born at 32 weeks gestation"
    status, got, _ = unit_guard("patient_age", 7.0, q)
    assert status == "converted" and got == pytest.approx(7 / 12, abs=1e-4)
    assert unit_guard("patient_age", 32.0, q)[:2] == ("value_mismatch", None)
    assert unit_guard("patient_age", 32.0, "delivered at 32 weeks' gestation")[:2] \
        == ("value_mismatch", None)


def test_an_age_the_quote_does_not_carry_is_rejected():
    assert unit_guard("patient_age", 0.0, "a 10-day-old infant")[:2] == ("value_mismatch", None)


def test_two_ages_the_value_fits_equally_are_left_alone():
    status, got, _ = unit_guard("patient_age", 3.0, "3 months old; her brother is 3 years old")
    assert (status, got) == ("ambiguous", 3.0)


@pytest.mark.parametrize("quote", ["a newborn, age 3", "walked 5m", "given 3 mg/kg daily"])
def test_an_age_without_a_recognisable_unit_is_untouched(quote):
    assert unit_guard("patient_age", 3.0, quote) == ("ok", 3.0, None)


def test_an_infant_age_reaches_the_tables_in_years():
    from scogs.evaluate import resolve_derived
    note = "An 18-month-old boy with HbSS presented with fever."
    feats, _, t = run_verify([{"feature": "patient_age", "value": 18,
                               "quote": "An 18-month-old boy"}], note=note)
    assert feats == {"patient_age": 1.5} and t.unit_converted == 1
    assert resolve_derived(feats)["age_stratum"] == "pediatric"


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
    """Several eGFR values across years.
    There is no rule that says which is 'the' eGFR, so none is invented."""
    value, clash = reconcile("egfr", [30.0, 45.0, 60.0])
    assert value is None and clash == [30.0, 45.0, 60.0]


def test_creatinine_reconciles_to_maximum():
    assert reconcile("creatinine", [0.9, 1.0, 1.6, 1.8]) == (1.8, None)


def test_repeated_identical_values_are_not_a_conflict():
    assert reconcile("creatinine", [1.8, 1.8, 1.8]) == (1.8, None)
    assert reconcile("patient_age", [30.0]) == (30.0, None)


def test_a_conflicted_feature_is_withheld_from_grading_and_reported():
    note = ("eGFR at the time of admission was 45. "
            "Her kidney function was stable, with eGFR values of 60.")
    feats, _, t = run_verify([
        {"feature": "egfr", "value": 45,
         "quote": "eGFR at the time of admission was 45"},
        {"feature": "egfr", "value": 60,
         "quote": "eGFR values of 60"},
    ], note=note)
    assert feats == {}                      # never silently 60
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
    assert reduce_policy("creatinine") == "max"         # "Highest serum creatinine..."
    assert reduce_policy("egfr") is None                # "Estimated glomerular..." - no rule
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


def test_harness_status_flags_missed_presence_when_criteria_met():
    assert harness_status("absent", False, criteria=True) == "missed_presence"
    assert harness_status("cannot_grade", None, criteria=True) == "missed_presence"
    assert harness_status("graded", True, criteria=True) == "graded"
    assert harness_status("absent", False, criteria=False) == "absent"
    assert harness_status("absent", True, criteria=False) == "refuted"



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


def test_default_outcomes_are_the_14_focus_conditions():
    from scogs.tables import TABLES

    expected_14 = {
        "10",  # Chronic Pain
        "11",  # Cognitive Dysfunction (CD)
        "12",  # Elevated TCD Ultrasonography Velocity (TCD Elevation)
        "15",  # Stroke
        "17",  # Sickle Cell Retinopathy (Retinopathy)
        "21",  # Chronic Kidney Disease (CKD)
        "24",  # Priapism
        "28",  # Acute Sickle Cell Pain Episode (Acute Pain)
        "29",  # Acute Splenic Sequestration (SS)
        "39",  # Avascular Necrosis of Joints (AVN)
        "40",  # Leg Ulcer
        "47",  # Depression
        "48",  # Acute Chest Syndrome (ACS)
        "49",  # Asthma Exacerbation (Asthma)
    }
    parsed = set(DEFAULT_OUTCOMES.split(","))
    assert parsed == expected_14
    assert len(DEFAULT_OUTCOMES.split(",")) == 14
    for outcome_id in parsed:
        assert outcome_id in TABLES


# ------------------------------------------------------------ patient context

def test_context_schema_and_prompt():
    from experiments.medgemma_extraction import stage, build_context_prompt, context_schema, CONTEXT_FEATURES
    st = stage("2b", patient_context=True)
    schema = context_schema(st)
    assert "present" not in schema.get("required", [])
    assert "present" not in schema.get("properties", {})
    assert "findings" in schema.get("properties", {})

    prompt = build_context_prompt(NOTE, st)
    for feat in CONTEXT_FEATURES:
        assert feat in prompt
    assert "Do not convert" in prompt


def test_patient_context_merge_precedence():
    ctx_feats = {"patient_age": 14.0, "patient_sex": "male"}
    outcome_feats = {"patient_age": 15.0, "fio2_pct": 60.0}
    # Outcome-level values win
    merged = {**ctx_feats, **outcome_feats}
    assert merged["patient_age"] == 15.0
    assert merged["patient_sex"] == "male"
    assert merged["fio2_pct"] == 60.0


def test_patient_context_tallies_stay_separate():
    from experiments.medgemma_extraction import stage, run
    notes = [{"patient_uid": "test-1", "patient": NOTE, "selection": "seeded:36", "gender": "M"}]
    t = Tally()
    ct = Tally()
    st = stage("2b", patient_context=True)
    results, context_results = run(notes, ["36"], backend="mock", model="medgemma-27b-f16",
                                   host="", tally=t, timeout=30, concurrency=1, st=st,
                                   ctx_tally=ct)
    assert ct.proposed > 0
    assert t.proposed > 0
    assert "test-1" in context_results
    ctx_extracted, _, _ = context_results["test-1"]
    assert "patient_age" in ctx_extracted or "patient_sex" in ctx_extracted
    feats, present, reply, detail = results["test-1"]["36"]
    for k, v in ctx_extracted.items():
        assert feats.get(k) == v


def test_expand_derived_with_computed_from(monkeypatch):
    from experiments.medgemma_extraction import expand_derived
    from scogs.features import FEATURES
    fake_features = dict(FEATURES)
    fake_features["mock_base_a"] = {"type": "num", "derived": None, "computed_from": None}
    fake_features["mock_base_b"] = {"type": "num", "derived": None, "computed_from": None}
    fake_features["mock_computed"] = {
        "type": "ord", "derived": None, "computed_from": ["mock_base_a", "mock_base_b"]
    }
    monkeypatch.setattr("experiments.medgemma_extraction.FEATURES", fake_features)
    expanded = expand_derived({"mock_computed"})
    assert expanded == {"mock_computed", "mock_base_a", "mock_base_b"}


# ---------------------------------------------------------------- Phase E: Feedback Retry

def test_precheck_clean_reply_has_no_issues():
    from experiments.verification import precheck
    reply = json.dumps({
        "present": True,
        "present_quote": "chest pain",
        "findings": [
            {"feature": "fio2_pct", "value": 60.0, "quote": "FiO2 was escalated to 60%"},
            {"feature": "resp_support", "value": "low_flow_o2", "quote": "presented with chest pain"},
        ]
    })
    issues = precheck(reply, NOTE, allowed_features={"fio2_pct", "resp_support"})
    assert issues == []


def test_precheck_flags_unfound_quotes():
    from experiments.verification import precheck
    reply = json.dumps({
        "present": True,
        "findings": [
            {"feature": "fio2_pct", "value": 60.0, "quote": "this quote is completely made up"},
        ]
    })
    issues = precheck(reply, NOTE)
    assert len(issues) == 1
    assert "was not found verbatim in the note" in issues[0]


def test_precheck_flags_missing_quote():
    from experiments.verification import precheck
    reply = json.dumps({
        "present": True,
        "findings": [
            {"feature": "fio2_pct", "value": 60.0, "quote": ""},
        ]
    })
    issues = precheck(reply, NOTE)
    assert len(issues) == 1
    assert "missing a quote" in issues[0]


def test_precheck_flags_unknown_and_disallowed_features():
    from experiments.verification import precheck
    reply = json.dumps({
        "findings": [
            {"feature": "non_existent_feature_123", "value": 5, "quote": "chest pain"},
            {"feature": "fio2_pct", "value": 60.0, "quote": "FiO2 was escalated to 60%"},
        ]
    })
    # non_existent_feature_123 is unknown
    issues = precheck(reply, NOTE)
    assert any("Unknown feature" in iss for iss in issues)

    # fio2_pct is not allowed if allowed_features is {"resp_support"}
    issues2 = precheck(reply, NOTE, allowed_features={"resp_support"})
    assert any("not in the schema for this outcome" in iss for iss in issues2)


def test_precheck_flags_quote_value_mismatch():
    from experiments.verification import precheck
    temp_note = "Patient had a temperature of 38.0 °C on admission."
    reply = json.dumps({
        "findings": [
            {"feature": "temperature", "value": 40.5, "quote": "temperature of 38.0 °C"},
        ]
    })
    issues = precheck(reply, temp_note)
    assert len(issues) == 1
    assert "does not match quote" in issues[0] or "neither the quote" in issues[0]


def test_precheck_flags_unfound_present_quote():
    from experiments.verification import precheck
    reply = json.dumps({
        "present": True,
        "present_quote": "hallucinated diagnosis statement",
        "findings": []
    })
    issues = precheck(reply, NOTE)
    assert len(issues) == 1
    assert "present_quote 'hallucinated diagnosis statement' was not found verbatim in the note" in issues[0]


def test_feedback_retry_end_to_end_mock():
    from experiments.medgemma_extraction import stage, run
    notes = [{"patient_uid": "uid-100", "patient": NOTE, "selection": "seeded:28", "gender": "M"}]
    
    # Without feedback retry: mock backend produces 1 unfound quote and 0 feedback retries
    t_no_fb = Tally()
    st_no_fb = stage("2b", feedback_retry=False)
    run(notes, ["28"], backend="mock", model="medgemma-27b-f16", host="",
        tally=t_no_fb, timeout=30, concurrency=1, st=st_no_fb)
    assert t_no_fb.feedback_retries == 0
    assert t_no_fb.quote_unfound == 1

    # With feedback retry: mock backend gets re-prompted, drops the hallucinated quote
    t_fb = Tally()
    st_fb = stage("2b", feedback_retry=True)
    run(notes, ["28"], backend="mock", model="medgemma-27b-f16", host="",
        tally=t_fb, timeout=30, concurrency=1, st=st_fb)
    assert t_fb.feedback_retries == 1
    assert t_fb.quote_unfound == 0
    assert t_fb.accepted == 1




