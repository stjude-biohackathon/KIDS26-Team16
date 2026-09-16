"""Canonical SCOGS evidence-feature registry (Layer 3 of the pipeline).

This file is the contract. Layer 3 extracts these features from a note; Layer 4
(`tables.py`) maps (outcome, features) -> grade using nothing else. Every
identifier appearing in any decision table must be declared here, and
`build_schema.py` fails the build if one is not.

Fields
  type        bool | num | ord | cat
              `ord` values are listed low-to-high and compare by rank, so
              `resp_support >= niv_bipap_cpap` means what it reads as.
  scope       encounter - one value per note/encounter (labs, care level, age)
              outcome   - one value per (note, outcome); the same word means
                          different things per outcome, so `per_outcome` carries
                          the outcome-specific reading the annotator is shown
  definition  what a human must see in the note to set this. The annotation form
              in the ground-truth protocol is generated from these strings, so
              they are written as verification questions, not as descriptions.
  review      True when the rubric wording is a clinical judgement rather than an
              observation. These are where inter-annotator agreement will be
              worst and where an external abstractor is most likely to disagree.
"""
from __future__ import annotations

FEATURES: dict[str, dict] = {}

def F(name, type, definition, *, scope="encounter", values=None, unit=None,
      outcomes=(), per_outcome=None, review=False, derived=None):
    if name in FEATURES:
        raise ValueError(f"duplicate feature {name!r}")
    FEATURES[name] = dict(type=type, scope=scope, values=list(values) if values else None,
                          unit=unit, definition=definition,
                          outcomes=sorted(outcomes), per_outcome=per_outcome or {},
                          review=review, derived=derived)
    return name

# ============================================================ patient / encounter

F("patient_age", "num", "Patient age at this encounter, in years. Required before "
  "grading 06, 19, 25, 26, 42, 53; grading returns `cannot grade: age unknown` without it.",
  unit="years", outcomes=["06","19","25","26","27","36","42","53"])

F("age_stratum", "ord", "Derived from patient_age: pediatric when age < 18, adult when >= 18.",
  values=["pediatric","adult"], outcomes=["42","53"], derived="patient_age")

F("care_setting", "ord",
  "Highest level of care this event actually reached. `home` = managed without a "
  "facility visit; `ed_treat_release` = seen and discharged; `inpatient` = admitted; "
  "`icu` = critical care.",
  values=["home","clinic_or_day_hospital","ed_treat_release","inpatient","icu"],
  outcomes=["20","24","28","33","38","42","51"])

F("resp_support", "ord",
  "Maximum respiratory support given during the event.",
  values=["room_air","low_flow_o2","high_flow","niv_bipap_cpap","invasive_ventilation"],
  outcomes=["48","49","52"])

F("fio2_pct", "num", "Highest documented FiO2 as a percentage. Room air is 21.",
  unit="%", outcomes=["48"],
  per_outcome={"48":"FiO2 is what is DELIVERED to the patient (the oxygen concentration in inspired gas). SpO2 / pulse oximetry is what is MEASURED in the patient. Do not report SpO2 as FiO2."})

F("vasopressors", "bool", "Vasoactive or vasopressor infusion given (norepinephrine, "
  "epinephrine, dopamine, dobutamine, milrinone).")
F("renal_replacement", "bool", "Renal replacement therapy given: haemodialysis, CRRT, "
  "haemofiltration, peritoneal dialysis.")
F("life_support_other", "bool", "Any other life-supporting treatment not covered by "
  "mechanical ventilation, vasopressors or renal replacement (e.g. ECMO).")
F("life_support", "bool",
  "The rubric's recurring clause: 'a life-threatening complication where the need for "
  "life-supporting treatment indicated, including use of mechanical ventilation, "
  "vasopressors, renal replacement therapy, or other life-supporting treatments'. "
  "True when any of invasive ventilation, vasopressors, renal replacement, or other "
  "life support was indicated.",
  derived="resp_support == invasive_ventilation or vasopressors or renal_replacement or life_support_other",
  outcomes=["01","04","05","13","18","20","28","30","32","33","35","37","43","48","49","51"])

F("transfusion_type", "ord", "Most intensive erythrocyte transfusion given for this event.",
  values=["none","simple","exchange"], outcomes=["48"])
F("anticoagulation", "bool", "Therapeutic anticoagulation started or continued for this event.",
  outcomes=["02"])
F("parenteral_antibiotics", "bool", "Antibiotics given by IV or IM route (not oral).",
  outcomes=["41"])
F("adl_limitation", "ord",
  "Highest level of activity limitation documented. `instrumental_adl` = limits "
  "cooking, shopping, managing money, telephone; `self_care_adl` = limits bathing, "
  "dressing, feeding, toileting.",
  values=["none","instrumental_adl","self_care_adl"], outcomes=["38","39","42","47","49"])
F("hemodynamic_instability", "bool",
  "Hypotension, shock, or circulatory collapse documented during the event.",
  outcomes=["02","05","29","51"], review=True)

# ============================================================ outcome-scoped generic

F("death_attributed", "bool",
  "The patient died and the note attributes the death to THIS outcome.",
  scope="outcome",
  outcomes=["01","02","04","05","06","13","15","18","19","20","21","26","27","28","29",
            "30","32","33","34","35","37","38","41","43","47","48","49","50","51","52"])

F("symptomatic", "bool", "The patient had symptoms attributable to this outcome.",
  scope="outcome", outcomes=["01","38","39","49","51"],
  per_outcome={"39":"Symptomatic from the AVN, as opposed to a radiological finding alone.",
               "51":"Symptomatic from the PE, as opposed to an incidental finding."})

F("treated", "bool",
  "Any treatment was given for THIS outcome during the event. What counts is "
  "outcome-specific - read the per-outcome wording.",
  scope="outcome", review=True,
  outcomes=["20","24","29","32","33","35","37","53"],
  per_outcome={
    "20":"Any treatment for papillary necrosis: erythrocyte transfusion, IV fluids, etc.",
    "24":"Medication given for the priapism episode.",
    "29":"Erythrocyte transfusion, IV fluids, or equivalent for the sequestration. "
         "NOTE: the rubric's Grade 3 both requires treatment 'i.e. erythrocyte "
         "transfusion, IV fluids, etc.' and excludes 'requiring splenectomy, fluids, "
         "etc.' - IV fluids appear on both sides. Implemented as: treatment counts, "
         "only splenectomy excludes. See tables.py note for 29.",
    "32":"Simple or exchange erythrocyte transfusion, or cholestatic agents.",
    "33":"Any treatment for the splenic infarction.",
    "35":"Erythrocyte transfusion or erythropoietin.",
    "37":"Treatment indicated for the sepsis."})

F("incidental_finding", "bool",
  "This outcome was found incidentally - on imaging or testing done for another "
  "reason - rather than being sought for symptoms.",
  scope="outcome", outcomes=["18","33","38","51"])

F("intervention_level", "ord",
  "Highest level of intervention indicated for THIS outcome.",
  values=["none","non_urgent_medical","urgent_medical","urgent_invasive"],
  scope="outcome", outcomes=["01"],
  per_outcome={"01":"`urgent_invasive` = ablation or equivalent urgent procedure."})

# ============================================================ laboratory values

F("hb_nadir", "num", "Lowest haemoglobin during this event.", unit="g/dL", outcomes=["35"])
F("hb_decline_pct", "num",
  "Fall in haemoglobin from the patient's steady-state baseline, as a percentage.",
  unit="%", outcomes=["30"])
F("creatinine", "num", "Serum creatinine.", unit="mg/dL", outcomes=["19"])
F("creatinine_x_baseline", "num",
  "Serum creatinine as a multiple of the patient's baseline (1.5 means 1.5x baseline).",
  unit="ratio", outcomes=["19"])
F("creatinine_increase_mg_dl", "num",
  "Absolute rise in serum creatinine within 48 hours.", unit="mg/dL", outcomes=["19"])
F("egfr", "num", "Estimated glomerular filtration rate.", unit="mL/min/1.73m2", outcomes=["19","21"])
F("albuminuria", "num", "Urine albumin-to-creatinine ratio.", unit="mg/g", outcomes=["21"])
F("ferritin", "num", "Serum ferritin at steady state.", unit="ng/mL", outcomes=["34"])
F("liver_iron_conc", "num", "Liver iron concentration.", unit="mg Fe/100g dry weight", outcomes=["34"])
F("mri_t2star", "num", "Cardiac MRI T2* relaxation time.", unit="msec", outcomes=["34"])
F("meld", "num", "Model for End-stage Liver Disease score.", unit="score", outcomes=["32"])
F("fsh", "num", "Follicle-stimulating hormone.", unit="mIU/mL", outcomes=["23"])
F("temperature", "num", "Highest documented body temperature.", unit="degC", outcomes=["36"])
F("blood_culture_positive", "bool", "Blood culture grew an organism (bacteraemia).", outcomes=["37"])
F("cardiac_biomarker_abnormal", "bool",
  "Cardiac enzymes above the reference range (troponin, CK-MB).", outcomes=["05"])
F("cytopenia_count", "ord",
  "How many of the three cell lines are depressed (anaemia, leucopenia, thrombocytopenia).",
  values=["0","1","2","3"], outcomes=["31"])
F("thrombocytopenia", "bool",
  "Platelets are a depressed cell line. Grade 1 requires the single depressed line "
  "to be platelets ('isolated thrombocytopenia'), not just any one cytopenia.",
  outcomes=["31"])

# ============================================================ imaging & measurements

F("stenosis_pct", "num", "Maximum arterial stenosis on MRA.", unit="%", outcomes=["09"])
F("vessel_segments", "num", "Number of arterial segments showing stenosis on MRA.",
  unit="count", outcomes=["09"])
F("moyamoya", "bool", "Moyamoya syndrome present.", outcomes=["09"])
F("tcd_velocity", "num", "Time-averaged mean velocity on non-imaging transcranial Doppler.",
  unit="cm/s", outcomes=["12"])
F("trv", "num", "Tricuspid regurgitant jet velocity on echocardiogram.", unit="m/s", outcomes=["08"])
F("lvef", "num", "Left ventricular ejection fraction.", unit="%", outcomes=["07"])
F("mpap", "num", "Mean pulmonary artery pressure by right heart catheterisation.",
  unit="mmHg", outcomes=["52"])
F("nyha_class", "ord", "New York Heart Association dyspnoea class.",
  values=["1","2","3","4"], outcomes=["52"])
F("right_heart_failure", "bool",
  "Right-sided congestive heart failure on physical exam.", outcomes=["52"])
F("low_cardiac_output", "bool",
  "Severely reduced cardiac output or cardiac index.", outcomes=["52"])
F("tlc_pct_pred", "num", "Total lung capacity as a percentage of predicted.",
  unit="%", outcomes=["50"])
F("ahi", "num", "Apnoea-hypopnoea index on polysomnography.", unit="events/hr", outcomes=["53"])
F("spo2_desat_over_3min", "bool",
  "HbO2 saturation below 90% for more than 3 minutes overnight.", outcomes=["53"])
F("diastolic_dysfunction_present", "bool",
  "Diastolic dysfunction present on echocardiogram per ASE criteria.", outcomes=["03"])

# ============================================================ per-outcome: cardiac

F("cardioversion", "bool", "Cardioversion performed for the arrhythmia.", outcomes=["01"])
F("hf_intervention", "bool",
  "Heart failure exacerbation treated with diuretics, nitric-oxide-containing "
  "compounds, and/or non-invasive ventilation.", outcomes=["04"])
F("mi_severity_criteria", "bool",
  "Any of the myocardial-infarction severity criteria listed in the rubric's "
  "diagnostic criteria were met (i.e. more than abnormal enzymes alone).",
  outcomes=["05"], review=True)
F("cardiovascular_compromise", "bool",
  "The infarction caused significant cardiovascular compromise.", outcomes=["05"], review=True)
F("cardiac_compromise", "bool",
  "Cardiac function compromise from the PE, including right heart strain on echo or "
  "ECG, or BNP elevated from baseline.", outcomes=["51"])
F("bp_stage", "ord",
  "ACC/AHA blood-pressure stage for the patient's age band, per the rubric's tables. "
  "`elevated` is above normal but below Stage 1.",
  values=["normal","elevated","1","2"], outcomes=["06"])
F("antihypertensive_count", "num", "Number of antihypertensive drugs the patient is on.",
  unit="count", outcomes=["06"])
F("end_organ_damage", "bool",
  "Elevated BP caused end-organ damage: acute kidney injury, sudden vision loss, "
  "stroke, myocardial infarction, or pulmonary oedema.", outcomes=["06"])

# ============================================================ per-outcome: vascular / CNS

F("limb_or_life_threatening", "bool", "The DVT had limb- or life-threatening consequences.",
  outcomes=["02"], review=True)
F("neurologic_instability", "bool", "Neurologic instability in the setting of the DVT.",
  outcomes=["02"], review=True)
F("thrombolysis", "bool", "Site-directed thrombolysis (tPA) given.", outcomes=["02"])
F("surgical_thrombectomy", "bool", "Surgical thrombectomy performed.", outcomes=["02"])
F("pres_confirmed", "bool",
  "PRES confirmed on MRI (gold standard) or head CT.", outcomes=["13"])
F("stroke_symptoms", "bool",
  "Neurological symptoms during an acute stroke, per NIHSS.", outcomes=["15"])
F("stroke_symptoms_resolved", "bool",
  "Stroke symptoms resolved completely (TIA).", outcomes=["15"])
F("mrs", "num", "Modified Rankin Scale at discharge (0-6).", unit="score", outcomes=["15"])
F("incidental_radiographic_only", "bool",
  "Incidental radiographic findings only - ischaemia, haemorrhage, encephalomalacia, "
  "haemosiderosis - with no history of neurologic stroke symptoms. Excludes silent "
  "cerebral infarct, which is outcome 14.", outcomes=["15"])
F("stroke_history", "bool", "Prior history of neurologic symptoms of stroke.", outcomes=["15"])
F("lesion_count", "ord", "Number of silent infarct lesions on brain MRI.",
  values=["none","one","more_than_one"], outcomes=["14"])
F("cognitive_deficit", "bool", "Cognitive deficits documented.", outcomes=["14"])
F("iq_sd_below_mean", "num",
  "How many standard deviations below the mean the IQ evaluation fell.",
  unit="SD", outcomes=["11"])

# ============================================================ per-outcome: chronic pain

F("unplanned_visits_12mo", "num",
  "Unplanned visits to medical facilities for pain treatment in the past 12 months.",
  unit="count", outcomes=["10"])
F("pain_hurt_score", "num",
  "PedsQL Sickle Cell Disease Module 'Pain and Hurt' score in the past 3 months.",
  unit="score", outcomes=["10"])
F("promis_interference_t", "num",
  "PROMIS T-score, pain interference domain, past 3 months.", unit="T", outcomes=["10"])
F("promis_behavior_t", "num",
  "PROMIS T-score, pain behavior domain, past 3 months.", unit="T", outcomes=["10"])
F("pro_severe_count", "num",
  "How many of the three severe patient-reported criteria are met: Pain and Hurt "
  "score < 60, PROMIS interference > 64, PROMIS behavior > 57. The rubric's Grade 4 "
  "needs at least two.",
  unit="count", outcomes=["10"],
  derived="count of (pain_hurt_score < 60, promis_interference_t > 64, promis_behavior_t > 57)")

# ============================================================ per-outcome: eyes / ENT

F("goldberg_stage", "ord", "Goldberg stage of proliferative sickle retinopathy.",
  values=["1","2","3","4","5"], outcomes=["17"])
F("vision_loss", "ord", "Vision loss attributable to the retinopathy.",
  values=["none","present","legal_blindness"], outcomes=["17"])
F("hearing_loss_db", "num",
  "Hearing loss in decibels at 2 kHz and higher frequencies.", unit="dB", outcomes=["16"])
F("hearing_lowest_affected_khz", "num",
  "Lowest frequency, in kHz, at which loss exceeds 20 dB. Per the rubric's notes: use "
  "8 kHz, or 6 kHz if 8 kHz was not recorded; if 2 kHz is recorded use 2 kHz, else 3 kHz.",
  unit="kHz", outcomes=["16"])

# ============================================================ per-outcome: GI / hepatic

F("gallstone_intervention", "ord",
  "Highest intervention for the gallstone disease. `medical` = oral or IV pain "
  "medication; `surgical` = stone retrieval or removal, stent placement, cholecystectomy.",
  values=["none","medical","surgical"], outcomes=["18"])
F("pancreatitis", "bool", "The gallstone disease was complicated by pancreatitis.", outcomes=["18"])

# ============================================================ per-outcome: renal / repro

F("esrd_progression", "bool",
  "The AKI progressed to permanent end-stage renal disease.", outcomes=["19"])
F("ovarian_reserve_state", "ord",
  "Most severe documented state of ovarian function.",
  values=["normal","diminished","premature_insufficiency","infertility"], outcomes=["22"])
F("semen_category", "ord",
  "Semen analysis category. `oligospermia` includes severe oligospermia and "
  "cryptozoospermia - the rubric groups all impaired spermatogenesis short of "
  "azoospermia into its Grade 2.",
  values=["normal","oligospermia","azoospermia"], outcomes=["23"])
F("infertility", "bool",
  "Infertility documented despite normal semen parameters (male).", outcomes=["23"])
F("fertility_intervention_indicated", "bool",
  "Intervention for fertility indicated (e.g., IUI, IVF, ICSI).", outcomes=["23"])
F("priapism_intervention", "ord",
  "Highest intervention for the priapism episode.",
  values=["none","medication_home","medication_facility","aspiration_irrigation","shunt_surgery"],
  outcomes=["24"])

# ============================================================ per-outcome: growth

F("no_breast_dev_by_13", "bool", "No breast development by age 13 (female).", outcomes=["25"])
F("no_breast_dev_by_14", "bool", "No breast development by age 14 (female).", outcomes=["25"])
F("testes_vol_under_3cc_by_14", "bool",
  "Testes volume below 3 cc at age 14 (male).", outcomes=["25"])
F("no_menses_by_16", "bool",
  "No menses by age 16, or within 4 years of breast development (female).", outcomes=["25"])
F("no_testes_increase_by_16", "bool",
  "No increase in testes volume by age 16 (male).", outcomes=["25"])
F("hormone_replacement_indicated", "bool",
  "Hormone replacement indicated for delayed puberty, either sex.", outcomes=["25"])
F("height_for_age_z", "num", "Height-for-age z-score at a single timepoint.",
  unit="z", outcomes=["26"])
F("height_z_decline", "num",
  "Decline in height-for-age z-score across at least two measurements.",
  unit="z", outcomes=["26"])
F("has_serial_height", "bool",
  "At least two height measurements are available, so the serial-decline table applies "
  "rather than the single-timepoint table.", outcomes=["26"])
F("weight_bmi_z", "num",
  "Weight-for-length z-score, or BMI z-score for age per the rubric's diagnostic criteria.",
  unit="z", outcomes=["27"])

# ============================================================ per-outcome: haematologic

F("pain_co_complication", "bool",
  "The pain episode was complicated by organ failure, acute chest syndrome, "
  "respiratory distress, DVT, PE, stroke, splenic sequestration, or hyperhaemolysis.",
  outcomes=["28"])
F("splenectomy", "bool", "Splenectomy performed or required.", outcomes=["29","31","33"])
F("hemolysis_intervention", "bool",
  "Intervention given for immune-mediated haemolysis: erythrocyte transfusion, "
  "steroids, IVIG, or immunosuppressors.", outcomes=["30"])
F("organ_dysfunction_iron", "bool",
  "Organ dysfunction due to iron overload.", outcomes=["34"])

# ============================================================ per-outcome: infection

F("organ_dysfunction", "bool",
  "Signs or symptoms of organ dysfunction in the setting of the infection.",
  outcomes=["37"], review=True)
F("life_threatening_sepsis", "bool",
  "Life-threatening consequences of sepsis, e.g. haemodynamic instability or "
  "respiratory failure.", outcomes=["37"], review=True)

# ============================================================ per-outcome: malignancy

F("neoplasm_grade", "ord",
  "Neoplasm category per the rubric's worked examples. `low_no_intervention` = "
  "e.g. CIN, incidental teratoma; `low_nonmetastatic` = e.g. carcinoma in situ, basal "
  "cell; `high_single_modality` = e.g. prostate, thyroid, glioma; `high_multimodal` = "
  "e.g. AML, lymphoma, osteosarcoma.",
  values=["none","low_no_intervention","low_nonmetastatic","high_single_modality","high_multimodal"],
  outcomes=["38"])
F("neoplasm_intervention", "ord",
  "Highest intervention for the neoplasm.",
  values=["none","noninvasive","single_modality","urgent_or_multimodal"], outcomes=["38"])
F("chemo_agent_count", "num", "Number of chemotherapy agents used.", unit="count", outcomes=["38"])

# ============================================================ per-outcome: musculoskeletal

F("avn_on_imaging", "bool", "Radiological findings of avascular necrosis.", outcomes=["39"])
F("avn_intervention", "ord",
  "Highest intervention for the AVN. `conservative` = physical therapy, analgesics, or "
  "bone-remodeling agents; `invasive_local` = steroid injection or core decompression; "
  "`joint_replacement` = total joint replacement.",
  values=["none","conservative","invasive_local","joint_replacement"], outcomes=["39"])
F("wound_depth", "ord", "Leg ulcer depth.",
  values=["intact_indurated","partial_thickness","full_thickness_subcutaneous",
          "full_thickness_fascia"], outcomes=["40"])
F("wound_area_cm2", "num", "Leg ulcer wound area.", unit="cm2", outcomes=["40"])
F("necrotic_pct", "num", "Percentage of the wound bed that is necrotic.", unit="%", outcomes=["40"])
F("exudate", "ord", "Leg ulcer exudate volume.",
  values=["minimal","moderate","heavy"], outcomes=["40"])
F("periwound_status", "ord",
  "Periwound skin. `compromised` = erythema, maceration, and clinical signs of infection.",
  values=["intact","compromised"], outcomes=["40"])
F("osteomyelitis_invasive_treatment", "bool",
  "Invasive treatment for the osteomyelitis: surgery to drain an abscess or address "
  "bone complications.", outcomes=["41"])
F("multifocal", "bool", "Multifocal osteomyelitis.", outcomes=["41"])
F("bmd_t_score", "num", "Bone mineral density T-score on DEXA (adults).", unit="T", outcomes=["42"])
F("bmd_z_score", "num", "Bone mineral density Z-score on DEXA (paediatric).", unit="z", outcomes=["42"])
F("low_bmd_on_imaging", "bool",
  "Radiologic evidence of osteoporosis (adult) or low BMD (paediatric).", outcomes=["42"])
F("significant_fracture_history", "bool",
  "History of significant fractures: a long bone fracture of the lower extremity, "
  "vertebral compression, or 2 or more long bone fractures of the upper extremities.",
  outcomes=["42"])
F("height_loss_cm", "num", "Loss of height, in centimetres.", unit="cm", outcomes=["42"])
F("bmd_therapy_indicated", "bool", "Therapy to improve bone mineral density indicated.",
  outcomes=["42"])

# ============================================================ per-outcome: other

F("multiorgan_failure", "bool", "Acute multiorgan failure present.", outcomes=["43"])
F("efw_percentile", "num", "Estimated fetal weight percentile for gestational age.",
  unit="%", outcomes=["44"])
F("fetal_loss", "bool", "Fetal loss at any gestational age.", outcomes=["45"])
F("gestational_age_weeks", "num", "Gestational age at delivery of a liveborn infant.",
  unit="weeks", outcomes=["46"])

# ============================================================ per-outcome: psychiatric

F("phq9", "num", "PHQ-9 score.", unit="score", outcomes=["47"])
F("depression_severity", "ord",
  "Documented depressive symptom severity, when no PHQ-9 score is recorded.",
  values=["none","mild","moderate","severe"], outcomes=["47"], review=True)
F("suicide_attempt", "bool", "Attempted suicide.", outcomes=["47"])
F("threat_of_harm", "bool", "Threats of harm to self or others.", outcomes=["47"])
F("treatment_recommended", "bool",
  "Treatment recommended for the depression: pharmacologic intervention or "
  "behavioral therapy.", outcomes=["47"])

# ============================================================ per-outcome: pulmonary

F("acs_other_support", "bool",
  "Erythropoietin or any other supportive treatment for the ACS, other than antibiotics.",
  outcomes=["48"])
F("asthma_medical_intervention", "bool",
  "Medical intervention sought for the asthma exacerbation.", outcomes=["49"])
F("saba_prn", "bool",
  "Intermittent asthma requiring short-acting beta-agonists as needed.", outcomes=["49"])
F("status_asthmaticus", "bool", "Status asthmaticus.", outcomes=["49"])
F("asthma_mild_symptoms", "bool",
  "Mild asthma symptoms with no medical intervention indicated.", outcomes=["49"])
F("sleep_apnea_treatment_indicated", "bool",
  "Treatment indicated for the sleep apnoea, e.g. CPAP or tonsillectomy.", outcomes=["53"])


def ordinal_rank(name: str, value) -> int | None:
    """Position of `value` in an ordinal feature's ladder, or None if not a member.

    Numeric ladders (Goldberg stage, NYHA class, cytopenia count) are declared as
    strings, so 4, 4.0 and "4" all have to land on the same rung.
    """
    vals = FEATURES[name]["values"]
    if not vals: return None
    if isinstance(value, bool):
        s = str(value).lower()
    elif isinstance(value, float) and value.is_integer():
        s = str(int(value))
    else:
        s = str(value)
    return vals.index(s) if s in vals else None


def review_features() -> list[str]:
    """Features whose rubric wording is a clinical judgement, not an observation."""
    return sorted(n for n, d in FEATURES.items() if d["review"])
