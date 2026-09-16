"""SCOGS decision tables - Layer 4. Deterministic, not learned.

Rows are ordered HIGHEST GRADE FIRST and evaluated top-down, first match wins.
The order is load-bearing: outcomes 19, 34 and 48 have grade cells whose trigger
clauses are identical and separated only by an added severity conjunct, so
bottom-up evaluation silently under-grades all three.

`eval` modes
    first_match  (default) ordered rows, first true row wins
    max_of       highest grade across independent axes (52, per its rubric note)
    stratified   a distinct table per stratum, chosen by `on` before grading

`notes` records places where the rubric itself is ambiguous or self-contradictory
and this file had to choose a reading. Those are the questions to put to a
clinical reviewer; they are not implementation freedom.
"""
from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class Table:
    name: str
    rows: list[tuple[int, str]] = field(default_factory=list)
    eval: str = "first_match"
    on: str | None = None                      # stratified: feature to switch on
    strata: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    axes: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def all_rows(self):
        yield from self.rows
        for rs in self.strata.values(): yield from rs
        for rs in self.axes.values():   yield from rs

    def grades(self) -> set[int]:
        return {g for g, _ in self.all_rows()}

LS = "life_support"          # the rubric's recurring life-supporting-treatment clause

TABLES: dict[str, Table] = {}

def T(num, name, rows=None, **kw):
    TABLES[num] = Table(name=name, rows=list(rows or []), **kw)

# ---------------------------------------------------------------- cardiovascular

T("01", "Arrhythmia", [
    (5, "death_attributed"),
    (4, f"cardioversion and {LS}"),
    (3, "symptomatic and intervention_level == urgent_invasive"),
    (2, "symptomatic and intervention_level == non_urgent_medical"),
    (1, "not symptomatic and intervention_level == none"),
])

T("02", "Deep Vein Thrombosis (DVT)", [
    (5, "death_attributed"),
    (4, "limb_or_life_threatening or hemodynamic_instability or neurologic_instability"),
    (3, "thrombolysis and not surgical_thrombectomy"),
    (2, "anticoagulation and not thrombolysis and not surgical_thrombectomy"),
    (1, "not anticoagulation and not thrombolysis and not surgical_thrombectomy"),
])

T("03", "Diastolic Dysfunction", [
    (2, "diastolic_dysfunction_present"),
], notes=["Grade 2 is the only reachable grade. The rubric marks 1, 3, 4 and 5 N/A "
          "because SCD diastolic dysfunction has no agreed severity gradation; the "
          "outcome is flagged exploratory and did not complete all Delphi rounds."])

T("04", "Heart Failure Exacerbation", [
    (5, "death_attributed"),
    (4, f"hf_intervention and {LS}"),
    (3, f"hf_intervention and not {LS}"),
    (2, "not hf_intervention"),
])

T("05", "Myocardial Infarction", [
    (5, "death_attributed"),
    (4, f"cardiovascular_compromise and {LS}"),
    (3, "mi_severity_criteria and not hemodynamic_instability"),
    (2, "cardiac_biomarker_abnormal and not mi_severity_criteria"),
])

T("06", "Systemic Arterial Hypertension", [
    (5, "death_attributed"),
    (4, "end_organ_damage"),
    (3, "bp_stage == 2 or antihypertensive_count >= 2"),
    (2, "bp_stage == 1 or antihypertensive_count == 1"),
    (1, "antihypertensive_count == 0 and bp_stage == elevated"),
], notes=["bp_stage is resolved against the age-appropriate ACC/AHA table (ages 1-13 "
          "use percentiles, > 13 use absolute mmHg), so patient_age is required."])

T("07", "Systolic Dysfunction", [
    (4, "lvef < 30"),
    (3, "30 <= lvef <= 39"),
    (2, "40 <= lvef <= 49"),
], notes=["The rubric says 'with or without the need for medical or surgical "
          "intervention' at every grade - intervention is explicitly irrelevant here."])

T("08", "TRV Elevation on Echocardiogram", [
    (4, "trv >= 3.0"),
    (3, "2.5 <= trv <= 2.9"),
])

# ------------------------------------------------------------- central nervous system

T("09", "Cerebral Vasculopathy", [
    (5, "(stenosis_pct >= 75 and vessel_segments >= 3) or moyamoya"),
    (4, "(50 <= stenosis_pct <= 74 and vessel_segments > 2) or "
        "(stenosis_pct >= 75 and vessel_segments <= 2)"),
    (3, "50 <= stenosis_pct <= 74 and vessel_segments <= 2"),
    (2, "25 <= stenosis_pct <= 49 and vessel_segments > 2"),
    (1, "25 <= stenosis_pct <= 49 and vessel_segments <= 2"),
], notes=["The segment count qualifies arterial segments, not the number of stenoses "
          "(booklet p.29; corrected in the 2026-08-27 rules.md audit)."])

T("10", "Chronic Pain", [
    (4, "unplanned_visits_12mo > 10 or pro_severe_count >= 2"),
    (3, "4 <= unplanned_visits_12mo <= 10 or pain_hurt_score < 60 or "
        "promis_interference_t > 64 or promis_behavior_t > 57"),
    (2, "1 <= unplanned_visits_12mo <= 3 or 61 <= pain_hurt_score <= 80 or "
        "(48 < promis_interference_t and promis_interference_t <= 64) or "
        "(41 < promis_behavior_t and promis_behavior_t <= 57)"),
    (1, "unplanned_visits_12mo == 0 or pain_hurt_score >= 81 or "
        "promis_interference_t <= 48 or promis_behavior_t <= 41"),
], notes=["RUBRIC GAP: Pain and Hurt score of exactly 60 matches no grade - Grade 3 is "
          "'< 60' and Grade 2 is '61-80'. Left as a gap rather than silently widening "
          "a band. A score of 60 returns cannot-determine on that axis.",
          "The two PROMIS domains have different thresholds (interference 48/64, "
          "behavior 41/57) and are separate features; collapsing them into one "
          "PROMIS score mis-grades every patient scored on behavior alone."])

T("11", "Cognitive Dysfunction", [
    (3, "iq_sd_below_mean >= 3"),
    (2, "2 <= iq_sd_below_mean and iq_sd_below_mean < 3"),
    (1, "1 <= iq_sd_below_mean and iq_sd_below_mean < 2"),
])

T("12", "Elevated TCD Ultrasonography Velocity", [
    (3, "tcd_velocity >= 200"),
    (2, "185 <= tcd_velocity <= 199"),
    (1, "170 <= tcd_velocity <= 184"),
], notes=["Non-imaging TCD only. The rubric deliberately excludes TCDi because "
          "reference ranges for imaging TCD were never established in STOP."])

T("13", "Posterior Reversible Encephalopathy Syndrome (PRES)", [
    (5, "death_attributed"),
    (4, f"pres_confirmed and {LS}"),
    (3, f"pres_confirmed and not {LS}"),
])

T("14", "Silent Cerebral Infarct", [
    (3, "lesion_count != none and cognitive_deficit"),
    (2, "lesion_count == more_than_one and not cognitive_deficit"),
    (1, "lesion_count == one and not cognitive_deficit"),
])

T("15", "Stroke (hemorrhagic or ischemic)", [
    (5, "death_attributed"),
    (4, "stroke_symptoms and not stroke_symptoms_resolved and 4 <= mrs <= 5"),
    (3, "stroke_symptoms and not stroke_symptoms_resolved and 1 <= mrs <= 3"),
    (2, "stroke_symptoms and (stroke_symptoms_resolved or mrs == 0)"),
    (1, "incidental_radiographic_only and not stroke_history"),
])

# ------------------------------------------------------------------------ eyes & ENT

T("16", "Hearing Loss (in at least one ear)", [
    (5, "hearing_loss_db >= 70 and hearing_lowest_affected_khz <= 2"),
    (4, "hearing_loss_db > 40 and hearing_loss_db < 70 and hearing_lowest_affected_khz <= 2"),
    (3, "hearing_loss_db > 20 and 2 <= hearing_lowest_affected_khz <= 3"),
    (2, "hearing_loss_db > 20 and hearing_lowest_affected_khz == 4"),
    (1, "hearing_loss_db > 20 and hearing_lowest_affected_khz > 4"),
], notes=["Grades are read off the rubric's Note lines, which are the operative "
          "definition: the grade is set by the LOWEST frequency at which loss exceeds "
          "20 dB (8 kHz -> G1, 4 kHz -> G2, 2-3 kHz -> G3), then by severity in dB at "
          "2 kHz and above (41-69 -> G4, >= 70 -> G5)."])

T("17", "Sickle Cell Retinopathy (SCR)", [
    (3, "goldberg_stage in (4,5) or vision_loss == legal_blindness"),
    (2, "goldberg_stage == 3 or (goldberg_stage in (1,2) and vision_loss == present)"),
    (1, "goldberg_stage in (1,2) and vision_loss == none"),
], notes=["This outcome has no Grade 5; the rubric notes it does not result in death "
          "and is a deliberate departure from the framework.",
          "Grade 2 reads 'Stage 3 SCR WITHOUT vision loss'; stage 3 with vision loss "
          "short of legal blindness matches no rubric cell. Compiled liberally: "
          "stage 3 grades 2 regardless, since Grade 3 (legal blindness) is tested "
          "first. A rubric gap resolved rather than left open - for clinical review."])

# --------------------------------------------------------------------- gastrointestinal

T("18", "Cholecystitis/Cholelithiasis (gallstones)", [
    (5, "death_attributed"),
    (4, f"pancreatitis or {LS}"),
    (3, "gallstone_intervention == surgical"),
    (2, "gallstone_intervention == medical"),
    (1, "gallstone_intervention == none"),
], notes=["'incidental finding' in Grade 1 is the rubric's example of requiring no "
          "intervention, not an additional conjunct."])

# ----------------------------------------------------- genitourinary / renal / reproductive

_AKI = ("(creatinine_x_baseline >= 3.0 or creatinine >= 4 or renal_replacement or "
        "(patient_age < 18 and egfr < 35))")

T("19", "Acute Kidney Injury (AKI)", [
    (5, "death_attributed"),
    (4, f"{_AKI} and esrd_progression"),
    (3, f"{_AKI} and not esrd_progression"),
    (2, "2.0 <= creatinine_x_baseline <= 2.9"),
    (1, "1.5 <= creatinine_x_baseline <= 1.9 or creatinine_increase_mg_dl >= 0.3"),
], notes=["ORDER IS LOAD-BEARING: Grades 3 and 4 share an identical trigger set and "
          "differ only by progression to ESRD. Evaluating upward would return 3 for "
          "every Grade 4 patient.",
          "The paediatric eGFR branch needs patient_age; without it the Grade 3/4 "
          "trigger is UNKNOWN rather than false."])

T("20", "Acute Papillary Necrosis", [
    (5, "death_attributed"),
    (4, LS),
    (3, f"care_setting >= inpatient and treated and not {LS}"),
    (2, "care_setting >= clinic_or_day_hospital and care_setting < inpatient and treated"),
    (1, "care_setting >= clinic_or_day_hospital and care_setting < inpatient and not treated"),
])

T("21", "Chronic Kidney Disease (CKD)", [
    (5, "egfr < 15 or death_attributed"),
    (4, "(45 <= egfr <= 59 and albuminuria > 300) or (30 <= egfr <= 44 and albuminuria > 30) "
        "or (15 <= egfr <= 29)"),
    (3, "(30 <= egfr <= 44 and albuminuria < 30) or (45 <= egfr <= 59 and 30 <= albuminuria <= 300) "
        "or (60 <= egfr <= 89 and albuminuria > 300)"),
    (2, "(45 <= egfr <= 59 and albuminuria < 30) or (egfr >= 60 and 30 <= albuminuria <= 300)"),
    (1, "60 <= egfr <= 89 and albuminuria < 30"),
], notes=["A 2-D grid over eGFR x albuminuria, not a ladder - both axes are required."])

T("22", "Female Ovarian Dysfunction", [
    (3, "ovarian_reserve_state == infertility"),
    (2, "ovarian_reserve_state == premature_insufficiency"),
    (1, "ovarian_reserve_state == diminished"),
])

T("23", "Male Impairments", [
    (3, "semen_category == azoospermia and fsh >= 7.6"),
    (2, "semen_category == oligospermia and fsh >= 7.6"),
    (1, "infertility and semen_category == normal and fsh < 7.6 and "
        "not fertility_intervention_indicated"),
], notes=["Grade 1 is the rubric's full conjunction - intervention not indicated AND "
          "infertility despite normal semen parameters AND FSH < 7.6 - not merely "
          "normal labs. An earlier compile dropped the infertility and "
          "no-intervention conjuncts."])

T("24", "Priapism", [
    (4, "priapism_intervention == shunt_surgery"),
    (3, "priapism_intervention == aspiration_irrigation"),
    (2, "priapism_intervention == medication_facility"),
    (1, "priapism_intervention <= medication_home"),
])

# ------------------------------------------------------------------ growth & development

T("25", "Delayed puberty", [
    (3, "no_breast_dev_by_14 or no_menses_by_16 or no_testes_increase_by_16 or "
        "hormone_replacement_indicated"),
    (2, "no_breast_dev_by_13 or testes_vol_under_3cc_by_14"),
])

T("26", "Malnutrition Leading to Stunting (Decreased Height Velocity)",
  eval="stratified", on="has_serial_height",
  strata={
    "false": [                                   # a single data point
        (5, "death_attributed"),
        (4, "height_for_age_z < -3"),
    ],
    "true": [                                    # at least two data points
        (5, "death_attributed"),
        (4, "height_z_decline >= 3"),
        (3, "height_z_decline >= 2"),
        (2, "height_z_decline >= 1"),
    ]},
  notes=["The stratum is data availability, not age: one height measurement grades on "
         "absolute z-score, two or more grade on the decline between them."])

T("27", "Underweight", [
    (5, "death_attributed"),
    (4, "weight_bmi_z <= -3"),
    (3, "-3 < weight_bmi_z and weight_bmi_z <= -2"),
    (2, "-2 < weight_bmi_z and weight_bmi_z <= -1"),
])

# ------------------------------------------------------------------------ hematologic

T("28", "Acute Sickle Cell Pain Episode", [
    (5, "death_attributed"),
    (4, f"pain_co_complication or {LS}"),
    (3, "care_setting >= inpatient and not pain_co_complication"),
    (2, "care_setting >= clinic_or_day_hospital and care_setting < inpatient"),
    (1, "care_setting == home"),
])

T("29", "Acute Splenic Sequestration", [
    (5, "death_attributed"),
    (4, "hemodynamic_instability or splenectomy"),
    (3, "treated and not hemodynamic_instability and not splenectomy"),
    (1, "not treated"),
], notes=["RUBRIC CONTRADICTION: Grade 3 requires 'treatment (i.e. erythrocyte "
          "transfusion, IV fluids, etc.)' and simultaneously excludes 'requiring "
          "splenectomy, fluids, etc.' - IV fluids appear as both a qualifying and a "
          "disqualifying treatment. Confirmed present in the booklet (p.73), so it is "
          "not a transcription error. Implemented as: any treatment qualifies, only "
          "splenectomy disqualifies, which is the only reading under which Grade 3 is "
          "reachable. Needs a clinical ruling."])

T("30", "Alloimmunization / Delayed Hemolytic Transfusion Reaction (DHTR)", [
    (5, "death_attributed"),
    (4, f"hb_decline_pct >= 20 and {LS}"),
    (3, "hb_decline_pct >= 20 or hemolysis_intervention"),
    (2, "hb_decline_pct < 20 and not hemolysis_intervention"),
    (1, "not hemolysis_intervention"),
], notes=["Grades 1 and 2 overlap by construction: Grade 1 is 'no intervention' and "
          "Grade 2 is 'decline < 20% AND no intervention', so every Grade 2 patient "
          "also satisfies Grade 1. Highest-first ordering resolves it - Grade 1 is "
          "reached only when the haemoglobin decline is not documented."])

T("31", "Chronic Hypersplenism", [
    (3, "cytopenia_count == 3 or splenectomy"),
    (2, "cytopenia_count == 2 and not splenectomy"),
    (1, "cytopenia_count == 1 and thrombocytopenia and not splenectomy"),
], notes=["Grade 1 is 'isolated THROMBOCYTOPENIA' specifically, not any single "
          "cytopenia. An isolated anaemia or leucopenia matches no cell and returns "
          "absent - a rubric gap, left as a gap rather than silently graded 1."])

T("32", "Hepatopathy", [
    (5, "death_attributed"),
    (4, f"meld >= 40 or {LS}"),
    (3, f"30 <= meld <= 39 and treated and not {LS}"),
    (2, f"meld <= 29 and treated and not {LS}"),
    (1, "meld <= 29 and not treated"),
])

T("33", "Splenic Infarction", [
    (5, "death_attributed"),
    (4, f"{LS} or splenectomy"),
    (3, f"care_setting >= inpatient and treated and not {LS}"),
    (2, f"care_setting >= clinic_or_day_hospital and care_setting < inpatient and not {LS}"),
    (1, "incidental_finding and not treated"),
])

T("34", "Transfusional Iron Overload (Hemochromatosis or Hemosiderosis)", [
    (5, "death_attributed"),
    (4, "(liver_iron_conc >= 15 and ferritin > 10000) or mri_t2star < 20 or organ_dysfunction_iron"),
    (3, "(liver_iron_conc >= 15 or 5000 <= ferritin <= 10000) and mri_t2star >= 20 "
        "and not organ_dysfunction_iron"),
    (2, "(7 <= liver_iron_conc <= 14.9 or 2000 <= ferritin <= 4999) and mri_t2star >= 20 "
        "and not organ_dysfunction_iron"),
    (1, "(2.5 <= liver_iron_conc <= 6.9 or 1000 <= ferritin <= 1999) and mri_t2star >= 20 "
        "and not organ_dysfunction_iron"),
], notes=["ORDER IS LOAD-BEARING: Grades 3 and 4 both trigger on LIC >= 15. Grade 4 "
          "additionally requires ferritin > 10,000, a T2* under 20 msec, or iron-"
          "attributable organ dysfunction, so it must be tested first."])

T("35", "Transient Aplastic Crisis Secondary to Parvovirus B19 Infection", [
    (5, "death_attributed"),
    (4, f"hb_nadir < 5 and treated and {LS}"),
    (3, f"hb_nadir < 5 and treated and not {LS}"),
    (2, "hb_nadir >= 5 and treated"),
    (1, "not treated"),
])

# --------------------------------------------------------------------- infectious disease

T("36", "Fever", [
    (2, "temperature >= 38.5"),
    (1, "38.0 <= temperature < 38.5"),
], notes=["RUBRIC GAP CLOSED (2026-08-31): the rubric's Celsius text - Grade 1 "
          "'38.0-38.4', Grade 2 '> 38.5' - leaves (38.4, 38.5] matching no grade, so a "
          "temperature of exactly 38.5 degC graded as absent. The rubric's own Fahrenheit "
          "annotations resolve it: Grade 1 is '(100.4 - 101.2 degrees Fahrenheit)' = "
          "38.0000-38.4444 degC, and Grade 2 is '(101.3 degrees Fahrenheit)' = exactly "
          "38.5000 degC. The bands are contiguous in Fahrenheit and the Celsius text is a "
          "one-decimal rounding of 38.4444. So the intent is [38.0, 38.5) and [38.5, inf), "
          "which is what these rows now say. Under the rubric's literal '<= 38.4' a note "
          "reporting 101.2 F - the rubric's OWN Grade 1 endpoint - converts to 38.4444 and "
          "grades absent. Deviation from rules.md, recorded in "
          "rules_vs_booklet_discrepancies.md. The neighbouring gap at outcome 10 "
          "(Pain and Hurt score of exactly 60) carries no such evidence and stays open."])

T("37", "Sepsis", [
    (5, "death_attributed"),
    (4, f"life_threatening_sepsis and {LS}"),
    (3, "blood_culture_positive and (organ_dysfunction or treated)"),
])

# ------------------------------------------------------------------------- malignancies

T("38", "Malignant Neoplasms", [
    (5, "death_attributed"),
    (4, "neoplasm_intervention == urgent_or_multimodal or neoplasm_grade == high_multimodal "
        "or chemo_agent_count > 1"),
    (3, "care_setting >= inpatient or adl_limitation == self_care_adl or "
        "neoplasm_grade == high_single_modality"),
    (2, "neoplasm_intervention == noninvasive or adl_limitation == instrumental_adl or "
        "neoplasm_grade == low_nonmetastatic"),
    (1, "not symptomatic or neoplasm_grade == low_no_intervention"),
])

# ------------------------------------------------------------------ muscular / skeletal / skin

T("39", "Avascular Necrosis of Joints (AVN)", [
    (4, "avn_on_imaging and adl_limitation >= instrumental_adl and "
        "avn_intervention == joint_replacement"),
    (3, "avn_on_imaging and adl_limitation >= instrumental_adl and "
        "avn_intervention >= conservative and avn_intervention < joint_replacement"),
    (2, "avn_on_imaging and adl_limitation >= instrumental_adl and avn_intervention == none"),
    (1, "avn_on_imaging and not symptomatic"),
], notes=["The rubric's 'AND/OR limiting instrumental ADL' clause is read as AND in "
          "Grades 2-4 alike (an earlier compile conjoined it in Grade 2 but dropped "
          "it from Grade 3). Read as OR, every asymptomatic incidental finding would "
          "out-rank Grade 1's own cell.",
          "'Limiting instrumental ADL' is compiled as >= instrumental_adl: a "
          "self-care limitation implies the instrumental one, and == would leave a "
          "joint-replacement patient with a self-care limitation matching no cell."])

T("40", "Leg Ulcer", [
    (4, "wound_depth == full_thickness_fascia and wound_area_cm2 > 8 and necrotic_pct >= 50 "
        "and exudate == heavy and periwound_status == compromised"),
    (3, "wound_depth == full_thickness_subcutaneous and wound_area_cm2 > 8 and necrotic_pct < 50 "
        "and exudate == moderate and periwound_status == intact"),
    (2, "wound_depth == partial_thickness and wound_area_cm2 < 8 and exudate == minimal "
        "and periwound_status == intact"),
    (1, "wound_depth == intact_indurated"),
], notes=["'Greater than 8 cm2 < 50% wound bed with necrotic tissue' is one clause - "
          "area AND necrotic fraction - not two (booklet p.99; the spurious extra AND "
          "was corrected in the 2026-08-27 rules.md audit).",
          "RUBRIC GAP: a wound of exactly 8 cm2 matches no cell - Grade 2 is '< 8' "
          "and Grades 3-4 are '> 8'. Same gap class as outcomes 10 and 36; left as "
          "a gap rather than widening a band."])

T("41", "Osteomyelitis", [
    (5, "death_attributed"),
    (4, "osteomyelitis_invasive_treatment and multifocal"),
    (3, "osteomyelitis_invasive_treatment and not multifocal"),
    (2, "parenteral_antibiotics and not osteomyelitis_invasive_treatment and not multifocal"),
    (1, "not parenteral_antibiotics and not osteomyelitis_invasive_treatment and not multifocal"),
])

T("42", "Osteoporosis",
  eval="stratified", on="age_stratum",
  strata={
    "adult": [
        (3, "height_loss_cm >= 2 or care_setting >= inpatient or adl_limitation == self_care_adl"),
        (2, "((bmd_t_score < -2.5 or height_loss_cm < 2 or adl_limitation == instrumental_adl) "
            "and significant_fracture_history) or bmd_therapy_indicated"),
        (1, "(low_bmd_on_imaging or (-2.5 <= bmd_t_score and bmd_t_score <= -1)) "
            "and not significant_fracture_history"),
    ],
    "pediatric": [
        (3, "adl_limitation == self_care_adl"),
        (2, "(bmd_z_score <= -2.0 and significant_fracture_history) or bmd_therapy_indicated"),
        (1, "low_bmd_on_imaging and bmd_z_score <= -2.0 and not significant_fracture_history"),
    ]},
  notes=["The booklet prints ONE table with 'Adult:' and 'Pediatric:' clauses inside "
         "each cell, not two tables. It is split here because the two readings never "
         "interact and patient_age selects between them.",
         "The fracture-history and BMD-therapy clauses are marked 'for both pediatrics "
         "and adults' and are applied to both strata: Grade 1 requires NO significant "
         "fracture history and Grade 2 requires it (or BMD therapy indicated). That "
         "conjunct was missing from the earlier draft tables, which made Grade 1 and "
         "Grade 2 indistinguishable on BMD alone.",
         "Pediatric Grade 1's 'radiologic evidence of low BMD WITH z-score <= -2.0' "
         "is one criterion - both conjuncts - not two alternatives; an earlier "
         "compile read the 'with' as OR."])

# ------------------------------------------------------------------------------- other

T("43", "Acute Multiorgan Failure", [
    (5, "death_attributed"),
    (4, f"multiorgan_failure and {LS}"),
    (3, f"multiorgan_failure and not {LS}"),
])

# ------------------------------------------------------- pregnancy / puerperium / perinatal

T("44", "Fetal Growth Restriction", [
    (4, "efw_percentile < 1"),
    (3, "efw_percentile < 5"),
    (2, "efw_percentile < 10"),
])

T("45", "Pregnancy Loss", [
    (4, "fetal_loss"),
], notes=["Grade 4 is the only reachable grade: fetal loss at any gestational age."])

T("46", "Premature Delivery", [
    (4, "gestational_age_weeks <= 24"),
    (3, "24 < gestational_age_weeks and gestational_age_weeks <= 28"),
    (2, "28 < gestational_age_weeks and gestational_age_weeks <= 34"),
    (1, "34 < gestational_age_weeks and gestational_age_weeks <= 37"),
])

# --------------------------------------------------------------- psychiatric / psychosocial

T("47", "Depression", [
    (5, "death_attributed"),
    (4, "20 <= phq9 <= 27 or suicide_attempt or threat_of_harm"),
    (3, "(15 <= phq9 <= 19 or depression_severity == severe) and adl_limitation != none "
        "and treatment_recommended"),
    (2, "(10 <= phq9 <= 14 or depression_severity == moderate) and "
        "adl_limitation == instrumental_adl and treatment_recommended"),
    (1, "(5 <= phq9 <= 9 or depression_severity == mild) and not treatment_recommended"),
], notes=["Each grade offers a symptom-severity route OR a PHQ-9 route; both are "
          "carried so a note without a PHQ-9 score is still gradeable.",
          "Grade 4's trailing 'AND treatment recommended' is not applied: read "
          "distributively it would make an attempted suicide ungradeable whenever "
          "treatment was not documented. Needs a clinical ruling."])

# ---------------------------------------------------------------------------- pulmonary

_ACS3 = "fio2_pct >= 50 or resp_support >= high_flow or transfusion_type == exchange"

T("48", "Acute Chest Syndrome (ACS)", [
    (5, "death_attributed"),
    (4, f"({_ACS3}) and {LS}"),
    (3, _ACS3),
    (2, "(21 < fio2_pct and fio2_pct < 50 or transfusion_type == simple) and "
        "resp_support < high_flow and transfusion_type != exchange"),
    (1, "resp_support == room_air and transfusion_type == none and not acs_other_support"),
], notes=["ORDER IS LOAD-BEARING: Grades 3 and 4 share an identical trigger set and "
          "differ only by the added critical-support conjunct.",
          "'BiPAP OR high flow O2' is one step on the resp_support ladder: "
          "resp_support >= high_flow covers both."])

T("49", "Asthma Exacerbation", [
    (5, "death_attributed"),
    (4, f"{LS} or resp_support >= niv_bipap_cpap"),
    (3, "adl_limitation == self_care_adl or resp_support >= low_flow_o2 or status_asthmaticus"),
    (2, "((symptomatic and asthma_medical_intervention and resp_support == room_air) or "
        "adl_limitation == instrumental_adl or saba_prn) and not status_asthmaticus"),
    (1, "asthma_mild_symptoms and not asthma_medical_intervention"),
])

T("50", "Chronic Restrictive Lung Physiology", [
    (5, "death_attributed"),
    (4, "tlc_pct_pred < 50"),
    (3, "50 <= tlc_pct_pred <= 60"),
    (2, "60 < tlc_pct_pred and tlc_pct_pred <= 70"),
    (1, "70 < tlc_pct_pred and tlc_pct_pred <= 80"),
])

T("51", "Pulmonary Embolism (PE)", [
    (5, "death_attributed"),
    (4, f"care_setting == icu and symptomatic and "
        f"(cardiac_compromise or hemodynamic_instability) and {LS}"),
    (3, f"care_setting >= inpatient and symptomatic and cardiac_compromise and "
        f"not hemodynamic_instability and not {LS}"),
    (2, "symptomatic and not cardiac_compromise and not hemodynamic_instability"),
    (1, "incidental_finding and not symptomatic"),
])

T("52", "Pulmonary Hypertension",
  eval="max_of",
  axes={
    "death": [(5, "death_attributed")],
    "mpap":  [(3, "mpap > 35"), (2, "25 <= mpap <= 35"), (1, "20 <= mpap and mpap < 25")],
    "nyha":  [(4, "nyha_class == 4"), (3, "nyha_class == 3"),
              (2, "nyha_class in (2,3)"), (1, "nyha_class in (1,2)")],
    "echo":  [(4, f"right_heart_failure and (low_cardiac_output or resp_support >= niv_bipap_cpap)")],
  },
  notes=["max_of, per the rubric's own worked example: 'if patient fits Grade 2 "
         "criteria based on mPAP, but Grade 3 based on NYHA and echocardiogram "
         "criteria, this patient will be classified as Grade 3'. First-match across "
         "conjoined axes returns nothing for a patient whose axes disagree, which is "
         "indistinguishable from absent - that note was truncated out of rules.md and "
         "restored by the 2026-08-27 audit.",
         "Grades 1-3 all carry 'NO evidence of right heart failure'; that negative "
         "is not encoded, so a patient with right heart failure who does not meet "
         "Grade 4's second conjunct still grades from the mPAP/NYHA axes. Liberal "
         "reading - for clinical review."])

T("53", "Sleep Apnea (obstructive or central)",
  eval="stratified", on="age_stratum",
  strata={
    "adult": [
        (4, "ahi > 30"),
        (3, "15 < ahi and ahi <= 30"),
        (2, "5 <= ahi <= 15 and sleep_apnea_treatment_indicated"),
        (1, "5 <= ahi <= 15 and not sleep_apnea_treatment_indicated"),
    ],
    "pediatric": [
        (4, "(15.1 <= ahi <= 24.9 and spo2_desat_over_3min) or ahi >= 25"),
        (3, "(8.0 <= ahi <= 15.0 and spo2_desat_over_3min) or "
            "(15.1 <= ahi <= 24.9 and not spo2_desat_over_3min)"),
        (2, "(3.1 <= ahi <= 7.9 and spo2_desat_over_3min) or "
            "(8.0 <= ahi <= 15.0 and not spo2_desat_over_3min)"),
        (1, "(ahi <= 3 and spo2_desat_over_3min) or "
            "(3.1 <= ahi <= 7.9 and not spo2_desat_over_3min)"),
    ]},
  notes=["Adult >= 18 / paediatric < 18, per the booklet's table sub-headings.",
         "The paediatric grades pair an AHI band with the overnight desaturation "
         "criterion; clause order was inverted in all four cells and corrected in the "
         "2026-08-27 rules.md audit."])

assert len(TABLES) == 53, f"expected 53 outcomes, got {len(TABLES)}"
