# Synthetic Notes — Ground-Truth Answer Key

> **Author:** Daffa Warsa  
> **Branch:** `Daffa-Synthetic-Notes`  
> **Data File:** [`data/synthetic_notes.csv`](data/synthetic_notes.csv)  
> **Grading Authority:** [`rules.md`](rules.md) + [`data/scogs_feature_schema.json`](data/scogs_feature_schema.json)

---

## Overview

Eight synthetic clinical notes were written with intentionally embedded clinical details that deterministically map to known SCOGS grades. Each note targets one or more outcomes across the full severity spectrum (Grades 1–5), providing ground-truth examples that the existing `clincal_notes.csv` dataset lacks (which only ever grades one outcome using grades 2–3).

| Note | Patient | Scenario | Outcomes Graded | Grade Range |
|:----:|:--------|:---------|:----------------|:------------|
| **1** | SYNTH_001 — 22F, HbSS | Mild — ED treat-and-release | Pain (#28) = 2, Fever (#36) = 1 | Grades 1–2 |
| **2** | SYNTH_002 — 17M, HbSS | Severe — Admitted with ACS | Pain (#28) = 4, ACS (#48) = 3, Fever (#36) = 2 | Grades 2–4 |
| **3** | SYNTH_003 — 29M, HbSC | Life-threatening — ICU | Pain (#28) = 4, ACS (#48) = 4, Fever (#36) = 2, AKI (#19) = 2 | Grades 2–4 |
| **4** | SYNTH_004 — 19F, HbSS | Mild — Home-managed pain | Pain (#28) = 1 | Grade 1 |
| **5** | SYNTH_005 — 34M, HbSS | Admitted — Pain + DVT | Pain (#28) = 3, DVT (#02) = 2 | Grades 2–3 |
| **6** | SYNTH_006 — 42F, HbSS | Fatal — ACS death | ACS (#48) = 5, Pain (#28) = 5, AKI (#19) = 3, Fever (#36) = 2 | Grades 2–5 |
| **7** | SYNTH_007 — 14F, HbSC | Mild ACS — Room air, antibiotics only | ACS (#48) = 1, Pain (#28) = 2, Fever (#36) = 2 | Grades 1–2 |
| **8** | SYNTH_008 — 16M, HbSS | Priapism — Aspiration/irrigation | Priapism (#24) = 3, Pain (#28) = 3 | Grade 3 |
| **9** | SYNTH_009 — 25M, HbSS | Moderate ACS — Simple transfusion + nasal cannula | ACS (#48) = 2, Pain (#28) = 3, Fever (#36) = 2 | Grades 2–3 |
| **10** | SYNTH_010 — 31F, HbSS | Ischemic stroke — Residual deficits, mRS 2 | Stroke (#15) = 3 | Grade 3 |
| **11** | SYNTH_011 — 8M, HbSS | Pediatric sepsis — Pneumococcal bacteremia | Sepsis (#37) = 3, Pain (#28) = 3, Fever (#36) = 2 | Grades 2–3 |
| **12** | SYNTH_012 — 27F, HbSS | Borderline — All values just below thresholds | Pain (#28) = 2 (5 outcomes tested as absent) | Grade 2 |

---

## Note 1: SYNTH_001 — Mild/Moderate ED Visit

**Patient:** 22-year-old female, HbSS, diagnosed age 2  
**Setting:** Emergency department, treated and released (not admitted)

### Graded Outcomes

#### Acute Sickle Cell Pain Episode (#28) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `care_setting` | `ed_treat_release` | *"Patient monitored in ED for 4 hours with improvement. Not admitted. Discharged home"* |
| `pain_co_complication` | `false` | *"No signs of acute chest syndrome, sepsis, or other infectious source identified"*; CXR clear |
| `life_support` | `false` | No mention of ventilation, vasopressors, or RRT |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `care_setting >= clinic_or_day_hospital AND care_setting < inpatient` → **Grade 2** ✓

#### Fever (#36) → Grade 1

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `temperature` | `38.3 °C` | *"Temperature 38.3 degrees Celsius (101.0 degrees Fahrenheit)"* |

**Rule applied:** `38.0 ≤ temperature < 38.5` → **Grade 1** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| ACS (#48) | *"Chest X-ray: No acute cardiopulmonary disease, no infiltrates, no pleural effusions"* — no lobar infiltrate = ACS diagnostic criteria not met |
| DVT (#02) | *"No history of DVT"*; no calf tenderness, no edema |
| PE (#51) | *"No history of... pulmonary embolism"*; SpO2 98%, clear lungs |
| Stroke (#15) | *"cranial nerves II-XII intact, no focal motor or sensory deficits"* |
| AKI (#19) | Creatinine 0.7 mg/dL (baseline 0.6), ratio 1.17x — below 1.5x threshold |
| Leg Ulcer (#40) | *"no leg ulcers"* |

---

## Note 2: SYNTH_002 — Severe Admission with ACS

**Patient:** 17-year-old male, HbSS, chronic transfusion program  
**Setting:** ER → admitted to medical floor (7-day stay)

### Graded Outcomes

#### Acute Sickle Cell Pain Episode (#28) → Grade 4

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `care_setting` | `inpatient` | *"Patient admitted to the medical floor"* |
| `pain_co_complication` | `true` | *"Acute sickle cell vaso-occlusive pain crisis complicated by acute chest syndrome"* |
| `life_support` | `false` | *"Patient did not require intubation, vasopressors, or other life-support interventions"* |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `pain_co_complication` → **Grade 4** ✓  
(First-match: Grade 5 skipped because `death_attributed = false`; Grade 4 matches because `pain_co_complication = true`)

#### Acute Chest Syndrome (#48) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `fio2_pct` | `60%` | *"high-flow nasal cannula at 40 L/min with FiO2 60%"* |
| `resp_support` | `high_flow` | *"Transitioned to high-flow nasal cannula"* |
| `transfusion_type` | `exchange` | *"Automated red cell exchange performed, reducing HbS from 78% to 28%"* |
| `acs_other_support` | `false` | No erythropoietin or other non-antibiotic support mentioned |
| `life_support` | `false` | *"Patient did not require intubation, vasopressors, or other life-support interventions"* |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `fio2_pct >= 50 OR resp_support >= high_flow OR transfusion_type == exchange` → **Grade 3** ✓  
(Not Grade 4 because `life_support = false`)

#### Fever (#36) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `temperature` | `39.2 °C` | *"Temperature 39.2 degrees Celsius (102.6 degrees Fahrenheit)"* |

**Rule applied:** `temperature >= 38.5` → **Grade 2** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| AKI (#19) | *"Serum creatinine 0.9 mg/dL (baseline 0.8 mg/dL, ratio 1.13x baseline)"* — below 1.5x threshold |
| DVT (#02) | *"No evidence of DVT on bilateral lower extremity Doppler ultrasound"* |
| PE (#51) | *"CT pulmonary angiography: No evidence of pulmonary embolism"* |
| Stroke (#15) | *"no focal neurological deficits, moving all extremities symmetrically"* |
| Arrhythmia (#01) | *"No arrhythmias noted on telemetry monitoring"* |
| Sepsis (#37) | *"Blood cultures returned negative"* |
| Depression (#47) | *"No depressive symptoms reported"* |
| Leg Ulcer (#40) | *"no leg ulcers"* |

---

## Note 3: SYNTH_003 — Life-Threatening ICU Admission

**Patient:** 29-year-old male, HbSC, history of AVN of right hip  
**Setting:** ER → ICU (14-day total stay, 6 days intubated)

### Graded Outcomes

#### Acute Sickle Cell Pain Episode (#28) → Grade 4

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `care_setting` | `icu` | *"Patient emergently intubated in the ED"*; ICU admission |
| `pain_co_complication` | `true` | *"Acute sickle cell vaso-occlusive crisis complicated by acute chest syndrome and requiring life-supporting interventions"* |
| `life_support` | `true` | Invasive mechanical ventilation + norepinephrine + vasopressin |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `pain_co_complication OR life_support` → **Grade 4** ✓

#### Acute Chest Syndrome (#48) → Grade 4

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `fio2_pct` | `100%` | *"Post-intubation ventilator settings: assist-control mode, FiO2 100%"* |
| `resp_support` | `invasive_ventilation` | *"Patient emergently intubated... for acute hypoxemic respiratory failure"* |
| `transfusion_type` | `exchange` | *"Urgent automated red cell exchange transfusion performed, reducing HbS from 52% to 18%"* |
| `acs_other_support` | `false` | No erythropoietin mentioned |
| `life_support` | `true` | Intubation + norepinephrine 0.3 mcg/kg/min + vasopressin 0.04 units/min |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `(fio2_pct >= 50 OR resp_support >= high_flow OR transfusion_type == exchange) AND life_support` → **Grade 4** ✓  
(All three base triggers are true: FiO2 100% ≥ 50, invasive ventilation ≥ high_flow, exchange transfusion; AND life_support = true → Grade 4, not just Grade 3)

#### Fever (#36) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `temperature` | `39.8 °C` | *"Temperature 39.8 degrees Celsius (103.6 degrees Fahrenheit)"* |

**Rule applied:** `temperature >= 38.5` → **Grade 2** ✓

#### Acute Kidney Injury (#19) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:------------|
| `creatinine_x_baseline` | `2.8` | *"Serum creatinine peaked at 2.8 mg/dL on day 2 (2.8 times baseline of 1.0 mg/dL)"* |
| `creatinine` | `2.8 mg/dL` | Same as above |
| `renal_replacement` | `false` | *"renal replacement therapy was not required"* |
| `esrd_progression` | `false` | *"Creatinine normalized to 1.1 mg/dL by discharge"* — transient, resolved |
| `patient_age` | `29 years` | 29-year-old male |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `2.0 ≤ creatinine_x_baseline ≤ 2.9` → **Grade 2** ✓  
(Not Grade 3: would need `creatinine_x_baseline >= 3.0` or `creatinine >= 4.0` or `renal_replacement`. 2.8x is below 3.0, creatinine 2.8 < 4.0, no RRT)

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| DVT (#02) | *"Lower extremity Doppler ultrasound: No evidence of deep vein thrombosis"* |
| PE (#51) | *"CT pulmonary angiography: No evidence of pulmonary embolism"* |
| Stroke (#15) | *"No seizures or focal neurological deficits observed throughout admission"*; *"Mental status returned to baseline"* |
| Arrhythmia (#01) | *"No arrhythmias on continuous telemetry"* |
| Depression (#47) | *"PHQ-9 score 3 (minimal depression)"* — below clinical threshold |
| Hearing Loss (#16) | *"Hearing intact bilaterally"* |
| Leg Ulcer (#40) | *"no leg ulcers"* |

> [!NOTE]
> **Sepsis (#37)** is clinically evident in Note 3 (positive blood cultures for *S. pneumoniae* + hemodynamic instability requiring vasopressors). The exact SCOGS grade should be determined by consulting [`rules.md` §37](rules.md) for the Sepsis grading table. This was intentionally left ungraded in the CSV to flag as an exercise or for team verification.

---

## Note 4: SYNTH_004 — Mild Home-Managed Pain

**Patient:** 19-year-old female, HbSS, diagnosed at birth  
**Setting:** Telephone encounter — managed entirely at home

### Graded Outcomes

#### Acute Sickle Cell Pain Episode (#28) → Grade 1

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `care_setting` | `home` | *"Patient called the sickle cell clinic nurse line"*; managed with oral analgesics at home, never visited a facility |
| `pain_co_complication` | `false` | *"No signs or symptoms concerning for acute chest syndrome, fever, infection, or other complications"* |
| `life_support` | `false` | Home management only |
| `death_attributed` | `false` | Patient is alive and improving |

**Rule applied:** `care_setting == home` → **Grade 1** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| Fever (#36) | *"Temperature at home measured 37.2 degrees Celsius"* — below 38.0 threshold |
| ACS (#48) | *"No signs or symptoms concerning for acute chest syndrome"*; no cough, no dyspnea |
| Depression (#47) | *"PHQ-9 score of 3 (minimal symptoms, no treatment recommended)"* — below Grade 1 threshold of 5 |

---

## Note 5: SYNTH_005 — Admitted Pain + DVT

**Patient:** 34-year-old male, HbSS  
**Setting:** ER → admitted to medical floor (4-day stay)

### Graded Outcomes

#### Acute Sickle Cell Pain Episode (#28) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `care_setting` | `inpatient` | *"Patient admitted to the medical floor for management"* |
| `pain_co_complication` | `false` | *"Acute sickle cell vaso-occlusive pain crisis requiring inpatient admission, without co-complications"*; DVT is a separate diagnosis, not listed as a pain co-complication (ACS/DVT/PE/stroke/sequestration/hyperhaemolysis) triggering the pain episode |
| `life_support` | `false` | No ventilation, vasopressors, or RRT |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `care_setting >= inpatient AND NOT pain_co_complication` → **Grade 3** ✓

> [!NOTE]
> DVT is listed in the Pain #28 Grade 4 co-complications. However, in this note the DVT is a **concurrent but separate diagnosis** discovered on imaging, not a complication arising from the pain crisis itself. The note explicitly states "without co-complications." If the team determines DVT should be coded as a co-complication here, the Pain grade would be **Grade 4** instead.

#### Deep Vein Thrombosis (#02) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `anticoagulation` | `true` | *"Heparin IV infusion initiated"*; *"transitioned to therapeutic enoxaparin"*; discharged on rivaroxaban |
| `thrombolysis` | `false` | *"No thrombolysis or surgical thrombectomy was required"* |
| `surgical_thrombectomy` | `false` | Same as above |
| `limb_or_life_threatening` | `false` | *"no signs of limb-threatening ischemia"*; *"hemodynamically stable"* |
| `hemodynamic_instability` | `false` | Stable vitals throughout |
| `neurologic_instability` | `false` | *"no focal deficits"* |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `anticoagulation AND NOT thrombolysis AND NOT surgical_thrombectomy` → **Grade 2** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| Fever (#36) | *"temperature remained below 38.0 degrees Celsius throughout"* |
| PE (#51) | *"CT pulmonary angiography: No evidence of pulmonary embolism"* |
| ACS (#48) | *"Chest X-ray: Clear lungs, no infiltrates"* |
| Stroke (#15) | *"no focal deficits"* |
| Arrhythmia (#01) | *"No arrhythmias on telemetry"* |
| Leg Ulcer (#40) | *"No leg ulcers bilaterally"* |

---

## Note 6: SYNTH_006 — Fatal ACS (Grade 5)

**Patient:** 42-year-old female, HbSS, CKD stage 3, chronic transfusions  
**Setting:** ER → ICU → death on ICU day 2

### Graded Outcomes

#### Acute Chest Syndrome (#48) → Grade 5

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `death_attributed` | `true` | *"Acute chest syndrome, fatal — death directly attributed to ACS with refractory respiratory failure"* |

**Rule applied:** `death_attributed` → **Grade 5** ✓

#### Acute Sickle Cell Pain Episode (#28) → Grade 5

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `death_attributed` | `true` | *"Acute sickle cell vaso-occlusive crisis — death in the setting of acute pain episode"* |

**Rule applied:** `death_attributed` → **Grade 5** ✓

#### Acute Kidney Injury (#19) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `creatinine_x_baseline` | `3.11` | *"Creatinine rose to 5.6 mg/dL (3.11 times baseline)"*; baseline 1.8 mg/dL |
| `creatinine` | `5.6 mg/dL` | Same as above |
| `renal_replacement` | `true` | *"Continuous renal replacement therapy (CRRT) initiated"* |
| `esrd_progression` | `false` | Patient died before ESRD could be assessed; acute injury, not chronic progression |
| `patient_age` | `42 years` | Adult |
| `death_attributed` | `false` | Death attributed to ACS, not AKI |

**Rule applied:** `(creatinine_x_baseline >= 3.0 OR creatinine >= 4 OR renal_replacement) AND NOT esrd_progression` → **Grade 3** ✓  
(Meets all three Grade 3 triggers: 3.11x ≥ 3.0, creatinine 5.6 ≥ 4.0, and CRRT. Not Grade 4 because `esrd_progression = false`. Not Grade 5 because death is attributed to ACS.)

#### Fever (#36) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `temperature` | `40.1 °C` | *"Temperature 40.1 degrees Celsius (104.2 degrees Fahrenheit)"* |

**Rule applied:** `temperature >= 38.5` → **Grade 2** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| DVT (#02) | *"No deep vein thrombosis on bilateral lower extremity Doppler"* |
| PE (#51) | *"CT pulmonary angiography: No pulmonary embolism"* |

---

## Note 7: SYNTH_007 — Mild ACS (Grade 1)

**Patient:** 14-year-old female, HbSC, diagnosed age 1  
**Setting:** Pediatric ER — observed 5 hours, discharged home

### Graded Outcomes

#### Acute Chest Syndrome (#48) → Grade 1

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `resp_support` | `room_air` | *"SpO2 97% on room air"*; *"No supplemental oxygen needed"* |
| `transfusion_type` | `none` | *"No transfusion indicated given stable hemoglobin near baseline"* |
| `acs_other_support` | `false` | *"managed with antibiotics alone on room air without transfusion or respiratory support"* — antibiotics are excluded from acs_other_support per schema |
| `fio2_pct` | `21%` | Room air = FiO2 21% |
| `life_support` | `false` | No ventilation, vasopressors, or RRT |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `resp_support == room_air AND transfusion_type == none AND NOT acs_other_support` → **Grade 1** ✓

#### Acute Sickle Cell Pain Episode (#28) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `care_setting` | `ed_treat_release` | *"Patient observed in ED for 5 hours... Not admitted. Discharged home"* |
| `pain_co_complication` | `false` | ACS is mild (Grade 1) — the note does not describe pain complicated by ACS; they are concurrent but both mild |
| `life_support` | `false` | No life support |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `care_setting >= clinic AND care_setting < inpatient` → **Grade 2** ✓

#### Fever (#36) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `temperature` | `38.6 °C` | *"Temperature 38.6 degrees Celsius (101.5 degrees Fahrenheit)"* |

**Rule applied:** `temperature >= 38.5` → **Grade 2** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| AKI (#19) | Creatinine 0.5 mg/dL = baseline (1.0x, below 1.5x threshold) |
| DVT (#02) | *"No swelling, no tenderness"*; no leg swelling or calf tenderness |
| Depression (#47) | *"No depressive symptoms"* |
| Leg Ulcer (#40) | *"no leg ulcers"* |

---

## Note 8: SYNTH_008 — Priapism with Aspiration/Irrigation

**Patient:** 16-year-old male, HbSS  
**Setting:** Pediatric ER → admitted to medical floor (2-day stay)

### Graded Outcomes

#### Priapism (#24) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `priapism_intervention` | `aspiration_irrigation` | *"bilateral corporal aspiration was performed"*; *"Irrigation performed with dilute phenylephrine solution"* |

**Rule applied:** `priapism_intervention == aspiration_irrigation` → **Grade 3** ✓  
(Not Grade 4 because *"No surgical shunt was required"*)

#### Acute Sickle Cell Pain Episode (#28) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `care_setting` | `inpatient` | *"Patient admitted to the medical floor for observation and concurrent vaso-occlusive pain crisis management"* |
| `pain_co_complication` | `false` | *"Acute sickle cell vaso-occlusive pain crisis requiring inpatient admission, without co-complications"* |
| `life_support` | `false` | No ventilation, vasopressors, or RRT |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `care_setting >= inpatient AND NOT pain_co_complication` → **Grade 3** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| Fever (#36) | *"afebrile throughout (temperature maximum 37.4 degrees Celsius, never reaching 38.0 degrees Celsius)"* |
| ACS (#48) | *"No chest pain, cough, or respiratory symptoms"*; *"Chest X-ray: Clear lungs, no infiltrates"* |
| DVT (#02) | *"no calf tenderness"*; no leg swelling |
| Stroke (#15) | *"no focal deficits"* |
| Leg Ulcer (#40) | *"no leg ulcers"* |

---

## Note 9: SYNTH_009 — Moderate ACS (Grade 2)

**Patient:** 25-year-old male, HbSS  
**Setting:** ER → admitted to medical floor (4-day stay)  
**Purpose:** Completes ACS grade coverage (was missing Grade 2)

### Graded Outcomes

#### Acute Chest Syndrome (#48) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `fio2_pct` | `28%` | *"nasal cannula at 2 L/min (FiO2 approximately 28%)"* |
| `resp_support` | `nasal_cannula` | Same as above |
| `transfusion_type` | `simple` | *"simple red blood cell transfusion... 2 units of packed red blood cells"*; *"No exchange transfusion was performed"* |
| `acs_other_support` | `false` | *"No erythropoietin was given"* |
| `life_support` | `false` | *"did not require high-flow nasal cannula, BiPAP, or any form of non-invasive or invasive ventilation. No vasopressors or other life-support interventions"* |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `(fio2_pct < 50 AND resp_support < high_flow AND transfusion_type != exchange) AND (transfusion_type == simple OR resp_support > room_air OR acs_other_support)` → **Grade 2** ✓  
(Has supplemental O₂ and simple transfusion, but none of the Grade 3 triggers: FiO₂ < 50%, not high-flow, not exchange)

#### Acute Sickle Cell Pain Episode (#28) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `care_setting` | `inpatient` | *"Patient admitted to the medical floor"* |
| `pain_co_complication` | `false` | *"without co-complications"* |

**Rule applied:** `care_setting >= inpatient AND NOT pain_co_complication` → **Grade 3** ✓

#### Fever (#36) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `temperature` | `39.0 °C` | *"Temperature 39.0 degrees Celsius"* |

**Rule applied:** `temperature >= 38.5` → **Grade 2** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| PE (#51) | *"CT pulmonary angiography: No pulmonary embolism"* |
| DVT (#02) | *"No swelling, no tenderness, no calf tenderness"* |
| Stroke (#15) | *"No neurological complications"*; *"no focal deficits"* |
| Arrhythmia (#01) | *"No arrhythmias on telemetry"* |

---

## Note 10: SYNTH_010 — Ischemic Stroke (Grade 3)

**Patient:** 31-year-old female, HbSS, history of elevated TCD velocities  
**Setting:** ER → neuro ICU → medical floor (8-day stay)  
**Purpose:** First CNS outcome graded. No concurrent pain crisis.

### Graded Outcomes

#### Stroke (#15) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `stroke_symptoms` | `true` | *"right-sided weakness and slurred speech"*; *"NIHSS score 8 at presentation"* |
| `stroke_symptoms_resolved` | `false` | *"stroke symptoms present and not fully resolved at discharge"*; *"NIHSS improved from 8 to 3"* (still abnormal) |
| `mrs` | `2` | *"Modified Rankin Scale (mRS) at discharge: 2 — slight disability"* |
| `incidental_radiographic_only` | `false` | Symptomatic stroke |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `stroke_symptoms AND NOT stroke_symptoms_resolved AND 1 <= mrs <= 3` → **Grade 3** ✓  
(Not Grade 4: mRS 2 is in the 1–3 range, not 4–5. Not Grade 2: symptoms have NOT fully resolved.)

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| Pain (#28) | *"No concurrent vaso-occlusive pain crisis — no pain reported during admission"* |
| Fever (#36) | *"No fever throughout admission (temperature remained below 38.0 degrees Celsius)"* |
| ACS (#48) | *"No chest pain or respiratory symptoms — no evidence of ACS"*; *"Chest X-ray: No infiltrates"* |
| Arrhythmia (#01) | *"No arrhythmias on telemetry"* |

---

## Note 11: SYNTH_011 — Sepsis (Grade 3)

**Patient:** 8-year-old male, HbSS, functional asplenia  
**Setting:** Pediatric ER → admitted to medical floor (4-day stay)  
**Purpose:** First sepsis grading. First pediatric patient under age 10.

### Graded Outcomes

#### Sepsis (#37) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `blood_culture_positive` | `true` | *"blood cultures (both sets) grew Streptococcus pneumoniae, serotype 19A"* |
| `organ_dysfunction` | `true` | *"elevated lactate"* (3.2 mmol/L), tachycardia (142 bpm), borderline hypotension, lethargy |
| `treated` | `true` | *"ceftriaxone 100 mg/kg IV... and vancomycin"*; *"Normal saline bolus 20 mL/kg"* |
| `life_threatening_sepsis` | `false` | Responded to fluids without vasopressors; *"did NOT require vasopressors, mechanical ventilation, or any other life-support interventions"* |
| `life_support` | `false` | Same as above |
| `death_attributed` | `false` | Patient discharged alive |

**Rule applied:** `blood_culture_positive AND (organ_dysfunction OR treated)` → **Grade 3** ✓  
(Not Grade 4: no life-threatening sepsis + no life support. Not Grade 5: patient survived.)

#### Acute Sickle Cell Pain Episode (#28) → Grade 3

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `care_setting` | `inpatient` | *"Patient admitted to the medical floor"* |
| `pain_co_complication` | `false` | *"no co-complications"* |

**Rule applied:** `care_setting >= inpatient AND NOT pain_co_complication` → **Grade 3** ✓

#### Fever (#36) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `temperature` | `39.6 °C` | *"Temperature 39.6 degrees Celsius (103.3 degrees Fahrenheit)"* |

**Rule applied:** `temperature >= 38.5` → **Grade 2** ✓

#### Outcomes Confirmed Absent

| Outcome | Evidence of Absence |
|:--------|:-------------------|
| ACS (#48) | *"Clear lungs bilaterally, no infiltrates"*; *"no respiratory distress, no new infiltrates — ACS ruled out"* |
| Stroke (#15) | *"no focal deficits, no neck stiffness"* |
| AKI (#19) | Creatinine 0.5 mg/dL (1.25x baseline 0.4) — below 1.5x threshold |

---

## Note 12: SYNTH_012 — Borderline/Edge Case (Precision Test)

**Patient:** 27-year-old female, HbSS  
**Setting:** ED treat-and-release  
**Purpose:** Tests the system's ability to correctly identify outcomes as ABSENT when clinical values are just below diagnostic thresholds. All borderline values should produce "absent" — a false positive here indicates a precision error.

### Graded Outcomes

#### Acute Sickle Cell Pain Episode (#28) → Grade 2

| Feature | Extracted Value | Source Quote |
|:--------|:---------------|:-------------|
| `care_setting` | `ed_treat_release` | *"After 3 hours in the ED... Patient does not require admission. Discharged home"* |
| `pain_co_complication` | `false` | No complications identified |

**Rule applied:** `care_setting >= clinic AND care_setting < inpatient` → **Grade 2** ✓

### Critical "Absent" Outcomes (Borderline Values)

> [!IMPORTANT]
> These outcomes are the key test cases in this note. Each has clinical values intentionally set **just below** the grading threshold. The correct answer for all five is **absent**.

| Outcome | Value in Note | Threshold | Margin | Correct Grade |
|:--------|:-------------|:----------|:-------|:-------------|
| **Fever (#36)** | 37.9 °C | ≥ 38.0 °C | **0.1° below** | **absent** |
| **AKI (#19)** | Creatinine 1.40x baseline | ≥ 1.5x baseline | **0.1x below** | **absent** |
| **ACS (#48)** | Subsegmental atelectasis | Lobar/segmental infiltrate required | Wrong finding type | **absent** |
| **DVT (#02)** | Calf tenderness, D-dimer 0.8 | Positive Doppler required | Doppler negative | **absent** |
| **Depression (#47)** | PHQ-2 = 2 | PHQ-9 ≥ 5 | No full PHQ-9 done | **absent** |

#### Supporting Quotes

| Outcome | Evidence |
|:--------|:--------|
| Fever | *"Temperature 37.9 degrees Celsius... below the 38.0 degrees Celsius threshold"* |
| AKI | *"creatinine 0.84 mg/dL (baseline 0.6 mg/dL, representing 1.40 times baseline — note: this is below the 1.5x threshold)"* |
| ACS | *"subsegmental atelectasis... No lobar or segmental consolidation. No infiltrate meeting criteria for acute chest syndrome"* |
| DVT | *"Normal compressibility of all deep veins bilaterally. No evidence of deep vein thrombosis"* |
| Depression | *"PHQ-2 score 2 — below the screening threshold; formal PHQ-9 not indicated"* |

---

## Summary: Grade Coverage

| Grade | Outcomes Using It |
|:-----:|:-----------------|
| **1** | Pain (Note 4), Fever (Note 1), ACS (Note 7) |
| **2** | Pain (Notes 1, 7, 12), Fever (Notes 2, 3, 6, 7, 9, 11), AKI (Note 3), DVT (Note 5), ACS (Note 9) |
| **3** | ACS (Note 2), Pain (Notes 5, 8, 9, 11), AKI (Note 6), Priapism (Note 8), Stroke (Note 10), Sepsis (Note 11) |
| **4** | Pain (Notes 2, 3), ACS (Note 3) |
| **5** | ACS (Note 6), Pain (Note 6) |
| **absent** | Multiple per note — Note 12 specifically tests 5 borderline-absent outcomes |

These 12 notes cover **8 distinct SCOGS outcomes** across **all 5 grades (1–5)**, with **explicit absence evidence** throughout. Note 12 specifically tests boundary precision with values intentionally set just below grading thresholds.
