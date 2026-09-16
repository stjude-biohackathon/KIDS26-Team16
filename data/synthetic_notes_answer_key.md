# Synthetic Notes — Ground-Truth Answer Key

> **Author:** Daffa Warsa  
> **Branch:** `Daffa-Synthetic-Notes`  
> **Data File:** [`data/synthetic_notes.csv`](data/synthetic_notes.csv)  
> **Grading Authority:** [`rules.md`](rules.md) + [`data/scogs_feature_schema.json`](data/scogs_feature_schema.json)

---

## Overview

Three synthetic clinical notes were written with intentionally embedded clinical details that deterministically map to known SCOGS grades. Each note targets multiple outcomes across a range of severity levels (Grades 1–4), providing ground-truth examples that the existing `clincal_notes.csv` dataset lacks (which only ever grades one outcome using grades 2–3).

| Note | Patient | Scenario | Outcomes Graded | Grade Range |
|:----:|:--------|:---------|:----------------|:------------|
| **1** | SYNTH_001 — 22F, HbSS | Mild — ED treat-and-release | Pain (#28) = 2, Fever (#36) = 1 | Grades 1–2 |
| **2** | SYNTH_002 — 17M, HbSS | Severe — Admitted with ACS | Pain (#28) = 4, ACS (#48) = 3, Fever (#36) = 2 | Grades 2–4 |
| **3** | SYNTH_003 — 29M, HbSC | Life-threatening — ICU | Pain (#28) = 4, ACS (#48) = 4, Fever (#36) = 2, AKI (#19) = 2 | Grades 2–4 |

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

## Summary: Grade Coverage

| Grade | Outcomes Using It |
|:-----:|:-----------------|
| **1** | Fever (Note 1) |
| **2** | Pain (Note 1), Fever (Notes 2 & 3), AKI (Note 3) |
| **3** | ACS (Note 2) |
| **4** | Pain (Notes 2 & 3), ACS (Note 3) |
| **absent** | Multiple per note (DVT, PE, Stroke, Leg Ulcer, etc.) |

These 3 notes cover **4 distinct SCOGS outcomes** across **grades 1–4**, with **explicit absence evidence** for 7+ additional outcomes per note — a significant improvement over the existing dataset's single outcome with only grades 2–3.
