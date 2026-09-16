# SCOGS-Scribe

### Automated Severity Grading for Sickle Cell Disease from Clinical Notes

**SCOGS-Scribe** is a St. Jude KIDS26 BioHackathon project exploring whether large language models can transform unstructured sickle cell disease clinical notes into standardized **Sickle Cell Outcome Grading System (SCOGS)** severity grades.

The project combines structured SCOGS criteria, synthetic clinical notes, local large language models, automated evaluation, and an interactive dashboard to create an initial computable framework for SCOGS.

> **Team Lead:** Joe Wardell
> **Project:** St. Jude KIDS26 BioHackathon
> **BioHackathon Dates:** September 16–18, 2026

---

## Project Profile

### Project Name

**SCOGS-Scribe**

### Question, Problem, or Opportunity

Can a large language model automatically identify sickle cell disease complications from clinical notes and assign standardized **SCOGS severity grades**?

Sickle cell disease can cause complications across nearly every organ system. The **Sickle Cell Outcome Grading System (SCOGS)** provides a standardized framework for describing the severity of these outcomes using grades from **1–5**, along with diagnostic criteria and frequency patterns.

Currently, applying SCOGS requires reviewers to manually examine clinical information and assign grades outcome by outcome.

This creates several challenges:

* Manual grading is time-intensive.
* Large patient cohorts require substantial abstraction effort.
* Clinical information is primarily stored as unstructured text.
* Grading may vary between reviewers.
* Manual abstraction is difficult to scale.
* SCOGS currently lacks a fully computable implementation.
* There is no established labeled benchmark for automated SCOGS grading.

**SCOGS-Scribe** explores whether large language models can help convert unstructured clinical information into reproducible and standardized SCOGS grades.

---

## Project Goal

The goal of SCOGS-Scribe is to build a working end-to-end prototype:

```text
Synthetic SCD Clinical Note
            ↓
     Clinical Information
            ↓
         LLM Grader
            ↓
      SCOGS Outcome
            ↓
    SCOGS Severity Grade
            ↓
 Evidence + Grading Rationale
            ↓
 Compare With Ground Truth
            ↓
       Model Evaluation
            ↓
   Interactive Dashboard
```

The goal is **not** to replace clinical judgment.

Instead, the project asks whether an LLM can reliably extract the information needed to apply standardized SCOGS criteria and identify cases where human review may still be necessary.

---

## Initial Outcome Scope

The BioHackathon prototype will focus on **14 SCOGS outcomes**.

|  # | Outcome                              |
| -: | ------------------------------------ |
|  1 | Acute Pain                           |
|  2 | Stroke                               |
|  3 | Splenic Sequestration (SS)           |
|  4 | Acute Chest Syndrome                 |
|  5 | Priapism                             |
|  6 | Chronic Pain                         |
|  7 | Chronic Kidney Disease (CKD)         |
|  8 | Retinopathy                          |
|  9 | CD                                   |
| 10 | Depression                           |
| 11 | Transcranial Doppler (TCD) Elevation |
| 12 | Asthma                               |
| 13 | Avascular Necrosis (AVN)             |
| 14 | Leg Ulcer                            |

These outcomes provide a diverse initial test set covering acute and chronic complications across multiple organ systems.

The BioHackathon goal is to establish a **working and reproducible grading pipeline across these selected outcomes** before expanding toward the complete SCOGS framework.

---

## What We Are Building

SCOGS-Scribe consists of five major components.

### 1. Machine-Readable SCOGS Criteria

SCOGS criteria for the selected outcomes will be converted from human-readable grading definitions into a structured format such as:

* JSON
* YAML
* Structured R objects

The structured criteria may include:

```json
{
  "outcome": "Acute Chest Syndrome",
  "grades": {
    "1": {
      "criteria": []
    },
    "2": {
      "criteria": []
    },
    "3": {
      "criteria": []
    },
    "4": {
      "criteria": []
    },
    "5": {
      "criteria": []
    }
  }
}
```

This machine-readable representation will serve as the grading reference for the LLM.

---

### 2. Synthetic Clinical Note Generator

Because access to real clinical notes involves privacy, governance, and data-access requirements, the BioHackathon prototype will begin with **synthetic clinical notes**.

Each synthetic case will be created with a known ground-truth SCOGS outcome and severity grade.

Synthetic notes may contain:

* Patient demographics
* Symptoms
* Clinical history
* Physical examination findings
* Laboratory results
* Imaging findings
* Treatments
* Medications
* Procedures
* Respiratory support
* Transfusions
* Level of care
* Hospital course
* Discharge information
* Longitudinal disease information

Where appropriate, cases may include multiple encounters representing progression from presentation through hospitalization, discharge, or death.

Each synthetic case will contain known labels that can be used to evaluate the grader.

Example structure:

```text
Patient ID
Outcome
Ground-Truth SCOGS Grade
Clinical Note
Relevant Clinical Evidence
Expected Grading Criteria
```

---

### 3. LLM-Based SCOGS Grader

The grader will ingest a clinical note and determine:

* Which SCOGS outcome is present
* Which SCOGS grade is supported
* What clinical evidence supports that grade
* Which SCOGS criteria were used
* Whether the evidence is sufficient
* Whether human review may be necessary

The model should produce **structured output** rather than only free-text responses.

Example:

```json
{
  "patient_id": "SYN-001",
  "outcome": "Acute Chest Syndrome",
  "predicted_grade": 3,
  "evidence": [
    "New pulmonary infiltrate",
    "Required supplemental oxygen",
    "Escalation of respiratory support"
  ],
  "rationale": "The documented clinical findings meet the encoded SCOGS Grade 3 criteria.",
  "confidence": "high",
  "human_review": false
}
```

The final schema may evolve during the BioHackathon.

---

### 4. Model Evaluation

Because synthetic cases have known ground-truth grades, model predictions can be evaluated directly.

Evaluation will focus on both overall performance and outcome-specific failure patterns.

#### Exact-Match Accuracy

The percentage of cases where:

```text
Predicted Grade = Ground-Truth Grade
```

#### Within-One-Grade Accuracy

The percentage of cases where:

```text
| Predicted Grade - Ground-Truth Grade | ≤ 1
```

#### Confusion Matrix

Confusion matrices will identify grades that are commonly confused with one another.

#### Per-Outcome Accuracy

Performance will be evaluated separately for each SCOGS outcome.

#### Cohen's Kappa

Agreement between model predictions and ground truth can also be measured using Cohen's kappa.

#### Error Analysis

Incorrect predictions will be reviewed to identify patterns such as:

* Missing clinical evidence
* Incorrect interpretation of thresholds
* Outcome misclassification
* Confusion between neighboring grades
* Insufficient information
* Hallucinated evidence
* Failure to recognize important clinical details

A transparent understanding of **why the grader fails** is as important as overall accuracy.

---

### 5. Interactive Dashboard

The final prototype will display grading results through an interactive dashboard.

Potential dashboard components include:

#### Patient-Level View

* Patient identifier
* Clinical note
* Identified SCOGS outcome
* Ground-truth grade
* Predicted grade
* Supporting evidence
* Model rationale
* Confidence
* Human-review flag

#### Patient Severity Profile

A patient-level profile may display severity across multiple SCOGS outcomes.

#### Severity Trajectory

For patients with multiple encounters:

```text
Presentation → Admission → Hospital Course → Discharge
```

The dashboard may visualize how disease severity changes across the episode.

#### Cohort-Level Views

Potential visualizations include:

* Grade distributions
* Outcome distributions
* Organ-system heatmaps
* Severity profiles
* Ground-truth vs. predicted grades
* Model accuracy by outcome
* Confusion matrices
* Cases flagged for human review

---

## Proposed Pipeline

```text
                         SCOGS RUBRIC
                              │
                              ▼
                 MACHINE-READABLE CRITERIA
                       JSON / YAML
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       SYNTHETIC NOTE GENERATOR         LLM GRADER
                │                           ▲
                │                           │
                ▼                           │
        SYNTHETIC SCD NOTES ────────────────┘
                │
                ▼
       STRUCTURED SCOGS OUTPUT
                │
                ├── Outcome
                ├── Grade
                ├── Evidence
                ├── Rationale
                ├── Confidence
                └── Human Review Flag
                │
                ▼
          GROUND-TRUTH COMPARISON
                │
                ▼
             EVALUATION
                │
                ├── Exact Accuracy
                ├── Within-One Accuracy
                ├── Confusion Matrix
                ├── Cohen's Kappa
                └── Outcome-Level Accuracy
                │
                ▼
       INTERACTIVE SCOGS DASHBOARD
```

---

## Tools and Technology Stack

### Primary Language

**R**

R will be used for:

* Data generation
* Data processing
* Model integration
* Evaluation
* Visualization
* Dashboard development

---

### Large Language Models

The project is designed to support locally hosted or open-weight models.

Potential tools include:

* Ollama
* MedGemma
* Llama
* Mistral
* Qwen
* Other locally available models

The model layer should remain modular so that different models can be tested using the same clinical cases and grading framework.

---

### LLM Integration

Potential R packages and interfaces:

```text
ellmer
httr2
jsonlite
```

Model prompts will request structured outputs that can be programmatically validated and analyzed.

---

### Synthetic Data

Potential packages:

```text
charlatan
wakefield
tidyverse
```

Synthetic clinical-note templates may also be generated using rule-based or LLM-assisted approaches.

---

### Data Processing

```text
dplyr
tidyr
purrr
stringr
readr
tibble
```

---

### Evaluation

```text
yardstick
dplyr
ggplot2
```

Metrics may include:

* Exact-match accuracy
* Within-one-grade accuracy
* Confusion matrices
* Cohen's kappa
* Outcome-specific accuracy

---

### Dashboard

The interactive dashboard will primarily use:

```text
Shiny
ggplot2
plotly
DT
```

---

### Development and Collaboration

```text
Git
GitHub
RStudio
VS Code
Ollama
```

---

## Repository Structure

The repository may follow a structure similar to:

```text
SCOGS-SCRIBE/
│
├── README.md
│
├── LICENSE
│
│
├── data/
│   ├── raw/
│   ├── synthetic/
│   └── outputs/
│
├── scogs-schema/
│   ├── acute-pain/
│   ├── stroke/
│   ├── splenic-sequestration/
│   ├── acute-chest-syndrome/
│   ├── priapism/
│   ├── chronic-pain/
│   ├── ckd/
│   ├── retinopathy/
│   ├── cd/
│   ├── depression/
│   ├── tcd-elevation/
│   ├── asthma/
│   ├── avn/
│   ├── leg-ulcer/
│   └── scogs_schema.json
│
├── synthetic-note-generator/
│   ├── templates/
│   ├── scripts/
│   └── outputs/
│
├── grader/
│   ├── prompts/
│   ├── scripts/
│   ├── schemas/
│   └── outputs/
│
├── evaluation/
│   ├── scripts/
│   ├── metrics/
│   ├── confusion-matrices/
│   └── reports/
│
├── dashboard/
│   ├── app.R
│   ├── modules/
│   └── www/
│
├── project-management/
│   ├── team.md
│   ├── tasks.md
│   ├── decisions.md
│   └── milestones.md
│
└── docs/
    ├── methods.md
    ├── limitations.md
    ├── architecture.md
    └── demo.md
```

The exact repository structure may evolve as development progresses.

---

## Team

### Team Lead

**Joe Wardell**

GitHub: `@ADD-GITHUB-HANDLE`

### Team Members

See:

[`project-management/team.md`](project-management/team.md)

Team roles may include:

| Role                | Primary Responsibility                               |
| ------------------- | ---------------------------------------------------- |
| Project Lead        | Scope, coordination, integration, presentation       |
| SCOGS Criteria Team | Convert grading criteria into machine-readable rules |
| Synthetic Data Team | Create clinical-note cases with known ground truth   |
| LLM / Prompt Team   | Develop and test grading prompts                     |
| Evaluation Team     | Measure agreement and analyze model errors           |
| Dashboard Team      | Build visualization and interactive interface        |
| Documentation Team  | Maintain GitHub documentation and methods            |

Team members may contribute across multiple areas.

---

## Communication

Primary team communication:

**Add team Slack / Teams / GitHub Discussion channel here**

GitHub Issues should be used for:

* Tasks
* Bugs
* Feature requests
* Documentation needs
* Decisions requiring team discussion

---

## Vision

Create a scalable and computable approach for applying standardized sickle cell disease severity criteria to clinical information.

Long term, a validated SCOGS-Scribe framework could support:

* Cohort-scale disease severity profiling
* Retrospective research
* Longitudinal outcomes research
* Standardized phenotype definitions
* Clinical research
* Trial endpoints
* Multi-institutional comparisons
* Automated clinical-data abstraction
* Human-assisted chart review

---

## Mission

During the KIDS26 BioHackathon, the team will build and evaluate a prototype that converts synthetic unstructured sickle cell disease clinical notes into structured SCOGS severity grades.

The project will prioritize:

**Reproducibility → Transparency → Evaluation → Scalability**

The objective is not simply to obtain an LLM answer.

The objective is to create a system where we can determine:

> What did the model predict?

> What clinical evidence did it use?

> Which SCOGS criteria support the prediction?

> Was the prediction correct?

> When should a human review the case?

---

## BioHackathon Roadmap

| Day       | Focus                                                                                                                                     | Expected Outcome                                                                                 |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Day 1** | Finalize outcomes, assign roles, encode SCOGS criteria, establish repository structure, create synthetic cases, and run first model tests | Initial SCOGS schemas, synthetic dataset, first successful note-to-grade model output            |
| **Day 2** | Expand synthetic cases, develop grading pipeline, test prompts/models, calculate preliminary metrics, and build dashboard                 | Working end-to-end grading pipeline with initial accuracy results                                |
| **Day 3** | Refine failure cases, integrate components, finalize evaluation, complete dashboard, document limitations, and prepare demo               | Stable demonstration, quantified results, documented repository, and presentation-ready workflow |

---

## Day 1 Priorities

* [ ] Confirm all 14 target outcomes
* [ ] Assign team members and roles
* [ ] Establish GitHub workflow
* [ ] Define SCOGS JSON/YAML schema
* [ ] Encode initial grading criteria
* [ ] Develop synthetic-note format
* [ ] Generate initial test cases
* [ ] Establish expected LLM output schema
* [ ] Run first LLM grading test
* [ ] Confirm output can be parsed programmatically
* [ ] Create initial dashboard framework

---

## Day 2 Priorities

* [ ] Expand synthetic cases across outcomes and grades
* [ ] Test multiple clinical-note formats
* [ ] Improve prompts
* [ ] Test selected LLMs
* [ ] Automate the grading workflow
* [ ] Capture evidence and rationale
* [ ] Calculate exact-match accuracy
* [ ] Calculate within-one-grade accuracy
* [ ] Generate confusion matrices
* [ ] Evaluate performance by outcome
* [ ] Begin error analysis
* [ ] Integrate outputs with dashboard

---

## Day 3 Priorities

* [ ] Stabilize grading pipeline
* [ ] Address major failure modes
* [ ] Finalize model evaluation
* [ ] Complete dashboard
* [ ] Add human-review flags
* [ ] Document methodology
* [ ] Document limitations
* [ ] Document repository structure
* [ ] Clean code
* [ ] Prepare demonstration cases
* [ ] Prepare final presentation
* [ ] Document next steps

---

## Minimum Viable Product

At minimum, the SCOGS-Scribe prototype should be able to:

* [ ] Load or generate a synthetic SCD clinical note
* [ ] Send the note to a locally available LLM
* [ ] Identify the relevant SCOGS outcome
* [ ] Assign a SCOGS severity grade
* [ ] Extract supporting clinical evidence
* [ ] Provide a structured rationale
* [ ] Return machine-readable output
* [ ] Compare the prediction with ground truth
* [ ] Calculate model performance
* [ ] Display the result in an interactive dashboard

The project does **not** need to achieve perfect accuracy during the BioHackathon.

A useful result may instead demonstrate:

* Where automated grading works well
* Which outcomes are difficult
* Which grades are frequently confused
* Which clinical evidence the model misses
* When human review is necessary

---

## Definition of Success

A successful BioHackathon prototype will demonstrate the complete workflow:

```text
Clinical Note
     ↓
SCOGS-Scribe
     ↓
Outcome + Grade + Evidence
     ↓
Ground Truth Comparison
     ↓
Performance Metrics
     ↓
Dashboard
```

Success means leaving the BioHackathon with:

* A functioning prototype
* Reproducible code
* Structured SCOGS criteria
* Synthetic benchmark cases
* Quantified performance
* Identified failure modes
* Clear documentation
* An interactive demonstration
* A roadmap for future validation

---

## Stretch Goals

If the minimum viable product is completed early:

* [ ] Increase the number of synthetic cases
* [ ] Add all grades represented within each selected outcome
* [ ] Support multiple outcomes in the same clinical note
* [ ] Add confidence estimates
* [ ] Add a **Needs Human Review** flag
* [ ] Compare multiple LLMs
* [ ] Compare different prompt strategies
* [ ] Compare zero-shot and rubric-grounded grading
* [ ] Add multi-encounter clinical trajectories
* [ ] Track severity from presentation through discharge
* [ ] Add longitudinal SCOGS visualization
* [ ] Develop patient-level severity profiles
* [ ] Develop organ-system heatmaps
* [ ] Export standardized analytic endpoints
* [ ] Expand beyond the initial 14 outcomes
* [ ] Move toward all 53 SCOGS outcomes

---

## Human Review

Automated grading should not assume every case can be confidently classified.

Potential reasons to trigger human review include:

```text
Insufficient clinical evidence
Conflicting clinical evidence
Multiple possible grades
Missing required diagnostic criteria
Ambiguous documentation
Low model confidence
Unsupported model rationale
Possible hallucinated evidence
```

A future version of SCOGS-Scribe could combine automated grading with targeted manual review rather than attempting complete automation.

---

## Project Principles

### 1. Ground the Model in SCOGS

The model should grade cases using the SCOGS criteria rather than relying only on general clinical knowledge.

### 2. Use Structured Outputs

Predictions should be machine-readable and easy to validate.

### 3. Capture Evidence

Every predicted grade should be linked to evidence from the clinical note.

### 4. Measure Performance

A model prediction is not useful without evaluating whether it is correct.

### 5. Make Errors Visible

Failure cases should be preserved and studied rather than hidden.

### 6. Start With Synthetic Data

Synthetic cases allow rapid prototype development without protected patient data.

### 7. Keep Humans in the Loop

Ambiguous or low-confidence cases should be identifiable for review.

### 8. Prioritize Reproducibility

Prompts, model versions, criteria, code, and evaluation procedures should be documented.

---

## Limitations

SCOGS-Scribe is an exploratory research prototype.

Important limitations include:

* Synthetic notes are not equivalent to real clinical documentation.
* Performance on synthetic cases does not establish performance on real patients.
* Synthetic cases may be cleaner and more explicit than real notes.
* LLM predictions may be incorrect.
* LLMs may hallucinate clinical evidence.
* Different models may produce different results.
* Clinical terminology varies across institutions.
* Clinical documentation varies across clinicians.
* Some SCOGS grades may require information unavailable in a single note.
* Longitudinal outcomes may require information across multiple encounters.
* Model confidence does not necessarily represent calibrated statistical uncertainty.
* Automated grading requires future expert validation.

The prototype should **not be used for clinical decision-making**.

---

## Data and Privacy

The BioHackathon prototype is designed to use **synthetic clinical notes**.

No protected patient information is required for the initial prototype.

Future validation using real clinical notes would require:

* Appropriate approvals
* Data governance review
* Secure computing environments
* Appropriate clinical-data access
* Collaboration with relevant data owners
* Appropriate research oversight

Synthetic-first development allows the engineering and evaluation framework to be created without making access to protected clinical notes a prerequisite for the BioHackathon.

---

## Future Directions

After the BioHackathon, potential next steps include:

1. Expand the number of synthetic benchmark cases.
2. Improve clinical realism of synthetic notes.
3. Expand machine-readable SCOGS criteria.
4. Add additional SCOGS outcomes.
5. Evaluate multiple open-weight and hosted LLMs.
6. Optimize prompting strategies.
7. Evaluate structured-output reliability.
8. Perform expert review of synthetic cases.
9. Establish inter-rater agreement benchmarks.
10. Validate automated grading against manually graded clinical notes.
11. Compare LLM predictions with expert SCOGS reviewers.
12. Develop uncertainty and human-review workflows.
13. Create longitudinal severity profiles.
14. Create standardized analytic datasets from model outputs.
15. Explore integration with retrospective SCD cohort studies.
16. Develop a reusable computable implementation of SCOGS.
17. Prepare technical documentation and methods manuscripts.
18. Expand toward all 53 SCOGS outcomes.

---

## Longer-Term Vision

The ultimate goal is not simply to classify clinical notes.

The longer-term opportunity is to create a system that transforms:

```text
Unstructured Clinical Documentation
                 ↓
       Standardized Evidence
                 ↓
          SCOGS Outcomes
                 ↓
        SCOGS Severity Grades
                 ↓
      Longitudinal Severity Data
                 ↓
     Research-Ready Endpoints
```

A validated system could potentially make large-scale SCOGS research substantially more efficient and reproducible.

---

## Contributing

Team members should:

1. Create or assign a GitHub Issue before starting major work.
2. Work on a dedicated branch.
3. Keep commits focused and descriptive.
4. Avoid committing protected or confidential data.
5. Document new scripts and major functions.
6. Submit changes through pull requests when practical.
7. Record major project decisions in `project-management/decisions.md`.

Example branch names:

```text
feature/synthetic-notes
feature/scogs-schema
feature/llm-grader
feature/evaluation
feature/dashboard
docs/readme
fix/json-output
```

Example commit messages:

```text
Add acute pain SCOGS schema

Create synthetic ACS cases

Add structured LLM grading output

Calculate exact-match accuracy

Add patient severity dashboard

Document grader limitations
```

---

## Documentation

Project documentation should be maintained throughout the BioHackathon rather than added only at the end.

Important documentation includes:

```text
README.md
docs/methods.md
docs/limitations.md
docs/architecture.md
project-management/team.md
project-management/tasks.md
project-management/decisions.md
```

---

## Reproducibility

Whenever possible, record:

* Model name
* Model version
* Prompt version
* SCOGS schema version
* Generation parameters
* Synthetic case version
* Evaluation date
* Software/package versions

This will allow team members to determine whether changes in model behavior are caused by:

* Model changes
* Prompt changes
* Criteria changes
* Dataset changes
* Code changes

---

## Disclaimer

**SCOGS-Scribe is a research and BioHackathon prototype.**

It is not intended for clinical use, diagnosis, treatment recommendations, or clinical decision-making.

Any future application to real clinical data will require appropriate validation, governance, approvals, and expert clinical oversight.

---

## The Question We Want to Answer

> **Can standardized SCOGS severity criteria be transformed into a computable framework that allows an LLM to reproducibly grade sickle cell disease outcomes from clinical text?**

The goal of the BioHackathon is not to build a perfect production system in three days.

The goal is to leave with a **clear, transparent, reproducible, and evaluable prototype that others can build on.**
