# SCOGS-Scribe GPT-OSS Dashboard

This package is designed to be unzipped directly into the `PCAI` folder.

## What it does

- Loads the 430 Qwen-generated SCD summaries.
- Uses St. Jude PCAI/Bifrost with `gpt-oss-120b`.
- Grades the 14 PI-finalized SCOGS outcomes.
- Uses small requests and automatic batch splitting to reduce the impact of the provider's 30-second timeout.
- Saves model predictions separately from human review decisions.
- Lets you review and revise each outcome after the model runs.
- Displays conformal prediction automatically if `conformal_calibration.json` is later added.

## Files

- `pcai_dashboard.py` — Streamlit dashboard.
- `pcai_grader_backend.py` — GPT-OSS/Bifrost grading backend.
- `SCD_summaries.csv` — 430 summarized cases.
- `scogs_14_outcomes.json` — finalized 14 SCOGS rubrics.
- `conformal_utils.py` — conformal helper functions.
- `SCD_grades_14_gptoss.csv` — partial model output already produced, if present.
- `SCD_reviewed_grades.csv` — created automatically by the dashboard.
- `requirements_pcai_dashboard.txt`
- `RUN_DASHBOARD.ps1`

## Run

Open PowerShell in the folder and set your key:

```powershell
$env:PCAI_API_KEY="YOUR_VIRTUAL_KEY"
```

Then either:

```powershell
.\RUN_DASHBOARD.ps1
```

or:

```powershell
python -m pip install -r requirements_pcai_dashboard.txt
python -m streamlit run pcai_dashboard.py
```

The dashboard opens in your browser.

## Recommended first test

Select `PMC10253251_01` (the 58-year-old ACS case) and click **Grade remaining**.

The dashboard defaults to **1 outcome per request** because the current Bifrost GPT-OSS provider has shown a 30-second server-side timeout. You can change it to 2 or 3 in the sidebar if the service is responding quickly.

## Human review

The model output is kept intact in `SCD_grades_14_gptoss.csv`.

Reviewer decisions are stored separately in:

`SCD_reviewed_grades.csv`

This makes model-vs-human evaluation straightforward.

## Reference labels

The source CSV contains known outcome-name labels for evaluation. They are hidden by default and are **never sent to GPT-OSS**. The dashboard has an optional evaluation-only toggle to reveal them.

## Conformal prediction

The dashboard does not invent conformal percentages before calibration.

If you later place a valid:

`conformal_calibration.json`

in this folder, conformal prediction sets will be displayed with the corresponding model results.


## v3 visual update
The dashboard layout now mirrors the supplied SCOGS-Scribe mockup: navy header, left clinical-note panel, right analysis-results card, confidence bar, evidence box, SCOGS criteria table, and all-14 outcome review grid.
