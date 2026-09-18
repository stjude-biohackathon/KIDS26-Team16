# Original Shiny Dashboard — PCAI / GPT-OSS Integration

This folder preserves the original Edward-branch Shiny dashboard structure and styling.

Changes in this overlay:
- Live note inference uses St. Jude PCAI/Bifrost.
- Model is `gpt-oss-120b`.
- The dashboard is restricted to the 14 PI-finalized SCOGS outcomes for live inference.
- `dashboard/SCD_summaries.csv` contains the 430 summarized cases.
- CSV mode now loads those summaries directly.
- GPT-OSS confidence is displayed as **uncalibrated model confidence**.
- Exact evidence quotes feed the original grounding ledger and note highlighter.
- If `dashboard/conformal_calibration.json` is later present, conformal prediction sets are displayed.
- Saved-run/explore infrastructure from the original dashboard remains intact.

## Run from repository root

```powershell
$env:PCAI_API_KEY="YOUR_VIRTUAL_KEY"
python -m pip install shiny openai pandas
python -m shiny run dashboard/interactive_dashboard.py
```

The virtual key is passed in the required `x-bf-vk` header.

Do not put the virtual key in source code.

## Important

The current Bifrost GPT-OSS provider has shown a 30-second server-side timeout.
The live dashboard therefore recommends one outcome request at a time and retries
an individual outcome up to three times.

Reference outcome labels in the 430-case CSV are not included in the GPT-OSS prompt.


## Dashboard v2 changes
- The outcome scope is now a 14-vs-53 toggle.
- The patient-triage filter defaults to **Present** only. Cannot-grade and Not-present outcomes remain available as optional filters.
- Live PCAI grading now performs true parallel requests; **2 parallel requests** is the recommended default, with 1 as safest and 3-4 experimental.
- 53-outcome mode uses the 14 finalized JSON rubrics for the core outcomes and GPT-OSS grounded feature extraction + Edward's deterministic SCOGS decision tables for the remaining 39.
- Optional **Experimental Fast 53 pre-screen** can reduce detailed calls. Screened-out outcomes are intentionally marked Cannot Grade, not Absent, to avoid overstating the screen.


## v4 — selectable PCAI grading model

The live dashboard now includes a **PCAI Grading Model** dropdown with:

- `gpt-oss-120b`
- `Qwen/Qwen3.8-27B-FP8`

Both models receive the same clinical note and the same SCOGS rubric prompt, which
makes their grades, evidence, and model-confidence outputs directly comparable.

Recommended comparison workflow:
1. Select GPT-OSS 120B and grade a case.
2. Save/export the run if desired.
3. Select Qwen3.8 27B FP8 and grade the same case.
4. Compare grade agreement, present/absent calls, exact evidence, and runtime.

The confidence values from both models remain uncalibrated. Conformal calibration
should be fitted separately for each model if model-specific conformal inference
is later used.
