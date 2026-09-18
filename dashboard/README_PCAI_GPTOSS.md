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


## v5 — simplified live clinical interface

The live dashboard is intentionally simplified for presentation:

- GPT-OSS 120B is fixed as the grading model.
- One outcome is graded per PCAI request.
- Model and concurrency selectors are retained internally but hidden.
- Input source, age, and sex controls are hidden.
- Live mode presents one large clinical-note text box.
- The 14-vs-53 outcome selector remains visible.
- Live execution telemetry is retained in code but hidden from the live UI.
- Explore Saved Runs remains available.


## v6 — fixed 14 outcomes + grading table

- Live evaluation is fixed to the 14 PI-finalized outcomes; the 53-outcome option is no longer shown.
- The sidebar lists all 14 included outcomes.
- Once an outcome receives a numeric SCOGS grade, its official Grade 1-5 criteria table is shown directly below the result.
- The selected model grade is highlighted in the table.
- Display grading-table text is stored separately in `scogs_14_display_tables.json` and is based on the SCOGS booklet.


## v7 — GPT-OSS empty-content fix

A prior 1800-token completion cap could be exhausted by GPT-OSS reasoning before
the final JSON was emitted, causing all outcomes to be recorded as inference
errors. The dashboard now retries single-outcome requests with completion budgets
of 3000, 4200, and 5200 tokens and records the provider finish reason when a
response still has no final content.


## v8 — 16,384 completion tokens by default

Every GPT-OSS single-outcome grading request now uses:

`max_tokens=16384`

This applies to the initial request and retries. The dashboard still grades one
outcome per request.


## v11 — speed diagnostics

This version keeps the existing behavior of one outcome at a time and
`max_tokens=16384`, but makes the bottleneck measurable.

Changes:
- Reuses a single PCAI/OpenAI client connection pool instead of recreating it
  for every outcome.
- Uses at most one retry per outcome instead of two retries after the first try.
- Prints per-outcome START / RESPONSE / ERROR timing to the PowerShell terminal.
- Shows the current outcome name in the Shiny progress indicator.
- Saves and displays response latency, finish reason, token usage (when returned),
  and a collapsed copy of the raw GPT-OSS response.


## v12 — no explicit max_tokens

The PCAI GPT-OSS request no longer sends a `max_tokens` parameter. The provider/model
now uses its default completion behavior. All other speed diagnostics remain in place.


## v13 — fast per-outcome confidence + conformal display

This version is based on the speed/no-explicit-max-tokens dashboard.

- Model-Level Confidence is outcome-specific validation grade accuracy.
- Individual-Level Confidence is expected-evidence support for the current
  labeled validation case.
- Confidence is computed once after the run and cached; it does not make
  additional PCAI calls and does not rescan saved results per card.
- Existing conformal prediction-set display remains enabled and is read from
  each outcome's `extracted_features`.
- No API key is included in the repository or ZIP. `PCAI_API_KEY` must be set
  in the terminal environment before Shiny is launched.

## v14 — timeout fix / fast PCAI request behavior

The GPT-OSS outcome request now matches the known-fast PCAI build again:

- Restores an explicit `max_tokens=16384` completion/reasoning cap by default.
- The cap is configurable with the `PCAI_MAX_TOKENS` environment variable.
- Uses a fresh OpenAI/PCAI client for each outcome request instead of a cached
  connection pool, avoiding stale pooled connections that can wait until timeout.
- Keeps `reasoning_effort="low"`, per-outcome timing diagnostics, validation
  confidence, and conformal display from PCAI_2.

PowerShell override example:

`$env:PCAI_MAX_TOKENS="16384"`
