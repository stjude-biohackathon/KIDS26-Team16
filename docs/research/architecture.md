# Implemented architecture

```text
Bundled PMC case cache
    -> deterministic cohort / note selection
    -> per-outcome prompt and output schema
    -> full F16/BF16 MedGemma 27B via local Ollama
    -> quote verification, value typing, unit checks, conflict reconciliation
    -> SCOGS decision tables
    -> result JSON
    -> notebook summaries and human-review CSVs
```

`rules.md` is the human-readable grading authority.
`scripts/scogs/features.py`, `predicates.py`, `tables.py`, and `evaluate.py`
implement the feature registry and deterministic grading. The generated
`data/scogs_feature_schema.json` must stay synchronized with them.

`scripts/experiments/` is split by responsibility:

| Module | Owns |
| --- | --- |
| `medgemma_extraction.py` | CLI arguments, prompt stages, backends, the run loop |
| `verification.py` | quote grounding, value typing, unit / age / TLC guards, conflict reconciliation |
| `grading.py` | verified features -> grade status (`grade_outcome`), shared with the dashboard |
| `cohort.py` | SCD cohort gate and stratified note selection |
| `run_output.py` | console report and the results-file format |
| `ollama_backend.py` | transport and model preflight |
| `review_results.py` | worksheets from already-saved results |

The two notebooks and `dashboard/` are interfaces to this same code, not
independent copies of it: the dashboard's live mode calls `run()` and
`grade_outcome()` and never builds its own prompt.

## Invariants

- Models extract features; grades are derived from the tables.
- Unsupported or ambiguous evidence is withheld rather than silently guessed.
- Missing findings are not automatically negative findings.
- A grounded quote proves text overlap, not that the value is correct.
- Review sheets use the run's original `accepted_findings` and `conflicts`.
- Metadata and transport failures stop real runs; they do not become empty,
  success-looking result files.
- Full-precision model policy is checked using reported metadata, not names.
  Operators remain responsible for verifying model source and conversion.
- Each result records the served digest, precision, prompt stage, cohort,
  sampling, decoding, and runtime settings.

The proposed encoder/ensemble training, calibration, and dashboard are not part
of the current runnable system. Previous planning documents are available in Git
history; local research notes and run artifacts are not rewritten by this cleanup.

## Rubric audit

Known booklet discrepancies and the documented fever-boundary correction are
recorded in [the verification log](../reference/rules_vs_booklet_discrepancies.md).
This cleanup does not change grading predicates or prompt-stage semantics.
The booklet and proposal are kept in `docs/reference/` for traceability.

The audit runs in three steps: `scripts/audit/extract_booklet.py` (PDF -> JSON),
`compare_grades.py` (exit 1 on a table-count mismatch) and `compare_prose.py`.
Each script's docstring documents its inputs, outputs and exit codes.
