# Data inventory

| Path | Purpose |
| --- | --- |
| `pmc_patients/scd_cache.json` | Bundled 978-case SCD-pattern cohort; primary extraction input |
| `pmc_patients/README.md` | Upstream dataset card, attribution, and CC BY-NC-SA 4.0 terms |
| `scogs_feature_schema.json` | Generated 137-feature, 53-outcome schema from `scripts/scogs/` |
| `clinical_notes.csv` | Legacy synthetic/scaffold notes and labels, preserved but not used by the runner |
| `clinical_notes_original.csv` | Original legacy CSV, preserved separately |

The legacy filenames were corrected from `clincal_notes*.csv`; file contents
were preserved. They are not SCOGS ground truth: the previous audit found
template-like notes, mostly empty outcome columns, and admission-derived labels.
Do not use them to support accuracy claims.

The cache contains published case summaries, not representative raw clinical
notes. The `loose` cohort includes any SCD-pattern match; `scd_primary` applies a
stricter patient-disease filter at runtime. Selection remains deterministic, and
the random holdout allows enrichment bias to be measured.

Optional raw files downloaded by `scripts/download_data.py` also live in
`pmc_patients/` and are ignored by Git. A download rebuilds the cache; freeze the
cache and selected patient IDs when reproducing an experiment. If later training
on derived corpora, split and deduplicate by article PMID, not just patient UID,
to prevent patients from the same article leaking across train/test.

Respect source and model licensing when redistributing data or derived material.
Do not add private patient notes or generated patient-level result exports to Git.
