# Extraction and review protocol

## Scope

The implemented pipeline is **full F16/BF16 MedGemma 27B alone for extraction,
followed by deterministic SCOGS grading**. It does not train an encoder or
predict grades with a language model. Older ensemble/training plans and session
handoffs are superseded by this document and the [architecture](architecture.md).

The question is whether the model can fill clinical feature values from a note.
Quotes must survive the harness's existing normalization/substring check; this
is evidence grounding, not an assessment of whether the quote supports the value.
The original stage-0 prompt is preserved as an ablation control.

## Reproducible run

1. Record the original weight revision and conversion, then run the Ollama
   preflight. Save its digest, precision, and settings with the results.
2. Optionally run `--backend mock --notes 2`. The mock intentionally generates one
   found quote and one invented quote per pair: grounding should be exactly 50%.
   This is never a model-quality result.
3. For reporting, use `--cohort scd_primary --stratify --holdout-frac 0.25`,
   20 notes, and the 14 focus outcomes `10,11,12,15,17,21,24,28,29,39,40,47,48,49` (or `all` for all 53). For broader absence audits, select
   `--cohort loose` deliberately and report the difference.
4. Run with `--repeat 2 --concurrency 1`; preserve per-outcome and
   seeded-versus-holdout denominators.
5. Export sheets from the saved JSON, not a new model response. Human reviewers
   check features against note text; the tables derive grades from those features.
6. Record wall-clock, tokens, model digest, context size, and prompt stage. When
   comparing stages, keep the model, selected notes, and settings fixed.

## Automatic checks

| Check | Useful interpretation |
| --- | --- |
| Quote-verified fraction of quoted proposals | Grounding; >=95% is a useful target, 85-95% needs review, <85% is concerning |
| Null-placeholder rate | Whether the omission/output contract is being followed |
| Invalid-value rate | Type/enum compliance; <=2% is a useful target, >10% is concerning |
| Unparseable/unusable replies | Inspect every failure; do not silently discard failed cases from denominators |
| Unit conversions and quote/value mismatch | Check that quoted numbers and units support accepted values |
| Conflicts | Several grounded values with no declared reduction policy; withheld, not guessed |
| Temperature-zero consistency | Fraction of pairs with identical features and presence across every repeat; 100% sequential consistency is expected |

Do not label the quote-verification percentage as precision. A quote can be in
the note and still support the wrong feature, wrong time period, or wrong value.
Higher accepted counts alone are not evidence of improvement. Concurrency can
confound consistency through batching.

## Human review

**Precision:** sample up to 100 accepted proposals from `handcheck.csv`.
For each, mark `supports_value` as `y` or `n` after reading the quote in context.
Calculate `y / (y + n)`; blank rows are unreviewed, not correct. A practical
viability target is >=90%, with uncertainty reported: at 100 observations the
interval near 90% remains several percentage points wide.

`accepted_findings` includes proposals that later conflict with each other.
`withheld_conflict` marks these in the worksheet so grounded-proposal precision
is not mistaken for precision of final grading inputs.

**Recall:** read at least five notes in full without relying solely on proposed
findings; list the supported features the model missed. This is directional
evidence at that sample size, not a precise recall estimate.

**Absence:** review up to 50 model-absent pairs in `absence_audit.csv`.
Mark `truly_absent` as `y/n`; `n / (y + n)` is the false-negative fraction
*within this sampled set of absence calls*, not population recall.
`refuted_audit.csv` separately records cases where the model said present but
the rule engine found no matching grade. Inspect conflicts and refutations in full.

For a future gold set, independently label feature values and evidence spans,
derive grades using the rubric, and freeze the evaluation split before any
training. Include some notes without preannotations to detect reviewer anchoring.
Deduplicate related corpora by article PMID.

## Limits on conclusions

- Published case reports are edited summaries, not representative clinical notes.
- Four common outcomes do not validate all 53 SCOGS outcomes.
- Neither deterministic tables nor quote matching validate clinical correctness.
- Missing evidence is not a documented negative. Keep `absent`, `refuted`,
  `grade_set`, `cannot_grade`, and `not_applicable` distinct.
- Prompt-stage comparisons need matched cases and human review. Current prompt
  fixes still require evaluation on real full-precision model runs.
- Do not mix mock results, model revisions, precisions, or cohort settings in
  one unqualified metric.
