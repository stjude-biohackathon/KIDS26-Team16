# Post-Rerun Testing Plan

Run these checks after the first rerun on the new `medgemma_27b_a100_run.ipynb` notebook with the prompt fixes applied.

## 1. Validate the three prompt fixes

Re-examine the same failure cases from the stage 3 handcheck to confirm the fixes worked.

| Fix | Handcheck row | What failed | What changed |
|:----|:-------------|:------------|:-------------|
| Plasma exchange exclusion | Row 8 | Plasma exchange scored as red cell exchange | Added "Plasma exchange (plasmapheresis) is NOT a red cell transfusion" to `transfusion_type` ordinal cue |
| FiO2 vs SpO2 clarifier | Row 24 | SpO2 reading reported as FiO2 | Added `per_outcome` clarifier to `fio2_pct` distinguishing delivered O2 from measured O2 |
| Quote word-count removal | General | Quotes were being truncated to hit "3-12 words" target | Removed word-count guidance, kept "shortest continuous run" |

**How to check:** In the new handcheck CSV, search for the same UIDs and features. If the same errors recur, the prompt fix was insufficient.

## 2. ED admission false positive

Handcheck row 25: an ED presentation was scored as an inpatient admission. This was **not fixed** in the current prompt.

**Proposed fix:** Add an ordinal cue to `admission_type` distinguishing ED visit from inpatient admission, similar to how `transfusion_type` distinguishes plasma exchange from red cell exchange.

**After rerun:** Check if this error recurs. If it does, implement the cue and rerun that outcome.

## 3. Long-note recall

Both false negatives in the absence audit came from uid 7361785-1, an 8,316-character Nigerian SCD+COVID case report. The model missed:
- Pain crisis (NRS = 8, IV morphine documented)
- Fever (>39°C in the opening sentence)

These are obvious findings that shorter notes captured without difficulty.

**After rerun:** Filter the new absence audit by note length. If false negatives cluster in notes above ~6,000 characters, consider:
- A two-pass strategy (extract once, then re-read missed outcomes)
- Note chunking with deduplication
- Increasing `num_ctx` for longer notes

## 4. Stage 2b vs Stage 3 regression check

The ladder showed stage 2b as the sweet spot (0% unfound quotes, lowest `cannot_grade` rate). Stage 3 added precision rules but regressed on grounding.

**After rerun:** Compare the new stage 3 results against the ladder's stage 2b numbers. If stage 3 still underperforms 2b on grounding, consider running production at stage 2b or selectively dropping the stage 3 rules that cause regression.
