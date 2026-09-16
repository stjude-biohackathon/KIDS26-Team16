# P11 Summary — What was tested and what improved accuracy

## The question

Can MedGemma 27B, served via Ollama on a Colab A100, read a clinical note and extract SCOGS features accurately enough to grade with? Grading is never what is being measured — the decision tables do that mechanically. Extraction quality is the variable.

---

## The prompt ablation ladder

Five cumulative stages were tested on 20 SCD-primary notes × 4 outcomes (19, 28, 36, 48), repeat 2. Each stage adds to the previous — nothing is turned off — so a number can be attributed to the change that produced it.

| Stage | What it adds | Rationale |
|:------|:-------------|:----------|
| **0** | The original prompt, unchanged | Baseline arm of the ablation. Never edited so comparisons don't drift. |
| **1** | Constrained decoding (JSON schema), field order (findings before present), rule grouping, `<note>` tags, content-level retry, repeat_penalty 1.0 | Mechanical fixes: stage 0 put `present` first so the decoder committed to the discard flag before reading any findings; features were alphabetical (burying grade-driving ones); rules mixed contracts with preferences; no retry on bad content. |
| **2a** | Rubric definition + criteria injected, `present_quote` field | Gives the model the rubric's own words for what counts as "present", and asks it to quote the evidence. Without this, the model had to guess what "Acute Chest Syndrome" means from the feature list alone. |
| **2b** | Chain-of-thought `evidence` field, generated first | The model reasons about what the note says before filling in findings and presence. Generated before the structured fields so the reasoning isn't post-hoc. |
| **3** | Episode scoping, negation rules, ordinal cues (clinical synonyms), unit wording, quote hardening | Precision rules. Expected to reduce accepted findings (stopping bad ones), not increase them. This is the stage hand-check sheets must be filled against. |

### Key ladder results

| Metric | Stage 0 | Stage 1 | Stage 2a | Stage 2b | Stage 3 |
|:-------|:--------|:--------|:---------|:---------|:--------|
| Quote verified % | 73.6% | 95.8% | 97.1% | 100% | 98.6% |
| Null placeholders | 31.5% | 22.8% | 20.7% | 19.6% | 20.7% |
| Run-to-run consistency | 96.3% | 100% | 100% | 100% | 100% |
| `cannot_grade` pairs | 14 | 10 | 10 | 8 | 9 |

**Stage 2b was the sweet spot** on automated metrics: 100% grounding, lowest `cannot_grade`, perfect consistency. Stage 3 regressed slightly on grounding (98.6%) because the tighter quote rules reject more, but its precision rules are expected to improve the hand-check numbers.

---

## Decisions and changes that improved accuracy

### 1. Field order — findings before present (Stage 1)

**Problem:** Stage 0 asked for `"present": true|false` as the first JSON field. The constrained decoder committed to that flag before generating any findings, so the model decided "present or not" without having reasoned about what was in the note.

**Fix:** Stage 1 reorders to `findings → present`. The model now extracts findings first, then the presence call they support.

**Impact:** Part of the stage 0→1 jump from 73.6% to 95.8% grounding.

### 2. Constrained decoding — JSON schema per outcome (Stage 1)

**Problem:** Stage 0 asked for free-form JSON. The model could (and did) invent feature names, emit wrong value types, and produce structurally invalid replies.

**Fix:** A per-outcome JSON schema with `feature` pinned to a `const` enum, `value` typed per feature (number, boolean, or enum), and `quote` as a required non-empty string.

**Impact:** Eliminated invalid feature names and wrong value types entirely.

### 3. Feature ordering — grade-driving features first (Stage 1)

**Problem:** Stage 0 listed features alphabetically. For outcome 48 (ACS) this opened with `acs_other_support` — a single-row, negated, grade-1 feature — burying the three features grades 2/3/4 separate on (`resp_support`, `transfusion_type`, `fio2_pct`).

**Fix:** Order by (highest grade the feature can trigger, number of table rows it appears in, name). Grade-driving features come first.

### 4. Content-level retry (Stage 1)

**Problem:** Stage 0 only retried on transport errors (HTTP failures). A reply that was valid JSON but contained null values, missing quotes, or violated the schema was accepted as-is.

**Fix:** Stage 1 retries when `reply_is_usable()` returns false — null placeholders, missing quotes, or unparseable content trigger a re-query.

### 5. Rubric definition injection (Stage 2a)

**Problem:** The model had to infer what "Acute Chest Syndrome" or "Fever" means from the feature names alone. Without the rubric's definition, `present` was often wrong — the model might call a note "present" for fever based on an infection mention with no temperature documented.

**Fix:** The rubric's own definition and criteria text is injected into the prompt: "The rubric defines this outcome as: ..."

### 6. `present_quote` — grounding the presence call (Stage 2a)

**Problem:** `present` was an ungrounded boolean. The model could assert presence or absence with no evidence, and the harness had no way to check.

**Fix:** An optional `present_quote` field asks the model to quote the note text evidencing the outcome. Optional because absence often has no quotable sentence — demanding one there buys fabrication.

**Impact:** Enabled tracking `present_quoted`, `present_unquoted`, and `present_quote_unfound` as metrics.

### 7. Chain-of-thought evidence field (Stage 2b)

**Problem:** The model went directly from reading the note to filling structured fields. Errors in reasoning were invisible — a wrong value appeared with no trace of why.

**Fix:** An `evidence` field generated *before* findings and present, where the model writes 2–4 sentences about what the note says and what it does/doesn't support.

**Impact:** Stage 2a→2b moved grounding from 97.1% to 100% and dropped `cannot_grade` from 10 to 8.

### 8. `death_attributed` extraction (pre-ladder fix, `909d8c4`)

**Problem:** Outcomes 28, 48, and 19 each have a grade 5 row keyed on `death_attributed`. While unknown, the engine can't rule out the top grade, so it returns `grade_set` (a range) instead of a definite grade. This feature was extracted zero times across 80 pairs — 16 pairs were stuck as `grade_set` because of it.

**Fix:** The prompt now explicitly asks for `death_attributed` (only on outcomes whose tables grade on it), with guidance that survival evidence (discharge, follow-up) supports `false`.

### 9. `unit_guard` — reading numbers from their own verified quote

**Problem:** The model sometimes converted units itself (e.g., 102.6°F → 38.9°C, when the correct conversion is 39.2°C). The error crosses a grade boundary. Other times it reported the number without units and the harness couldn't tell Fahrenheit from Celsius.

**Fix:** `unit_guard()` reads the number out of the verified quote, next to the unit token it was written with, and converts from there. The quote is the authority, not the model's value.

### 10. `reconcile` — multi-value collapse with declared policy

**Problem:** A note carries five creatinine values across twelve years; one belongs to the transplant donor. The old code overwrote until the last value, picking by emission order.

**Fix:** `reconcile()` reads the aggregation policy off the feature definition ("highest", "maximum" → max; "lowest", "nadir" → min). Where no policy is declared, disagreement is a conflict — reported and withheld, never guessed.

### 11. `refuted` split from `absent`

**Problem:** The engine returned `absent` for two different situations: (a) the model never evidenced the outcome, and (b) the model evidenced it but the tables overruled the call (e.g., 36.5°C "fever"). Pooled together, type (b) landed in the absence audit asking a reviewer to confirm an absence the *rule engine*, not the model, produced.

**Fix:** `harness_status()` returns `refuted` when `rule_status == "absent"` but `present == true`. Separate category, separate audit.

### 12. Fever rubric gap closure (`b865eb2`)

**Problem:** The rubric's Celsius bands were not contiguous: Grade 1 `38.0–38.4°C`, Grade 2 `> 38.5°C`. A temperature of exactly 38.5°C — a fever — graded as `absent`.

**Fix:** The rubric's own Fahrenheit annotations are contiguous and resolve the ambiguity: 101.2°F = 38.4444°C (Grade 1 endpoint), 101.3°F = 38.5°C. Both bounds moved to `[38.0, 38.5)` and `[38.5, ∞)`.

### 13. Prompt fixes from hand-check review (post-ladder, `8df0c7f`)

Three fixes based on manually reviewing 34 hand-check rows (precision = 88.2%) and 50 absence audit rows (FNR = 4.0%):

| Fix | What failed | Change |
|:----|:------------|:-------|
| Plasma exchange exclusion | Plasma exchange scored as red cell exchange (row 8) | Added "Plasma exchange (plasmapheresis) is NOT a red cell transfusion" to `transfusion_type` ordinal cue |
| FiO2 vs SpO2 clarifier | SpO2 reading reported as FiO2 (row 24) | Added `per_outcome` clarifier distinguishing delivered O2 from measured O2 |
| Quote word-count removal | Quotes truncated to hit "3-12 words" target | Removed word-count guidance, kept "shortest continuous run" |

**These fixes have not yet been validated with a rerun.** See `tasks/post_rerun_tests.md`.

---

## Current accuracy numbers (Stage 3, pre-fix rerun)

| Metric | Value | Gate |
|:-------|:------|:-----|
| Quote-verified (grounding) | 98.6% of quoted | ≥95% |
| Run-to-run consistency | 100% | ≥98% |
| Null-placeholder rate | 20.7% | ≤5% (not met) |
| Hand-check precision | 88.2% (30/34) | Manual review |
| Absence audit FNR | 4.0% (2/50) | Manual review |

The null-placeholder rate remains above the ≤5% gate. This is partly a corpus limit — features like `creatinine_x_baseline` require a baseline measurement that case reports rarely state.

---

## What has not been tested yet

See `tasks/post_rerun_tests.md` for the full plan. Key gaps:

1. **The three prompt fixes** (plasma exchange, FiO2/SpO2, quote length) need a rerun to validate
2. **ED admission false positive** (row 25) — no fix implemented yet
3. **Long-note recall** — both false negatives came from one 8,316-char note
4. **Stage 2b vs Stage 3 production choice** — automated metrics favor 2b, precision rules favor 3; needs a rerun comparison
