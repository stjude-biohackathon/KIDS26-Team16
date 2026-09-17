# Running `data/SCD_summaries.csv` through the pipeline

Runs the three labelled SCD cases as **full case text** and again as **summaries**,
then scores the two arms against the `true_outcome` column.

> **Read [What these numbers are](#what-these-numbers-are-and-are-not) before quoting
> anything.** This comparison measures whether the model *spots* an outcome. It does
> not measure whether the grade is right, and on the current pipeline you should
> expect `graded = 0`. That is a known defect in the grading path, not something you
> have done wrong.

---

## 0. Prerequisites

| | |
| --- | --- |
| Python | 3.10+, repo root as the working directory |
| Ollama | running, serving **unquantized F16/BF16 MedGemma 27B** |
| Disk | ~1 MB of results per arm |
| Time | 42 model calls per arm (3 notes x 14 outcomes) |

Check the model before anything else. This fails closed, and it is the most common
reason a run produces no output at all:

```bash
python scripts/experiments/medgemma_extraction.py --check-model
```

`Preflight: medgemma-27b-f16 | F16 | sha256:…` means you are ready. Anything else is
a hard stop — see [Troubleshooting](#troubleshooting). For a BF16 import add
`--model medgemma-27b-bf16` to every command below.

To rehearse the whole sequence without a model, put `--backend mock` on both
extraction commands. The mock emits one synthetic feature and verifies exactly half
its quotes, so **its numbers are meaningless** — it proves only that the plumbing runs.

---

## 1. The two runs

```bash
mkdir -p results/scd_summaries
FOCUS="10,11,12,15,17,21,24,28,29,39,40,47,48,49"   # the repo default 14

# Arm A - full case text
python scripts/experiments/medgemma_extraction.py \
  --notes-file data/SCD_summaries.csv \
  --note-column original_case_text \
  --no-stratify \
  --outcomes "$FOCUS" \
  --patient-context \
  --out results/scd_summaries/original.json

# Arm B - the summaries, everything else identical
python scripts/experiments/medgemma_extraction.py \
  --notes-file data/SCD_summaries.csv \
  --note-column summary \
  --no-stratify \
  --outcomes "$FOCUS" \
  --patient-context \
  --out results/scd_summaries/summary.json
```

Windows PowerShell: use `` ` `` instead of `\`, and `$FOCUS = "10,11,…"`.

**Every flag here is load-bearing.** Dropping one silently changes what you measured:

| Flag | Why it is there |
| --- | --- |
| `--notes-file` | Reads the CSV instead of the bundled 978-case cache. Without it you are not running your data at all. |
| `--note-column` | **The only difference between the two arms.** This is the experiment. |
| `--no-stratify` | Without it the sampler seeds on outcome vocabulary and holds part back — on a 3-row file that selects **1 note**, verified. The CLI now warns when this happens; do not ignore the warning. |
| `--outcomes` | The project's 14 focus outcomes, passed explicitly so the run does not drift if the CLI default ever changes. Only 6 of them (15, 21, 28, 39, 40, 48) appear in your labels; the other 8 are never labelled, which makes them a useful false-positive probe. See [the coverage caveat](#what-these-numbers-are-and-are-not). |
| `--patient-context` | Off by default. Without it `patient_age`/`patient_sex` are never extracted, and your CSV has no age/sex columns to fall back on. Age-stratified tables cannot grade without it. |
| `--out` | Required for scoring — the scorer reads `detailed_records`. Existing files are never overwritten. |

`--notes` is deliberately omitted: with `--notes-file` and no `--notes`, every row runs.
Passing `--notes` still caps the count.

### Optional: widen the outcome set

The 14 above ask about 8 outcomes your labels never mention, so precision already has
somewhere to go wrong — but 39 outcomes still go unasked. For the fullest picture, swap
in `--outcomes all`: 53 × 3 = **159 calls per arm**, so budget hours, not minutes.

The other direction, if you want every label measured: `--outcomes
"01,02,06,10,11,12,15,17,21,24,28,29,36,39,40,47,48,49,51,52"` is the 20-outcome union
of the focus 14 and your 12 labelled outcomes — 60 calls per arm, no label left out.

### Optional: repeatability

Add `--repeat 2` to either arm to get run-to-run consistency at temperature 0. Keep
`--concurrency 1` — batching changes floating-point results and confounds the measure.

---

## 2. Score and compare

```bash
python scripts/experiments/score_presence.py \
  results/scd_summaries/original.json \
  results/scd_summaries/summary.json \
  --labels data/SCD_summaries.csv \
  --json-out results/scd_summaries/scores.json
```

Output, per arm and then side by side. **The shape below is real, from a `--backend
mock` rehearsal — the mock invents its answers, so treat only the layout as
meaningful:**

```
original.json   note_column=original_case_text   run_id=20260917T191502-038bb6a6
  3 note(s) x 14 outcome(s) scored

   id outcome                             TP  FP  FN  TN  prec%   rec%    F1%
   10 Chronic Pain                         0   2   0   1    0.0    n/a    n/a
   15 Stroke (hemorrhagic or ischemic)     1   2   0   0   33.3  100.0   50.0
   21 Chronic Kidney Disease (CKD)         1   0   0   2  100.0  100.0  100.0
   28 Acute Sickle Cell Pain Episode       3   0   0   0  100.0  100.0  100.0
   40 Leg Ulcer                            0   1   1   1    0.0    0.0    0.0
   48 Acute Chest Syndrome (ACS)           2   1   0   0   66.7  100.0   80.0
   ...
  OVERALL                                  8  23   1  10   25.8   88.9   40.0

  grade status (42 pairs): absent=11, cannot_grade=31
  gradeable: 0/42 (0.0%) - coverage, NOT accuracy: these labels carry no severity grades.

==============================================================================
COMPARISON
==============================================================================
  metric                 original_case_        summary
  presence precision %             25.8           26.7   delta +0.9
  presence recall %                88.9           88.9   delta +0.0
  presence F1 %                    40.0           41.0   delta +1.0
  gradeable %                       0.0            0.0   delta +0.0

  15 pair(s) where the presence call differs:
    PMC10166222_01 outcome 15 (Stroke (hemorrhagic or ische): True -> False
    PMC10332925_01 outcome 47 (Depression): True -> False
    ...
```

`n/a` is not 0%. It means the denominator was empty. Chronic Pain above shows
`rec% n/a` because your labels never mark it present, so there is nothing to recall —
its row can only ever produce false positives and true negatives.

**Read the pair-level diff, not just the rates.** With three notes, one flipped call
moves F1 by several points. Two arms can share an F1 and still disagree everywhere —
the per-pair list is where the actual finding is.

Outcomes that were never labelled *and* never called are omitted from the table: they
contribute only true negatives and would pad every rate upward.

### Hand-check the evidence

Rates over 3 notes are anecdote. The claim worth making comes from the review sheets:

```bash
python scripts/experiments/review_results.py results/scd_summaries/original.json
python scripts/experiments/review_results.py results/scd_summaries/summary.json
```

This writes `handcheck.csv`, `conflicts.csv`, `absence_audit.csv` and
`refuted_audit.csv` next to each run. `handcheck.csv` is the one that matters:
quote verification only checks that the quoted words appear in the note, **not that
they support the value**. Precision is not measured until a human marks that sheet.

---

## What these numbers are, and are not

**Measured:** presence detection — did the model say outcome *N* is evidenced in this
note, and was it right? Labels are read closed-world: an outcome the run evaluated and
the row does not list counts as a true absence. That is what makes recall meaningful,
and it is sound only because `true_outcome` enumerates every outcome each case has.

**Not measured — severity grade accuracy.** `true_outcome` names outcomes; it carries
no grades. There is no ground-truth grade anywhere in this repository, so nothing here
can tell a correct Grade 3 from a wrong one. A run that grades everything correctly and
a run that grades nothing at all **score identically**. Read `gradeable %` as coverage.

**Recall here covers 9 of your 16 labelled positives, not all 16.** The 14 focus
outcomes miss 6 that your labels use — Arrhythmia, DVT, Systemic Arterial Hypertension,
Fever, Pulmonary Embolism, Pulmonary Hypertension. Those 7 labelled positives (6 on
`PMC10253251_01`, 1 on `PMC10332925_01`) are never asked about, so they can be neither
found nor missed. Recall is therefore measured over the 9 positives that fall inside
the focus set. Run the 20-outcome union above if you need all 16.

**Precision is the harsher number here.** 8 of the 14 are never labelled, so every
positive call on them counts as a false positive. That is the point — it is the only
false-positive signal in the run — but it means precision and recall are being measured
over different outcome sets, and the pooled F1 mixes them. Read the per-outcome rows.

**Three notes is not a sample.** 36 pairs, of which 16 are positive. A single flipped
call moves overall F1 by ~3 points. Treat any delta here as a direction to investigate,
never as a result. If you need a defensible number, the bundled cache has 978 cases.

**The summary arm is confounded.** The summaries were produced by some earlier process,
and if that process was itself an LLM reading the case text, then "summaries score
better" may mean the summarizer surfaced the findings, not that summarization helps
extraction. Worth stating explicitly in any write-up.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Cannot reach Ollama at …` | service down, or wrong `--host` | start Ollama; `--host http://localhost:11434` |
| `Only full 27B models are supported; parameter count is …` | quantized or smaller weights | import F16/BF16 27B; see `docs/ollama_setup.md` |
| `Only F16/BF16 weights are allowed` | a Q4/Q8 tag | same — the check reads GGUF metadata, not the tag name |
| `Preflight model did not follow the JSON instruction` | broken chat template in the import | re-import the GGUF |
| `Output already exists: …` | results are never overwritten, by design | choose a new `--out`, or move the old run aside |
| `WARNING: 1/3 rows selected` | `--stratify` still on | add `--no-stratify` |
| `--cohort does not apply to --notes-file` | cohort gate does not run on a notes file | drop `--cohort`; every row is used |
| `no column 'summary'` | wrong `--note-column` | the error lists the columns that exist |
| `label(s) match no decision table` | `true_outcome` text drifted from a table name | fix the label; the error names it |
| `no note in the run matched a label row` | id mismatch | check `--id-column` against the CSV's `case_id` |
| Every pair is `cannot_grade` | expected on the current pipeline | nothing to do yet |

---

## Reproducing without a model

Proves the plumbing in about a second. Numbers are synthetic; ignore them.

```bash
python scripts/experiments/medgemma_extraction.py --backend mock \
  --notes-file data/SCD_summaries.csv --note-column original_case_text \
  --no-stratify --outcomes "10,11,12,15,17,21,24,28,29,39,40,47,48,49" \
  --patient-context --out /tmp/original.json

python scripts/experiments/medgemma_extraction.py --backend mock \
  --notes-file data/SCD_summaries.csv --note-column summary \
  --no-stratify --outcomes "10,11,12,15,17,21,24,28,29,39,40,47,48,49" \
  --patient-context --out /tmp/summary.json

python scripts/experiments/score_presence.py /tmp/original.json /tmp/summary.json \
  --labels data/SCD_summaries.csv
```

---

## What this file's claims rest on

| Claim | Evidence |
| --- | --- |
| Loader could not read the CSV | `cohort.py` `load_notes()` reads only `scd_cache.json`; no CLI flag existed |
| Stratify keeps 1 of 3 notes | `select_notes(pool_of_3, 3, focus_outcomes)` → 1 note |
| `graded = 0` | mock run: `cannot_grade 20 (71.4%)`, `absent 8 (28.6%)`, `graded 0` |
| Three-valued blocking | `grade('28', {'care_setting':'inpatient'})` → `cannot_grade`; adding `pain_co_complication`/`life_support`/`death_attributed` as `False` → `Grade 3` |
| All 12 labels map to tables | every `true_outcome` value resolves to a table id; pinned by a test |
| 9 of 16 positives measurable under the 14 | per-row label counts 2/10/4; 6 labelled outcomes fall outside the focus set |
| Suite green | `pytest -q` → **580 passed** (538 before, 42 added); `ruff check .` clean |
