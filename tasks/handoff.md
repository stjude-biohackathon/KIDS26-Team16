# Handoff — P11 MedGemma extraction harness

**Last updated:** 2026-08-31
**Branch:** `p11-harness-measurement-fixes`, in sync with origin at `909d8c4`
**Read first:** `tasks/plan.md` (§7 defines the review protocol), `tasks/medgemma_extraction_test.md` (the gates)

---

## Where things stand

The question P11 answers is *"can MedGemma 27B read a clinical note and fill in SCOGS
features well enough to grade with?"* — measured by running the model alone, straight into
the real decision tables. Grading is never what is being measured; extraction is.

The serving path and the harness both work now. **The measurement is not finished**, because
the two review sheets that produce the actual numbers have never been filled in.

| | state |
|:--|:--|
| Model | `medgemma-27b-bf16`, BF16, digest `7322761276de…`, served by Ollama on a Colab A100 80GB |
| Weights | merged GGUF + built Ollama store, both cached in `MyDrive/scogs_ollama_models` |
| Last full run | `run_id 20260901T002248-e3a484ed`, cohort `scd_primary`, 20 notes × 4 outcomes (28/48/36/19), repeat 2 |
| Review sheets | **never filled in** — no precision number, no false-negative rate |

That run: 92 proposed findings, 71 quote-verified (98.6% of quoted), 68 accepted, 19 null
placeholders (20.7%), 0 tokenizer artifacts, 100% run-to-run consistency. Grades: 5 graded,
16 grade_set, 9 cannot_grade, 47 absent, 3 refuted.

**That run is now stale.** The prompt and the Fever table both changed after it, so its
numbers are not comparable to anything produced from here on.

---

## What changed this session

| commit | what and why |
|:--|:--|
| `1895a13` | **Harness measurement fixes.** Multi-value reconciliation (was last-write-wins), a unit guard that reads numbers out of their own verified quote, `refuted` split out of `absent`, real Ollama digest + `run_id` stamped on every artifact, quote-verified relabelled as a grounding check rather than precision. |
| `b99bdab` | **Step 5 proves the model generates** before claiming it built one. Daemon log kept at `/content/ollama.log`; `ollama create` output streamed. |
| `39c63ee` | (not mine) cohort switched to `scd_primary`. |
| `af756e9` | BF16 is ggml file type **32**, not 30 (30 is IQ4_XS). The A100 run had recorded `quant: "file_type_32"`. |
| `b865eb2` | **Fever rubric gap closed.** See below — this one overrode a deliberate prior decision. |
| `909d8c4` | **Prompt asks for `death_attributed`**, plus three smaller rules. See below — this is the change most likely to move yield. |

### The Fever gap (`b865eb2`)

The rubric's Celsius bands are not contiguous: Grade 1 `38.0–38.4`, Grade 2 `> 38.5`, so
38.5 °C — a fever — graded as `absent`. This had been recorded as a deliberate `RUBRIC GAP`
and left open. It was closed because the rubric's **own Fahrenheit annotations** are
contiguous and decide it:

```
100.4 F = 38.0000 C     101.2 F = 38.4444 C     101.3 F = 38.5000 C
```

Grade 1's `<= 38.4` excludes 38.4444 — which is 101.2 °F, the rubric's own Grade 1 endpoint.
Since the unit guard now converts Fahrenheit quotes, that was reachable. Both bounds moved:
`[38.0, 38.5)` and `[38.5, ∞)`. Recorded in `rules_vs_booklet_discrepancies.md` under a new
section (tables.py deviating from rules.md is a different axis from the rest of that file).

**The equivalent gap at outcome 10** (Pain-and-Hurt score of exactly 60) has the same shape
and *no* equivalent evidence of intent. It is deliberately still open. Don't "fix" it by
analogy.

### The prompt change (`909d8c4`)

Outcomes 28, 48 and 19 each carry `grade 5: death_attributed`. While that value is unknown
the engine cannot rule out the top grade, so it returns `grade_set` instead of closing.
`death_attributed` was extracted **zero times** in 80 pairs. Fever is the only one of the
four with no death row — which is exactly why every graded pair in that run was Fever.

```
{care_setting: inpatient, pain_co_complication: true}                       -> grade_set
{care_setting: inpatient, pain_co_complication: true, death_attributed: F}  -> graded 4
```

The prompt now asks for it, only for tables that grade on it, in a form the §2 quote contract
already allows: "discharged home on day 5" is verbatim text and it supports `false`.

---

## What to do next, in order

1. **Re-run Steps 7–12.** ~12 min; the model is already in the Drive store. Everything below
   depends on having sheets that match the current prompt and tables.
2. **Fill `results/absence_audit.csv`** (`truly_absent` = y/n). This is the false-negative
   rate and it is the number nothing else can substitute for — ~47 rows. Then
   `results/handcheck.csv` (`supports_value` = y/n) for precision.
3. **Drive cleanup.** `blobs/` holds ~54 GB of orphan — `ollama create` writes the imported
   blob *and* a compatibility rewrite, and the manifest references only one. `ollama list`
   says 54 GB; the store is 108 GB. Step 5c (`RUN_STEP_5C = True`) prunes, or walk the
   manifests and delete unreferenced blobs by hand. Also `.ipynb_checkpoints`.

### What to watch on the next run

- **`death_attributed` appearing at all**, and outcomes 28/48 producing definite grades
  instead of `grade_set`. That is the test of `909d8c4`. 16 pairs are currently stuck one
  step short.
- **Null-placeholder rate.** 20.7% on the last run; the gate is ≤5%.
- **`death_attributed: false` rows in the hand-check sheet.** New failure mode: a model
  asserting "didn't die" off a weak quote. Read those rows first.
- **AKI will probably not move.** Every AKI row needs `creatinine_x_baseline` or
  `creatinine_increase_mg_dl`, which need a *baseline* measurement case reports rarely state.
  That is a corpus limit, not a prompt one. Don't chase it with prompt edits.

---

## Traps

- **The notebook runs from what is PUSHED**, never a working copy — cell 4 clones GitHub and
  checks out `BRANCH`. Uncommitted fixes are invisible in Colab. Cell 4 now asserts the
  harness carries `unit_guard`/`reconcile`/`harness_status` and fails in Step 2 rather than
  an hour downstream.
- **Colab wipes `/mnt/local-scratch` every session.** The Ollama store never survives; only
  Drive does.
- **`HAVE_STORE` (cell 8) and `gguf_complete` (cell 10) are presence checks, not working
  checks** — `HAVE_STORE` trusts one manifest file without verifying the blobs it references
  exist. **This is unfixed and it already cost a session**: Drive held manifests without the
  model blob, Step 5 "restored" it, and Ollama refused every `/api/generate` with
  `does not support generate`. Same for the restore, which is `cp -r` of *everything* in
  Drive into the store dir — that is what dumped a 54 GB `.gguf` into `blobs/`'s parent.
- **`call_ollama` swallows backend errors into `""`** after 3 retries and prints a warning.
  A serving failure therefore reads as "the model proposed nothing" rather than an error. The
  Step 6 preflight is currently the only thing standing between that and a wasted run.
- **`results/old/*.csv`** are orphaned: an earlier run, no `run_id` column, and their results
  JSON was deleted. They cannot be matched to anything. Do not fill them in.
- **`results/*.csv` is gitignored** (sheets are regenerated per run). If a *filled-in* sheet
  should ever be committed, that rule needs narrowing first.

---

## Corrections — claims made this session that were wrong

Recorded so they are not repeated.

- **"`patient_age` recall is 4/20."** Real but misleading. `grade()` returns `absent` whenever
  `present` is false, so features on absent pairs are ignored entirely. Where age mattered
  (`present: true`) it was extracted 4/4. AKI's zero grades were then blamed on this; the
  actual cause is `death_attributed` plus the missing creatinine baseline.
- **"BF16 is a late ggml type Ollama's Go decoder can't read."** Wrong. The daemon log said
  `couldn't open model file` — the blob was *absent*, not corrupt. Get the log before
  theorising; that is why Step 5 now keeps it.
- **"`rules.md:2004` defines fever as ≥ 38.5."** That line is the *ACS* diagnostic criteria
  (outcome 48), not the Fever grading table. The Fahrenheit-band argument above is the one
  that actually holds.

---

## Open, not done

- `HAVE_STORE` blob validation, and restoring only `manifests/` + `blobs/` instead of `cp -r`.
- `call_ollama` failing loudly instead of returning `""`.
- `data/scogs_feature_schema.json` says `patient_age` is *"Required before grading … 19"*, but
  table 19 only uses it inside `(patient_age < 18 and egfr < 35)` — `{creatinine_x_baseline: 2.0}`
  alone reaches `grade_set` without it. Doc/behaviour mismatch, unresolved.
- Uncommitted in the working tree, not mine: `README.md` and `docs/windows_gpu_setup.md`
  (Windows/GPU docs), `results/local.json` shows as deleted, `notebooks/medgemma_a100_extraction.ipynb`
  and `temp/` untracked.

---

## Conventions worth keeping

- **A presence check is not a working check.** This bug appeared three times: `ollama list`
  showing a name, `HAVE_STORE` seeing a manifest, `gguf_complete` seeing a magic number. Each
  time the failure surfaced far from its cause. Prefer asserting the thing works.
- **Quote-verified is grounding, not precision.** It asks whether the quoted words are in the
  note, never whether they support the value. `fio2_pct = 21` quoting *"needed increasing
  oxygen by nasal cannula"* verifies at 100%. Precision only comes from `supports_value`.
- **Don't pool different questions into one status.** Splitting `refuted` out of `absent`
  surfaced the Fever rubric gap on its first run; pooled into 47 absences nobody would have
  looked.
