# P11 — Does MedGemma extract features well enough?

**Status:** harness built and unit-tested; not yet run against a model
**Script:** `scripts/experiments/medgemma_extraction.py`
**Owner decision this feeds:** whether Version B of `tasks/plan.md` §3 is viable

---

## The question

Everything currently in the plan assumes MedGemma can read a clinical note and
fill in feature values. Version B assumes it entirely. Even Version A hands
MedGemma the *"which of these three ferritin values is the current one?"* job,
which the §2 reconciler routes to it by name. **Nothing has checked.**

This is the cheapest experiment on the board and it can prune the most work.

## What it measures, and what it does not

Measures **feature extraction quality** — Layers 1–3. Grades always come from
the decision tables, which are already built and tested (P3). Grading is never
what is being evaluated here, and a run of this test can never change a grade
that the tables would not also produce.

Runs MedGemma **alone**, so it measures **Version B**. Version A cannot be
measured until the lexicon (P5) exists and a BERT is trained (P6–P8). That
ordering is the point: if Version B is strong, Version A may never need
building, and P6/P7/P8 go with it.

## Setup

| | |
| :-- | :-- |
| `local` tier | `google/medgemma-1.5-4b-it` — Apple Silicon, the iteration loop |
| `full` tier | `google/medgemma-27b-text-it` — needs a real GPU, the reporting tier |
| Serving | Ollama (installed on the dev Mac; `ollama serve` must be running) |
| Notes | 20, sampled with a fixed seed from A's 978 SCD-pattern records |
| Outcomes | `28, 48, 36, 19` — pain episode, ACS, fever, AKI: common enough that 20 notes will actually contain them |
| Decoding | temperature 0, seed 0, JSON-constrained |

Model ids are pinned in `TIERS` in the script. **Record the revision hash**, not
just the repo id — §2 provenance requires it, and `-it` repos get updated.

## Protocol

1. `ollama serve`; import both models under the tags in `TIERS` (or pass `--model`)
2. `--backend mock --notes 2` first — confirms the plumbing end to end with no
   model; the mock emits one real quote and one fabricated one, so the verifier
   should report exactly 50%
3. `--tier local --notes 20 --repeat 2 --out results/local.json`
4. `--tier full --notes 20 --repeat 2 --out results/full.json` on a GPU box
5. Hand-check (below) — the only step that measures accuracy rather than behaviour
6. Record wall-clock and tokens per note; cohort-scale profiling is the project's
   stated goal and per-note cost is the thing that decides whether it is feasible

## Automated metrics — behaviour, no labels required

These need no ground truth, which is why they come first.

| Metric | Good | Workable | Concerning |
| :-- | :-- | :-- | :-- |
| **Quote-verified %** — proposals whose quote appears verbatim in the note | ≥ 95% | 85–95% | < 85% |
| **Run-to-run consistency** — same note twice at temperature 0 | ≥ 98% | 90–98% | < 90% |
| **Invalid-value rate** — value not of the declared type or not a declared enum | ≤ 2% | 2–10% | > 10% |
| **Unparseable replies** | 0 | ≤ 2% | > 2% |

Two of these are load-bearing beyond this test:

- **Quote verification is the hallucination rate.** A rejected proposal is safe —
  it becomes `cannot_determine`, not a wrong grade — but a high rate means recall
  is being paid for with fabrication, and the §2 evidence contract is doing all
  the work.
- **Consistency below ~90% breaks §4.** Conformal calibration assumes inference
  is a pure function. If temperature-0 output drifts, the coverage guarantee is
  unsound as specified and has to be fixed or qualified before any coverage
  number is reported.

## Hand-check — the part that measures accuracy

There is no gold standard yet (that is P9), so accuracy has to come from a human
reading. Budget **~1.5 hours**.

**Precision — is what it extracted correct?**
Take up to 100 accepted proposals. For each, read the quote and the value and
answer: *does this quote support this value?* Roughly 15–20 seconds each.

| Precision | Read |
| :-- | :-- |
| ≥ 90% | Version B viable — proceed to compare against Version A |
| 75–90% | Good enough as a third engine (Version A), not alone |
| < 75% | A labeller at best, not an extractor. Revisit the prompt before the model |

At n=100 the 95% interval is roughly ±6pp near 90%, so treat these as bands, not
points.

**Recall — what did it miss?**
Read 5 notes in full and list every feature that *should* have been extracted for
the four outcomes, then diff against what it produced. ~10 minutes per note. At
n=5 this is directional only, and that is fine — the question is whether it is
missing things that are plainly stated, which shows up immediately if it is
happening.

Omission is the safer failure: it yields `cannot_determine` rather than a wrong
grade. But an extractor that omits most of what is present is not useful, however
precise the rest is.

**Grade-status distribution** is the sanity check on both. If most note-outcome
pairs come back `cannot_grade`, extraction is too sparse to grade with regardless
of how precise the few extractions are.

## Local vs full

Run both tiers on the same 20 notes and report the delta as **"these two
models"**, never "4B vs 27B" — `27b-text-it` is the text-only variant, so the
two tiers are not the same family and the gap is not purely parameter count.

Per §3 tier discipline: no number from a `local` run appears in the dashboard or
the write-up. The delta itself is the one `local` result worth reporting, and it
answers the question a site without an A100 will ask.

## Known limitations — state these with any result

- **The notes are not clinical notes.** PMC-Patients records are published case
  summaries: already compressed, well-written, and edited. Performance here is an
  optimistic estimate of performance on real notes. Corpus B's full-text `CASE`
  sections are closer and should be the follow-up.
- **No ground truth.** This compares MedGemma against a careful reader, not
  against SCOGS gold. It is a viability screen, not an accuracy claim, and no
  number from it belongs in a results table without that caveat attached.
- **Four outcomes of 53.** Chosen for prevalence, so the rare tail — the case the
  lexicon exists for — is untested here by construction.
- **Prompt sensitivity is unmeasured.** A bad result may be a bad prompt. Before
  concluding anything about the model, try one prompt variant.

## What each outcome means for the plan

| Result | Action |
| :-- | :-- |
| Precision ≥ 90%, quotes ≥ 95%, consistency ≥ 98% | Version B is live. Build it out, and treat P6/P7/P8 as probably-unnecessary rather than scheduled |
| Precision 75–90% | Version A as designed. MedGemma is the third engine and the mention-adjudicator; the lexicon and BERT stay |
| Quote rate < 85% or consistency < 90% | Fix before judging. Prompt first, then decoding settings, then the model |
| Precision < 75% after a prompt variant | Drop MedGemma from extraction. Keep it for Layer 5 rationale (which renders, and cannot fabricate) and as the P6 labeller with human review |

## Next

Blocked on nothing. `ollama serve` plus the two model imports is the whole setup.
Runs before P10 or after it — P10 changes what the extractors *emit*, not
whether MedGemma can read a note.
