# SCOGS-Scribe — Implementation Plan

**Status:** P2 and P3 complete; extraction layers not started
**Last updated:** 2026-08-28 (rules.md re-verified against the booklet; compiled
tables audited cell-by-cell against the booklet and 8 compile drifts fixed or
noted; feature schema reopened for two of those fixes and re-closed at 137
features; decision tables executable and under test; P1 dropped; P6 decided;
P7 revised)
**Event:** St. Jude KIDS26 BioHackathon, Sept 16–18 2026 (~3 weeks out)

Supersedes the earlier PMC/MTSamples harvesting plan, which was withdrawn — it
generated its own ground truth with a rules engine, which makes any accuracy
number a measure of imitating the rules engine rather than of applying SCOGS.

---

## 1. Objective

Read an unstructured SCD clinical note; emit, for each of the 53 SCOGS health
outcomes, whether it is present in this patient now and at what severity grade
(1–5), with rubric-linked rationale and a calibrated uncertainty flag.

Two hard requirements that shape everything below:

- **Grades must be derived, not predicted.** `rules.md` is the authority. A model
  that learns grades end-to-end cannot be audited by a clinician.
- **Absent, not-applicable, and cannot-determine are three different answers.**
  Collapsing them into a blank is the failure mode that hides errors.

---

## 2. Architecture

```
                 ┌── grep lexicon + numeric parser ──┐
   note ────────►├── BioClinical-ModernBERT ─────────┤──► reconcile ──► features
                 └── MedGemma ──────────────────────-┘         │
                                                               ▼
                                           decision tables (rules.md) ──► grade set
                                                               │
                                                               ▼
                                   MedGemma renders the evidence record as prose
```

| Layer | Job | Implementation | Learned? |
| :-- | :-- | :-- | :-- |
| **1** | Which of the 53 outcomes are *mentioned* | lexicon ∪ BERT spans ∪ MedGemma | partly |
| **2** | Assertion: present / negated / historical / hypothetical / family / planned-not-done | medspaCy ConText + BERT assertion head + MedGemma | partly |
| **3** | 137 evidence features (interventions, care level, labs, scores) | numeric parser + BERT + MedGemma | **yes — all labels go here** |
| **4** | grade = f(outcome, features) | deterministic decision tables | **no** |
| **5** | Rationale in prose | MedGemma, **rendering the evidence record** (§2) | generative, constrained |

**Three engines, in parallel rather than cascaded.** In a cascade, Layer 1
recall is a hard ceiling — anything the first stage misses is unrecoverable.
Run as parallel extractors into one reconciler:

- **grep wins** on exact numeric thresholds (`2000 <= ferritin <= 4999`), on
  unambiguous rare terms (*deferasirox*, *Winter shunt*, *moyamoya*), on
  structured regions (vitals blocks, med lists), and on **the rare tail** —
  leg ulcer, hepatopathy and aplastic crisis will be training-starved under any
  data plan, and a lexicon works with zero examples. It is also auditable:
  "matched literal string X at offset Y" is defensible to a clinician.
- **BERT wins** on synonymy, assertion in context, and implicit evidence —
  *"weaned to room air"* implies prior supplemental O2 and no keyword catches it.
- **MedGemma wins** on the rare tail without training data (an instruction-tuned
  medical model knows what a leg ulcer is with zero examples — this is the
  lexicon's strongest card, matched), and on **which mention** a value comes
  from: baseline vs current, ordered vs administered, this admission vs last
  year. That is precisely the span-selection ambiguity §4 identifies as the
  source of feature-level uncertainty.

**They are not three of a kind.** One engine is deterministic and two are
learned, and the reconciler has to respect that:

| Engine | Nature | Fails how |
| :-- | :-- | :-- |
| lexicon + numeric parser | deterministic, exact, auditable by construction | misses, or picks the wrong mention |
| BioClinical-ModernBERT | learned, calibrated, span-emitting | confidently wrong |
| MedGemma | learned, zero-shot, generative | fluently wrong |

**Reconciliation is by authority per feature type, not by majority vote.**
Reliability is asymmetric: if the parser matches a lab value and both learned
models disagree, a flat 2-of-3 vote would overrule the one engine that cannot
hallucinate a digit.

```
numeric values        parser authoritative where it matches; others may only flag
enum / surface cues   any engine; agreement raises confidence
implicit / inferred   BERT + MedGemma only (the lexicon cannot reach these)
which mention         MedGemma adjudicates, feeding the §4 candidate set

all three agree       -> auto-accept, high confidence
one silent            -> accept (they cover different things)
mutually exclusive    -> review flag
```

**Unanimity is weaker evidence than it looks.** BERT and MedGemma are both
transformers over overlapping medical text; when they misread *"weaned to room
air"* they will misread it the same way, agree, and be auto-accepted. The
lexicon is the genuinely independent engine — an argument for weighting it more,
not treating it as the junior partner. Agreement rate is a *measurable* proxy for
confidence, not a proof of correctness, and its correlation with correctness gets
measured once P9 produces gold.

**Free benefit: an active-learning signal before any gold exists.** Where the
three engines agree, annotation is probably wasted; where they split, it is
worth the 10–15 minutes. P9 is the project's bottleneck, so annotating
disagreements first buys more information per hour than random order.

### Evidence and audit

**The two engines run independently and neither justifies itself at inference
time. Evidence is captured as a byproduct of extraction and rendered only when
the user asks for it.** Asking for evidence never re-runs extraction — it reads
what was already recorded, so what the reviewer sees is what actually produced
the grade, not a re-derivation that might differ.

**One record, both engines.** A regex is auditable because `matched "deferasirox"
at offset 4412` is checkable in seconds without understanding the matcher. BERT
gets the same property only if it is asked for a *span* rather than a label: a
classifier emitting `transfusion_type = exchange` gives the reviewer nothing to
check. So Layer 3 is built as a span extractor, and both engines emit the same
record — the engines differ, the artifact does not:

```
feature      transfusion_type
value        exchange
span         4412-4437
quote        "underwent RBC exchange"
engine       bert | lexicon | both
evidence     explicit | inferred
confidence   0.94                       # temperature-scaled; lexicon = 1.0
```

This is the same rule §7 puts on the LLM pre-annotator — **no span, no
proposal** — and it makes the audit record and the human annotation record the
same object, so every hour of P9 verification measures extraction accuracy and
span faithfulness at once.

**`explicit` vs `inferred` is not cosmetic.** BERT earns its place on implicit
evidence: *"weaned to room air"* establishes prior supplemental oxygen and no
keyword catches it. There the span implies the value rather than stating it, and
the reviewer's task changes from *read the quote* to *judge the inference*.
Different act, different error rate. Report accuracy separately for the two;
collapsing them into one number hides the half that is actually contestable.

**Silence is evidence.** The record carries what *each* engine said, including
"lexicon: no matching term" or "BERT: no span above threshold". One engine
firing alone is a materially different trust level from both agreeing, and the
reviewer needs to see which they are looking at. On the subset the lexicon
covers, the regex is a partial oracle for BERT: agreement is measurable and
every disagreement is already a review flag under the reconciliation rule above.

**Absent and cannot-determine carry evidence too.** This is the case that gets
forgotten, and §1 turns on it. `absent` must show what was searched for and
found missing; `cannot_grade` must name the feature that was undetermined and
show both engines silent on it. `grade()` already returns the undecided rules
and the missing features — the engine-silence detail attaches to that.

**Show the clauses that mattered, not all of them.** A fired rule is often a
disjunction where one branch carried it. The predicate AST is walked to mark the
minimal satisfying subset, so the chain shows the evidence that made the grade
and marks the rest not-required:

```
Outcome 48 (ACS) -> Grade 4
  rule: (fio2_pct >= 50 or resp_support >= high_flow or transfusion_type == exchange) and life_support
    fio2_pct = 60             lexicon "FiO2 60%" @2841 | bert @2836-2851   [agree]
    life_support = true       derived
      vasopressors = true     lexicon "norepinephrine gtt" @4001 | bert: silent
    resp_support              not required for this rule
    transfusion_type          not required for this rule
```

**What is deliberately not built.** Attention maps and gradient saliency are not
explanations — attention can be changed substantially without changing the
prediction, and a clinician cannot validate a heatmap. Building one would give
the *appearance* of auditability, which is worse than not having it.
Faithful-by-construction beats post-hoc attribution.

**Prose rationale is RENDERED from the evidence record, never generated freely.**
Layer 5 is the one generative step in the pipeline and the one place a fluent
model can do real damage. A generated explanation that is not causally connected
to the decision is a *rationalisation*, and it is more dangerous than an
attention heatmap because it reads authoritatively.

The concrete failure: the table fires Grade 3 on `transfusion_type == exchange`;
MedGemma, reading the same note, writes *"Grade 3 because FiO2 reached 60%."*
Every fact true, the grade correct, and the clinician has been handed a chain
that does not exist.

So Layer 5 is constrained, and the constraint is checkable:

- input is the evidence record only — the fired rule, the **minimal satisfying
  subset**, and the spans — never the raw note plus a free instruction
- every feature named in the output must appear in that record; a generation
  naming anything else is **rejected, not down-weighted**
- the check is programmatic and belongs in the serving path, not in review

This keeps the natural-language rationale the proposal promises without letting
it drift from what actually happened.

**The span claim is testable, and the test is the metric.** A cited span could be
decoration — the model may have predicted from elsewhere and the pointer is a
lie. Occlusion settles it: delete the span and re-run (necessity), or feed the
span alone (sufficiency). Two extra forward passes, so this runs on demand or
in batch over the eval set, never in the serving path. It yields a hard number —
*the cited span was necessary in N% of predictions* — which belongs in §9 next
to conformal coverage.

**Cost.** Spans are free at extraction time: `match.span()` for the lexicon, and
for a span extractor the offsets *are* the output. On-demand means rendering,
not recomputing. Only the occlusion test costs anything.

**Provenance is pinned.** Every record carries the model-weights hash, lexicon
version, tables version and schema version. Without it, "show me the evidence"
a month later renders a chain that no longer corresponds to the stored grade.

### P10 — implementation layout

§2 above is the design. This is what gets built, and it is deliberately small:
P10 defines the contract and wires it through `grade()`. The engines that *fill*
it come later (P5 for the lexicon, P11 onward for MedGemma, the BERT heads after
training). Doing it in this order is the whole point — a head trained to emit a
label cannot be made to emit a span afterwards without retraining.

**1. The record.** `scripts/scogs/evidence.py`

```python
@dataclass(frozen=True)
class Evidence:
    feature: str
    value: Any
    engine: str          # "lexicon" | "bert" | "medgemma" | "derived"
    start: int | None    # character offsets into the note; None only for derived
    end: int | None
    quote: str | None
    kind: str            # "explicit" | "inferred"   (§2 - different audit act)
    confidence: float    # lexicon = 1.0; learned engines temperature-scaled
    from_: tuple[str, ...] = ()   # for engine="derived": the features it came from
```

`EvidenceSet` holds `{feature: [Evidence, ...]}` for one note — a list, because
more than one engine may speak to the same feature, and disagreement must be
visible rather than collapsed. A helper reports agreement / one-silent /
contradiction per feature, which is what the §2 reconciler routes on.

**2. Wire it through.** `GradeResult` gains `evidence: EvidenceSet | None` and
`required: tuple[str, ...]`. `grade()` takes an optional `EvidenceSet`, carries
it onto the result, and computes `required` — the next item. Nothing about
grading changes; the tables and the Kleene evaluation are untouched.

**3. The minimal satisfying subset.** The part with actual logic.

A fired rule is usually a disjunction where one branch carried it. Showing all
of the clause is noise; showing the branch that fired is the audit. Two mutually
recursive walks over the AST that already exists:

```
witness_true(node)                       witness_false(node)
  Truth/Cmp/Between/In  -> {feature}       Truth/Cmp/Between/In -> {feature}
  Not(c)                -> witness_false(c)  Not(c)  -> witness_true(c)
  And(cs)               -> union of all      And(cs) -> first false child
  Or(cs)                -> first true child  Or(cs)  -> union of all
```

Everything in the rule but outside the returned set is rendered *not required*.
Derived features expand one level, so `life_support` shows which of ventilation
/ vasopressors / renal replacement / other actually fired.

**4. The Layer-5 rejection check.** `check_rationale(text, result) -> ok, offenders`
— every feature name appearing in generated prose must be in the evidence chain.
Returns the offenders so the failure is legible. This lives in the serving path,
and the rejection rate is a reported number (§9), not a silent filter.

**5. Provenance.** A `Provenance` block pinned onto every `EvidenceSet`: schema
version, tables version, lexicon version, model repo id **and revision hash**,
and the MedGemma tier. Rendering evidence for a stored grade under a different
provenance is refused, never silently re-derived.

**Tests — the invariants worth asserting**

- *Sufficiency*: re-running `grade()` with **only** the witness features returns
  the same grade. This is the strong property; it makes the chain a claim that
  can be falsified rather than a rendering.
- *Minimality* on disjunctions: dropping the witnessed branch changes the result.
- *Coverage*: every feature in a witness set has at least one `Evidence`, and a
  value with no span from any engine is a **build failure**, not low confidence.
- *Negation*: `not treated` witnesses `treated`, and the record shows the value
  that made it false — the absent/cannot-determine distinction must survive into
  the evidence, not just the grade.
- *Rationale check*: prose naming a feature outside the chain is rejected, and
  prose naming only in-chain features passes.
- *Provenance mismatch* is refused.

**Not in P10:** engines emitting spans (P5, P11, BERT), the rendering UI, and the
occlusion / span-necessity test — that one needs a model to occlude and belongs
with the first real extractor.

---

## 3. Model decisions

| Decision | Choice | Rationale |
| :-- | :-- | :-- |
| Encoder | **BioClinical-ModernBERT-base (8k ctx)** — committed 2026-08-27 | grade evidence is scattered across a note — FiO2 in vitals, disposition at the end; 8k holds an entire p90-capped case (~1,200 tokens) whole, no chunk-and-aggregate anywhere in the pipeline |
| Fallback | Bio_ClinicalBERT (512 tokens) | only if ModernBERT fine-tuning misbehaves within the time budget; would force chunking — the *median* case is already ~660 tokens |
| Not using | Med-BERT (Rasmy) | trained on diagnosis/med/procedure **codes**, cannot read text |
| Not using | Charangan/MedBERT | real text NER model, MIT, but max_seq_length 256 — strictly dominated |
| Head | **ordinal (CORAL/CORN)**, not 5-way softmax | softmax treats grade 1 and 5 as equally distant from 2; ordinal makes off-by-one cheap and confidence interpretable along the severity axis the eval metrics use |
| Calibration | temperature scaling | BERT softmax is overconfident out of the box; one parameter, free |
| Uncertainty | **Mondrian conformal, feature-level** | see §4 |
| Reasoning / rationale | **MedGemma**, two tiers — see below | open weights, so notes never leave the machine |

MIMIC-derived weights (Bio_ClinicalBERT) are openly released — no PhysioNet
credentialing needed to use them, only to use the data.

### MedGemma tiers

Two tiers behind one interface. The model is **configuration, not code**: same
prompts, same output schema, same evidence contract, same decision tables —
only the backend and the model id change.

| Tier | Weights | Where | Role |
| :-- | :-- | :-- | :-- |
| `local` | **`google/medgemma-1.5-4b-it`** | development, Apple Silicon | the loop you iterate in; small enough to run on the author's Mac |
| `full` | **`google/medgemma-27b-text-it`** | event machines, any box with a real GPU | the tier every reported number comes from |

The tiers are not the same variant — `27b-text-it` is text-only — so the
`local`→`full` delta measures more than parameter count. Input here is text
either way, so it does not change what can be extracted; it does mean the delta
is reported as "these two models", never as "4B vs 27B".

Rules that follow from having two tiers:

- **The tier is provenance.** §2 pins the model id and weights hash into every
  evidence record. A grade produced by 4B and one produced by 27B are not
  interchangeable and must never be silently mixed in a results table.
- **Never quote a dev-run number.** Results reported at `local` are for
  iteration only. Anything that goes in the dashboard or the write-up comes from
  `full`.
- **Measure the gap deliberately.** Run both tiers over the same eval notes once
  and report the delta. That number is not a curiosity — the project's stated
  goal is cohort-scale severity profiling at other institutions, and "what does
  this cost on a laptop" is exactly what a site without an A100 will ask.

Pin the revision hash alongside the repo id, and check the licence terms —
MedGemma ships under
Google's Health AI Developer Foundations terms, not Apache or MIT, which matters
for §6's licensing section if anything derived gets redistributed.

### Two arms — run both, compare, then choose

The ensemble is the plan of record, but it is not obviously better than the
simplest thing that could work. Both arms are built and evaluated on the same
notes so the choice is settled by numbers.

| | Arm A — ensemble | Arm B — MedGemma alone |
| :-- | :-- | :-- |
| Layer 1–3 | lexicon + BERT + MedGemma, reconciled by authority (§2) | MedGemma only |
| Layer 4 | decision tables | decision tables |
| Layer 5 | rendered rationale | rendered rationale |
| Costs | three systems to build, debug and keep in sync | one |
| Buys | an independent non-hallucinating engine on numbers; a disagreement signal that is free confidence and free active learning | simplicity, and no fine-tuning at all |

**Both arms keep Layer 4.** This is not negotiable and it is not what is being
compared. §1 requires grades to be derived; an arm that lets a model emit grades
is a different project. What the arms compare is *feature extraction quality*.

**Comparison is only meaningful if these are held fixed:** the same frozen eval
notes, the same feature definitions in the prompt (straight from the schema),
the same evidence contract with verbatim quote-verification, the same decision
tables, the same metrics with per-outcome n, and **the same MedGemma tier** —
comparing Arm A at 27B against Arm B at 4B measures nothing.

**The result that would matter most.** If Arm B is within noise of Arm A, ship
Arm B: the BERT fine-tuning pipeline goes away, and with it P6's labeling, P7's
training split and P8's silver-label mapping — the corpora revert to an eval set
only, and the whole contamination question becomes moot. That is the largest
available simplification on the board, and it is one experiment away.

**Optional Arm C — baseline only, cannot ship.** MedGemma predicting grades
end-to-end with no tables. It answers the question a reviewer will ask — *what
do the decision tables actually buy you?* — and it is nearly free, being the
same prompts with a different output schema. Label it a baseline in every table
it appears in. It violates §1 and is not a candidate architecture.

---

## 4. Uncertainty

Conformal **wraps** softmax, it does not replace it:

```
encoder -> ordinal head -> temperature scaling -> conformal -> prediction SET
```

**Applied at the feature level, not the grade.** Most grade logic is not learned
— the tables are deterministic and ~20 outcomes are pure threshold lookups.
There is no model uncertainty there. Uncertainty lives in Layer 3, and
propagates through the table by cartesian product:

```
fio2        ∈ {36, 60}        # competing spans, not regression uncertainty
transfusion ∈ {simple, exchange}

  (36, simple)   -> Grade 2       (60, simple)   -> Grade 3
  (36, exchange) -> Grade 3       (60, exchange) -> Grade 3
                              grade set = {2, 3}
```

This yields a legible reason, which grade-level conformal never would:
*"Grade 2 or 3 — could not determine whether FiO2 reached 50%."* That is the
rubric-linked rationale the proposal promises, and it tells the reviewing CRA
exactly what to look at.

Rules:
- Candidate numeric values come from **span-selection ambiguity** (which mention,
  current vs historical, ordered vs administered) — not from a fuzzy distribution
  over values. Numbers are parsed, not predicted.
- Joint coverage degrades as `1 - k·α` across k uncertain features (union bound).
  Tighten per-feature α accordingly.
- **Cap the combinatorics.** More than 3–4 uncertain features → flag for review
  rather than enumerating 16+ branches.
- **Coverage must be per-outcome (Mondrian).** 90% marginal coverage across 53
  outcomes at wildly different prevalence hides terrible coverage on rare ones.

**Determinism.** Inference is a pure function: `{g : p(g) >= 1 - q̂}` with `q̂`
frozen. Nothing is sampled at serving time. To remove build-time variance too,
**do not randomly split** — designate a fixed, curated calibration set as a
named versioned artifact (ideally the CRA gold set) and ship `q̂` alongside the
weights. CV+/cross-conformal reclaims that data later at K× training cost.

Review triggers: `|grade set| >= 2`, Type-B engine contradiction, >3 uncertain
features, or `cannot grade: age unknown`.

---

## 5. Findings that constrain the design

Established by inspection of `rules.md` and the existing data — carry these
forward, they are all load-bearing.

**The seed dataset is unusable as-is.** `data/clincal_notes.csv`: 75,016 rows,
52 of 53 outcome columns 100% empty. The one populated column is a deterministic
function of `admitted` (Yes→3 33,793/33,793; No→2 33,738/33,738), and the note
text leaks it verbatim. Grades 1, 4, 5 never occur. Notes are 217–241 chars over
~565 templates. Contains contradictions (`"no pain rated at 4/10"` + opioids +
admission, graded blank). Treat as scaffold, not training data.

**21% of the naive label space does not exist.** 275 grade cells → 217 real, 58
N/A. Only 24 of 53 outcomes use the full 1–5 range; one can only ever be Grade 2,
one only Grade 4. Generating 53×5 produces 58 impossible labels.

**Rule order is load-bearing.** Outcomes 19, 34, 48 have grade cells with
*identical* trigger clauses separated only by an added conjunct (ACS G4 = G3 +
critical support; AKI G4 = G3 + ESRD progression). Bottom-up evaluation silently
under-grades all three. Tables are ordered highest-grade-first.

**Outcome 52 is `max_of`, not first-match.** Its grade cells are conjunctions
across mPAP × NYHA × echo, so a patient whose axes disagree (mPAP 28, NYHA IV)
matches *no cell* and returns null — indistinguishable from "absent". The rubric
note resolves it: take the highest grade any axis supports. That note had been
truncated out of `rules.md` and was restored by the discrepancy audit; the
current file is correct and the table is built from it.

**Demographics are required input, not metadata.** Outcomes 26, 42, 53 have two
parallel rubrics and the note may not say which applies — sleep apnea AHI 10 is
Grade 1–2 in an adult and Grade 2–3 in a child; osteoporosis uses T-score in
adults and Z-score in children (different measurements, not different cutoffs);
malnutrition splits on single-timepoint vs serial, not age at all. `patient_age`
additionally gates outcomes 06, 19, 25, 26, 27, 36, 53. Resolve stratum **before**
grading, and emit `cannot grade: age unknown` rather than null.

**The feature schema is roughly half-size.** The 53 decision tables reference 174
identifiers; 86 are absent from the current 64-feature schema. Stripping ~15
enum values and parse artifacts leaves **~70 genuine missing features — the real
schema is ~130.** Cue-matching found only what had surface cues in the grade
text; writing the logic found what the logic needs.

---

## 6. Data sources — committed corpora and roles

Corpora verified 2026-08-27 (local inspection, live API tests, license checks).
The corpora are committed; the training **method** is the one decision still
open — end of section.

| # | Source | License | Role |
| :-- | :-- | :-- | :-- |
| A | **PMC-Patients V2** — local, `PMC-Patients/PMC-Patients-V2.json` | CC BY-NC-SA 4.0 | real-text **eval set**; distillation input |
| B | **PMC OA full-text case reports** — BioC-PMC API | per-article; ~64% CC BY | rare-tail mining; full-note eval supplement |
| C | **ClinicalTrials.gov v2 AE tables** | public domain | `priors.json` (P4) |
| D | **AGBonnet/augmented-clinical-notes** (HuggingFace) | MIT | Layer-3 silver-label bootstrap |
| E | **Asclepius-Synthetic-Clinical-Notes** (HuggingFace) | CC BY-NC-SA 4.0 | style donor; domain-adaptation text |

**Length cap — p90, case narratives only.** The case-narrative corpora (A
summaries, B `CASE` sections) are capped at the 90th percentile of the SCD
subset: **~5,700 chars (~1,200 tokens)**. Texts above the cap are excluded,
not truncated — the long tail is dominated by multi-patient case series and
longitudinal reports where per-patient, per-episode attribution is ambiguous,
so the cap is a data-quality filter as much as a length one. Every retained
note fits the 8k encoder whole. **C, D, and E are used in full** — no cap.

**This inventory is closed.** A–E are the complete data set for the project;
nothing else gets added without reopening this section.

**A — PMC-Patients V2.** 250,294 case summaries with structured `age` and
`gender` on every record (feeds the stratified outcomes 26/42/53 directly).
1,037 match an SCD pattern (`sickle cell|SCD|HbSS|HbSC`), text 268–48,691 chars
(median ~3k). ~185 of those mention sickle cell *trait* and some keyword hits
are non-SCD — **deliberately not filtered out**: outcomes are graded from
evidence in the note, not from the underlying diagnosis, so any note bearing
`rules.md` outcomes is usable. The SCD keyword filter is an enrichment
heuristic, not a diagnosis gate; the other 249k notes remain a pool for
outcome-lexicon mining if specific features are starved.
Roles: (1) **the** real-text eval set — freeze a split before anything trains;
(2) source passages for feature-level LLM labeling if distillation is chosen.

**B — PMC full-text harvest.** Tested end-to-end recipe (Gemini's original
query returns 0 — `case reports[filter]` does not exist in the PMC db):

```
1. esearch  db=pubmed  '"sickle cell"[Title/Abstract] AND case reports[Publication Type]'   → 4,292 PMIDs
2. elink    pubmed→pmc                                                    → ~55% have PMC full text
3. GET ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/PMC{id}/unicode
                                                                          → ~73% of those in OA subset
4. filter   doc.infons['license']  (per-document: CC BY ~64%, rest NC/ND variants)
5. extract  passages where section_type == 'CASE'                         → the clinical narrative
```

Funnel ≈ 4,292 → ~2,400 in PMC → ~1,700 retrievable → ~1,100 CC BY. Full texts
carry what the summaries compress away: labs with units, imaging reports,
day-by-day course. No auth, no key; per-document license and section tags make
the harvest auditable. Precision caveat confirmed in sampling (one "sickle
cell" case report *excluded* SCD by electrophoresis) — same rule as A applies:
grade what the note evidences.
Roles: (1) mine the rare tail (leg ulcer, hepatopathy, aplastic crisis — 50–100
keyword-bearing articles each); (2) full-length-note eval supplement, closer to
real notes than A's summaries.

**C — ClinicalTrials.gov.** Verified live: 255 completed SCD studies with
results via API v2, no auth. Unchanged role: P4 compiles CTCAE-graded AE tables
into `priors.json` (grade priors per outcome), plus CDC SCDC rates. Priors and
structure only — never text, never labels.

**D — AGBonnet/augmented-clinical-notes.** The 30K *longest* PMC-Patients
notes, each paired with a GPT-4-extracted structured patient summary (JSON:
symptoms, treatments, labs, history). MIT — the cleanest license in the stack.
Role: **silver-label bootstrap for Layer 3.** Map their generic JSON onto our
~130-feature schema; where a feature maps (transfusion given, O2 support,
named labs), a free training label exists before we run any LLM ourselves.
Their schema will not cover all ~130 features — treat as a head start that
cuts LLM-labeling cost, not as complete labels. Mapping coverage is measured
in P8 before anything depends on it.

**E — Asclepius.** 157k synthetic discharge summaries (GPT-3.5, seeded from
PMC-Patients case reports). Role: style donor for generated passages (discharge-
summary register, section headers, clinical shorthand) and optional unlabeled
domain-adaptation text for the encoder. **Never a label source** — its notes
are model-generated and carry no ground truth.

### Contamination rule (load-bearing)

A, D, and E all descend from the same PMC-Patients corpus. **Revised
2026-08-28:** all three are used for training; validation and test are drawn
from **A and D only** — E stays training-only, because its notes are
model-generated and carry no ground truth, though its different register is
exactly why it earns a place in training.

**Dedupe by PMID, not `patient_uid`** (measured 2026-08-28). A's 978 SCD-pattern
records span only **911 distinct PMIDs** — 112 of them (11.5%) share an article
with another patient, and one article carries 8. Holding out patient A from
article X while training on patient B from article X puts the article's text on
both sides of the split. Simulated over a 150-patient eval draw, PMID-level
dedupe removes ~174 training patients against `patient_uid`'s 150 — an extra
~24, **2.5% of A's SCD subset**, worst draw 191. Cheap to get right and
invisible if you get it wrong.

The contamination rule is unchanged and now does more work, not less. Every
case selected for validation or test must be excluded from every training
corpus **by PMID**, E included: D indexes the same patients as A,
and E's notes are *generated from* them, so a note in E derived from a held-out
patient is that patient's data wearing a different style. Deduping A against D
and forgetting E leaves the leak wide open. Split first, dedupe all three, then
train. Skipping this makes every accuracy number meaningless.

The cost of deduping E is ~174 notes of ~157,000 — **0.1%**, estimate pending
the real PMID match in P8. E's value is paraphrase variety, which regularises
the encoder toward content-invariant representations; that is exactly the
property that also transfers a held-out patient's facts across a rewording, so
the ~174 come out and the other ~156,800 stay. D's overlap is unmeasured until
P8.

### Licensing

CC BY-NC-SA on A and E: fine for hackathon/research use; anything derived and
redistributed carries the same license. B: keep the per-article license infon
in provenance metadata; redistribute only the CC BY subset. D (MIT) and C
(public) are unrestricted.

### Training method — decided 2026-08-28 (P6)

**LLM-label real text (distillation), at feature level.** Labels are reviewed by
the user before anything trains on them. The other two options stay available as
supplements — generated passages for combinations real text never exhibits, and
D's GPT-4 summaries as a free head start — but distillation is the spine.

The options as mapped:

| Approach | Data mapping |
| :-- | :-- |
| **Generate labeled passages** | sample from the 217 valid combos → narrate in E's style; labels exact by construction |
| **LLM-label real text (distill)** | label A/B passages at **feature** level (never grade level); D's GPT-4 summaries pre-pay part of this |
| **Weak supervision (Snorkel)** | `rules.md` grade cells as labeling functions over A/B; label model resolves conflicts |

Labeling is at **feature** level, never grade level: grades are derived by the
decision tables, so an LLM that proposed grades directly would be training the
model to imitate an LLM's rubric reading rather than the rubric. Ruled out
(unchanged): MIMIC-IV-Note and n2c2 (per-person
DUAs, no team sharing; PhysioNet DUA also bars sending notes to third-party
LLM APIs, killing distillation there); MTSamples (licensing, near-zero SCD);
Synthea (no SCD module).

---

## 7. Ground truth protocol

**Principle: humans verify features, not grades.** Layer 4 is deterministic
and unit-tested, so `gold grades = decision_tables(gold features)` — free and
exact. "Was an exchange transfusion given?" is verifiable by any careful
reader in seconds; "what SCOGS grade is this hepatopathy?" needs rubric
expertise. Label at the level where verification is cheap and objective;
derive everything above it.

### Pipeline

```
frozen eval split (P7, ~100–150 notes)
   │
   ├─ 1. pre-annotate   lexicon + LLM propose feature values,
   │                    each with a QUOTED supporting span (no span → no proposal)
   ├─ 2. verify         human accepts/corrects each proposal against its quote
   │                    (~3–6 outcomes present per note ⇒ ~10–15 min/note)
   ├─ 3. derive         validated decision tables over verified features → gold grades
   └─ 4. audit          two honesty checks, below
```

Tooling: Label Studio, or a generated verification sheet with note-text links.
The span requirement is load-bearing — it makes rubber-stamping hard and every
accepted label auditable.

### Note selection — stratify by outcome, not by note

Random notes yield 40 pain crises and zero leg ulcers. Pick eval notes to hit
a minimum count per outcome (target ≥10 where the corpus allows it); B's
rare-tail mining exists to feed exactly this. Report per-outcome n alongside
every metric — an outcome with n=3 gets a dash, not a percentage.

### No external gold — `rules.md` is the authority

**Decided 2026-08-28: the external-abstractor (CRA) step is dropped.** Ground
truth is `rules.md` as re-verified against the booklet on 2026-08-28 (266/275
grade cells identical, 53/53 frequency classifications identical, every residual
difference a PDF artifact or a booklet typo `rules.md` already corrects — see
`rules_vs_booklet_discrepancies.md`).

The error class this gives up on is real and should be stated plainly: where our
reading of a rule differs from how a trained abstractor applies it, nothing in
the pipeline will notice, because the annotator, the decision table and the model
all inherit the same reading. Two partial substitutes are in place:

- **The rubric is self-checking where it is broken.** Compiling all 53 tables
  surfaced defects the prose hides — a Grade 1 that no patient can ever be
  assigned (30), two cells that no value falls into (10, 36), one cell that
  requires and forbids the same treatment (29). Those are logged as table
  `notes` and asserted in the tests; they are the shortlist to put to a
  clinician if one becomes available.
- **The conformal calibration set** (§4) now comes from the verified eval split
  rather than from external gradings. The guarantee is unchanged in form; its
  provenance is our own verified features, which is worth saying out loud when
  reporting coverage.

### Honesty checks

- **Inter-annotator agreement.** Double-annotate 15–20% of notes
  independently; report κ. Without it, "model disagrees with gold" and "gold
  is noisy" are indistinguishable.
- **Absence audit.** Verification effort concentrates on flagged outcomes, so
  misses go unexamined. Human-confirm a random sample of (note, outcome) pairs
  marked *absent* → false-negative rate estimate. This is the number that
  protects the absent / cannot-determine distinction (§1).

### What needs no labeling

Generated training passages: labels exact by construction. D's GPT-4
summaries: silver, training-only, never eval (contamination rule, §6).
E: never labeled. C: priors, not labels.

---

## 8. Work breakdown

### Pre-event (now → Sept 15) — long-lead items first

| # | Task | Blocked by | Why now |
| :-- | :-- | :-- | :-- |
| ~~P1~~ | ~~Email the CRA~~ — **dropped 2026-08-28**; `rules.md` is the authority (§7) | — | — |
| ~~P2~~ | ~~Complete the feature schema~~ — **done 2026-08-28**: 137 features, one canonical source, 0 undeclared identifiers, 0 orphans (reopened once, same day, for the booklet-audit fixes to outcomes 23 and 31) | — | — |
| ~~P3~~ | ~~Compile decision tables to executable + unit tests~~ — **done 2026-08-28**: 53 tables, 224 rules, 245 tests green; compiled predicates audited cell-by-cell against the booklet | — | — |
| P4 | Build `priors.json` from ClinicalTrials.gov AE tables + CDC SCDC rates | — | zero-approval download; turns a day of guesswork into a config file |
| P5 | Build the lexicon: `rules.md` terms + scispaCy/UMLS synonym expansion; emits the §2 evidence record with character offsets | P10 | non-regex, non-training synonym coverage; also carries the detection-only vocabulary deliberately kept out of the grading schema |
| ~~P6~~ | ~~Decide the training method~~ — **decided 2026-08-28**: LLM-label real text at feature level, user reviews the labels (§6) | — | — |
| P7 | **Split the corpora** — harvest A (+B full notes) with the p90 cap and license/provenance metadata; draw validation and test from A and D only; **dedupe by PMID** so no eval article appears in any training corpus, E included | — | E is generated *from* PMC-Patients, so deduping A and D alone leaves the leak open; 112 of A's 978 SCD patients share an article, so `patient_uid` dedupe misses 11.5% of the overlap |
| P8 | Pull D + E; dedupe against the eval split **by PMID**; map D's GPT-4 JSON onto the 137-feature schema and **measure mapping coverage** | P7 | tells us how much of the LLM labeling in P6 D pre-pays for |
| P10 | **Evidence record contract** (§2) — one record type all three engines emit, span-first; `grade()` carries the chain from grade to characters; minimal-satisfying-subset walk over the predicate AST; the Layer-5 rejection check that no rationale may name a feature outside the record | — | binds P5, the BERT heads and MedGemma from day one; retrofitting auditability onto a trained classifier is not possible |
| P11 | **MedGemma extraction test** — measures **Version B only** (Version A needs P5 and a trained BERT). Protocol, thresholds and hand-check procedure in `tasks/medgemma_extraction_test.md`; harness in `scripts/experiments/` | — | Versions A and B, and the §2 reconciler's "MedGemma adjudicates which mention", all assume it extracts well. That is an assumption, not a measurement — and it is the cheapest experiment available, so it runs first |
| P9 | **Execute the ground-truth protocol** (§7): pre-annotate, verify, derive gold grades, IAA slice, absence audit | P7, P10 | the eval set is the product every metric depends on; P2 and P3 are done, and P10 makes the annotation record and the audit record the same object |

### Event (72h)

**Drop order, decided now while it is cheap.** Three extractors, a three-way
reconciler and a rationale layer is a lot for 72 hours with a team nobody has
met yet.

1. **Keep regardless** — decision tables (done), numeric parser, evidence record
2. **Highest demo value per hour** — Layer 5 rendered rationale. Cheap, because
   it renders a record that already exists, and it is what makes the demo land
3. **First to cut** — MedGemma as a third *extractor*. Purely additive; nothing
   downstream breaks if it does not ship

That ordering front-loads the work that does not depend on MedGemma being good
at extraction — which is the part still unmeasured (P11).

**Day 1 — extraction.** Layers 1–2 running: lexicon + BERT candidate detection,
ConText assertion, MedGemma pass. Output: evidence records (§2) — spans with
outcome type, assertion label, engine and offsets. Checkpoint:
end-to-end on 10 hand-picked notes.

**Day 2 — grading and uncertainty.** Layer 3 feature extraction, Layer 4 tables
wired, temperature scaling, Mondrian conformal on the frozen calibration set,
propagation through the tables. Checkpoint: grade sets with rationale on the
eval set.

**Day 3 — evaluation and dashboard.** Per-outcome confusion matrices, exact-match
and within-one-grade accuracy, Cohen's κ vs the gold set, conformal coverage per
outcome, % auto-accepted vs % flagged, span-necessity rate. Shiny dashboard:
patient severity trajectories, organ-system heatmap, cohort grade distribution,
and **evidence on demand** — click a grade, get the fired rule, the features that
satisfied it, and the quoted span from each engine with the note text highlighted
(§2).

---

## 9. Verification

No completion claim without one of these:

- **Rubric fidelity — done 2026-08-28.** `rules.md` re-verified against
  `SCOGS_Booklet.pdf` with an independently written parser: 266/275 grade cells
  identical after normalization, 53/53 frequency classifications identical, and
  every one of the 9 residual differences read against the rendered page. None
  substantive. Reproduce with `scripts/audit/`.
- **Decision tables — done 2026-08-28.** `python3 -m pytest tests/` → **245
  passed**. Covers: all 53 tables' grade sets match the rubric's non-N/A grades
  exactly (0 mismatches); every declared grade reachable by search over each
  outcome's own boundary values; the ordering traps (19, 34, 48) each verified
  to flip on the added conjunct; `max_of` (52) verified against the rubric's own
  worked example; the stratified outcomes (26, 42, 53) verified to grade the same
  measurement differently per stratum; the 2026-08-28 booklet-audit fixes (23,
  31, 39, 42-pediatric, 51) each pinned by a targeted test.
- **Compiled tables vs booklet — done 2026-08-28.** All 275 grade cells read
  against the compiled predicates directly (booklet extraction, not via
  `rules.md`); 53/53 grade sets match the booklet's non-N/A cells
  programmatically. 8 compile drifts found: 5 fixed in `tables.py`
  (23 G1 conjuncts, 31 G1 isolated thrombocytopenia, 39 ADL clause
  consistency, 42 pediatric G1 conjunction, 51 G4 symptoms), 3 recorded as
  `notes` (17 stage-3-with-vision-loss, 40 exactly-8 cm² gap, 52 un-encoded
  right-heart-failure negative).
- **Stratified outcomes — done.** `cannot grade: age unknown` is returned rather
  than null when age is missing (42, 53), and outcome 26 names `has_serial_height`
  as its own missing prerequisite.
- **Three-valued semantics — done.** An undocumented finding evaluates UNKNOWN,
  never False: a note that does not mention treatment yields a grade set, not a
  silent downgrade.
- **Evidence completeness (§2).** Every non-absent feature value carries a span
  and a quote from at least one engine; a value with no span is a build failure,
  not a low-confidence result. Assert it over the whole eval set.
- **Span faithfulness (§2).** Occlusion over the eval set: report the rate at
  which the cited span was *necessary* for the prediction, per engine and split
  by `explicit` vs `inferred`. A high accuracy with a low necessity rate means
  the audit trail is decorative and the number cannot be trusted.
- **Rationale faithfulness (§2, Layer 5).** Every feature named in a generated
  rationale appears in the evidence record it was rendered from. Violations are
  rejections, and the rejection rate is reported — a rising rate means the
  constraint is being leaned on, which is itself a finding.
- **Arm A vs Arm B (§3).** Both arms reported on the same notes at the same
  MedGemma tier, per-outcome n beside every number. Arm C, if run, labelled a
  baseline everywhere it appears.
- **Tier discipline.** No number sourced from a `local` (4B) run appears in the
  dashboard or the write-up; the 4B-vs-27B delta is measured once and reported
  as its own result.
- **Evidence reproducibility.** Rendering evidence for a stored grade returns the
  captured record, never a re-run. Pin weights/lexicon/tables/schema versions and
  test that a version mismatch is refused rather than silently re-derived.
- Conformal: empirical per-outcome coverage on held-out data vs the nominal 1−α.
- Grader: exact-match and within-one-grade accuracy against the gold set, never
  against LLM-generated labels.

---

## 10. Risks

| Risk | Mitigation |
| :-- | :-- |
| No external abstractor validates our reading of `rules.md` (P1 dropped) | accepted, not mitigated. The compiled tables surface the rubric's *internal* defects (§7), but a shared misreading stays invisible. Report accuracy as agreement with our verified features, and say so in those words |
| Conformal exchangeability broken (calibrate on generated, test on real) | calibrate on whatever we deploy against. With E in training but never in validation or test, the calibration set stays real-text |
| Audit trail looks explanatory but is not faithful (attention maps, saliency) | not built by design (§2). Auditability is span-first and occlusion-tested; the necessity rate is reported so a decorative trail is visible as a number rather than assumed |
| Generated rationale drifts from the actual decision path | Layer 5 renders the evidence record and may not name a feature outside it; violations rejected in the serving path, rejection rate reported (§9) |
| Three engines do not fit in 72 hours | drop order fixed in §8 before the event; MedGemma-as-extractor is additive and first to cut |
| BERT and MedGemma share a failure mode and agree while both wrong | the lexicon is the independent engine and holds authority on numerics; agreement is treated as a measurable proxy for confidence, never as proof (§2) |
| Rare-tail outcomes have no training signal | lexicon carries them; report per-outcome coverage so the gap is visible rather than hidden in an average |
| Feature schema churn after implementation starts | closed: P2 completed before P3, and `build_schema.py` fails the build on any undeclared identifier or orphan feature |
| Team-wide DUA blockers | no gated corpus is in the critical path by design |

---

## 11. Current state

```
rules.md                          53 outcomes, re-verified against the booklet 2026-08-28
rules_vs_booklet_discrepancies.md audit log + the 2026-08-28 independent re-verification
scripts/audit/                    booklet parser + grade/prose comparators (reproducible)

scripts/scogs/features.py         137 features - THE contract; types, units, enums,
                                  annotator-facing definitions, 9 flagged for review
scripts/scogs/tables.py           53 decision tables, 224 rules, rubric ambiguities
                                  recorded as notes rather than silently resolved
scripts/scogs/predicates.py       predicate DSL: tokenizer, parser, Kleene logic
scripts/scogs/evaluate.py         grade() -> graded | grade_set | absent |
                                  cannot_grade | not_applicable
scripts/scogs/build_schema.py     regenerates the JSON; fails on undeclared/orphan
scripts/experiments/              MedGemma extraction test (P11) - see its own runbook
                                  at tasks/medgemma_extraction_test.md
data/scogs_feature_schema.json    generated artifact, 137 features x 53 outcomes
tests/                            264 tests, all passing

data/clincal_notes.csv            degenerate - see §5, do not train on
PMC-Patients/PMC-Patients-V2.json 250,294 case summaries, 1,037 SCD-pattern (§6-A)
BioC-PMC API                      tested; harvest recipe in §6-B, not yet run
AGBonnet, Asclepius (HF)          identified, licenses verified, not yet pulled (P8)
```

### Rubric defects found by compiling the tables

Not implementation choices — places where `rules.md` (and the booklet behind it)
is internally inconsistent. Each is recorded as a `notes` entry on its table and
asserted in the tests, so none can be silently "fixed" later.

| Outcome | Defect |
| :-- | :--- |
| **29** Acute Splenic Sequestration | Grade 3 requires treatment "i.e. erythrocyte transfusion, **IV fluids**, etc." and simultaneously excludes "requiring splenectomy, **fluids**, etc." Present in the booklet (p.73), so not a transcription error. Implemented as: any treatment qualifies, only splenectomy disqualifies — the only reading under which Grade 3 is reachable. |
| **30** Alloimmunization/DHTR | Grade 1 is "not requiring intervention"; Grade 2 is "decline < 20% **and** not requiring intervention". Every Grade 1 patient therefore also satisfies Grade 2, so **Grade 1 is unreachable** for any fully-documented patient. It survives only as a member of a grade set when the haemoglobin decline is undocumented. |
| **36** Fever | Grade 1 is 38.0–38.4 °C, Grade 2 is "> 38.5 °C". A temperature of exactly **38.5 falls in neither**. Left as a gap rather than widening a band. |
| **10** Chronic Pain | Grade 3 is Pain-and-Hurt "< 60", Grade 2 is "61–80". A score of exactly **60 falls in neither**. |
| **42** Osteoporosis | The fracture-history and BMD-therapy clauses are printed after the Pediatric line but read "for both pediatrics and adults". Applied to both strata; without that conjunct Grades 1 and 2 are indistinguishable on BMD alone. |
| **47** Depression | Grade 4's trailing "AND treatment recommended" read distributively would make an attempted suicide ungradeable whenever treatment was not documented. Not applied. |
| **19** AKI | Booklet prints "0.3 mg/dL (26.5x mmol/L)"; the correct value is 26.5 **µ**mol/L (KDIGO). `rules.md` is right; the printed page still wants a human eye. |
| **17** Retinopathy | Grade 2 is "Stage 3 SCR **without** vision loss"; stage 3 *with* vision loss short of legal blindness matches no cell. Compiled liberally (stage 3 → Grade 2 regardless; legal blindness tested first). |
| **31** Hypersplenism | Grade 1 is "isolated **thrombocytopenia**" specifically; an isolated anaemia or leucopenia matches no cell. Left as a gap — returns absent. |
| **40** Leg Ulcer | A wound of exactly **8 cm² falls in neither** band (Grade 2 "< 8", Grades 3–4 "> 8"). Same class as the 10 and 36 gaps. |
| **52** Pulm. HTN | Grades 1–3 all require "NO evidence of right heart failure", but Grade 4 needs a second conjunct too — RHF without it matches no cell. The negative is not encoded; such a patient grades from the mPAP/NYHA axes. |

### What P2 and P3 changed

The three earlier scripts (`build_feature_schema.py`, `schema_corrections.py`,
`decision_tables_backbone.py`) are retired — they held five competing definitions
of what a feature was. Beyond consolidating them:

- **The tables now execute.** 224 predicate rows had never been parsed; one
  (`testes_volume < 3 by 14`, outcome 25) was not valid syntax in any language.
- **86 undeclared identifiers** are now declared features; **38 enum values** that
  had leaked in as bare identifiers (`invasive_ventilation` across 16 outcomes)
  are `feature == value` comparisons against a declared ladder.
- **Enum mismatches fixed.** `invasive_procedure` declared `none/bedside_or_
  percutaneous/surgical` while tables compared it against `urgent`,
  `aspiration_irrigation`, `surgical_shunt`. Replaced by outcome-scoped ladders.
- **`treated` is outcome-scoped.** One global boolean meant transfusion in 29 and
  antibiotics in 37; it now carries a per-outcome definition the annotator sees.
- **Missing conjuncts restored** in 40 (exudate, periwound), 42 (fracture history),
  25 (hormone replacement), 48 (other supportive treatment), 49 (ADL and SABA
  routes), 23 (infertility, no-intervention) and 31 (isolated *thrombocytopenia*,
  not any single cytopenia — the earlier compile would have graded an isolated
  anaemia as Grade 1), and outcome 10's two PROMIS domains split apart — they
  have different thresholds and collapsing them mis-graded every behavior-only
  patient.
