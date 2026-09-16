# SCOGS `rules.md` vs `SCOGS_Booklet.pdf` — verification log

**Audit date:** 2026-08-27  
**Source of truth:** `SCOGS_Booklet.pdf` (St. Jude Children's Research Hospital, 144 pp., created 2026-04-29)  
**Scope:** all 53 health outcomes — grading rules (Grades 1–5), frequency classification, definition, diagnostic criteria, methodology, references — plus the framework front matter and the Grading Frequency Guide.

**Method:** the booklet was text-extracted per page and split into 53 per-outcome sections keyed to the booklet's own table of contents; `rules.md` was split into the matching 53 sections. Each section pair was compared with a normalization that ignores only footnote-marker style, bullet/line-wrap formatting, and whitespace. Every flagged difference was then confirmed against the rendered PDF page.

---

## Verified identical (no action needed)

- **Grading rules (Grades 1–5) for all 53 outcomes** — content matches the booklet after the fixes in the next section.
- **Frequency classification for all 53 outcomes** — matches in all three places it appears: the booklet's per-outcome page header, the booklet's Grading Frequency Guide (pp. 132–135), the `rules.md` per-outcome attribute table, and the `rules.md` Master 53 Summary Matrix.
- **Grade 2, 3, 4, 5 tier definitions** in the framework front matter — verbatim.
- **Grading Frequency Guide (§16)** — all 53 outcome-to-pattern assignments and all four per-event counting notes (hearing loss, malnutrition, leg ulcer, fetal growth restriction) match.

---

## Discrepancies found and FIXED in `rules.md` on 2026-08-27

| # | Outcome | Section | Booklet | Was in `rules.md` |
| :-- | :--- | :--- | :--- | :--- |
| 1 | **09 Cerebral Vasculopathy** | Grades 1–5 | "Mild stenosis (25-49%) of **≤ 2** arterial segments as per MRA" (segment count qualifies *segments*) | "**≤ 2** Mild stenosis (25-49%) of arterial segments" — segment count moved to the front of every one of the five grade cells, reading as a count of stenoses rather than of segments |
| 2 | **19 Acute Kidney Injury** | Definition | Opens "**AKI:** An abrupt (within hours) decrease…" | Leading "AKI:" label dropped |
| 3 | **40 Leg Ulcer** | Grades 3 & 4 | "**AND** Greater than 8 cm² < 50% wound bed with necrotic tissue" (single clause) | An extra `AND` inserted: "Greater than 8 cm² **AND** < 50% wound bed…", turning one criterion into two |
| 4 | **52 Pulmonary Hypertension** | Methodology note | Note ends "…the highest grade will be applied. **For instance, if patient fits Grade 2 criteria based on mPAP, but Grade 3 based on NYHA and echocardiogram criteria, this patient will be classified as Grade 3 pulmonary hypertension.**" | The worked example was truncated away |
| 5 | **53 Sleep Apnea** | Pediatric Grades 1–4 | "AHI 0 – 3 **AND** Hb02 sat < 90% for > 3 min overnight **OR** AHI 3.1 – 7.9 **AND NO** Hb02 sat…" | Clause order inverted in all four pediatric grades (saturation criterion led, AHI followed) |
| 6 | **53 Sleep Apnea** | Table sub-headings | "Adults (age ≥ 18)" / "Children (age < 18 yrs)" | "(age ≥ 18 years)" / "(age < 18 years)" |
| 7 | **06 Systemic Arterial Hypertension** | Reference 3 | "…Management of High Blood Pressure in Adults**: A Report of the American College of Cardiology/American Heart Association Task Force on Clinical Practice Guidelines**." | Subtitle dropped |
| 8 | **06 Systemic Arterial Hypertension** | Methodology | "separate guidelines for adults³ and pediatrics²" | Merged to "adults and pediatrics [3, 2]", losing the per-guideline attribution |
| 9 | **11 Cognitive Dysfunction** | Reference 6 | "Kaufman, S A. Kaufman brief intelligence test: KBIT. 1990." | Rewritten as "Kaufman AS, Kaufman NL." |
| 10 | **11 Cognitive Dysfunction** | Reference 8 | "…with preschoolers. **In: Alfonso VC, Bracken BA, Nagle RJ, eds.** Psychoeducational Assessment…" | Editors' statement dropped |
| 11 | **26 Malnutrition → Stunting** | Diagnostic criteria | "Height (i.e., stature) is measured in the standing position. Length-for-age refers to measurements taken in the recumbent position and is recommended for children ≤ 2 years of age." | Compressed paraphrase |
| 12 | **26 Malnutrition → Stunting** | Reference 4 | Full 9-author list, plus "Epub 2013 Mar 25. PMID: 23528324." | Truncated to "Mehta NM, Corkins MR, Lyman B, et al"; Epub/PMID dropped |
| 13 | **26 / 27** | Diagnostic criteria bullets | "For infants aged < 2 years **of age**…" / "For children aged ≥ 2 years **of age**…" | "of age" dropped in both outcomes |
| 14 | **10 Chronic Pain** | Reference 4 | "…Highlight Summary Document. **PDF.** Updated May 6, 2021." | "PDF." dropped |
| 15 | **52 Pulmonary Hypertension** | Methodology | Footnotes: "dyspnea³,⁴" and "CTCAE v5¹" | Merged to a single "[1, 3, 4]" after "CTCAE v5" |
| 16 | **Framework — Grade 1** | Front matter | "Evidence of specific health outcome event…" / "…for treatment, **they** will not require admission" | "Evidence of **a** specific…" and "they" dropped |
| 17 | **Framework — frequency** | Front matter | Two sentences: "Guidance for these frequency patterns are denoted in SCOGS health outcome classification tables as acute, chronic, or chronic with exacerbation." and "Reference the Grading Frequency Guide for a full table of health outcomes by frequency category." | Both sentences absent |
| 18 | **Framework — frequency** | Front matter | The booklet's Framework defines **three** patterns only | `rules.md` presented "Acute + Chronic" as a fourth booklet category. It is retained (Outcome 26 needs it, and the booklet's Grading Frequency Guide does mark 26 under both Acute and Chronic) but is now explicitly labelled as derived rather than quoted. |

---

## Deliberate deviations RETAINED — `rules.md` silently corrects errors in the booklet

These were left as-is in `rules.md`. Each is a place where the booklet is wrong or typographically damaged; propagating it verbatim would degrade the rules file. **Flagging them so the booklet can be corrected at source.**

| Outcome | Booklet text (p.) | `rules.md` text | Why |
| :--- | :--- | :--- | :--- |
| **19 AKI** | "Increase in sCr by 0.3 mg/dL **(26.5x mmol/L)**" (p. 53) | "(26.5 µmol/L)" | **Clinically significant.** KDIGO's threshold is 26.5 **µ**mol/L; "mmol/L" is off by 1000×, and the stray "x" is a mis-mapped µ glyph. Highest-priority booklet correction. |
| **34 Transfusional Iron Overload** | Grade 4: "serum ferritin > 10,000 ng/mL (at steady state) **=** irrespective of iron chelation" (p. 83) | stray "=" removed | Typographical artifact |
| **50 Chronic Restrictive Lung** | Grade 1: "TLC > 70% and TLC ≤ 80% **%**; mild restrictive disease" (p. 123) | duplicate "%" removed | Typographical artifact |
| **53 Sleep Apnea** | "**Hb02** sat < 90%" — letter O typed as digit zero (pp. 129–130) | "HbO2" | Typographical artifact |
| **53 Sleep Apnea** | p. 129 heading "Children (age **<** 18 yrs)" vs p. 130 continuation "Children (age **≤** 18 yrs)" | "< 18 yrs" | Booklet is internally inconsistent; "<" chosen to complement "Adults (age ≥ 18)" |
| **30 Alloimmunization/DHTR** | "relative **reticuloctyopenia**" | "reticulocytopenia" | Spelling |
| **19 AKI / 21 CKD** | "Kidney International **Suppliments**" and duplicated year "January 2013, 2013;3(1)" | "Supplements", single year | Spelling / duplication |
| **29 Acute Splenic Sequestration** | "a drop **from** ≥ 2 g/dL from baseline" | "a drop **of** ≥ 2 g/dL from baseline" | Grammatical slip |
| **52 Pulmonary Hypertension** | NYHA Class II: "…or syncope**.** right heart cath" | comma | Punctuation |
| **05 MI / 06 HTN / 11 Cognitive** | "doi: doi:10.…" duplicated prefix | single "doi:" | Duplication |
| **35 Transient Aplastic Crisis** | "< 10,000/**ul**" (µ glyph mis-mapped) | "< 10,000/µL" | Glyph |

---

## Decision-table deviations from `rules.md` — `scripts/scogs/tables.py`

A different axis from the rest of this file: everything above compares `rules.md` against
the booklet. This section records where the executable decision tables deviate from
`rules.md`, so a grade that does not follow from the rubric as written can always be
traced to a decision rather than a bug.

| Outcome | `rules.md` text | `tables.py` rule | Why |
| :--- | :--- | :--- | :--- |
| **36 Fever** | Grade 1: "Temperature 38.0 – 38.4 °C **(100.4 - 101.2 degrees Fahrenheit)**"; Grade 2: "Temperature **> 38.5** Celsius **(101.3 degrees Fahrenheit)**" (l. 1572–1573) | Grade 1: `38.0 <= temperature < 38.5`<br>Grade 2: `temperature >= 38.5` | **Clinically significant.** As written the Celsius bands are not contiguous: `(38.4, 38.5]` matches no grade, so 38.5 °C — a fever — grades as `absent`. **The rubric's own Fahrenheit annotations resolve it.** 100.4 °F = 38.0000 °C, 101.2 °F = **38.4444** °C, 101.3 °F = **38.5000** °C. The Fahrenheit bands *are* contiguous, and the Celsius text is a one-decimal rounding of 38.4444 — so the intended bands are `[38.0, 38.5)` and `[38.5, ∞)`. Note the literal reading is worse than a lone gap at 38.5: a note reporting **101.2 °F, the rubric's own Grade 1 endpoint**, converts to 38.4444 and grades as `absent`. Closed 2026-08-31. Surfaced by two pairs in the 2026-09-01 A100 run that landed on exactly 38.5 and were reported as `refuted` — the model read the fever correctly and the tables overruled it. **Flagging so the booklet can be corrected at source: the Celsius bands should read 38.0–38.4 and ≥ 38.5.** |

**Gaps deliberately left open.** Outcome 10 (Pain and Hurt score of exactly 60) has the
same shape — Grade 3 and Grade 2 bands that do not meet — but no equivalent evidence of
intent, so it stays open and returns cannot-determine rather than being rounded into a
band. See the `RUBRIC GAP` note on that table.

---

## Intentional formatting differences (not discrepancies)

`rules.md` is a restructured reference manual, not a transcription. The following are by design and were not changed:

- Grading criteria are rendered as Markdown tables with an added **Severity Tier** column (Mild / Moderate / Severe / Life-Threatening / Fatal) and an added **Primary Modality** attribute row. Neither exists in the booklet; both are derived labels.
- Booklet prose that enumerates items is rendered as bulleted lists with bolded lead-in terms (e.g. Outcome 17 Sickle Cell Retinopathy definition, Outcome 26/27 age bands).
- Superscript footnote markers are rendered as `[n]`.
- A few citations are reformatted to a consistent style (e.g. Outcome 22 ref 3 "Volume 107, Issue 2, February 2022, Pages 309–323" → "February 2022;107(2):309-323"; Outcome 15 ref "doi:https://doi.org/10.1002/ana.22427" → "doi:10.1002/ana.22427"; Outcome 47 ref 2 drops the stray "2021/04/06").
- The Forward / Development / Applications front matter is summarized rather than quoted; the Grade 1–5 tier definitions and frequency-pattern definitions within it are verbatim.

---

## Independent re-verification — 2026-08-28

Re-run from scratch with a separately written parser, to check the 2026-08-27
audit rather than trust it. **Result: confirmed. No substantive discrepancy
between `rules.md` and the booklet's grading rules.**

**Method.** `pdftotext -layout` over all 131 content pages; outcomes 01–53 keyed
to the booklet's own table of contents (printed page == PDF page, verified);
grade cells recovered as blank-line-delimited blocks with orphan blocks
re-attached (the "Grade N" label is vertically centred, so a tall cell splits
around it); `rules.md` split on `### NN.` and its `| **Grade N** |` rows read
directly. Both sides passed through one identical normalizer (Unicode folding,
Markdown stripping, footnote-marker removal, punctuation folding) and compared
cell by cell, with a word-level diff on every mismatch.

**Grading rules — the load-bearing comparison**

| | |
| :-- | :-- |
| Outcomes parsed | 53 / 53 |
| Grade tables per outcome | matches `rules.md` in all 53 (26 and 53 two-table; 42 single table with inline Adult:/Pediatric: clauses) |
| Grade cells compared | 275 |
| **Identical after normalization** | **266** |
| Differing | 9 — all explained below, none substantive |

The 9: three are PDF line-wrap spacing inside slash constructions (`40` G1
`pain/itching/burning/ warmth`, `09` G5 `stenosis/ occlusion`, `23` G2 `and/ or`);
four are outcome 53's pediatric `Hb02`/`HbO2` digit-zero typo; one is outcome 34
G4's stray `=`; one is outcome 50 G1's doubled `%`. The last three are booklet
typos already listed under *Deliberate deviations RETAINED* — `rules.md` is
correct in each.

**Other sections**

| Section | Identical | Remaining differences |
| :-- | :-- | :-- |
| Frequency classification | **53 / 53** | — |
| Definition | 43 / 53 | footnote-marker residue; outcome 32 is a parser artifact (booklet heading reads `Diagnostic Criteria3`, footnote fused to the heading) |
| Diagnostic Criteria | 35 / 53 | footnote residue, unit spacing (`7cm/sec`), µ-glyph mis-mapping, and two booklet typos already logged (30 `reticuloctyopenia`, 29 `drop from`) |
| Methodology | 43 / 53 | footnote residue; outcomes 28 and 22 are parser artifacts (a mid-sentence "Diagnostic Criteria," at column 0 opens a phantom section) |

Every difference above ratio 0.94 was opened and read against the rendered page.
No case was found where `rules.md` misstates the booklet.

**Confirmed independently: outcome 19 AKI.** The booklet's PDF text stream reads
`0.3 mg/dL (26.5x mmol/L)`. The same font mis-maps µ to plain `u` at outcome 35
(`< 10,000/ul`), so the `x` is a mis-mapped µ glyph — but the doubled `mm`
means the printed page still needs a human eye. Either way `rules.md`'s
`26.5 µmol/L` is the correct value (0.3 mg/dL × 88.4 = 26.5 µmol/L, matching
KDIGO) and is what the decision tables implement.

Reproduce: `scripts/audit/extract_booklet.py`, `compare_grades.py`, `compare_prose.py`.
