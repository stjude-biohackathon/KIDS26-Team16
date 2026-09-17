"""The check pipeline: decide which model findings are trustworthy enough to grade.

Every finding the model proposes passes through here before any decision table
sees it: quote grounding -> type/enum coercion -> unit, age and TLC guards ->
conflict reconciliation. `verify()` returns the accepted features; `precheck()`
runs the same checks without counting, to build re-prompt feedback.

Called by `medgemma_extraction.run()` and by the tests. Nothing here calls a
model or the grading tables (grading is experiments/grading.py). All counters
live on `Tally`, which is only ever changed from the main thread.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from scogs.features import FEATURES


# ==============================================================================
# CLINICAL EXTRACTION VERIFICATION & CHECK PIPELINE
# ==============================================================================
# This section implements the multi-stage verification and grounding pipeline
# that checks and validates LLM-extracted clinical features against the source
# clinical note and the SCOGS schema before any decision table evaluation:
#
#   1. Verbatim Quote Grounding (normalize, verify):
#      Checks that quoted evidence exists verbatim in the source clinical note.
#   2. Schema Type & Enum Coercion (coerce):
#      Enforces declared schema types (bool, num, ord, cat) and enum values.
#   3. Unit Guards & Normalization (unit_guard):
#      Reconciles numbers against quote units, handles conversions, and flags mismatches.
#   4. Age Guard (age_guard):
#      Reconciles compound age units into decimal years; drops gestational age.
#   5. TLC Guard (tlc_guard):
#      Reconciles tlc_pct_pred against raw lung volume (L) vs. percent predicted (%).
#   6. Conflict Reconciliation (reconcile, reduce_policy):
#      Reconciles multiple findings per feature; withholds unresolvable conflicts.
#   7. Side-Effect-Free Feedback Precheck (precheck):
#      Pre-validates output JSON for targeted error feedback re-prompting.
#   8. Diagnostic Status Reconciliation (harness_status, in experiments/grading.py):
#      Reconciles model presence calls against rule engine and objective criteria.
# ==============================================================================

# SentencePiece byte-token wreckage from a badly converted GGUF. Its presence
# means the served weights are corrupt, not that the model hallucinated - so it
# is counted and reported, never quietly normalized into a passing quote.
ARTIFACT = re.compile(r"\[UNK_BYTE_|\u2581")


def normalize(s: str) -> str:
    """Normalize text for verbatim quote comparison.

    Checks and transforms performed:
    1. SentencePiece Byte Artifacts: Strips [UNK_BYTE_...] and \u2581 from corrupt GGUF conversions.
    2. Whitespace Normalization: Collapses consecutive whitespace (tabs, newlines, spaces) to a single space.
    3. Punctuation Spacing: Strips scraping artifact spaces before closing punctuation (e.g. '(Figure )' -> '(Figure)').
    4. Casing: Lowercases text so case differences between note and quote do not cause false quote mismatches.

    Returns:
        Cleaned, lowercased string ready for verbatim substring search.
    """
    # Strip community GGUF SentencePiece byte token artifacts (e.g. [UNK_BYTE_0xe29681▁...])
    s = re.sub(r"\[UNK_BYTE_[^\]]+\]", " ", s)
    s = s.replace("\u2581", " ")
    s = re.sub(r"\s+", " ", s)
    # Strip whitespace before closing punctuation from dataset scraping artifacts (e.g. '(Figure )' -> '(Figure)')
    s = re.sub(r"\s+([\)\]\.,;:])", r"\1", s)
    return s.strip().lower()


# ---------------------------------------------------------- units & aggregation

# A `num` feature declares its unit in the schema, but `coerce` only ever checked
# that the value parses as a float. A note reading "serum creatinine was at 7 mg/L"
# therefore landed 7.0 in an mg/dL field - a 10x error that grades an AKI at its
# ceiling, and that no type check can see. The quote is already verified verbatim
# against the note, so it is the one trustworthy place to read the unit the number
# was actually written in.
UNIT_TOKENS = {
    "mg/dl":   r"mg\s*/\s*dl|mg\s+dl\s*(?:\u2212|-)?\s*1|milligrams?\s+per\s+decilit",
    "mg/l":    r"mg\s*/\s*l(?![a-z/])|mg\s+l\s*(?:\u2212|-)?\s*1|milligrams?\s+per\s+lit",
    "umol/l":  r"[\u00b5u]mol\s*/\s*l|micromol",
    "mmol/l":  r"mmol\s*/\s*l|millimol",
    # Not after a letter, and not after a micro sign or Greek mu (both occur in the
    # corpus): the `g/L` inside "µg/L" is micrograms, a millionfold from grams.
    "g/dl":    r"(?<![a-z\u00b5\u03bc])g\s*/\s*dl|(?<![a-z])grams?\s+per\s+decilit",
    "g/l":     r"(?<![a-z\u00b5\u03bc])g\s*/\s*l(?![a-z/])|(?<![a-z])grams?\s+per\s+lit",
    "ug/l":    r"(?<![a-z])(?:[\u00b5\u03bcu]|mc)g\s*/\s*l(?![a-z/])|micrograms?\s+per\s+lit",
    "ng/ml":   r"(?<![a-z])ng\s*/\s*ml\b|nanograms?\s+per\s+millilit",
    "pmol/l":  r"(?<![a-z])pmol\s*/\s*l\b|picomol",
    "miu/ml":  r"(?<![a-z])m(?:iu|u)\s*/\s*ml\b|milli-?international\s+units?\s+per\s+millilit",
    "iu/l":    r"(?<![a-z])(?:iu|u)\s*/\s*l(?![a-z/])|international\s+units?\s+per\s+lit",
    "degf":    r"\u00b0\s*f\b|\u00ba\s*f\b|\bfahrenheit",
    "degc":    r"\u00b0\s*c\b|\u00ba\s*c\b|\bcelsius|\bcentigrade",
    "cm/s":    r"(?<![a-z])cm\s*/\s*s(?:ec)?\b|centimeters?\s+per\s+sec",
    "m/s":     r"(?<![a-z])m\s*/\s*s(?:ec)?\b|meters?\s+per\s+sec",
    "khz":     r"(?<![a-z])khz\b|kilohertz",
    "hz":      r"(?<![a-z])(?<!k)hz\b|(?<!kilo)hertz",
    "mg/g":    r"(?<![a-z])mg\s*/\s*g\b|mg\s+g\s*(?:\u2212|-)?\s*1|milligrams?\s+per\s+gram",
    "ug/mg":   r"(?<![a-z])[\u00b5u]g\s*/\s*mg|mcg\s*/\s*mg|micrograms?\s+per\s+milligram",
    "mg/mmol": r"(?<![a-z])mg\s*/\s*mmol|milligrams?\s+per\s+millimol",
    "cm2":     r"(?<![a-z])cm\s*(?:\^2|2|\u00b2)\b|sq(?:uare)?\s*cm",
    "mm2":     r"(?<![a-z])mm\s*(?:\^2|2|\u00b2)\b|sq(?:uare)?\s*mm",
    "mg_fe_g": r"(?<![a-z])mg\s+fe\s*/\s*g\b|(?<![a-z])mg\s*/\s*(?:g\s+fe|fe\s*g)\b|milligrams?\s+(?:of\s+)?iron\s+per\s+gram",
    "umol/g":  r"(?<![a-z])(?:[\u00b5\u03bcu]|micro)mol\s*/\s*g\b|micromol(?:es)?\s+per\s+gram",
    "mg_fe_100g": r"(?<![a-z])mg(?:\s+fe)?\s*/\s*100\s*g\b",
}

# Generic words like cm, mm, inch must not be in global UNIT_TOKENS to avoid
# collision with unrelated quotes (e.g. "3 cm x 4 cm" for wound_area_cm2).
SCOPED_UNIT_TOKENS = {
    "cm": {
        "cm":   r"(?<![a-z])cm\b|centimeters?",
        "mm":   r"(?<![a-z])mm\b|millimeters?",
        "inch": r"(?<![a-z])(?:inches|inch)\b|(?<=\d)\s*in(?:\.|\b)",
    },
}

# Keyed by the unit the SCHEMA declares, so a factor can never be applied to a
# feature it was not derived for. The umol/L -> mg/dL divisor is creatinine's
# molar mass; mg/dL is declared only by creatinine features, which is what makes
# it safe to sit in this table. Add a unit here only with the same guarantee.
UNIT_CONVERSIONS = {
    "mg/dL": {"mg/dl": lambda v: v,
              "mg/l":  lambda v: v / 10.0,
              "umol/l": lambda v: v / 88.4},
    "g/dL":  {"g/dl": lambda v: v,
              "g/l":  lambda v: v / 10.0},
    "degC":  {"degc": lambda v: v,
              "degf": lambda v: (v - 32.0) * 5.0 / 9.0},
    "cm/s":  {"cm/s": lambda v: v,
              "m/s":  lambda v: v * 100.0},
    "m/s":   {"m/s":  lambda v: v,
              "cm/s": lambda v: v / 100.0},
    "mg/g":  {"mg/g": lambda v: v,
              "ug/mg": lambda v: v,
              "mg/mmol": lambda v: v * 8.84},
    "cm2":   {"cm2": lambda v: v,
              "mm2": lambda v: v / 100.0},
    "cm":    {"cm":   lambda v: v,
              "mm":   lambda v: v / 10.0,
              "inch": lambda v: v * 2.54},
    # ug/L and ng/mL are the same unit for any analyte (1 ug/L = 1 ng/mL), so this
    # factor needs no molar mass. Declared only by ferritin.
    "ng/mL": {"ng/ml": lambda v: v,
              "ug/l":  lambda v: v,
              "pmol/l": lambda v: v / 2.247},
    "mIU/mL": {"miu/ml": lambda v: v,
               "iu/l":   lambda v: v},
    "kHz":   {"khz": lambda v: v,
              "hz":  lambda v: v / 1000.0},
    "mg Fe/g dry weight": {
              "mg_fe_g": lambda v: v,
              "mg/g":    lambda v: v,
              "umol/g":  lambda v: v / 17.9},
}

UNIT_OK, UNIT_CONVERTED, UNIT_AMBIGUOUS = "ok", "converted", "ambiguous"
UNIT_BAD, UNIT_VALUE_MISMATCH = "bad", "value_mismatch"

# A decimal point is a decimal; a comma is a thousands separator only in groups of
# three ("27,469 ng/mL", "1,200 mg/g"). "1,5" stays two numbers rather than being
# read as a European decimal - guessing wrong there is a 10x error either way.
NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|-?\d+(?:\.\d+)?")


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _number_for_unit(q: str, pat: str):
    """The number the unit token is attached to - the nearest one before it, else
    the first after. -> float | None."""
    m = re.search(pat, q, re.I)
    if not m:
        return None
    before = NUMBER.findall(q[:m.start()])
    if before:
        return _number(before[-1])
    after = NUMBER.search(q[m.end():])
    return _number(after.group()) if after else None


def _agrees(a: float, b: float) -> bool:
    """Strict numeric equivalence check with a tight 0.5% relative tolerance.

    Check rationale:
        Clinical grade thresholds frequently sit on fine margins. For instance,
        a body temperature of 38.9 vs. 39.2 °C represents a 0.8% difference, yet
        straddles a Grade 2 vs. Grade 3 boundary; similarly, TRV 2.49 vs 2.50 m/s
        straddles elevated vs normal. Any loose tolerance (e.g. 1-5%) would defeat
        the rubric's decision boundaries.

    Formula:
        abs(a - b) <= max(abs(b), abs(a), 1.0) * 0.005
    """
    return abs(a - b) <= max(abs(b), abs(a), 1.0) * 0.005


# ------------------------------------------------------------------ patient age
#
# The schema holds age in years because every rule that reads it is written in
# years: paediatric below 18, fever's 0-59-day exclusion, the stratified tables.
# Notes write "10-day-old", "18-month-old", "2 years 10-month-old", and the model
# is told to copy a number as the note writes it - so an 18-month-old reached the
# tables as 18 and was graded on the adult strata. The age is read out of the
# quote and naturalised to years here, on the same terms as every other unit: the
# quote is the authority, and the model's number only says which age it meant.

DAYS_PER_YEAR = 365.25
AGE_UNITS = {                        # unit -> (years per unit, order in a compound)
    "years": (1.0, 0),
    "months": (1 / 12, 1),
    "weeks": (7 / DAYS_PER_YEAR, 2),
    "days": (1 / DAYS_PER_YEAR, 3),
}
HYPHEN = r"[-\u2010\u2011\u2012\u2013]"
# Longest spellings first. A bare letter ("2y 3m") counts only when written against
# its number AND inside a compound or an explicit age: "walked 5m" is not an age.
AGE_PART = re.compile(
    rf"(?<![\d.,])(?P<num>\d+(?:\.\d+)?)(?P<gap>\s*{HYPHEN}?\s*)"
    r"(?P<unit>y/o|y\.o\.?|yo|years?|yrs?|m/o|months?|mos?|weeks?|wks?|days?|(?P<letter>[ymwd]))"
    r"(?![a-z/])", re.I)
AGE_SELF_MARKED = {"y/o", "yo", "y.o", "y.o.", "m/o"}    # "year(s) old" in one token
AGE_JOIN = re.compile(r"[\s,]*(?:(?:and|&|\+)\s*)?", re.I)
AGE_AFTER = re.compile(rf"\s*{HYPHEN}?\s*(?:old\b|of\s+age\b)", re.I)
AGE_BEFORE = re.compile(r"(?:\baged?|\bage\s+of|\bat\s+age)\s*[:=]?\s*$", re.I)
AGE_DAY_OF_LIFE = re.compile(r"\b(?:day\s+of\s+life|postnatal\s+day|dol)\s*#?\s*(?P<num>\d+)\b", re.I)
# Gestational, postmenstrual and corrected ages are ages of a pregnancy or of a
# premature infant's development - not how old the patient is.
GEST_AFTER = re.compile(r"\s*['\u2019]?\s*(?:of\s+)?(?:gestation|gestational|ga\b|pma\b|postmenstrual|corrected)", re.I)
GEST_BEFORE = re.compile(r"(?:\bborn\s+at|\bdelivered\s+at|\bgestational\s+age|\bga|\bpma|"
                         r"\bpostmenstrual\s+age|\bcorrected\s+age)\s*(?:of\s*)?[:=]?\s*$", re.I)


@dataclass(frozen=True)
class AgeReading:
    text: str
    years: float
    numbers: tuple[float, ...]     # each number as written, for matching the model's
    marked: bool                   # "-old", "of age", "aged", "day of life"
    gestational: bool


def age_readings(q: str) -> list[AgeReading]:
    """Every age-shaped phrase in the quote, naturalised to years.

    Adjacent parts in descending units are one age - "4 years, 2 months and 10
    days" is 4.194 - in any combination of years, months, weeks and days.
    """
    parts = []
    for m in AGE_PART.finditer(q):
        if m.group("letter") and m.group("gap"):
            continue                                   # "5 m" is metres, not months
        unit = {"y": "years", "m": "months", "w": "weeks", "d": "days"}[m.group("unit")[0].lower()]
        parts.append((m, unit))

    groups: list[list] = []
    for m, unit in parts:
        if groups:
            prev, prev_unit = groups[-1][-1]
            if (AGE_UNITS[unit][1] > AGE_UNITS[prev_unit][1]
                    and AGE_JOIN.fullmatch(q, prev.end(), m.start())):
                groups[-1].append((m, unit))
                continue
        groups.append([(m, unit)])

    out = []
    for g in groups:
        start, end = g[0][0].start(), g[-1][0].end()
        marked = bool(AGE_AFTER.match(q, end)
                      or AGE_BEFORE.search(q[max(0, start - 20):start])
                      or any(m.group("unit").lower() in AGE_SELF_MARKED for m, _ in g))
        if any(m.group("letter") for m, _ in g) and not (marked or len(g) > 1):
            continue
        out.append(AgeReading(
            text=q[start:end],
            years=sum(float(m.group("num")) * AGE_UNITS[u][0] for m, u in g),
            numbers=tuple(float(m.group("num")) for m, _ in g),
            marked=marked,
            gestational=bool(GEST_AFTER.match(q, end)
                             or GEST_BEFORE.search(q[max(0, start - 40):start]))))
    for m in AGE_DAY_OF_LIFE.finditer(q):
        n = float(m.group("num"))
        out.append(AgeReading(m.group(), n / DAYS_PER_YEAR, (n,), True, False))
    return out


def age_guard(value: float, q: str):
    """Reconcile extracted patient age against quote and convert to decimal years.

    Clinical intent:
        The SCOGS schema standardizes all patient ages in years (`patient_age`).
        Every rule reading age depends on years (e.g. pediatric stratification < 18 yr,
        infant fever exclusion for 0-59 days [0.1642 yr]). Clinical notes express
        infant and child ages in days, weeks, months, or compound phrases ("18-month-old",
        "day of life 3", "2 years 4 months"). The quote is the ground truth.

    Checks performed:
        1. Compound Age Parsing: Parses adjacent units in descending order
           (years, months, weeks, days) and computes equivalent decimal years
           using 365.25 days/year and 12 months/year.
        2. Gestational Age Filter Check: Rejects numbers marked with gestational/postmenstrual
           descriptors ("born at 32 weeks", "GA", "PMA", "corrected"). If the extracted
           value matches only a gestational age, returns UNIT_VALUE_MISMATCH.
        3. Age Marker Prioritization Check: Outranks bare durations ("fever for 3 days" is
           not age 3) with explicit age markers ("-old", "aged", "day of life", "DOL").
        4. Value Agreement Check:
           - Matches model value against parsed raw numbers or converted years (`_agrees`).
           - Ambiguity: If multiple distinct candidate ages match, returns UNIT_AMBIGUOUS.
           - Direct match: If value matches converted years, returns (UNIT_OK, truth, None).
           - Raw match: If value copied raw number (e.g. 18 for 18 months), returns
             (UNIT_CONVERTED, 1.5, detail).
           - Mismatch: If value matches no valid age candidate, returns (UNIT_VALUE_MISMATCH, None, detail).
           - No age candidates found: Returns (UNIT_OK, value, None) [silence default].

    Returns:
        tuple[str, float | None, str | None]: (status, reconciled_years, detail)
    """
    readings = age_readings(q)
    ages = [r for r in readings if not r.gestational]
    if not ages:
        if any(_agrees(value, n) for r in readings for n in r.numbers):
            return UNIT_VALUE_MISMATCH, None, f"value {value} is a gestational age, not the patient's"
        return UNIT_OK, value, None                    # nothing in the quote reads as an age
    pool = [r for r in ages if r.marked] or ages
    hits = [r for r in pool
            if _agrees(value, r.years) or any(_agrees(value, n) for n in r.numbers)]
    if not hits:
        return UNIT_VALUE_MISMATCH, None, (
            f"value {value} is none of the quote's ages: {'; '.join(r.text for r in pool)}")
    truths = sorted({round(r.years, 4) for r in hits})
    if len(truths) > 1:
        return UNIT_AMBIGUOUS, value, " / ".join(r.text for r in hits)
    truth = truths[0]
    if _agrees(value, truth):
        return UNIT_OK, truth, None
    return UNIT_CONVERTED, truth, f"{hits[0].text} -> {truth} years"


TLC_VOL_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(liters?|litres?|l(?![a-z/])|ml\b)", re.I)
TLC_PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(%|percent(?:age)?\b|pct\b)", re.I)


def tlc_guard(value: float, q: str):
    """Reconcile Total Lung Capacity (tlc_pct_pred) against its verified quote.

    Clinical intent:
        Total Lung Capacity (Outcome 50: Chronic Restrictive Lung Physiology) is graded
        strictly on TLC percent predicted (% predicted), NOT raw gas volume in Liters.
        Notes routinely state both: 'TLC 3.2 L (78% predicted)'. If the model extracts
        the raw volume 3.2, evaluating without this guard checks '3.2 < 50%' and
        erroneously classifies a patient with mild disease as Grade 4 Life-Threatening.

    Checks performed:
        1. Percentage Value Agreement Check:
           If extracted `value` matches any percentage in the quote, it passes as
           correct (UNIT_OK).
        2. Raw Volume Rescue Check:
           If extracted `value` matches a volume in L or mL:
           - Exactly one percentage in quote: Rescues value to that percentage
             (UNIT_CONVERTED, truth_pct, detail).
           - Multiple percentages in quote: Flags as ambiguous to avoid arbitrary selection
             (UNIT_AMBIGUOUS, value, detail).
           - No percentage in quote (volume-only): Rejects extraction because raw volume
             cannot be graded on percentage thresholds (UNIT_BAD, None, detail).
        3. Fallback Validation:
           - Quote has neither volume nor %: Passes untouched (UNIT_OK, value, None).
           - Quote has % but value does not agree: Rejects as mismatch (UNIT_VALUE_MISMATCH, None, detail).

    Returns:
        tuple[str, float | None, str | None]: (status, reconciled_pct, detail)
    """
    vol_entries = [(float(n), u.strip()) for n, u in TLC_VOL_PATTERN.findall(q)]
    pct_entries = [(float(n), u.strip()) for n, u in TLC_PCT_PATTERN.findall(q)]

    for p, _ in pct_entries:
        if _agrees(value, p):
            return UNIT_OK, p, None

    for v, u in vol_entries:
        if _agrees(value, v):
            unit_str = "L" if u.lower() == "l" else ("mL" if u.lower() == "ml" else u)
            if len(pct_entries) == 1:
                truth = pct_entries[0][0]
                return UNIT_CONVERTED, truth, f"{v} {unit_str} -> {truth}% predicted"
            if len(pct_entries) > 1:
                return UNIT_AMBIGUOUS, value, "/".join(str(p) for p, _ in pct_entries)
            return UNIT_BAD, None, f"{v} {unit_str} is a lung volume, not percent predicted"

    if not vol_entries and not pct_entries:
        return UNIT_OK, value, None
    if pct_entries:
        return UNIT_VALUE_MISMATCH, None, f"value {value} does not match quote percentage ({pct_entries[0][0]}%)"
    return UNIT_BAD, None, "quote contains lung volume in liters, not percent predicted"


def unit_guard(name: str, value: float, quote: str):
    """Reconcile an extracted numeric value against its own verified quote.

    Clinical intent:
        A numeric feature declares its expected unit in the schema (e.g. mg/dL, m/s, °C).
        The verified quote is the clinical source of truth, not the model's value: the model
        may extract a number without its unit ("creatinine 7 mg/L" extracted as 7.0 mg/dL,
        a 10x dosing/grading error), or botch mental arithmetic ("102.6 °F" -> 38.9 °C
        instead of 39.2 °C, crossing a grade boundary).

    Checks performed:
        1. Schema Unit & Delegation Check:
           - If feature has no declared unit or no conversion family, returns (UNIT_OK, value, None).
           - If declared unit is 'years', delegates to `age_guard()`.
           - If feature is 'tlc_pct_pred', delegates to `tlc_guard()`.
        2. Unit Token Detection Check:
           - Scans quote for declared unit tokens and scoped tokens (preventing collisions
             like "cm" on wound area).
           - No unit found in quote: Returns (UNIT_OK, value, None) [silence default].
           - Multiple distinct unit tokens: Returns (UNIT_AMBIGUOUS, value, detail).
           - Unit token not convertible to declared unit: Returns (UNIT_BAD, None, detail).
        3. Anchor Number Association Check:
           - Locates number attached to the unit token via `_number_for_unit()`.
           - If no number is attached to the unit token: Returns (UNIT_OK, value, None).
        4. Value & Conversion Verification Check:
           - Converts raw number to schema's canonical unit using `UNIT_CONVERSIONS`.
           - If value agrees with converted truth (`_agrees`): Returns (UNIT_OK, truth, None).
           - If value agrees with raw number: Converts and returns (UNIT_CONVERTED, truth, detail).
           - If value agrees with neither: Returns (UNIT_VALUE_MISMATCH, None, detail).

    Returns:
        tuple[str, float | None, str | None]: (status, reconciled_value, detail)
    """
    declared = FEATURES[name].get("unit")
    if declared == "years":
        return age_guard(value, normalize(quote))
    if name == "tlc_pct_pred":
        return tlc_guard(value, normalize(quote))
    table = UNIT_CONVERSIONS.get(declared)
    if not table:
        return UNIT_OK, value, None       # no unit declared, or no family for it
    q = normalize(quote)
    tokens = {**UNIT_TOKENS, **SCOPED_UNIT_TOKENS.get(declared, {})}
    found = [u for u, pat in tokens.items() if re.search(pat, q, re.I)]
    if not found:
        return UNIT_OK, value, None
    if len(found) > 1:
        return UNIT_AMBIGUOUS, value, "/".join(sorted(found))
    src = found[0]
    if src not in table:
        return UNIT_BAD, None, f"{src} is not convertible to {declared}"
    raw = _number_for_unit(q, tokens[src])
    if raw is None:
        return UNIT_OK, value, None       # a unit, but no number to anchor it to
    truth = round(table[src](raw), 4)
    if _agrees(value, truth):
        return UNIT_OK, truth, None       # already right; canonicalise the rounding
    if _agrees(value, raw):
        # the number was copied across without the unit coming with it
        return UNIT_CONVERTED, truth, f"{raw} {src} -> {truth} {declared}"
    return UNIT_VALUE_MISMATCH, None, (
        f"value {value} is neither the quote's {raw} {src} nor its {truth} {declared}")


# How to collapse several verified proposals for one feature into the single value
# the decision tables take. The schema already answers this: `ord` values are listed
# low-to-high and compare by rank, and the definitions say which end wins ("Highest
# level of care this event actually reached", "Maximum respiratory support given").
# Reading the policy off the definition keeps this file from re-stating the contract.
AGG_MAX = re.compile(r"\bhighest\b|\bmaximum\b|\bmax\b|\bpeak\b|\bworst\b|\bmost intensive\b", re.I)
AGG_MIN = re.compile(r"\blowest\b|\bminimum\b|\bnadir\b", re.I)


def reduce_policy(name: str) -> str | None:
    """Check feature definition in schema for an aggregation rule ('max' | 'min').

    Checks performed:
        1. Categorical Feature Check: Returns None (unordered; no defensible winner).
        2. Peak / Worst Search: Searches feature definition text for peak indicators
           ('highest', 'maximum', 'max', 'peak', 'worst', 'most intensive').
           If found, returns 'max'.
        3. Nadir / Minimum Search: Searches feature definition text for nadir indicators
           ('lowest', 'minimum', 'nadir').
           If found, returns 'min'.
        4. Default: If no policy keywords exist, returns None.
    """
    if FEATURES[name]["type"] == "cat":
        return None                       # unordered: no defensible winner
    d = FEATURES[name]["definition"]
    if AGG_MAX.search(d):
        return "max"
    if AGG_MIN.search(d):
        return "min"
    return None


def _rank(name: str, v):
    spec = FEATURES[name]
    if spec["type"] == "ord":
        return (spec["values"] or []).index(v)
    return v


def reconcile(name: str, values: list):
    """Check and reconcile multiple quoted candidate values for a single feature.

    Clinical intent:
        In clinical notes spanning multiple years or ICU stays, notes routinely mention
        multiple values for the same lab (e.g. 5 creatinines, or a donor's lab value).
        Arbitrarily taking the first or last value by emission order silently corrupts grading.

    Checks performed:
        1. Uniqueness Check: Deduplicates candidate values. If only one unique value
           exists, returns (unique_value, None).
        2. Aggregation Policy Check: Queries `reduce_policy(name)`:
           - If 'max': Returns the maximum value (using numeric value or ordinal rank).
           - If 'min': Returns the minimum value.
        3. Unresolved Conflict Check: If multiple distinct values exist and the feature
           has no defined aggregation policy, the values are NOT guessed. They are
           withheld from grading and flagged as an unresolvable conflict:
           returns (None, list_of_conflicting_values).

    Returns:
        tuple[Any | None, list | None]: (resolved_value, conflicting_values_list)
    """
    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    if len(uniq) == 1:
        return uniq[0], None
    policy = reduce_policy(name)
    if policy is None:
        return None, uniq
    pick = max if policy == "max" else min
    try:
        return pick(uniq, key=lambda v: _rank(name, v)), None
    except (ValueError, TypeError):
        return None, uniq


@dataclass
class Tally:
    proposed: int = 0
    quote_ok: int = 0
    quote_missing: int = 0      # no quote field at all
    quote_unfound: int = 0      # quote not verbatim in the note -> hallucination
    value_bad: int = 0          # value not of the declared type / not a declared enum
    unknown_feature: int = 0
    accepted: int = 0
    bad_json: int = 0
    tokenizer_artifacts: int = 0   # quotes carrying corrupt-GGUF byte tokens
    unit_converted: int = 0        # value rewritten into the schema's unit
    unit_ambiguous: int = 0        # quote carried several units -> left alone
    unit_mismatch: int = 0         # quote's unit cannot reach the declared one
    quote_value_mismatch: int = 0  # number is not the one its own quote carries
    value_conflicts: int = 0       # several verified values, no aggregation rule
    # `present` gates every other result - `grade()` returns absent whenever it is
    # false - and stage 0 is the only field in the reply carrying no quote at all.
    # From stage 2a it is asked for and these count what comes back.
    present_true: int = 0          # pairs the model called present
    present_quoted: int = 0        # ... of those, with a quote that is in the note
    present_quote_unfound: int = 0 # ... with a quote that is not
    present_unquoted: int = 0      # ... with no quote offered
    presence_contradicted: int = 0 # criteria met but model said not present
    content_retries: int = 0       # calls redone because the reply was unusable
    unusable_replies: int = 0      # ... and still unusable when the tries ran out
    feedback_retries: int = 0      # calls re-prompted once with specific error feedback
    per_feature: Counter = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_sec: float = 0.0

    def report(self):
        p = self.proposed or 1
        # A null-quote placeholder is a prompt-compliance failure, not a grounding
        # result. Reporting one grounding number lets the denominator be chosen to
        # flatter the model, so both are always emitted together.
        quoted = self.quote_ok + self.quote_unfound
        q = quoted or 1
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "quoted": quoted,
            "quote_verified": self.quote_ok,
            "quote_unfound": self.quote_unfound,
            "null_placeholder": self.quote_missing,
            "null_placeholder_pct": round(100 * self.quote_missing / p, 1),
            "quote_verified_pct": round(100 * self.quote_ok / p, 1),
            "quote_verified_pct_of_quoted": round(100 * self.quote_ok / q, 1),
            "hallucinated_quote_pct": round(100 * self.quote_unfound / p, 1),
            "hallucinated_pct_of_quoted": round(100 * self.quote_unfound / q, 1),
            "tokenizer_artifacts": self.tokenizer_artifacts,
            "unit_converted": self.unit_converted,
            "unit_ambiguous": self.unit_ambiguous,
            "unit_mismatch": self.unit_mismatch,
            "quote_value_mismatch": self.quote_value_mismatch,
            "value_conflicts": self.value_conflicts,
            "present_true": self.present_true,
            "present_quoted": self.present_quoted,
            "present_quote_unfound": self.present_quote_unfound,
            "present_unquoted": self.present_unquoted,
            "presence_contradicted": self.presence_contradicted,
            "present_quoted_pct": round(100 * self.present_quoted / (self.present_true or 1), 1),
            "content_retries": self.content_retries,
            "unusable_replies": self.unusable_replies,
            "feedback_retries": self.feedback_retries,
            "missing_quote": self.quote_missing,
            "invalid_value": self.value_bad,
            "unknown_feature": self.unknown_feature,
            "unparseable_replies": self.bad_json,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_clock_sec": round(self.wall_clock_sec, 2),
        }


def coerce(name: str, value):
    """Enforce declared schema type and enum constraints without type widening.

    Checks performed:
        1. Feature Registry Check: Verifies `name` is declared in `FEATURES`.
           If undeclared, returns (False, None).
        2. Boolean Type Check ('bool'):
           - Accepts bool instances directly.
           - Accepts string booleans ('true', 'yes' -> True; 'false', 'no' -> False).
           - Rejects all other types/strings (returns False, None).
        3. Numeric Type Check ('num'):
           - Parses value as float.
           - Rejects values failing float conversion (returns False, None).
        4. Categorical & Ordinal Enum Check ('cat', 'ord'):
           - Lowercases and strips value string.
           - Checks value against allowed enum values whitelist (`spec['values']`).
           - Rejects values not in whitelist (returns False, None).

    Returns:
        tuple[bool, Any]: (True, coerced_value) if all checks pass, else (False, None).
    """
    spec = FEATURES.get(name)
    if spec is None:
        return False, None
    t = spec["type"]
    if t == "bool":
        if isinstance(value, bool):
            return True, value
        if str(value).lower() in {"true", "yes"}:
            return True, True
        if str(value).lower() in {"false", "no"}:
            return True, False
        return False, None
    if t == "num":
        try:
            return True, float(value)
        except (TypeError, ValueError):
            return False, None
    v = str(value).strip().lower()
    return (True, v) if v in (spec["values"] or []) else (False, None)


def precheck(reply: str, note: str, allowed_features: set[str] | None = None) -> list[str]:
    """Side-effect-free pre-validation inspection of an extraction reply against note text.

    Clinical intent:
        Performs a non-destructive dry run of all verification checks without altering
        `Tally` counters or pipeline state. When `--feedback-retry` is enabled, this
        identifies precise extraction flaws and formats targeted feedback so the model
        can correct its own output in a second turn.

    Checks performed:
        1. JSON Syntax Check: Validates that `reply` is valid JSON and parses as a dictionary.
        2. Schema Registration Check: Confirms every finding's `feature` is declared in `FEATURES`.
        3. Outcome Scope Check: If `allowed_features` is provided, ensures the feature belongs to this outcome.
        4. Quote Grounding Check:
           - Verifies `quote` string is non-empty.
           - Normalizes quote and note; checks quote appears verbatim in `note`.
        5. Type Coercion Check: Runs `coerce(name, value)` to verify value type and enum constraints.
        6. Unit & Number Guard Check: For numeric features, runs `unit_guard()`:
           - Rejects invalid/unconvertible units (UNIT_BAD).
           - Rejects numbers that disagree with quote text (UNIT_VALUE_MISMATCH).
           - Flags ambiguous units (UNIT_AMBIGUOUS).
        7. Presence Quote Grounding Check: If `present` is True, verifies that `present_quote`
           appears verbatim in the note text.

    Returns:
        list[str]: List of human-readable issues describing failed checks. Empty list if all checks pass.
    """
    try:
        data = json.loads(reply)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []

    hay = normalize(note)
    issues: list[str] = []

    for f in data.get("findings", []) or []:
        if not isinstance(f, dict):
            continue
        name = f.get("feature")
        if not name or name not in FEATURES:
            issues.append(f"Unknown feature '{name}'. Only report features listed in the schema.")
            continue
        if allowed_features is not None and name not in allowed_features:
            issues.append(f"Feature '{name}' is not in the schema for this outcome.")
            continue
        quote = f.get("quote")
        if not quote or not isinstance(quote, str) or not quote.strip():
            issues.append(f"Finding for '{name}' is missing a quote from the note.")
            continue
        if normalize(quote) not in hay:
            issues.append(f"Finding for '{name}': quote {quote!r} was not found verbatim in the note.")
            continue
        ok, val = coerce(name, f.get("value"))
        if not ok:
            issues.append(f"Finding for '{name}': value {f.get('value')!r} is not a valid {FEATURES[name]['type']}.")
            continue
        if FEATURES[name]["type"] == "num":
            status, converted, detail = unit_guard(name, val, quote)
            if status == UNIT_BAD:
                issues.append(f"Finding for '{name}': {detail or 'unit is not convertible'}.")
            elif status == UNIT_VALUE_MISMATCH:
                issues.append(f"Finding for '{name}': {detail or f'value {val} does not match quote {quote!r}'}.")
            elif status == UNIT_AMBIGUOUS:
                issues.append(f"Finding for '{name}': unit in quote {quote!r} is ambiguous ({detail}).")

    if data.get("present") is True:
        pq = data.get("present_quote")
        if pq and isinstance(pq, str) and normalize(pq) not in hay:
            issues.append(f"present_quote {pq!r} was not found verbatim in the note.")

    return issues


def verify(reply: str, note: str, tally: Tally) -> tuple[dict, bool | None, dict]:
    """Execute the core extraction verification and grounding pipeline against a clinical note.

    Clinical intent:
        Acts as the primary quality gate separating raw model generation from deterministic
        grading. Enforces strict evidence grounding, schema conformance, unit safety, and
        conflict resolution. All metrics (accepted, hallucinated quotes, unit mismatches)
        are tallied synchronously.

    Checks performed:
        1. JSON Parse Check: Validates JSON syntax. Bad JSON increments `tally.bad_json`.
        2. Tokenizer Artifact Check: Scans quotes for SentencePiece corruption (`[UNK_BYTE_...]`).
        3. Feature Whitelist Check: Validates feature name in `FEATURES`.
        4. Verbatim Quote Grounding Check:
           - Verifies quote is non-empty (`quote_missing`).
           - Normalizes text and checks that quote exists verbatim in note (`quote_ok` vs `quote_unfound`).
           - Quotes absent from the note are rejected as hallucinations and excluded from accepted findings.
        5. Type Coercion Check: Validates type and enum constraints via `coerce()`.
           Non-compliant values increment `value_bad` and are rejected.
        6. Unit Guard Verification Check:
           - Reconciles numeric features against quote units via `unit_guard()`.
           - Detects and counts `unit_mismatch`, `quote_value_mismatch`, `unit_ambiguous`, `unit_converted`.
        7. Finding Acceptance: Findings passing quote grounding, type coercion, and unit checks
           are admitted to `accepted` findings.
        8. Presence Call & Quote Check: Evaluates `present` boolean; verifies verbatim grounding
           of `present_quote` in note (`present_quoted` vs `present_quote_unfound` vs `present_unquoted`).
        9. Value Conflict Reconciliation Check: Groups accepted findings by feature and calls
           `reconcile()`:
           - If a reduction rule exists, collapses by rule (`max` / `min`).
            - If multiple conflicting values exist without a reduction policy, withholds
              the feature from grading and flags it in `conflicts` (`tally.value_conflicts`).

    Returns:
        tuple[dict, bool | None, dict]:
            - features (dict): Reconciled feature values ready for decision tables.
            - present (bool | None): Model presence call.
            - detail (dict): Telemetry carrying 'accepted' findings, 'conflicts', 'present_quote', 'evidence'.
    """
    empty = {"accepted": [], "conflicts": {}, "present_quote": None, "evidence": None}
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        tally.bad_json += 1
        return {}, None, empty
    hay = normalize(note)
    cand: dict[str, list] = {}
    accepted: list[dict] = []
    for f in data.get("findings", []) or []:
        tally.proposed += 1
        name = f.get("feature")
        if name not in FEATURES:
            tally.unknown_feature += 1
            continue
        quote = f.get("quote")
        if not quote:
            tally.quote_missing += 1
            continue
        if ARTIFACT.search(quote):
            tally.tokenizer_artifacts += 1
        if normalize(quote) not in hay:
            tally.quote_unfound += 1          # the §2 rule doing its job
            continue
        tally.quote_ok += 1
        ok, val = coerce(name, f.get("value"))
        if not ok:
            tally.value_bad += 1
            continue
        unit_detail = None
        if FEATURES[name]["type"] == "num":
            status, converted, unit_detail = unit_guard(name, val, quote)
            if status == UNIT_BAD:
                tally.unit_mismatch += 1
                continue
            if status == UNIT_VALUE_MISMATCH:
                tally.quote_value_mismatch += 1
                continue
            if status == UNIT_AMBIGUOUS:
                tally.unit_ambiguous += 1
            elif status == UNIT_CONVERTED:
                tally.unit_converted += 1
            val = converted
        cand.setdefault(name, []).append(val)
        accepted.append({"feature": name, "value": val, "quote": quote,
                         "unit": unit_detail})
        tally.accepted += 1
        tally.per_feature[name] += 1

    present = data.get("present")
    if present is True:
        tally.present_true += 1
        pq = data.get("present_quote")
        if not pq or not isinstance(pq, str):
            tally.present_unquoted += 1
        elif normalize(pq) not in hay:
            tally.present_quote_unfound += 1
        else:
            tally.present_quoted += 1

    out, conflicts = {}, {}
    for name, vals in cand.items():
        picked, clash = reconcile(name, vals)
        if clash is not None:
            conflicts[name] = clash
            tally.value_conflicts += 1
            continue
        out[name] = picked
    return out, present, {"accepted": accepted, "conflicts": conflicts,
                          "present_quote": data.get("present_quote"),
                          "evidence": data.get("evidence")}
