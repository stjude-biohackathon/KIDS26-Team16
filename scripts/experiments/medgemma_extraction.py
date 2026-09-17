"""Extract SCOGS features using full 16-bit MedGemma 27B through Ollama.

Grades come from deterministic tables, never the model. Proposed findings must
carry a quote found in the note. Quote grounding is not clinical correctness;
review the exported worksheets before reporting precision.

Usage examples:
    python scripts/experiments/medgemma_extraction.py --check-model
    python scripts/experiments/medgemma_extraction.py --cohort scd_primary --notes 20 --repeat 2 --out results/full.json
    python scripts/experiments/medgemma_extraction.py --cohort scd_primary --prompt-stage 3 --notes 20 --out results/stage3.json
    python scripts/experiments/medgemma_extraction.py --backend mock --notes 2
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Ensure UTF-8 output on Windows consoles (prevents charmap / cp1252 encode errors on °, µ, ×, etc.)
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.ollama_backend import (
    DEFAULT_HOST, DEFAULT_MODEL, call_ollama, preflight,
)
from experiments.verification import Tally, precheck, verify
from experiments.cohort import load_notes, select_notes
from experiments import run_output
from experiments.run_output import Repeats, ServedModel
from scogs.definitions import presence_brief
from scogs.features import FEATURES
from scogs.predicates import parse
from scogs.tables import TABLES


# Default 14 focus outcomes: Chronic Pain (10), CD (11), TCD Elevation (12),
# Stroke (15), Retinopathy (17), CKD (21), Priapism (24), Acute Pain (28),
# SS (29), AVN (39), Leg Ulcer (40), Depression (47), ACS (48), Asthma (49).
DEFAULT_OUTCOMES = "10,11,12,15,17,21,24,28,29,39,40,47,48,49"

# ------------------------------------------------------------------- prompting
#
# Every prompt and decoding change sits behind `--prompt-stage`, and stage "0"
# reproduces the prompt this file shipped with, character for character. That is
# not deference to the old prompt: an ablation only reads if one arm is the
# untouched baseline, and the baseline has to come out of the same binary as the
# other arms or it is measuring the binary too. The stages are ordered and
# CUMULATIVE - each is the one before it plus one group of changes - so the arms
# nest and a run is described by a single token.
#
#   0    the prompt as it was
#   1    mechanical: constrained decoding, field order, rule grouping, framing
#   2a   presence: the rubric's own definition, a quote behind `present`
#   2b   an `evidence` field, generated first (chain-of-thought inside the JSON)
#   3    precision: episode scope, negation, ladders, unit wording
#
# Stages 1 and 2 move counters this harness already prints. Stage 3 is expected
# to move `accepted` DOWN - it exists to stop findings that should never have
# been proposed - so it cannot be ranked on the automatic counters and is the
# stage the hand-check sheets have to be filled against.

STAGES = ("0", "1", "2a", "2b", "3")

# What a run uses when --prompt-stage is not given. Stage 3 incorporates precision
# rules (negation handling, episode scoping, ordinal cues, and unit hardening).
# Any rung in STAGES ("0", "1", "2a", "2b", "3") can still be chosen via --prompt-stage.
# Stage "0" stays the frozen baseline arm, and prompt-building functions still default to it.
DEFAULT_PROMPT_STAGE = "3"


@dataclass(frozen=True)
class Stage:
    """Which groups of changes are live. Build with `stage()`, never by hand."""
    name: str = "0"
    schema: bool = False        # 1  constrained decoding from a per-outcome JSON schema
    structure: bool = False     # 1  field order, rule grouping, framing, delimiters
    flat_penalty: bool = False  # 1  repeat_penalty 1.1 -> 1.0
    retry: bool = False         # 1  retry on invalid CONTENT, not just transport
    presence: bool = False      # 2a rubric definition + criteria, quoted `present`
    death_both: bool = False    # 2a a death from another cause is also `false`
    cot: bool = False           # 2b an `evidence` field, generated first
    precision: bool = False     # 3  episode scope, negation, ladders, unit wording
    note_first: bool = False    # independent of the ladder; see --note-first
    patient_context: bool = False  # independent of the ladder; see --patient-context
    feedback_retry: bool = False   # independent of the ladder; see --feedback-retry

    @property
    def rank(self) -> int:
        return STAGES.index(self.name)


def stage(name: str, note_first: bool = False, patient_context: bool = False,
          feedback_retry: bool = False) -> Stage:
    if name not in STAGES:
        raise ValueError(f"unknown prompt stage {name!r}; expected one of {', '.join(STAGES)}")
    i = STAGES.index(name)
    return Stage(name=name, note_first=note_first, patient_context=patient_context,
                 feedback_retry=feedback_retry,
                 schema=i >= 1, structure=i >= 1, flat_penalty=i >= 1, retry=i >= 1,
                 presence=i >= 2, death_both=i >= 2,
                 cot=i >= 3,
                 precision=i >= 4)


STAGE0 = Stage()


# How an ordinal ladder actually appears in a case report. The schema's own
# `values` are identifiers - `niv_bipap_cpap` - and the note says "started on
# BiPAP overnight". Nothing in the prompt bridged that, on features the grade
# turns on: outcome 48's grades 2, 3 and 4 separate on `resp_support` and
# `transfusion_type` alone. These are prompting aids, not schema, which is why
# they live here and not in `features.py`.
ORD_CUES = {
    "resp_support":
        "room_air = no supplemental oxygen; low_flow_o2 = nasal cannula, face mask, "
        "simple mask, up to about 6 L/min; high_flow = high-flow nasal cannula, HFNC, "
        "Optiflow, Airvo, venturi mask at high flow; niv_bipap_cpap = BiPAP, CPAP, "
        "non-invasive ventilation; invasive_ventilation = intubation, mechanical "
        "ventilation, ventilated via tracheostomy.",
    "transfusion_type":
        "none = no red cell transfusion; simple = top-up, packed red cells, straight "
        "transfusion; exchange = exchange transfusion, red cell exchange, "
        "erythrocytapheresis, manual or automated exchange. "
        "Plasma exchange (plasmapheresis) is NOT a red cell transfusion - do not report it here.",
    "care_setting":
        "home = managed without a facility visit; clinic_or_day_hospital = outpatient "
        "clinic, day unit, infusion centre; ed_treat_release = seen in the emergency "
        "department and discharged; inpatient = admitted to a ward; icu = intensive "
        "care, critical care, HDU.",
}


def feature_brief(name: str, outcome: str, st: Stage = STAGE0) -> str:
    spec = FEATURES[name]
    bits = [f'"{name}" ({spec["type"]}']
    if spec["values"]:
        bits.append(f', one of: {", ".join(spec["values"])}')
    if spec["unit"]:
        # Stage 0 renders the schema's unit as though it were an instruction
        # ("in mg/dL") three lines above a rule telling the model not to convert.
        # The unit is the pipeline's target, not the model's job: `unit_guard`
        # reaches it from the quote.
        bits.append(f', in {spec["unit"]}' if not st.precision
                    else f'; the pipeline converts to {spec["unit"]}')
    bits.append(")")
    line = "".join(bits) + " - " + spec["definition"]
    clarifier = spec["per_outcome"].get(outcome)
    if clarifier:
        line += f" For this outcome specifically: {clarifier}"
    if st.precision and name in ORD_CUES:
        line += f" In notes this reads as: {ORD_CUES[name]}"
    return line


def expand_derived(names: set[str]) -> set[str]:
    """Replace derived features with the features they are computed from.

    A table row says `life_support`, but nothing in a note says that - it is
    resolved from ventilation, vasopressors, renal replacement and other life
    support. Asking the model for the derived name yields nothing, and every
    outcome that uses it would silently cap one grade below its true ceiling.
    """
    out, seen = set(), set()
    stack = list(names)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        spec = FEATURES[n]
        if spec.get("computed_from"):
            out.add(n)
            for inp in spec["computed_from"]:
                if inp in FEATURES and inp not in seen:
                    stack.append(inp)
            continue
        if not spec["derived"]:
            out.add(n)
            continue
        inputs = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", spec["derived"])
                  if w in FEATURES and w != n]
        if inputs:
            stack += inputs
        else:
            out.add(n)
    return out


def grade_weight(outcome: str) -> dict[str, tuple[int, int]]:
    """-> feature -> (highest grade it can trigger, how many rows it appears in).

    Stage 0 lists features alphabetically. For outcome 48 that opens with
    `acs_other_support` - one row, negated, grade 1 - and pushes the three
    features grades 2, 3 and 4 actually separate on into the middle of the list.
    A derived feature hands its weight to the features it expands into, because
    those are the names the model is asked for.
    """
    table = TABLES[outcome]
    seen: dict[str, list[int]] = {}
    for g, pred in table.all_rows():
        for n in parse(pred).names():
            for leaf in expand_derived({n}):
                seen.setdefault(leaf, []).append(g)
    if table.on:
        top = max(table.grades(), default=0)
        for leaf in expand_derived({table.on}):
            seen.setdefault(leaf, []).append(top)
    return {n: (max(gs), len(gs)) for n, gs in seen.items()}


def order_features(names: list[str], outcome: str, st: Stage = STAGE0) -> list[str]:
    """Highest grade first, then most rows, then name. Alphabetical at stage 0."""
    if not st.structure:
        return sorted(names)
    w = grade_weight(outcome)
    return sorted(names, key=lambda n: (-w.get(n, (0, 0))[0], -w.get(n, (0, 0))[1], n))


# ------------------------------------------------------- constrained decoding

def value_schema(name: str) -> dict:
    spec = FEATURES[name]
    if spec["type"] == "bool":
        return {"type": "boolean"}
    if spec["type"] == "num":
        return {"type": "number"}
    return {"enum": list(spec["values"] or [])}


def reply_schema(needed: list[str], st: Stage) -> dict:
    """The JSON Schema handed to the decoder.

    Stage 0 sends the bare string "json", which buys a syntactically valid object
    and nothing else. `unknown_feature`, `value_bad` and the null-quote
    placeholders were all still reachable, and all three were being COUNTED as
    model failures when the decoder could have made them unreachable. Here the
    feature name is an enum of what this outcome grades on, each value is typed
    from the schema, and a quote is a non-empty string.
    """
    finding = [{
        "type": "object",
        "properties": {"feature": {"const": n},
                       "value": value_schema(n),
                       "quote": {"type": "string", "minLength": 1}},
        "required": ["feature", "value", "quote"],
        "additionalProperties": False,
    } for n in needed]

    props: dict[str, dict] = {}
    required: list[str] = []
    if st.cot:
        props["evidence"] = {"type": "string"}
        required.append("evidence")
    # `findings` ahead of `present`, and the schema has to agree with the prompt:
    # for a decoder that honours property order this IS the generation order.
    props["findings"] = {"type": "array", "items": {"anyOf": finding}}
    props["present"] = {"type": "boolean"}
    required += ["findings", "present"]
    if st.presence:
        # Deliberately NOT required. A note that does not evidence an outcome
        # frequently contains no sentence saying so, and a schema that demands a
        # quote there buys a fabricated one instead of an honest omission. The
        # prose asks for it when `present` is true; `present_unquoted` counts
        # what comes back without one.
        props["present_quote"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


CONTEXT_FEATURES = ("patient_age", "patient_sex")


def context_schema(st: Stage = STAGE0) -> dict:
    finding = [{
        "type": "object",
        "properties": {"feature": {"const": n},
                       "value": value_schema(n),
                       "quote": {"type": "string", "minLength": 1}},
        "required": ["feature", "value", "quote"],
        "additionalProperties": False,
    } for n in CONTEXT_FEATURES]

    props: dict[str, dict] = {}
    required: list[str] = []
    if st.cot:
        props["evidence"] = {"type": "string"}
        required.append("evidence")
    props["findings"] = {"type": "array", "items": {"anyOf": finding}}
    required.append("findings")
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def reply_is_usable(reply: str) -> bool:
    """Is this reply worth keeping, or should the call be retried?

    Stage 0 retries only on a transport exception, so an HTTP 200 carrying a
    findings list of null placeholders was accepted first time, every time. This
    is the content-level gate: parseable, and no finding that the verifier would
    throw away before it ever reached the note.
    """
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(data, dict) or "findings" not in data:
        return False
    findings = data.get("findings")
    if not isinstance(findings, list):
        return False
    for f in findings:
        if not isinstance(f, dict):
            return False
        if not f.get("feature") or not f.get("quote"):
            return False
        if f.get("value") is None:
            return False
    return True


# ------------------------------------------------------------ prompt assembly

def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + ln if ln.strip() else ln for ln in text.split("\n"))


def _prompt_v0(note: str, outcome: str, table, needed: list[str], lines: str) -> str:
    """The stage 0 prompt, unchanged. Do not edit - it is the baseline arm."""
    survival = ""
    if "death_attributed" in needed:
        survival = (
            '\n- Absence of an event can be explicitly supported. If the note shows the patient '
            'survived\n  this episode - discharge, follow-up, recovery, ongoing care - report '
            '"death_attributed": false\n  and quote that text. Leaving it unknown is not neutral: '
            'no outcome can be graded\n  while it is unknown whether the patient died.')

    return f"""You are reading a clinical note and extracting specific findings. \
You are NOT assigning a severity grade - that is done separately by a rule engine.

Health outcome under consideration: {outcome} - {table.name}

Extract only these findings:
{lines}

Rules:
- Report a finding ONLY if the note explicitly supports it. Omit anything the note does not address.
  Omitting is correct and expected; guessing is not.{survival}
- Report each feature at most ONCE. Where its definition names an extreme ("highest",
  "maximum", "most intensive"), report that one value, not every occurrence.
- For a number, copy it as the note states it, in the note's OWN units, and let the quote carry
  both the number and its unit. Do not convert - the pipeline does that.
- Every finding MUST include "quote": a concise sentence or phrase copied EXACTLY from the note,
  character for character. If you cannot copy an exact supporting quote, do not report the finding.
- NEVER emit a finding whose value or quote is null. Leave it out of the list entirely. An empty
  list is a valid and often correct answer.
- "present" is whether the note evidences this health outcome at all.

Reply with JSON only, no prose:
{{"present": true|false,
  "findings": [{{"feature": "<name>", "value": <value>, "quote": "<exact text from the note>"}}]}}

NOTE:
\"\"\"{note}\"\"\""""


def _rules_v1(needed: list[str], st: Stage) -> str:
    """The rules, grouped. Stage 0 is one flat list of seven bullets that mixes a
    hard contract (the quote) with a preference (report once) and a definition
    (`present`), giving the reader nothing to tell them apart."""
    g = ["## How to report", (
        "Grounding - a contract, not a preference:\n"
        "- Every finding MUST carry a quote copied EXACTLY from the note, character for "
        "character.\n"
        "- A finding whose quote is not found in the note is discarded, not down-weighted. "
        "Reporting\n  nothing costs you less than reporting something you cannot quote.\n"
        "- Never emit a finding whose value or quote is null. Leave it out of the list "
        "entirely.")]

    if st.precision:
        g.append(
            "Quote the shortest continuous run of the note's own words that proves the value. "
            "Never join text from different sentences, never write '...', "
            "never fix a typo,\nnever paraphrase. Copy one continuous span or report nothing.")

    g.append(
        "What to report:\n"
        "- Report a finding ONLY if the note explicitly supports it. Omit anything the note "
        "does not\n  address. Omitting is correct and expected; guessing is not. An empty "
        "findings list is a\n  valid and often correct answer.\n"
        '- Report each feature at most ONCE. Where its definition names an extreme ("highest",\n'
        '  "maximum", "most intensive"), report that one value, not every occurrence.')

    if st.precision:
        g.append(
            "What does NOT support a finding:\n"
            '- Negation. "remained afebrile", "did not require intubation", "no transfusion was\n'
            '  given" are evidence the finding is FALSE or absent - never evidence it happened.\n'
            '- Things considered but not done. "exchange transfusion was considered", "planned\n'
            '  for ICU admission", "would have required ventilation" are not the event.\n'
            "- Anything belonging to another person, or to a different admission.")

    if "death_attributed" in needed:
        d = ("Death:\n"
             "- Absence of an event can be explicitly supported. If the note shows the patient "
             "survived\n  this episode - discharge, follow-up, recovery, ongoing care - report "
             '"death_attributed": false\n  and quote that text. Leaving it unknown is not '
             "neutral: no outcome can be graded while\n  it is unknown whether the patient died.")
        if st.death_both:
            d += ('\n- If the patient died but the note attributes the death to a different '
                  'cause, that is\n  ALSO "death_attributed": false - quote the cause the note '
                  "gives. Only a death this note\n  ties to THIS outcome is true.")
        g.append(d)

    if any(FEATURES[n]["type"] == "num" for n in needed):
        u = ("Numbers:\n"
             "- Copy the number as the note states it, in the note's OWN units. Do not convert "
             "- the\n  pipeline does that.")
        if st.precision:
            u += ("\n- The quote for a number MUST contain both the number and its unit. Without "
                  "the unit the\n  pipeline cannot tell which one you meant, and the finding is "
                  "discarded.")
        g.append(u)

    return "\n\n".join(g)


def _output_v1(st: Stage) -> str:
    fields = []
    if st.cot:
        fields.append(
            '  "evidence": "<2-4 sentences. What the note says that bears on this outcome, and\n'
            '               which of the findings above it does and does not support. Reason\n'
            '               here, so the fields below are read off rather than guessed at.>",')
    fields.append('  "findings": [{"feature": "<name>", "value": <value>, '
                  '"quote": "<exact text from the note>"}],')
    fields.append('  "present": true|false' + ("," if st.presence else ""))
    if st.presence:
        fields.append('  "present_quote": "<exact text from the note evidencing this outcome, '
                      'when present is true>"')
    empty = ('{"evidence": "...", "findings": [], "present": false}' if st.cot
             else '{"findings": [], "present": false}')
    joined = "\n".join(fields)
    # Field order is the point. Stage 0 puts `present` first, so the decoder
    # commits to the field that discards every other result before it has read
    # out a single finding.
    return ("## Output\n"
            "Reply with JSON only, no prose. Fill the fields in the order shown - the findings "
            "first,\nthen the presence call they support.\n\n"
            f"{{\n{joined}\n}}\n\n"
            f"When the note supports nothing, that is a complete answer:\n{empty}")


def _prompt_v1(note: str, outcome: str, table, needed: list[str], lines: str, st: Stage) -> str:
    blocks = [
        "You are reading a clinical note and reporting what it says about specific findings.\n"
        "A separate rule engine turns what you report into a severity grade, so your job is\n"
        "observation only.",
        f"Health outcome under consideration: {outcome} - {table.name}",
    ]

    if st.precision:
        blocks.append(
            "## The episode\n"
            "Case reports often span years and several admissions, and often describe more than\n"
            "one person - a donor, a relative, a comparison case. Report the ONE episode of this\n"
            "outcome the note is about, for the patient the report is about. Values from an\n"
            "earlier or later admission, from the patient's baseline or steady state, or from\n"
            "anyone else are not findings for this episode.")

    if st.presence:
        head = ('## What "present" means\n'
                '"present" is whether the note evidences this health outcome at all. The tables\n'
                "consult it first: when it is false, nothing else you report is used.")
        brief = presence_brief(outcome)
        if brief:
            head += "\n\nThe rubric defines this outcome as:\n\n" + _indent(brief)
        blocks.append(head)

    blocks += ["## Findings to extract\nReport only these, and only what the note supports:\n" + lines,
               _rules_v1(needed, st),
               _output_v1(st)]

    body = "\n\n".join(blocks)
    doc = f"<note>\n{note}\n</note>"
    return f"{doc}\n\n{body}" if st.note_first else f"{body}\n\n{doc}"


def prompt_features(outcome: str, st: Stage = STAGE0) -> list[str]:
    """The features this outcome's prompt asks for, in the order it asks for them.

    The schema's `feature` enum is built from this, so the two cannot drift: a
    name the prompt asks for and the grammar forbids is an unanswerable question,
    and one the grammar allows and the prompt never mentions is `unknown_feature`
    coming back through the front door.
    """
    table = TABLES[outcome]
    needed = {n for _, pred in table.all_rows() for n in parse(pred).names()}
    if table.on:
        needed.add(table.on)
    return order_features(sorted(expand_derived(needed)), outcome, st)


def build_prompt(note: str, outcome: str, st: Stage = STAGE0) -> str:
    table = TABLES[outcome]
    needed = prompt_features(outcome, st)
    lines = "\n".join(f"  - {feature_brief(n, outcome, st)}" for n in needed)
    if not st.structure:
        return _prompt_v0(note, outcome, table, needed, lines)
    return _prompt_v1(note, outcome, table, needed, lines, st)


def build_context_prompt(note: str, st: Stage = STAGE0) -> str:
    lines = "\n".join(f"  - {feature_brief(n, '', st)}" for n in CONTEXT_FEATURES)
    blocks = [
        "You are reading a clinical note and reporting the patient's baseline demographics (age and sex).\n"
        "Observation only: report only what is explicitly supported.",
        "## Findings to extract\nReport only these, and only what the note supports:\n" + lines,
        _rules_v1(list(CONTEXT_FEATURES), st),
    ]
    fields = []
    if st.cot:
        fields.append(
            '  "evidence": "<1-2 sentences. What the note says about the patient\'s age and sex.>",'
        )
    fields.append('  "findings": [{"feature": "<name>", "value": <value>, "quote": "<exact text from the note>"}]')
    joined = "\n".join(fields)
    empty = '{"evidence": "...", "findings": []}' if st.cot else '{"findings": []}'
    blocks.append(
        "## Output\n"
        "Reply with JSON only, no prose.\n\n"
        f"{{\n{joined}\n}}\n\n"
        f"When the note supports nothing, that is a complete answer:\n{empty}"
    )
    body = "\n\n".join(blocks)
    doc = f"<note>\n{note}\n</note>"
    return f"{doc}\n\n{body}" if st.note_first else f"{body}\n\n{doc}"

# -------------------------------------------------------------------- backends

def call_mock(prompt: str, model: str, host: str, stats: dict | None = None, **kwargs) -> str:
    """Deterministic stand-in so the harness itself can be tested and reviewed
    without a GPU. Emits one findable quote and one deliberate hallucination, so
    the verifier must report exactly 50%.

    `present` is flipped deterministically per (note, outcome) so the ABSENT path
    is exercised too - otherwise the absence audit has nothing to run on and a
    broken absent branch ships unnoticed. It does not touch the quote tally,
    which counts proposals regardless of presence."""
    # Stage 0 fences the note with triple quotes, stage 1+ with <note> tags.
    m = (re.search(r"<note>\n(.*)\n</note>", prompt, re.S)
         or re.search(r'NOTE:\n"""(.*)"""', prompt, re.S))
    note = m.group(1) if m else ""
    first = next((w for w in re.findall(r"[A-Za-z]{6,}", note)), "unknown")
    if stats is not None:
        stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + 100
        stats["eval_count"] = stats.get("eval_count", 0) + 50
        stats["eval_duration_sec"] = stats.get("eval_duration_sec", 0.0) + 0.01
        stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + 0.01

    outcome = (re.search(r"Health outcome under consideration: (\S+)", prompt) or [None, "0"])[1]
    present = zlib.crc32(f"{outcome}:{note[:120]}".encode()) % 3 != 0

    if "Your previous reply had these problems:" in prompt:
        if "baseline demographics" in prompt:
            return json.dumps({"findings": [
                {"feature": "patient_age", "value": 14.0, "quote": first},
            ]})
        return json.dumps({"present": present, "findings": [
            {"feature": "death_attributed", "value": False, "quote": first},
        ]})

    if "baseline demographics" in prompt:
        return json.dumps({"findings": [
            {"feature": "patient_age", "value": 14.0, "quote": first},
            {"feature": "patient_sex", "value": "male",
             "quote": "a sentence that is definitely not in this note"},
        ]})

    return json.dumps({"present": present, "findings": [
        {"feature": "death_attributed", "value": False, "quote": first},
        {"feature": "death_attributed", "value": True,
         "quote": "a sentence that is definitely not in this note"},
    ]})


BACKENDS = {
    "ollama": call_ollama,
    "mock": call_mock,
}


# ------------------------------------------------------------------ run loop



# How many times one (note, outcome) call may be redone because the CONTENT came
# back unusable. Stage 0 retries only on a transport exception, so a 200 carrying
# a list of null placeholders was taken first time; the placeholders were then
# counted as a model failure the harness never gave the model a chance to fix.
MAX_CONTENT_TRIES = 3


def generate(backend, prompt, model, host, stats, st: Stage, fmt, **kw) -> tuple[str, int]:
    """-> (reply, retries). One backend call, redone while the reply is unusable.

    The seed moves on every retry. Retrying at temperature 0 with a fixed seed
    re-runs the same arithmetic and returns the same bytes, so a retry that does
    not change the seed is a wasted call by construction.
    """
    tries = MAX_CONTENT_TRIES if st.retry else 1
    reply = ""
    for attempt in range(tries):
        reply = BACKENDS[backend](prompt, model, host, stats=stats, fmt=fmt,
                                  seed=attempt, **kw)
        if not st.retry or reply_is_usable(reply):
            return reply, attempt
    return reply, tries - 1


def run(notes, outcomes, backend, model, host, tally, timeout=300,
        concurrency=1, st: Stage = STAGE0, num_predict: int | None = None,
        num_ctx: int = 16384, ctx_tally: Tally | None = None):
    """Model calls first (optionally in parallel), then verification - always serial.

    Verification stays on the main thread in the original note-major order, so the
    Tally accumulates identically at any concurrency and `Tally` itself never needs
    a lock. Only the backend calls fan out.

    Each task carries its own stats dict; `call_ollama` does an unlocked
    read-modify-write on that dict, which is only safe because no two tasks share
    one. They are summed here.
    """
    if st.patient_context and ctx_tally is None:
        ctx_tally = Tally()

    ctx_tasks = list(notes) if st.patient_context else []
    outcome_tasks = [(rec, num) for rec in notes for num in outcomes]
    all_tasks = [(True, rec, None) for rec in ctx_tasks] + [(False, rec, num) for rec, num in outcome_tasks]
    total, done = len(all_tasks), 0
    replies, stats_parts = [None] * total, [None] * total
    print_lock = threading.Lock()
    t0 = time.time()

    # An `evidence` field is prose the model writes before the findings, so the
    # budget that fitted findings alone truncates the reply mid-object and the
    # whole pair is lost as bad JSON.
    budget = num_predict if num_predict else (2048 if st.cot else 1024)
    retries = [0] * total
    feedback_retries = [0] * total

    def call(i):
        nonlocal done
        is_ctx, rec, num = all_tasks[i]
        local = {"prompt_eval_count": 0, "eval_count": 0,
                 "eval_duration_sec": 0.0, "total_duration_sec": 0.0}
        if is_ctx:
            prompt = build_context_prompt(rec["patient"], st)
            fmt = context_schema(st) if st.schema else None
            reply, n_retry = generate(backend, prompt, model, host, local, st, fmt,
                                      timeout=timeout, num_predict=budget, num_ctx=num_ctx,
                                      repeat_penalty=1.0 if st.flat_penalty else 1.1)
            if st.feedback_retry and reply_is_usable(reply):
                issues = precheck(reply, rec["patient"], set(CONTEXT_FEATURES))
                if issues:
                    fb_prompt = (
                        prompt + "\n\nYour previous reply had these problems:\n"
                        + "\n".join(f"- {issue}" for issue in issues)
                        + "\n\nReply again with the full JSON."
                    )
                    fb_reply, _ = generate(backend, fb_prompt, model, host, local, st, fmt,
                                           timeout=timeout, num_predict=budget, num_ctx=num_ctx,
                                           repeat_penalty=1.0 if st.flat_penalty else 1.1)
                    if reply_is_usable(fb_reply):
                        reply = fb_reply
                    feedback_retries[i] = 1
            replies[i], stats_parts[i], retries[i] = reply, local, n_retry
            with print_lock:
                done += 1
                print(f"  [{done}/{total}] UID {rec['patient_uid']} context (age, sex)", flush=True)
        else:
            prompt = build_prompt(rec["patient"], num, st)
            fmt = reply_schema(prompt_features(num, st), st) if st.schema else None
            reply, n_retry = generate(backend, prompt, model, host, local, st, fmt,
                                      timeout=timeout, num_predict=budget, num_ctx=num_ctx,
                                      repeat_penalty=1.0 if st.flat_penalty else 1.1)
            if st.feedback_retry and reply_is_usable(reply):
                allowed = set(prompt_features(num, st))
                issues = precheck(reply, rec["patient"], allowed)
                if issues:
                    fb_prompt = (
                        prompt + "\n\nYour previous reply had these problems:\n"
                        + "\n".join(f"- {issue}" for issue in issues)
                        + "\n\nReply again with the full JSON."
                    )
                    fb_reply, _ = generate(backend, fb_prompt, model, host, local, st, fmt,
                                           timeout=timeout, num_predict=budget, num_ctx=num_ctx,
                                           repeat_penalty=1.0 if st.flat_penalty else 1.1)
                    if reply_is_usable(fb_reply):
                        reply = fb_reply
                    feedback_retries[i] = 1
            replies[i], stats_parts[i], retries[i] = reply, local, n_retry
            with print_lock:
                done += 1
                print(f"  [{done}/{total}] UID {rec['patient_uid']} outcome {num} "
                      f"({TABLES[num].name})", flush=True)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(call, range(total)))
    else:
        for i in range(total):
            call(i)

    context_results = {}
    offset = len(ctx_tasks)
    if st.patient_context and ctx_tally is not None:
        for i, rec in enumerate(ctx_tasks):
            reply = replies[i] or ""
            ctx_tally.content_retries += retries[i]
            ctx_tally.feedback_retries += feedback_retries[i]
            if st.retry and not reply_is_usable(reply):
                ctx_tally.unusable_replies += 1
            feats, _, detail = verify(reply, rec["patient"], ctx_tally)
            context_results[rec["patient_uid"]] = (feats, reply, detail)
        for part in stats_parts[:offset]:
            if part:
                ctx_tally.prompt_tokens += part["prompt_eval_count"]
                ctx_tally.completion_tokens += part["eval_count"]

    results = {}
    for j, (rec, num) in enumerate(outcome_tasks):
        i = offset + j
        reply = replies[i] or ""
        tally.content_retries += retries[i]
        tally.feedback_retries += feedback_retries[i]
        if st.retry and not reply_is_usable(reply):
            tally.unusable_replies += 1
        feats, present, detail = verify(reply, rec["patient"], tally)
        if st.patient_context:
            ctx_feats = context_results.get(rec["patient_uid"], ({}, "", {}))[0]
            feats = {**ctx_feats, **feats}
        results.setdefault(rec["patient_uid"], {})[num] = (feats, present, reply, detail)

    for part in stats_parts[offset:]:
        if part:
            tally.prompt_tokens += part["prompt_eval_count"]
            tally.completion_tokens += part["eval_count"]
    tally.wall_clock_sec += (time.time() - t0)
    return results, context_results


# ---------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    """-> the CLI parser. Its help text is user documentation (README.md shows examples)."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["full"], default="full", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="local Ollama tag for verified F16/BF16 MedGemma 27B weights")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="ollama",
                    help="ollama for inference; mock only checks harness plumbing")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
                    help="Ollama URL including http:// (default OLLAMA_HOST or localhost:11434)")
    ap.add_argument("--check-model", action="store_true",
                    help="check 27B/16-bit metadata, digest, and synthetic generation, then exit")
    ap.add_argument("--num-ctx", type=int, default=16384,
                    help="Ollama context window; increase for long notes (uses more memory)")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-request timeout in seconds (default 300)")
    ap.add_argument("--notes", type=int, default=20)
    ap.add_argument("--cohort", choices=["scd_primary", "loose"], default="loose",
                    help="pool definition. loose (default): any SCD mention - keeps notes "
                         "where outcomes are genuinely absent, which the absence audit needs. "
                         "scd_primary: SCD-primary notes only")
    ap.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True,
                    help="pick notes to hit every target outcome, plus a random holdout")
    ap.add_argument("--holdout-frac", type=float, default=0.25,
                    help="fraction of notes drawn at random, to measure the seeds' bias")
    ap.add_argument("--outcomes", default=DEFAULT_OUTCOMES,
                    help=f"comma-separated outcome IDs, '14'/'focus' for the 14 focus outcomes, "
                         f"or 'all' for all 53 (default: {DEFAULT_OUTCOMES})")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run N times and report run-to-run consistency at temperature 0")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="in-flight backend requests (default 1 = sequential). Needs a "
                         "server that batches, e.g. OLLAMA_NUM_PARALLEL>=N. See the "
                         "warning printed when this is combined with --repeat")
    ap.add_argument("--prompt-stage", choices=list(STAGES), default=DEFAULT_PROMPT_STAGE,
                    help=f"cumulative ablation rung (default {DEFAULT_PROMPT_STAGE}). "
                         "0 is the original prompt, kept unchanged as the baseline; "
                         "1 adds constrained decoding, field order, grouped rules and a "
                         "flat repeat penalty; 2a adds the rubric's presence definition "
                         "and a quoted `present`; 2b adds an `evidence` field; 3 adds "
                         "episode scope, negation, ordinal cues and the unit wording. "
                         "Stage 3 is expected to LOWER `accepted` and can only be judged "
                         "from the hand-check sheets")
    ap.add_argument("--note-first", action="store_true",
                    help="put the note above the instructions (stage >= 1). Off by "
                         "default: at a median 469 words the gain is small and it costs "
                         "the shared prefix across calls")
    ap.add_argument("--patient-context", action=argparse.BooleanOptionalAction, default=False,
                    help="extract patient-level context (age, sex) once per note and share across outcomes")
    ap.add_argument("--feedback-retry", action=argparse.BooleanOptionalAction, default=False,
                    help="re-prompt once with specific error feedback if extracted quotes or values fail verification")
    ap.add_argument("--num-predict", type=int, default=None,
                    help="completion token budget (default 1024, or 2048 once --prompt-stage "
                         "adds the evidence field)")
    ap.add_argument("--out", default=None)
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject argument combinations argparse cannot express; exits through ap.error()."""
    for name in ("notes", "repeat", "concurrency", "timeout", "num_ctx", "num_predict"):
        value = getattr(args, name)
        if value is not None and value < 1:
            ap.error(f"--{name.replace('_', '-')} must be >= 1")
    if not 0 <= args.holdout_frac <= 1:
        ap.error("--holdout-frac must be between 0 and 1")
    # A results file is the evidence for one specific run; never overwrite one.
    if args.out and pathlib.Path(args.out).exists():
        ap.error(f"Output already exists: {args.out}. Choose a new path to preserve prior results.")
    if args.check_model and args.backend != "ollama":
        ap.error("--check-model requires --backend ollama")


def parse_outcomes(ap: argparse.ArgumentParser, raw: str) -> list[str]:
    """-> outcome ids from --outcomes: "all", "14" / "focus", or a comma-separated list."""
    choice = raw.strip().lower()
    if choice == "all":
        return sorted(TABLES)
    if choice in ("14", "focus"):
        return [o.strip() for o in DEFAULT_OUTCOMES.split(",") if o.strip()]
    outcomes = [o.strip() for o in raw.split(",") if o.strip()]
    if not outcomes or len(outcomes) != len(set(outcomes)):
        ap.error("--outcomes must contain unique outcome ids")
    for outcome in outcomes:
        if outcome not in TABLES:
            raise SystemExit(f"unknown outcome {outcome!r}")
    return outcomes


def check_served_model(ap: argparse.ArgumentParser, args: argparse.Namespace) -> ServedModel:
    """-> what Ollama is really serving, after the fail-closed 27B / 16-bit preflight."""
    try:
        checked = preflight(args.model, args.host, timeout=args.timeout, num_ctx=args.num_ctx)
    except (RuntimeError, ValueError) as exc:
        ap.error(str(exc))
    served = ServedModel(model_info=checked["model_info"], digest=checked["model_digest"],
                         quant=checked["quant"])
    print(f"Preflight: {args.model} | {served.quant} | {served.digest}", flush=True)
    return served


def run_repeats(args: argparse.Namespace, st: Stage, notes: list[dict],
                outcomes: list[str]) -> Repeats:
    """-> every repeat's results and tallies; prints each repeat's tally as it finishes."""
    repeats = Repeats()
    for i in range(args.repeat):
        tally = Tally()
        context_tally = Tally() if st.patient_context else None
        started = time.time()
        try:
            results, context_results = run(notes, outcomes, args.backend, args.model, args.host, tally,
                                           timeout=args.timeout, concurrency=args.concurrency,
                                           st=st, num_predict=args.num_predict, num_ctx=args.num_ctx,
                                           ctx_tally=context_tally)
        except RuntimeError as exc:
            raise SystemExit(f"Extraction failed; no result file written: {exc}") from exc
        repeats.runs.append(results)
        repeats.context_runs.append(context_results)
        # The repeat's wall clock replaces run()'s own figure: it includes setup.
        tally.wall_clock_sec = time.time() - started
        repeats.tallies.append(tally)
        if context_tally is not None:
            repeats.context_tallies.append(context_tally)
        run_output.print_run_tally(i + 1, tally, context_tally, len(notes))
    return repeats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. -> process exit code. `argv=None` reads sys.argv (tests rely on this)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    st = stage(args.prompt_stage, note_first=args.note_first,
               patient_context=bool(args.patient_context),
               feedback_retry=bool(args.feedback_retry))
    validate_args(ap, args)
    outcomes = parse_outcomes(ap, args.outcomes)

    served = ServedModel()
    if args.backend == "ollama":
        served = check_served_model(ap, args)
        if args.check_model:
            return 0

    pool = load_notes(cohort=args.cohort)
    notes, selection = select_notes(pool, args.notes, outcomes,
                                    holdout_frac=args.holdout_frac, stratify=args.stratify)
    if not notes:
        ap.error(f"No notes available for cohort {args.cohort}")

    run_output.print_run_banner(args, notes, outcomes)
    if args.concurrency > 1 and args.repeat > 1:
        run_output.print_concurrency_warning(args.concurrency)

    repeats = run_repeats(args, st, notes, outcomes)

    # Grades come from the first repeat; later repeats only measure consistency.
    summary = run_output.summarize_grades(repeats.runs[0], outcomes, selection, repeats.tallies[0])
    run_output.print_grade_summary(summary, len(notes), outcomes)
    consistency_pct = run_output.report_consistency(repeats.runs, outcomes, args.repeat,
                                                    args.concurrency)

    # Read the tally only now: summarize_grades() adds presence contradictions to it.
    first_report = repeats.tallies[0].report()
    invalid_value_pct = run_output.print_metrics_assessment(first_report, consistency_pct,
                                                            args.concurrency)

    if args.out:
        results = run_output.build_results(args, st, served, notes, selection, outcomes, repeats,
                                           summary, first_report, consistency_pct, invalid_value_pct)
        run_output.write_results(args.out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
