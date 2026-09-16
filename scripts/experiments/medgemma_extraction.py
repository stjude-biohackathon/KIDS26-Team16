"""P11 - Does MedGemma actually extract features well enough?

Everything in the current plan assumes MedGemma is good at reading a note and
filling in feature values. Nothing has checked that. This script checks it.

It runs MedGemma ALONE - note in, features out, straight into the real decision
tables - so what it measures is Version B of the plan (§3). Version A cannot be
measured until the lexicon (P5) and a trained BERT exist. If the numbers here
are bad, delete this file and the whole MedGemma direction with it. If they are
good, this grows into Version B.

Grades always come from the tables. Extraction quality is what is measured;
grading never is.

The contract from §2 is enforced, not assumed: every proposed value must carry a
quote, and the quote must appear VERBATIM in the note. A value whose quote
cannot be found is rejected, not down-weighted - that turns hallucination into a
counted failure rather than a silent one.

Usage examples:
    # 1. Local 4B tier (Apple Silicon / CPU / GPU via Ollama):
    python3 scripts/experiments/medgemma_extraction.py --tier local --notes 20

    # 2. Full 27B tier on NVIDIA Windows / Linux via Ollama:
    python scripts/experiments/medgemma_extraction.py --tier full --notes 20 --repeat 2 --out results/full.json

    # 3. Full 27B tier on NVIDIA Windows / Linux directly via PyTorch / HuggingFace (4-bit quant for 16-24GB GPUs):
    python scripts/experiments/medgemma_extraction.py --tier full --backend hf --quant 4bit --notes 20 --out results/full.json

    # 4. Mock backend (no GPU/model required):
    python3 scripts/experiments/medgemma_extraction.py --backend mock --notes 2

    # 5. Throughput run on a GPU with VRAM to spare (needs OLLAMA_NUM_PARALLEL>=4).
    #    Measure run-to-run consistency separately, at --concurrency 1:
    python scripts/experiments/medgemma_extraction.py --tier full --notes 20 --concurrency 4 --out results/full.json
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
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Ensure UTF-8 output on Windows consoles (prevents charmap / cp1252 encode errors on °, µ, ×, etc.)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scogs.definitions import presence_brief
from scogs.evaluate import grade
from scogs.features import FEATURES
from scogs.predicates import parse
from scogs.tables import TABLES

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Model ids are configuration, never hardcoded logic (§3).
TIERS = {
    "local": {
        "hf": "google/medgemma-1.5-4b-it",
        "ollama": "medgemma-1.5-4b-it",
        "note": "dev tier, Apple Silicon; never quote its numbers (§3)",
    },
    "full": {
        "hf": "google/medgemma-27b-text-it",
        "ollama": "medgemma-27b-text-it",
        "note": "reporting tier; needs a real GPU (NVIDIA CUDA / A100 / RTX 3090/4090)",
    },
}

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

    @property
    def rank(self) -> int:
        return STAGES.index(self.name)


def stage(name: str, note_first: bool = False) -> Stage:
    if name not in STAGES:
        raise ValueError(f"unknown prompt stage {name!r}; expected one of {', '.join(STAGES)}")
    i = STAGES.index(name)
    return Stage(name=name, note_first=note_first,
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
    if spec["values"]: bits.append(f', one of: {", ".join(spec["values"])}')
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
    if clarifier: line += f" For this outcome specifically: {clarifier}"
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
        if n in seen: continue
        seen.add(n)
        spec = FEATURES[n]
        if not spec["derived"]:
            out.add(n); continue
        inputs = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", spec["derived"])
                  if w in FEATURES and w != n]
        if inputs: stack += inputs
        else: out.add(n)
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
    if spec["type"] == "bool": return {"type": "boolean"}
    if spec["type"] == "num":  return {"type": "number"}
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
        if not isinstance(f, dict): return False
        if not f.get("feature") or not f.get("quote"): return False
        if f.get("value") is None: return False
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
    if table.on: needed.add(table.on)
    return order_features(sorted(expand_derived(needed)), outcome, st)


def build_prompt(note: str, outcome: str, st: Stage = STAGE0) -> str:
    table = TABLES[outcome]
    needed = prompt_features(outcome, st)
    lines = "\n".join(f"  - {feature_brief(n, outcome, st)}" for n in needed)
    if not st.structure:
        return _prompt_v0(note, outcome, table, needed, lines)
    return _prompt_v1(note, outcome, table, needed, lines, st)

# -------------------------------------------------------------------- backends

def get_model_info(model: str, host: str) -> dict:
    try:
        body = json.dumps({"name": model}).encode("utf-8")
        req = urllib.request.Request(f"{host}/api/show", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return {}


# `general.file_type` is a GGUF quantisation enum, not an identity. Recording it as
# the digest answered "which weights ran?" with the number 1.
GGUF_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1_SOME_F16", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
    21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S",
    27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0"}


def get_model_digest(model: str, host: str) -> str:
    """The content hash Ollama holds for these weights - the only field that actually
    identifies what ran. /api/show does not return it; /api/tags does."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=10) as r:
            for m in json.loads(r.read().decode("utf-8")).get("models", []):
                if m.get("name") in (model, f"{model}:latest"):
                    return m.get("digest", "")
    except Exception:
        pass
    return ""


def call_ollama(prompt: str, model: str, host: str, stats: dict | None = None,
                timeout: int = 300, fmt=None, seed: int = 0, temperature: float = 0.0,
                num_predict: int = 1024, repeat_penalty: float = 1.1, **kwargs) -> str:
    # keep_alive -1 pins the model in VRAM. Without it Ollama unloads after 5 idle
    # minutes and the next call silently pays a full reload - 50+ GB at the F16 tier.
    #
    # `repeat_penalty` defaults to the stage 0 value of 1.1 and is dropped to 1.0
    # from stage 1. A repetition penalty on a task whose output must contain text
    # copied verbatim out of the note is pushing against the job: quotes reuse the
    # note's tokens, feature names recur, and JSON punctuation recurs hardest of
    # all. At temperature 0 under a grammar it buys nothing and can walk the
    # decoder off an exact span, which lands as `quote_unfound`.
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "format": fmt if fmt is not None else "json", "keep_alive": -1,
                       "options": {"temperature": temperature, "seed": seed,
                                   "num_predict": num_predict,
                                   "repeat_penalty": repeat_penalty}}).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
                if stats is not None:
                    stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + resp.get("prompt_eval_count", 0)
                    stats["eval_count"] = stats.get("eval_count", 0) + resp.get("eval_count", 0)
                    stats["eval_duration_sec"] = stats.get("eval_duration_sec", 0.0) + resp.get("eval_duration", 0) / 1e9
                    stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + (time.time() - t0)
                return resp.get("response", "")
        except Exception as e:
            if attempt == 2:
                print(f"    [warning] ollama call failed after 3 attempts: {e}", flush=True)
                return ""
            time.sleep(2)
    return ""


_HF_PIPELINE = None
_HF_MODEL_NAME = None

def get_hf_pipeline(model_name: str, quant: str = "none"):
    global _HF_PIPELINE, _HF_MODEL_NAME
    if _HF_PIPELINE is not None and _HF_MODEL_NAME == model_name:
        return _HF_PIPELINE
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        raise SystemExit(
            "Hugging Face backend requires PyTorch and transformers:\n"
            "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
            "    pip install transformers accelerate bitsandbytes\n"
        )

    print(f"\n[HF Backend] Loading model {model_name} on CUDA/GPU (quant={quant})...", flush=True)
    kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32

    if quant == "4bit":
        kwargs["load_in_4bit"] = True
    elif quant == "8bit":
        kwargs["load_in_8bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    _HF_PIPELINE = pipeline("text-generation", model=model, tokenizer=tokenizer)
    _HF_MODEL_NAME = model_name
    print(f"[HF Backend] Model {model_name} loaded successfully.\n", flush=True)
    return _HF_PIPELINE


def call_hf(prompt: str, model: str, host: str, stats: dict | None = None,
            quant: str = "none", num_predict: int = 1024, repeat_penalty: float = 1.1,
            seed: int = 0, temperature: float = 0.0, **kwargs) -> str:
    pipe = get_hf_pipeline(model, quant)
    t0 = time.time()
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # No grammar here: transformers has no schema-constrained decoding in this
    # path, so the reply is fenced out below and the content gate does the rest.
    sample = temperature > 0
    gen = {"max_new_tokens": num_predict, "do_sample": sample,
           "repetition_penalty": repeat_penalty}
    if sample:
        gen["temperature"] = temperature
    out = pipe(prompt_formatted, **gen)
    gen_text = out[0]["generated_text"][len(prompt_formatted):].strip()
    dur = time.time() - t0
    if stats is not None:
        stats["eval_duration_sec"] = stats.get("eval_duration_sec", 0.0) + dur
        stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + dur
        stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + len(prompt_formatted.split())
        stats["eval_count"] = stats.get("eval_count", 0) + len(gen_text.split())

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", gen_text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*\})", gen_text, re.S)
    return m.group(1) if m else gen_text


def call_openai_compatible(prompt: str, model: str, host: str, stats: dict | None = None,
                           timeout: int = 300, fmt=None, seed: int = 0,
                           temperature: float = 0.0, num_predict: int = 1024,
                           **kwargs) -> str:
    url = f"{host.rstrip('/')}/v1/chat/completions"
    # A schema goes as `json_schema`; without one this falls back to `json_object`,
    # which only guarantees the reply parses.
    response_format = ({"type": "json_schema",
                        "json_schema": {"name": "scogs_findings", "strict": True,
                                        "schema": fmt}}
                       if isinstance(fmt, dict) else {"type": "json_object"})
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": num_predict,
        "response_format": response_format,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
            usage = resp.get("usage", {})
            if stats is not None:
                stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + usage.get("prompt_tokens", 0)
                stats["eval_count"] = stats.get("eval_count", 0) + usage.get("completion_tokens", 0)
                stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + (time.time() - t0)
            return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"    [warning] openai-compatible call failed: {e}", flush=True)
        return ""


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
    outcome = (re.search(r"Health outcome under consideration: (\S+)", prompt) or [None, "0"])[1]
    present = zlib.crc32(f"{outcome}:{note[:120]}".encode()) % 3 != 0
    first = next((w for w in re.findall(r"[A-Za-z]{6,}", note)), "unknown")
    if stats is not None:
        stats["prompt_eval_count"] = stats.get("prompt_eval_count", 0) + 100
        stats["eval_count"] = stats.get("eval_count", 0) + 50
        stats["eval_duration_sec"] = stats.get("eval_duration_sec", 0.0) + 0.01
        stats["total_duration_sec"] = stats.get("total_duration_sec", 0.0) + 0.01
    return json.dumps({"present": present, "findings": [
        {"feature": "death_attributed", "value": False, "quote": first},
        {"feature": "death_attributed", "value": True,
         "quote": "a sentence that is definitely not in this note"},
    ]})


BACKENDS = {
    "ollama": call_ollama,
    "hf": call_hf,
    "openai": call_openai_compatible,
    "vllm": call_openai_compatible,
    "mock": call_mock,
}


# ------------------------------------------------------- verification & scoring

# SentencePiece byte-token wreckage from a badly converted GGUF. Its presence
# means the served weights are corrupt, not that the model hallucinated - so it
# is counted and reported, never quietly normalized into a passing quote.
ARTIFACT = re.compile(r"\[UNK_BYTE_|\u2581")


def normalize(s: str) -> str:
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
    "mg/dl":  r"mg\s*/\s*dl|mg\s+dl\s*(?:\u2212|-)?\s*1|milligrams?\s+per\s+decilit",
    "mg/l":   r"mg\s*/\s*l(?![a-z/])|mg\s+l\s*(?:\u2212|-)?\s*1|milligrams?\s+per\s+lit",
    "umol/l": r"[\u00b5u]mol\s*/\s*l|micromol",
    "mmol/l": r"mmol\s*/\s*l|millimol",
    "g/dl":   r"(?<![a-z])g\s*/\s*dl|grams?\s+per\s+decilit",
    "g/l":    r"(?<![a-z])g\s*/\s*l(?![a-z/])|grams?\s+per\s+lit",
    "degf":   r"\u00b0\s*f\b|\u00ba\s*f\b|\bfahrenheit",
    "degc":   r"\u00b0\s*c\b|\u00ba\s*c\b|\bcelsius|\bcentigrade",
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
}

UNIT_OK, UNIT_CONVERTED, UNIT_AMBIGUOUS = "ok", "converted", "ambiguous"
UNIT_BAD, UNIT_VALUE_MISMATCH = "bad", "value_mismatch"

# Decimal point only. A comma here would read a thousands separator as a decimal.
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _number_for_unit(q: str, pat: str):
    """The number the unit token is attached to - the nearest one before it, else
    the first after. -> float | None."""
    m = re.search(pat, q, re.I)
    if not m:
        return None
    before = NUMBER.findall(q[:m.start()])
    if before:
        return float(before[-1])
    after = NUMBER.search(q[m.end():])
    return float(after.group()) if after else None


def _agrees(a: float, b: float) -> bool:
    """Tight on purpose. 38.9 against 39.2 is 0.8% and straddles a grade boundary,
    so anything loose enough to call those equal defeats the check."""
    return abs(a - b) <= max(abs(b), abs(a), 1.0) * 0.005


def unit_guard(name: str, value: float, quote: str):
    """-> (status, value, detail). Reconcile a number against its own verified quote.

    The quote is the authority, not the value: the model may already have tried the
    conversion and botched it ("102.6 Fahrenheit" -> 38.9, which is 39.2 and lands on
    the far side of a grade boundary). So the number is read out of the quote, next
    to the unit it is written in, and converted from there.

    Silence is the default. No recognisable unit in the quote, several of them, or no
    number beside one, and the value is returned untouched - this guard only speaks
    where the note itself said what unit it meant.
    """
    declared = FEATURES[name].get("unit")
    table = UNIT_CONVERSIONS.get(declared)
    if not table:
        return UNIT_OK, value, None       # no unit declared, or no family for it
    q = normalize(quote)
    found = [u for u, pat in UNIT_TOKENS.items() if re.search(pat, q, re.I)]
    if not found:
        return UNIT_OK, value, None
    if len(found) > 1:
        return UNIT_AMBIGUOUS, value, "/".join(sorted(found))
    src = found[0]
    if src not in table:
        return UNIT_BAD, None, f"{src} is not convertible to {declared}"
    raw = _number_for_unit(q, UNIT_TOKENS[src])
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
    """-> 'max' | 'min' | None, read off the feature's own definition."""
    if FEATURES[name]["type"] == "cat":
        return None                       # unordered: no defensible winner
    d = FEATURES[name]["definition"]
    if AGG_MAX.search(d): return "max"
    if AGG_MIN.search(d): return "min"
    return None


def _rank(name: str, v):
    spec = FEATURES[name]
    if spec["type"] == "ord":
        return (spec["values"] or []).index(v)
    return v


def harness_status(rule_status: str, present) -> str:
    """-> the status the harness reports, which splits one of the engine's.

    `absent` pools two different findings: the model never evidenced the outcome,
    and the model DID evidence it but the tables overruled the call (a 36.5 degC
    "fever"). Pooled, the second kind lands in the absence audit, where it asks a
    reviewer to confirm an absence that the rule engine, not the model, produced.
    """
    return "refuted" if rule_status == "absent" and present else rule_status


def reconcile(name: str, values: list):
    """-> (value, conflict). Several quoted values for one feature is the norm, not
    an edge case: a note carries five creatinines across twelve years, and one of
    them belongs to the transplant donor. Overwriting until the last one wins picks
    by emission order, silently. Where the schema states an aggregation the values
    collapse by it; where it does not, disagreement is a conflict that is reported
    and withheld, never guessed at.
    """
    uniq = []
    for v in values:
        if v not in uniq: uniq.append(v)
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
    content_retries: int = 0       # calls redone because the reply was unusable
    unusable_replies: int = 0      # ... and still unusable when the tries ran out
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
            "present_quoted_pct": round(100 * self.present_quoted / (self.present_true or 1), 1),
            "content_retries": self.content_retries,
            "unusable_replies": self.unusable_replies,
            "missing_quote": self.quote_missing,
            "invalid_value": self.value_bad,
            "unknown_feature": self.unknown_feature,
            "unparseable_replies": self.bad_json,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_clock_sec": round(self.wall_clock_sec, 2),
        }


def coerce(name: str, value):
    """-> (ok, coerced). Enforces the declared type; never widens it."""
    spec = FEATURES.get(name)
    if spec is None: return False, None
    t = spec["type"]
    if t == "bool":
        if isinstance(value, bool): return True, value
        if str(value).lower() in {"true", "yes"}:  return True, True
        if str(value).lower() in {"false", "no"}:  return True, False
        return False, None
    if t == "num":
        try: return True, float(value)
        except (TypeError, ValueError): return False, None
    v = str(value).strip().lower()
    return (True, v) if v in (spec["values"] or []) else (False, None)


def verify(reply: str, note: str, tally: Tally) -> tuple[dict, bool | None, dict]:
    """-> (features, present, detail).

    `detail` carries every accepted finding and every unresolved conflict for this
    one (note, outcome). The review sheets are built from it rather than re-deriving
    acceptance downstream, which is how a hand-check sheet ends up disagreeing with
    the harness that produced it.

    A verified quote means the words are in the note. It does NOT mean the words
    support the value - "needed increasing oxygen by nasal cannula" verifies
    perfectly behind fio2_pct=21. That judgement is human, and it is what the
    hand-check sheet's `supports_value` column exists to record.
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


# ------------------------------------------------------------------------ main

# Cohort gate. Three traps live in this corpus and each one silently poisons the
# eval set with a note that can only ever score `absent`:
#   1. "sickle cell trait" is the heterozygous carrier state, not the disease.
#   2. In cardiology, SCD means *sudden cardiac death* - so the bare abbreviation
#      never qualifies a note on its own.
#   3. The mention may be negated, or belong to the mother rather than the patient.
SICKLE_EXPLICIT = re.compile(r"sickle[- ]cell(?!\s+trait)|HbSS|HbSC|\bHb\s?S\b", re.I)
SCD_TERM = re.compile(r"sickle[- ]cell(?!\s+trait)|HbSS|HbSC|\bHb\s?S\b|\bSCD\b", re.I)
NEG_SCD = re.compile(r"(?:denie[sd]|den(?:y|ying)|no|without|negative for|ruled out|"
                     r"family history|maternal|paternal|mother|father|sibling|"
                     r"brother|sister|cousin)\b[^.]{0,70}?"
                     r"(?:sickle[- ]cell|\bSCD\b|HbSS|HbSC)", re.I)
TRAIT = re.compile(r"sickle[- ]cell\s+trait", re.I)


def _clean(text: str) -> str:
    """Drop trait mentions, then mentions that are negated or somebody else's."""
    return NEG_SCD.sub(" ", TRAIT.sub(" ", text or ""))


def scd_mentions(text: str) -> int:
    """SCD mentions that are the patient's own and not negated."""
    return len(SCD_TERM.findall(_clean(text)))


def is_scd_primary(rec: dict) -> bool:
    """Is this note ABOUT sickle cell disease, or does it merely say the words?

    The loose mention regex admits notes whose only SCD reference is a denial
    ("denied a family history of SCD"), a carrier state, the mother's diagnosis,
    or a cardiology note using SCD for sudden cardiac death. Those land in the
    eval set as guaranteed `absent`, inflating the absent count and deflating
    every rate computed over the sample - a measurement artifact indistinguishable
    from a model that simply extracts nothing.
    """
    raw_title = rec.get("title", "") or ""
    if TRAIT.search(raw_title) and not SICKLE_EXPLICIT.search(TRAIT.sub(" ", raw_title)):
        return False                       # a paper titled "Sickle Cell Trait: ..." is about trait
    if SICKLE_EXPLICIT.search(_clean(raw_title)):
        return True                        # the paper names the disease in its title
    body = _clean(rec.get("patient", "") or "")
    if not SICKLE_EXPLICIT.search(body):
        return False                       # "SCD" alone is not evidence of sickle cell
    return len(SCD_TERM.findall(body)) >= 2


def load_notes(cohort: str = "loose") -> list[dict]:
    """The candidate pool. Selection happens in select_notes()."""
    cache = ROOT / "PMC-Patients" / "scd_cache.json"
    if cache.exists():
        scd = json.loads(cache.read_text(encoding="utf-8"))
    else:
        pat = re.compile(r"sickle cell|\bSCD\b|HbSS|HbSC", re.I)
        data = json.loads((ROOT / "PMC-Patients" / "PMC-Patients-V2.json").read_text(encoding="utf-8"))
        scd = [r for r in data if pat.search(r.get("patient", ""))]
        scd.sort(key=lambda r: r["patient_uid"])          # deterministic before sampling
        try:
            cache.write_text(json.dumps(scd), encoding="utf-8")
        except Exception:
            pass
    if cohort == "scd_primary":
        kept = [r for r in scd if is_scd_primary(r)]
        print(f"cohort=scd_primary: {len(kept)}/{len(scd)} kept "
              f"({len(scd) - len(kept)} dropped as mention-only)")
        scd = kept
    return scd


# ------------------------------------------------------------- note selection
#
# The unit of evaluation is the (note, outcome) PAIR, not the note. `absent` is a
# first-class answer (plan §1), so an eval set needs outcomes that are genuinely
# present AND outcomes that are genuinely not - otherwise the absent decision,
# which is what actually gates whether anything gets graded, goes unmeasured.
#
# Seeds pick notes only. They never touch extraction, features or grading, so
# they cannot bias a grade - but they DO bias which cases get seen, toward the
# lexically obvious ones. That is what the unstratified holdout is for: it is
# drawn at random from the same pool, so the size of the bias is measurable
# rather than merely disclosed.

OUTCOME_SEEDS = {
    "10": r"chronic pain|daily pain|persistent pain",
    "19": r"acute kidney injur|\bAKI\b|renal failure|rising creatinine|creatinine",
    "24": r"priapism",
    "28": r"pain(ful)? (crisis|crises|episode)|vaso-?occlusive|\bVOC\b|sickle cell crisis|pain control",
    "29": r"splenic sequestration|sequestration crisis",
    "30": r"alloimmuni|delayed h(a)?emolytic|\bDHTR\b",
    "34": r"iron overload|h(a)?emochromatosis|h(a)?emosiderosis|ferritin|chelat",
    "35": r"aplastic crisis|parvovirus",
    "36": r"\bfever|febrile|pyrexia|temperature of \d",
    "37": r"sepsis|septic|bacter(a)?emia",
    "40": r"leg ulcer|ankle ulcer|venous ulcer",
    "43": r"multiorgan failure|multi-organ failure|\bMOF\b",
    "48": r"acute chest|chest syndrome|\bACS\b",
    "49": r"asthma|wheez|bronchodilator",
    "04": r"heart failure|\bCHF\b|cardiac decompensation",
    "05": r"myocardial infarction|\bMI\b|troponin",
    "06": r"hypertension|hypertensive|elevated blood pressure",
    "11": r"cognitive|neurocognitive|memory (loss|impairment)",
    "15": r"stroke|infarct|h(a)?emorrhage|\bCVA\b",
    "18": r"cholecyst|cholelith|gallstone|gallbladder",
    "21": r"chronic kidney disease|\bCKD\b|nephropathy|proteinuria",
    "31": r"hypersplenism|splenomegaly",
    "32": r"hepatopathy|hepatic|liver (failure|dysfunction)|transaminas",
    "39": r"avascular necrosis|osteonecrosis|\bAVN\b",
    "42": r"osteoporo|osteopeni|bone mineral density",
    "47": r"depress|\bPHQ",
    "52": r"pulmonary hypertension|\bPAH\b|elevated TRV",
    "53": r"sleep apn(o)?ea|\bOSA\b|polysomnograph",
}


def outcome_seed(num: str) -> re.Pattern:
    """A lexical prior for 'this note probably discusses outcome `num`'.

    Falls back to the outcome's own name, which is usually enough ("Priapism",
    "Leg Ulcer", "Osteomyelitis"); OUTCOME_SEEDS covers the ones where the
    rubric's phrasing is not what a clinician writes.
    """
    if num in OUTCOME_SEEDS:
        return re.compile(OUTCOME_SEEDS[num], re.I)
    name = TABLES[num].name
    alts = []
    paren = re.search(r"\(([^)]*)\)", name)
    base = re.sub(r"\s*\([^)]*\)", "", name).strip()
    alts += [re.escape(x.strip()) for x in base.split("/") if x.strip()]
    if paren and re.fullmatch(r"[A-Z]{2,6}", paren.group(1).strip()):
        alts.append(r"\b" + re.escape(paren.group(1).strip()) + r"\b")
    return re.compile("|".join(alts), re.I)


def select_notes(pool: list[dict], n: int, outcomes: list[str], *, seed: int = 20260828,
                 holdout_frac: float = 0.25, stratify: bool = True):
    """-> (notes, selection) where selection maps uid -> 'seeded:<outcome>' | 'holdout'."""
    import random
    rng = random.Random(seed)
    if not stratify:
        picked = rng.sample(pool, min(n, len(pool)))
        return picked, {r["patient_uid"]: "random" for r in picked}

    n_hold = max(1, round(n * holdout_frac))
    n_strat = max(0, n - n_hold)
    per = [n_strat // len(outcomes)] * len(outcomes)
    for i in range(n_strat - sum(per)):
        per[i] += 1

    picked, selection = [], {}
    taken = set()
    for num, want in zip(outcomes, per):
        pat = outcome_seed(num)
        cands = [r for r in pool
                 if r["patient_uid"] not in taken and pat.search(r.get("patient", "") or "")]
        got = rng.sample(cands, min(want, len(cands)))
        if len(got) < want:
            print(f"  seeded {num:>3s} ({TABLES[num].name[:28]}): only {len(got)}/{want} "
                  f"candidates in the pool")
        for r in got:
            taken.add(r["patient_uid"])
            selection[r["patient_uid"]] = f"seeded:{num}"
        picked += got

    rest = [r for r in pool if r["patient_uid"] not in taken]
    hold = rng.sample(rest, min(n_hold, len(rest)))
    for r in hold:
        selection[r["patient_uid"]] = "holdout"
    picked += hold

    seeded_n = len(picked) - len(hold)
    print(f"selection: {seeded_n} seeded across {len(outcomes)} outcomes + "
          f"{len(hold)} unstratified holdout = {len(picked)} notes")
    return picked, selection


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


def run(notes, outcomes, backend, model, host, tally, quant="none", timeout=300,
        concurrency=1, st: Stage = STAGE0, num_predict: int | None = None):
    """Model calls first (optionally in parallel), then verification - always serial.

    Verification stays on the main thread in the original note-major order, so the
    Tally accumulates identically at any concurrency and `Tally` itself never needs
    a lock. Only the backend calls fan out.

    Each task carries its own stats dict; `call_ollama` does an unlocked
    read-modify-write on that dict, which is only safe because no two tasks share
    one. They are summed here.
    """
    tasks = [(rec, num) for rec in notes for num in outcomes]
    total, done = len(tasks), 0
    replies, stats_parts = [None] * total, [None] * total
    print_lock = threading.Lock()
    t0 = time.time()

    # An `evidence` field is prose the model writes before the findings, so the
    # budget that fitted findings alone truncates the reply mid-object and the
    # whole pair is lost as bad JSON.
    budget = num_predict if num_predict else (2048 if st.cot else 1024)
    retries = [0] * total

    def call(i):
        nonlocal done
        rec, num = tasks[i]
        local = {"prompt_eval_count": 0, "eval_count": 0,
                 "eval_duration_sec": 0.0, "total_duration_sec": 0.0}
        prompt = build_prompt(rec["patient"], num, st)
        fmt = reply_schema(prompt_features(num, st), st) if st.schema else None
        reply, n_retry = generate(backend, prompt, model, host, local, st, fmt,
                                  quant=quant, timeout=timeout, num_predict=budget,
                                  repeat_penalty=1.0 if st.flat_penalty else 1.1)
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

    results = {}
    for i, (rec, num) in enumerate(tasks):
        reply = replies[i] or ""
        tally.content_retries += retries[i]
        if st.retry and not reply_is_usable(reply):
            tally.unusable_replies += 1
        feats, present, detail = verify(reply, rec["patient"], tally)
        results.setdefault(rec["patient_uid"], {})[num] = (feats, present, reply, detail)

    for part in stats_parts:
        if part:
            tally.prompt_tokens += part["prompt_eval_count"]
            tally.completion_tokens += part["eval_count"]
    tally.wall_clock_sec += (time.time() - t0)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIERS), default="local")
    ap.add_argument("--model", help="override the tier's model id")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="ollama",
                    help="inference backend (ollama, hf, openai/vllm, mock)")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="Ollama host or OpenAI-compatible server URL")
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none",
                    help="quantization for Hugging Face backend (4bit/8bit recommended for 24GB GPUs)")
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
    ap.add_argument("--outcomes", default="28,48,36,19",
                    help="comma-separated; default is a common-outcome sample")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run N times and report run-to-run consistency at temperature 0")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="in-flight backend requests (default 1 = sequential). Needs a "
                         "server that batches, e.g. OLLAMA_NUM_PARALLEL>=N. See the "
                         "warning printed when this is combined with --repeat")
    ap.add_argument("--prompt-stage", choices=list(STAGES), default="0",
                    help="cumulative ablation rung (default 0 = the prompt unchanged). "
                         "1 adds constrained decoding, field order, grouped rules and a "
                         "flat repeat penalty; 2a adds the rubric's presence definition "
                         "and a quoted `present`; 2b adds an `evidence` field; 3 adds "
                         "episode scope, negation, ordinal cues and the unit wording. "
                         "Stage 3 is expected to LOWER `accepted` and can only be judged "
                         "from the hand-check sheets")
    ap.add_argument("--note-first", action="store_true",
                    help="put the note above the instructions (stage >= 1). Off by "
                         "default: at a median 469 words the gain is small and it costs "
                         "the shared prefix across an outcome's four calls")
    ap.add_argument("--num-predict", type=int, default=None,
                    help="completion token budget (default 1024, or 2048 once --prompt-stage "
                         "adds the evidence field)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    st = stage(a.prompt_stage, note_first=a.note_first)

    if a.concurrency > 1 and a.backend == "hf":
        print("!! --backend hf holds one lazily-initialised global pipeline and is not "
              "safe to call\n   concurrently. Forcing --concurrency 1.")
        a.concurrency = 1
    if a.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    tier = TIERS[a.tier]
    if a.model:
        model = a.model
    elif a.backend in {"hf", "openai", "vllm"}:
        model = tier["hf"]
    else:
        model = tier["ollama"]

    outcomes = [o.strip() for o in a.outcomes.split(",") if o.strip()]
    for o in outcomes:
        if o not in TABLES: raise SystemExit(f"unknown outcome {o!r}")

    pool = load_notes(cohort=a.cohort)
    notes, selection = select_notes(pool, a.notes, outcomes,
                                    holdout_frac=a.holdout_frac, stratify=a.stratify)
    model_info = get_model_info(model, a.host) if a.backend == "ollama" else {}
    model_digest = (get_model_digest(model, a.host) if a.backend == "ollama" else "") \
        or model_info.get("details", {}).get("parent_model", "")
    file_type = model_info.get("model_info", {}).get("general.file_type")
    if a.backend == "hf":
        served_quant = a.quant
    elif file_type is None:
        served_quant = None
    else:
        served_quant = GGUF_FILE_TYPES.get(file_type, f"file_type_{file_type}")

    print("=" * 70)
    print(f"P11 MedGemma Extraction Test")
    print(f"tier={a.tier}  weights={tier['hf']}  served-as={model}  backend={a.backend}")
    if a.backend == "hf":
        print(f"hf_quant={a.quant}")
    print(f"notes={len(notes)}  outcomes={','.join(outcomes)}  repeat={a.repeat}  "
          f"concurrency={a.concurrency}")
    print(f"tier_note={tier['note']}")
    print("=" * 70)

    if a.concurrency > 1 and a.repeat > 1:
        print()
        print("!! CONCURRENCY WARNING - run-to-run consistency is confounded.")
        print(f"   At --concurrency {a.concurrency} the server batches requests, and a batch's")
        print("   composition depends on timing, so it differs between repeats. Batched float")
        print("   reductions are not bit-identical, so a token can flip at temperature 0 for")
        print("   reasons that have nothing to do with the model. Mismatches below are then")
        print("   'model nondeterminism OR batching', and you cannot tell which.")
        print("   Measure consistency with --concurrency 1. Use >1 for throughput and cost.")
        print()

    runs, tallies = [], []
    for i in range(a.repeat):
        t = Tally()
        t_start = time.time()
        try:
            runs.append(run(notes, outcomes, a.backend, model, a.host, t,
                            quant=a.quant, timeout=a.timeout, concurrency=a.concurrency,
                            st=st, num_predict=a.num_predict))
        except (urllib.error.URLError, TimeoutError) as e:
            raise SystemExit(f"backend unreachable at {a.host}: {e}\n"
                             f"start it, or use --backend mock to exercise the harness")
        t.wall_clock_sec = time.time() - t_start
        tallies.append(t)
        rep = t.report()
        sec_per_note = t.wall_clock_sec / len(notes) if notes else 0
        tok_per_sec = t.completion_tokens / t.wall_clock_sec if t.wall_clock_sec > 0 else 0
        print(f"\nRun {i+1} completed in {t.wall_clock_sec:.1f}s ({sec_per_note:.2f}s/note, {tok_per_sec:.1f} tok/s):")
        print(f"  Proposed findings:    {rep['proposed']}")
        print(f"    null placeholders:  {rep['null_placeholder']:4d}  {rep['null_placeholder_pct']:5.1f}%  (no quote -> prompt not followed)")
        print(f"    quote verified:     {t.quote_ok:4d}  {rep['quote_verified_pct_of_quoted']:5.1f}% of quoted | {rep['quote_verified_pct']:.1f}% of all")
        print(f"    quote not in note:  {t.quote_unfound:4d}  {rep['hallucinated_pct_of_quoted']:5.1f}% of quoted | {rep['hallucinated_quote_pct']:.1f}% of all")
        print(f"  Accepted findings:    {rep['accepted']}")
        print(f"     ^ a verified quote means the words are in the note, NOT that they")
        print(f"       support the value. Precision needs the hand-check sheet.")
        print(f"  Invalid values:       {rep['invalid_value']}")
        if rep['unit_converted'] or rep['unit_mismatch'] or rep['unit_ambiguous']:
            print(f"  Unit guard:           {rep['unit_converted']} converted into the "
                  f"schema's unit, {rep['unit_mismatch']} rejected as unconvertible, "
                  f"{rep['unit_ambiguous']} left alone (quote carried several units)")
        if rep['quote_value_mismatch']:
            print(f"  Number not in quote:  {rep['quote_value_mismatch']} rejected - the "
                  f"value is neither the number its quote carries nor its conversion")
        if rep['present_true']:
            print(f"    present=true:       {rep['present_true']:4d}  of which quoted "
                  f"{rep['present_quoted']} ({rep['present_quoted_pct']:.1f}%), "
                  f"unfound {rep['present_quote_unfound']}, unquoted {rep['present_unquoted']}")
        if rep['content_retries'] or rep['unusable_replies']:
            print(f"    content retries:    {rep['content_retries']:4d}  "
                  f"still unusable after retrying: {rep['unusable_replies']}")
        if rep['value_conflicts']:
            print(f"  !! VALUE CONFLICTS:   {rep['value_conflicts']} feature(s) had several "
                  f"verified values and no aggregation rule.")
            print(f"     Withheld from grading rather than guessed at; listed per note in --out.")
        if rep['tokenizer_artifacts']:
            print(f"  !! TOKENIZER ARTIFACTS: {rep['tokenizer_artifacts']} quotes carry corrupt GGUF byte tokens.")
            print(f"     The served weights are broken; these numbers are not a clean measurement.")
        print(f"  Unparseable replies:  {rep['unparseable_replies']}")
        print(f"  Prompt tokens:        {rep['prompt_tokens']} (~{rep['prompt_tokens']//len(notes)} tok/note)")
        print(f"  Completion tokens:    {rep['completion_tokens']} (~{rep['completion_tokens']//len(notes)} tok/note)")

    # grades, from the run-1 features through the real decision tables
    statuses = Counter()
    by_outcome = {num: Counter() for num in outcomes}
    by_selection = {"seeded": Counter(), "holdout": Counter(), "random": Counter()}
    grade_results_detail = {}
    for uid, per in runs[0].items():
        grade_results_detail[uid] = {}
        for num, (feats, present, *_) in per.items():
            res = grade(num, feats, present=bool(present))
            # "the model never saw this outcome" and "the model called it and the
            # tables overruled the call" are different questions. Pooled as one
            # `absent` they send a reviewer to confirm an absence the rule engine
            # produced, on a note where the model actually said present.
            status = harness_status(res.status, present)
            statuses[status] += 1
            by_outcome[num][status] += 1
            by_selection[selection[uid].split(":")[0]][status] += 1
            grade_results_detail[uid][num] = {
                "status": status,
                "rule_status": res.status,
                "grade": res.grade,
                "features": feats,
                "present": present,
                "reason": res.reason,
            }

    print(f"\nGrade status over {len(notes)}x{len(outcomes)} note-outcome pairs:")
    for k, v in statuses.most_common():
        print(f"   {k:14s} {v:3d} ({100*v/(len(notes)*len(outcomes)):.1f}%)")

    cols = ["graded", "grade_set", "cannot_grade", "absent", "refuted", "not_applicable"]
    print(f"\nPer outcome (n={len(notes)} each) - a pooled number hides this shape:")
    print(f"   {'':>3s} {'outcome':30s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for num in outcomes:
        c = by_outcome[num]
        print(f"   {num:>3s} {TABLES[num].name[:30]:30s} "
              + " ".join(f"{c.get(col, 0):>12d}" for col in cols))

    print("\nSeeded vs unstratified holdout - the size of the selection bias:")
    for k, c in by_selection.items():
        if sum(c.values()):
            print(f"   {k:8s} n={sum(c.values()):3d}  " + "  ".join(
                f"{col}={c.get(col, 0)}" for col in cols if c.get(col)))

    consistency_pct = None
    if a.repeat > 1:
        same = tot = 0
        for uid in runs[0]:
            for num in outcomes:
                tot += 1
                same += runs[0][uid][num][0] == runs[1][uid][num][0]
        consistency_pct = round(100 * same / tot, 1)
        print(f"\nTemperature-0 consistency across runs 1-{a.repeat}: {consistency_pct}% "
              f"({same}/{tot} note-outcome pairs identical)")
        if a.concurrency == 1:
            print("   At --concurrency 1 with greedy decoding this is close to a tautology:")
            print("   100% is the expected result and evidences nothing about the model.")
            print("   It is a smoke test for a nondeterministic serving stack, not a metric.")

    # Threshold assessment
    rep0 = tallies[0].report()
    print("\n" + "=" * 70)
    print("Automated Metrics Evaluation (Tasks/medgemma_extraction_test.md §Automated metrics):")
    def band(v, good, workable):
        return f"GOOD (≥{good}%)" if v >= good else (f"WORKABLE ({workable}-{good}%)" if v >= workable else f"CONCERNING (<{workable}%)")
    qv_quoted = rep0["quote_verified_pct_of_quoted"]
    qv_status = band(qv_quoted, 95, 85)
    if consistency_pct is None:
        cs_status = "NOT MEASURED (needs --repeat 2)"
    elif consistency_pct < 90:
        cs_status = "CONCERNING (<90%)"
    elif consistency_pct < 98:
        cs_status = "WORKABLE (90-98%)"
    elif a.concurrency == 1:
        cs_status = "EXPECTED - greedy and unbatched; a near-tautology, not evidence"
    else:
        cs_status = "GOOD (≥98%) and meaningful - it held under batching"
    iv_pct = 100 * rep0["invalid_value"] / (rep0["proposed"] or 1)
    iv_status = "GOOD (≤2%)" if iv_pct <= 2 else ("WORKABLE (2-10%)" if iv_pct <= 10 else "CONCERNING (>10%)")
    print(f"  - Quote-verified % (of quoted proposals):  {qv_quoted}% -> {qv_status}")
    print(f"  - Quote-verified % (of ALL proposals):     {rep0['quote_verified_pct']}%")
    print(f"      NB: quote-verified is a GROUNDING check, not precision. It asks only")
    print(f"      whether the quoted words appear in the note - a quote that does not")
    print(f"      support its value passes it. Precision comes from the hand-check sheet.")
    print(f"  - Null-placeholder rate:   {rep0['null_placeholder_pct']}% -> "
          f"{'GOOD (≤5%)' if rep0['null_placeholder_pct'] <= 5 else 'CONCERNING - the prompt omission rule is being ignored'}")
    cs_value = "  n/a" if consistency_pct is None else f"{consistency_pct}%"
    print(f"  - Run-to-run consistency:  {cs_value} -> {cs_status}")
    print(f"  - Invalid-value rate:      {iv_pct:.1f}% -> {iv_status}")
    print(f"  - Unit conversions:        {rep0['unit_converted']} applied, "
          f"{rep0['unit_mismatch']} unconvertible, "
          f"{rep0['quote_value_mismatch']} value/quote mismatches")
    print(f"  - Value conflicts:         {rep0['value_conflicts']} withheld "
          f"(several verified values, no aggregation rule)")
    print(f"  - Unparseable replies:     {rep0['unparseable_replies']} -> {'GOOD (0)' if rep0['unparseable_replies']==0 else 'CONCERNING'}")
    print("=" * 70)

    if a.out:
        out_path = pathlib.Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_records = []
        for rec in notes:
            uid = rec["patient_uid"]
            per_outcome_details = {}
            for num in outcomes:
                feats, present, reply, detail = runs[0][uid][num]
                per_outcome_details[num] = {
                    "outcome_name": TABLES[num].name,
                    "present": present,
                    "extracted_features": feats,
                    # Every finding that cleared §2, in emission order, and every
                    # feature withheld for disagreeing with itself. The review sheets
                    # read these directly: re-deriving acceptance downstream is how a
                    # hand-check sheet silently stops describing the run it came from.
                    "accepted_findings": detail["accepted"],
                    "conflicts": detail["conflicts"],
                    "grade_result": grade_results_detail[uid][num],
                    "raw_reply": reply,
                }
            detailed_records.append({
                "patient_uid": uid,
                "selection": selection[uid],          # seeded:<outcome> | holdout | random
                "scd_primary": is_scd_primary(rec),   # a label now, not a filter
                "title": rec.get("title", ""),
                "age": rec.get("age"),
                "gender": rec.get("gender"),
                "patient_note": rec["patient"],
                "outcomes": per_outcome_details,
            })

        provenance = {
            "tier": a.tier,
            "weights": tier["hf"],
            "served_as": model,
            "backend": a.backend,
            "quant": served_quant,
            "model_digest": model_digest,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes_count": len(notes),
            "cohort": a.cohort,
            "stratified": a.stratify,
            "holdout_frac": a.holdout_frac if a.stratify else None,
            "outcomes": outcomes,
            "repeat": a.repeat,
            "concurrency": a.concurrency,
            # The rung, and what it turned on. A results file that does not say which
            # prompt produced it cannot be placed on the ladder, and the ladder is the
            # only thing that says which change bought which number.
            "prompt_stage": st.name,
            "prompt_stage_flags": {k: v for k, v in vars(st).items()
                                   if k != "name" and v},
            # True when consistency was measured under batching, which can flip a
            # token at temperature 0 for reasons unrelated to the model.
            "consistency_confounded_by_batching": a.concurrency > 1 and a.repeat > 1,
        }
        # Every artifact derived from this run carries this id. Without it a review
        # sheet and a results file cannot be told apart from a review sheet and a
        # DIFFERENT run's results file, and the reviewer's hours land on the wrong run.
        provenance["run_id"] = "{}-{:08x}".format(
            provenance["timestamp"].replace("-", "").replace(":", "").rstrip("Z"),
            zlib.crc32(json.dumps(provenance, sort_keys=True).encode("utf-8")))

        out_data = {
            "provenance": provenance,
            "profiling": {
                "total_wall_clock_sec": round(tallies[0].wall_clock_sec, 2),
                "sec_per_note": round(tallies[0].wall_clock_sec / len(notes), 2),
                "total_prompt_tokens": tallies[0].prompt_tokens,
                "total_completion_tokens": tallies[0].completion_tokens,
                "prompt_tokens_per_note": tallies[0].prompt_tokens // len(notes),
                "completion_tokens_per_note": tallies[0].completion_tokens // len(notes),
                "completion_tokens_per_sec": round(tallies[0].completion_tokens / tallies[0].wall_clock_sec, 1) if tallies[0].wall_clock_sec > 0 else 0,
            },
            "automated_metrics": {
                "proposed": rep0["proposed"],
                "quoted": rep0["quoted"],
                "quote_verified": rep0["quote_verified"],
                "quote_unfound": rep0["quote_unfound"],
                "null_placeholder": rep0["null_placeholder"],
                "null_placeholder_pct": rep0["null_placeholder_pct"],
                "quote_verified_pct": rep0["quote_verified_pct"],
                "quote_verified_pct_of_quoted": rep0["quote_verified_pct_of_quoted"],
                "hallucinated_quote_pct": rep0["hallucinated_quote_pct"],
                "hallucinated_pct_of_quoted": rep0["hallucinated_pct_of_quoted"],
                "tokenizer_artifacts": rep0["tokenizer_artifacts"],
                "invalid_value_count": rep0["invalid_value"],
                "invalid_value_pct": round(iv_pct, 1),
                "unit_converted": rep0["unit_converted"],
                "unit_ambiguous": rep0["unit_ambiguous"],
                "unit_mismatch": rep0["unit_mismatch"],
                "quote_value_mismatch": rep0["quote_value_mismatch"],
                "value_conflicts": rep0["value_conflicts"],
                "accepted": rep0["accepted"],
                "unknown_feature": rep0["unknown_feature"],
                # `present` gates every other result, so its grounding belongs beside the
                # findings' grounding rather than only in the per-run tally.
                "present_true": rep0["present_true"],
                "present_quoted": rep0["present_quoted"],
                "present_quote_unfound": rep0["present_quote_unfound"],
                "present_unquoted": rep0["present_unquoted"],
                "present_quoted_pct": rep0["present_quoted_pct"],
                "content_retries": rep0["content_retries"],
                "unusable_replies": rep0["unusable_replies"],
                # Quote verification is a grounding check: it asks whether the quoted
                # words are in the note, never whether they support the value. There is
                # no automated precision number here, and there should not appear to be.
                "quote_verified_measures": "quote presence in the note, not value support",
                "unparseable_replies": rep0["unparseable_replies"],
                "run_to_run_consistency_pct": consistency_pct,
            },
            "runs": [t.report() for t in tallies],
            "grade_status": dict(statuses),
            "grade_status_by_outcome": {k: dict(v) for k, v in by_outcome.items()},
            "grade_status_by_selection": {k: dict(v) for k, v in by_selection.items() if sum(v.values())},
            "features_extracted": dict(tallies[0].per_feature),
            "detailed_records": detailed_records,
        }
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nWrote full test results and note texts to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
