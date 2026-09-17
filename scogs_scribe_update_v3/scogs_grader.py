from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "medgemma:4B"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def normalize_outcome_name(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_model_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"parse_error": True, "raw_text": raw}


def quote_is_in_note(note: str, quote: Any) -> bool:
    """Verify a quote after whitespace normalization, without semantic guessing."""
    if quote is None:
        return False
    q = normalize_ws(str(quote))
    if not q:
        return False
    return q.casefold() in normalize_ws(note).casefold()


def load_rules(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_rule(target_outcome: str, raw_rules: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    wanted = normalize_outcome_name(target_outcome)
    for item in raw_rules.get("outcomes", []):
        if not isinstance(item, dict):
            continue
        names = [item.get("outcome", ""), *item.get("aliases", [])]
        if any(normalize_outcome_name(x) == wanted for x in names if x):
            return item.get("outcome"), item
    return None, None


# -----------------------------------------------------------------------------
# Ollama
# -----------------------------------------------------------------------------

def ollama_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = 240,
    num_predict: int = 900,
) -> Tuple[Dict[str, Any], str]:
    """Call Ollama's native API. No openai package is required."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s while running {model}. "
            "Use a smaller model or increase --timeout."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Confirm Ollama is running and the host is correct."
        ) from exc

    raw = str(body.get("response", "")).strip()
    return clean_model_json(raw), raw


# -----------------------------------------------------------------------------
# ACS extraction + deterministic SCOGS grading
# -----------------------------------------------------------------------------

ACS_FEATURES = [
    "death_attributed_to_acs",
    "fio2_pct",
    "oxygen_flow_lpm",
    "supplemental_oxygen",
    "bipap",
    "high_flow_o2",
    "simple_transfusion",
    "exchange_transfusion",
    "erythropoietin",
    "other_support",
    "vasopressor",
    "intubation",
    "invasive_mechanical_ventilation",
    "other_critical_support",
]


# -----------------------------------------------------------------------------
# Deterministic ACS text extraction (safety net for small local LLMs)
# -----------------------------------------------------------------------------

def _sentences(note: str) -> List[str]:
    # Normalize whitespace only; quote_is_in_note() also normalizes whitespace.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalize_ws(note)) if s.strip()]


def _first_sentence(note: str, *, contains_all: Iterable[str] = (), contains_any: Iterable[str] = ()) -> Optional[str]:
    all_terms = [x.casefold() for x in contains_all]
    any_terms = [x.casefold() for x in contains_any]
    for sentence in _sentences(note):
        low = sentence.casefold()
        if all(term in low for term in all_terms) and (not any_terms or any(term in low for term in any_terms)):
            return sentence
    return None


def extract_acs_deterministic(note: str) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[bool], Optional[str]]:
    """Extract high-value ACS facts directly from note text.

    This layer is deliberately narrow. It only captures facts that can be supported
    by explicit wording in the note. It never converts L/min to FiO2 and never
    assumes an undocumented negative. These facts override conflicting LLM output.
    """
    features = {k: None for k in ACS_FEATURES}
    evidence = {k: None for k in ACS_FEATURES}

    # ----- ACS presence -----
    presence_patterns = [
        r"diagnosed with acute chest syndrome[^.]*",
        r"developed acute chest syndrome[^.]*",
        r"acute chest syndrome \(acs\)[^.]*",
    ]
    present_quote = None
    for pat in presence_patterns:
        m = re.search(pat, normalize_ws(note), flags=re.I)
        if m:
            present_quote = m.group(0).strip()
            break
    outcome_present = True if present_quote else None

    # ----- Oxygen flow in L/min -----
    flow_pat = re.compile(
        r"(?:oxygen(?:\s+requirement)?(?:\s+(?:increased|escalated))?(?:\s+to)?|"
        r"(?:increased|escalating)\s+oxygen\s+requirement\s+to)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:L\s*/\s*min|LPM)",
        flags=re.I,
    )
    m = flow_pat.search(normalize_ws(note))
    if not m:
        # looser fallback: capture a sentence containing both oxygen and an L/min value
        for s in _sentences(note):
            if "oxygen" in s.casefold():
                mm = re.search(r"(\d+(?:\.\d+)?)\s*(?:L\s*/\s*min|LPM)", s, flags=re.I)
                if mm:
                    m = mm
                    quote = s
                    features["oxygen_flow_lpm"] = float(mm.group(1))
                    evidence["oxygen_flow_lpm"] = quote
                    features["supplemental_oxygen"] = True
                    evidence["supplemental_oxygen"] = quote
                    break
    else:
        quote = m.group(0).strip()
        features["oxygen_flow_lpm"] = float(m.group(1))
        evidence["oxygen_flow_lpm"] = quote
        features["supplemental_oxygen"] = True
        evidence["supplemental_oxygen"] = quote

    # ----- Explicit FiO2 only -----
    fio2_patterns = [
        re.compile(r"FiO\s*2\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%", re.I),
        re.compile(r"FiO2\s*(?:of|=|:)?\s*(0\.\d+)", re.I),
    ]
    for pat in fio2_patterns:
        mm = pat.search(normalize_ws(note))
        if mm:
            raw = float(mm.group(1))
            features["fio2_pct"] = raw * 100 if raw <= 1 else raw
            evidence["fio2_pct"] = mm.group(0).strip()
            break

    # ----- High-flow / BiPAP -----
    q = _first_sentence(note, contains_any=["high-flow", "high flow", "hfnc", "hhfnc"] )
    if q:
        features["high_flow_o2"] = True
        evidence["high_flow_o2"] = q
        features["supplemental_oxygen"] = True
        evidence["supplemental_oxygen"] = evidence["supplemental_oxygen"] or q

    q = _first_sentence(note, contains_any=["bipap", "bi-pap"] )
    if q:
        features["bipap"] = True
        evidence["bipap"] = q

    # ----- Intubation / invasive ventilation -----
    q = _first_sentence(note, contains_any=["intubated", "intubation", "mechanical ventilation", "invasive ventilation", "ventilator"] )
    if q:
        if "intubat" in q.casefold():
            features["intubation"] = True
            evidence["intubation"] = q
        if any(term in q.casefold() for term in ["mechanical ventilation", "invasive ventilation", "ventilator"]):
            features["invasive_mechanical_ventilation"] = True
            evidence["invasive_mechanical_ventilation"] = q

    # ----- RBC transfusion -----
    for s in _sentences(note):
        low = s.casefold()
        if "transfus" in low and any(term in low for term in ["prbc", "packed red blood", "red blood cell", "rbc"]):
            if "exchange" in low:
                features["exchange_transfusion"] = True
                evidence["exchange_transfusion"] = s
            else:
                features["simple_transfusion"] = True
                evidence["simple_transfusion"] = s
            break

    # A later explicit pRBC unit can also establish simple transfusion.
    if features["simple_transfusion"] is None:
        q = _first_sentence(note, contains_any=["prbc unit", "pRBC unit", "packed red blood cell unit"] )
        if q:
            features["simple_transfusion"] = True
            evidence["simple_transfusion"] = q

    # ----- Vasopressors / other critical support -----
    q = _first_sentence(note, contains_any=[
        "vasopressor", "vasoactive", "norepinephrine", "epinephrine",
        "phenylephrine", "vasopressin"
    ])
    if q:
        features["vasopressor"] = True
        evidence["vasopressor"] = q

    # ----- Erythropoietin -----
    q = _first_sentence(note, contains_any=["erythropoietin", "epoetin"] )
    if q:
        features["erythropoietin"] = True
        evidence["erythropoietin"] = q

    return features, evidence, outcome_present, present_quote


def merge_acs_features(
    llm_features: Dict[str, Any],
    llm_evidence: Dict[str, Any],
    deterministic_features: Dict[str, Any],
    deterministic_evidence: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Merge verified LLM facts with deterministic text facts.

    Deterministic facts have precedence because they are anchored to explicit strings.
    The LLM can fill only fields that the deterministic layer did not establish.
    """
    features = {k: llm_features.get(k) for k in ACS_FEATURES}
    evidence = {k: llm_evidence.get(k) for k in ACS_FEATURES}
    sources = {k: ("llm_verified" if features.get(k) is not None else None) for k in ACS_FEATURES}

    for k in ACS_FEATURES:
        if deterministic_features.get(k) is not None:
            features[k] = deterministic_features[k]
            evidence[k] = deterministic_evidence.get(k)
            sources[k] = "deterministic_text"

    return features, evidence, sources


def _acs_prompt(note: str, rule: Dict[str, Any], feedback: str = "") -> str:
    grades = rule.get("grade_definitions", {})
    grade_text = "\n".join(f"Grade {k}: {v}" for k, v in grades.items())
    feedback_text = f"\nCORRECTION FEEDBACK FROM THE PREVIOUS ATTEMPT:\n{feedback}\n" if feedback else ""

    # Note first is intentional: small local models tend to retain the clinical facts better.
    return f"""
CLINICAL NOTE
-------------
{note}

TASK
----
Extract ONLY the documented facts needed to grade Acute Chest Syndrome (ACS) with SCOGS.
Do NOT assign the SCOGS grade yourself. Python will apply the grading rule.

Important extraction rules:
- Consider the MOST SEVERE point during the ACS event, not only the discharge condition.
- Every non-null feature MUST have a short supporting quote copied from the clinical note.
- The quote must support that specific feature. Do not reuse an unrelated discharge sentence.
- If a finding is not documented, use null. Do NOT turn an unmentioned finding into false.
- Do NOT infer room air from improvement or discharge.
- FiO2 (%) is NOT the same thing as oxygen flow in L/min.
- If only liters/min are documented, record oxygen_flow_lpm and leave fio2_pct null.
- Set high_flow_o2 true only when the note supports high-flow oxygen; otherwise use null.
- "intubated" supports intubation=true. Do not invent a FiO2 value from intubation.
- A simple packed-RBC transfusion is not an exchange transfusion.
- death_attributed_to_acs is true only when the note supports death due to ACS.
{feedback_text}

OFFICIAL SCOGS ACS RUBRIC
-------------------------
Definition: {rule.get('definition', '')}
Diagnostic criteria: {rule.get('diagnostic_criteria', '')}
{grade_text}

Return ONLY valid JSON in exactly this shape:
{{
  "outcome_present": true,
  "present_quote": "exact quote supporting ACS presence",
  "features": {{
    "death_attributed_to_acs": null,
    "fio2_pct": null,
    "oxygen_flow_lpm": null,
    "supplemental_oxygen": null,
    "bipap": null,
    "high_flow_o2": null,
    "simple_transfusion": null,
    "exchange_transfusion": null,
    "erythropoietin": null,
    "other_support": null,
    "vasopressor": null,
    "intubation": null,
    "invasive_mechanical_ventilation": null,
    "other_critical_support": null
  }},
  "evidence": {{
    "death_attributed_to_acs": null,
    "fio2_pct": null,
    "oxygen_flow_lpm": null,
    "supplemental_oxygen": null,
    "bipap": null,
    "high_flow_o2": null,
    "simple_transfusion": null,
    "exchange_transfusion": null,
    "erythropoietin": null,
    "other_support": null,
    "vasopressor": null,
    "intubation": null,
    "invasive_mechanical_ventilation": null,
    "other_critical_support": null
  }}
}}
""".strip()


def _acs_semantic_problem(feature: str, value: Any, quote: str) -> Optional[str]:
    """Reject feature/quote pairs that are verbatim but do not actually support the field.

    This is deliberately conservative.  Exact quote matching alone is insufficient for small
    models because a real sentence can still be attached to the wrong feature (for example,
    treating "15 L/min" as "FiO2 100%").
    """
    q = normalize_ws(quote).casefold()

    if feature == "fio2_pct":
        # Flow (L/min) is not FiO2. Require an explicit FiO2/percentage statement.
        has_fio2_label = bool(re.search(r"\bfio\s*2\b|\bfio2\b", q))
        has_percent = "%" in q or "percent" in q
        if not (has_fio2_label or has_percent):
            return (
                f"fio2_pct={value!r} is unsupported: the quote does not explicitly state FiO2 or a percent. "
                "Do not convert oxygen flow in L/min into FiO2. If flow is documented, use oxygen_flow_lpm instead."
            )
        try:
            v = float(value)
        except (TypeError, ValueError):
            return f"fio2_pct={value!r} is not numeric."
        nums = [float(x) for x in re.findall(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?)", q)]
        # Accept decimal FiO2 notation (e.g. 0.50) or percent notation (e.g. 50%).
        supported = any(abs(n - v) < 0.01 or abs(n * 100 - v) < 0.01 for n in nums)
        if nums and not supported:
            return f"fio2_pct={v:g} does not match the numeric FiO2/percent in its quote."

    elif feature == "oxygen_flow_lpm":
        if not re.search(r"\b(?:l\s*/\s*min|lpm|lit(?:er|re)s?\s*(?:per|/)\s*min)\b", q):
            return f"oxygen_flow_lpm={value!r} is unsupported: quote does not explicitly state L/min or LPM."
        try:
            v = float(value)
            nums = [float(x) for x in re.findall(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?)", q)]
            if nums and not any(abs(n - v) < 0.01 for n in nums):
                return f"oxygen_flow_lpm={v:g} does not match the numeric flow in its quote."
        except (TypeError, ValueError):
            return f"oxygen_flow_lpm={value!r} is not numeric."

    elif feature == "intubation" and _as_bool(value) is True:
        if "intubat" not in q:
            return "intubation=true requires a quote that explicitly documents intubation/intubated."

    elif feature == "invasive_mechanical_ventilation" and _as_bool(value) is True:
        if not any(term in q for term in ["mechanical ventilation", "invasive ventilation", "ventilator", "intubat"]):
            return "invasive_mechanical_ventilation=true requires explicit invasive ventilation evidence."

    elif feature == "high_flow_o2" and _as_bool(value) is True:
        if not any(term in q for term in ["high flow", "high-flow", "hfnc", "hhfnc"]):
            return "high_flow_o2=true requires explicit high-flow/HFNC wording; a high L/min value alone is not enough."

    elif feature == "bipap" and _as_bool(value) is True:
        if "bipap" not in q and "bi-pap" not in q:
            return "bipap=true requires explicit BiPAP wording."

    elif feature == "simple_transfusion" and _as_bool(value) is True:
        if not ("transfus" in q and any(term in q for term in ["prbc", "packed red", "red blood cell", "rbc"])):
            return "simple_transfusion=true requires an explicit RBC/pRBC transfusion quote."
        if "exchange" in q:
            return "simple_transfusion=true cannot be supported only by an exchange-transfusion quote."

    elif feature == "exchange_transfusion" and _as_bool(value) is True:
        if not ("exchange" in q and "transfus" in q):
            return "exchange_transfusion=true requires explicit exchange-transfusion wording."

    elif feature == "vasopressor" and _as_bool(value) is True:
        if not any(term in q for term in ["vasopressor", "vasoactive", "norepinephrine", "epinephrine", "phenylephrine", "vasopressin"]):
            return "vasopressor=true requires explicit vasopressor/vasoactive treatment evidence."

    return None


def _validated_acs_extraction(note: str, obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    features_in = obj.get("features") if isinstance(obj.get("features"), dict) else {}
    evidence_in = obj.get("evidence") if isinstance(obj.get("evidence"), dict) else {}
    features: Dict[str, Any] = {}
    evidence: Dict[str, Any] = {}
    problems: List[str] = []

    for feature in ACS_FEATURES:
        value = features_in.get(feature)
        quote = evidence_in.get(feature)
        if value is None:
            features[feature] = None
            evidence[feature] = None
            continue
        if not quote_is_in_note(note, quote):
            features[feature] = None
            evidence[feature] = None
            problems.append(f"{feature}: value {value!r} had a missing/non-verbatim quote {quote!r}")
            continue

        clean_quote = normalize_ws(str(quote))
        semantic_problem = _acs_semantic_problem(feature, value, clean_quote)
        if semantic_problem:
            features[feature] = None
            evidence[feature] = None
            problems.append(f"{feature}: {semantic_problem} Quote: {clean_quote!r}")
            continue

        features[feature] = value
        evidence[feature] = clean_quote

    present = obj.get("outcome_present")
    present_quote = obj.get("present_quote")
    if present is not None and not quote_is_in_note(note, present_quote):
        problems.append(f"outcome_present: quote was missing/non-verbatim: {present_quote!r}")
        present = None
        present_quote = None

    return (
        {"outcome_present": present, "present_quote": present_quote, "features": features},
        evidence,
        problems,
    )


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "yes", "1"}:
            return True
        if low in {"false", "no", "0"}:
            return False
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class GradeResult:
    grade: Optional[int]
    status: str
    reason: str


def grade_acs_strict(features: Dict[str, Any], outcome_present: Optional[bool]) -> GradeResult:
    """Apply the ACS SCOGS rubric with a monotonic respiratory-support rule.

    Important: intubation/invasive mechanical ventilation is treated as respiratory
    support at least as intensive as the listed Grade-3 modalities. This does NOT
    invent an FiO2 value; it only preserves the obvious severity ordering of support.
    This operationalization should be clinically reviewed before production use.
    """
    if outcome_present is False:
        return GradeResult(-1, "not_present", "ACS was not supported by the note.")

    death = _as_bool(features.get("death_attributed_to_acs"))
    fio2 = _as_float(features.get("fio2_pct"))
    bipap = _as_bool(features.get("bipap"))
    high_flow = _as_bool(features.get("high_flow_o2"))
    simple_tx = _as_bool(features.get("simple_transfusion"))
    exchange_tx = _as_bool(features.get("exchange_transfusion"))
    supplemental_o2 = _as_bool(features.get("supplemental_oxygen"))
    epo = _as_bool(features.get("erythropoietin"))
    other_support = _as_bool(features.get("other_support"))
    vasopressor = _as_bool(features.get("vasopressor"))
    intubation = _as_bool(features.get("intubation"))
    invasive_mv = _as_bool(features.get("invasive_mechanical_ventilation"))
    other_critical = _as_bool(features.get("other_critical_support"))

    if death is True:
        return GradeResult(5, "graded", "Death attributed to ACS complications was documented.")

    explicit_grade3_trigger = (
        (fio2 is not None and fio2 >= 50)
        or bipap is True
        or high_flow is True
        or exchange_tx is True
    )

    invasive_resp_support = intubation is True or invasive_mv is True
    critical_support = (
        vasopressor is True
        or invasive_resp_support
        or other_critical is True
    )

    # Invasive ventilation/intubation is operationalized as respiratory support
    # at least as intensive as high-flow/BiPAP. We do NOT infer FiO2.
    grade3_or_greater_resp_support = explicit_grade3_trigger or invasive_resp_support

    if critical_support and grade3_or_greater_resp_support:
        if invasive_resp_support and not explicit_grade3_trigger:
            return GradeResult(
                4,
                "graded",
                "Intubation/invasive ventilation was explicitly documented. The grader treats invasive respiratory support as at least as intensive as the Grade-3 respiratory modalities, without inventing an FiO2 value.",
            )
        return GradeResult(
            4,
            "graded",
            "A Grade-3 respiratory/transfusion trigger plus critical cardio-respiratory support was documented.",
        )

    if explicit_grade3_trigger:
        return GradeResult(
            3,
            "graded",
            "At least one Grade-3 trigger was documented (FiO2 >=50%, BiPAP, high-flow O2, or exchange transfusion).",
        )

    grade2_trigger = (
        (fio2 is not None and 21 < fio2 < 50)
        or simple_tx is True
    )
    grade2_exclusion = bipap is True or high_flow is True or exchange_tx is True or invasive_resp_support
    if grade2_trigger and not grade2_exclusion:
        return GradeResult(
            2,
            "graded",
            "FiO2 >21% and <50% or simple transfusion was documented, without higher respiratory support.",
        )

    # Grade 1 requires affirmative evidence that all listed supports were absent.
    grade1_known_absent = all(
        value is False
        for value in [supplemental_o2, simple_tx, exchange_tx, epo, bipap, other_support]
    )
    if grade1_known_absent:
        return GradeResult(1, "graded", "None of the Grade-1 exclusionary supports were documented as required.")

    return GradeResult(
        None,
        "cannot_grade",
        "ACS may be present, but the available documented treatment/severity features are insufficient for a strict SCOGS grade.",
    )

def grade_acs(
    note: str,
    rule: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = 240,
    retry: bool = True,
) -> Dict[str, Any]:
    # Deterministic extraction runs first and acts as a safety net.
    det_features, det_evidence, det_present, det_present_quote = extract_acs_deterministic(note)

    first_obj, first_raw = ollama_json(_acs_prompt(note, rule), model=model, host=host, timeout=timeout)
    validated, llm_evidence, problems = _validated_acs_extraction(note, first_obj)
    attempts = [{"raw": first_raw, "problems": problems}]

    if retry and (problems or first_obj.get("parse_error")):
        feedback = (
            "Your previous extraction failed quote/semantic verification. Correct only unsupported fields. "
            "Use null when the note does not document a value. Do not convert L/min to FiO2. Problems:\n- "
            + "\n- ".join(problems)
        )
        second_obj, second_raw = ollama_json(
            _acs_prompt(note, rule, feedback=feedback),
            model=model,
            host=host,
            timeout=timeout,
        )
        validated2, llm_evidence2, problems2 = _validated_acs_extraction(note, second_obj)
        attempts.append({"raw": second_raw, "problems": problems2})

        # Keep any newly corrected LLM fields.
        for k, v in validated2["features"].items():
            if v is not None:
                validated["features"][k] = v
                llm_evidence[k] = llm_evidence2.get(k)
        if validated2.get("outcome_present") is not None:
            validated["outcome_present"] = validated2.get("outcome_present")
            validated["present_quote"] = validated2.get("present_quote")

    features, evidence, sources = merge_acs_features(
        validated["features"],
        llm_evidence,
        det_features,
        det_evidence,
    )

    # Deterministic presence evidence wins when present.
    outcome_present = det_present if det_present is not None else validated.get("outcome_present")
    present_quote = det_present_quote if det_present_quote else validated.get("present_quote")

    grade_result = grade_acs_strict(features, outcome_present)
    return {
        "outcome": rule.get("outcome", "Acute Chest Syndrome (ACS)"),
        "outcome_present": outcome_present,
        "present_quote": present_quote,
        "features": features,
        "evidence": evidence,
        "feature_sources": sources,
        "grade": grade_result.grade,
        "status": grade_result.status,
        "reason": grade_result.reason,
        "grading_method": "hybrid_deterministic_text_plus_llm_acs_v2",
        "model": model,
        "attempts": attempts,
        "implementation_note": (
            "For ACS, explicit intubation/invasive mechanical ventilation is operationalized as respiratory support "
            "at least as intensive as Grade-3 modalities. No FiO2 is inferred from L/min or intubation."
        ),
    }


# -----------------------------------------------------------------------------
# Generic rubric grader (for the other 52 outcomes)
# -----------------------------------------------------------------------------

def _generic_prompt(note: str, rule: Dict[str, Any], target_outcome: str, feedback: str = "") -> str:
    grade_text = "\n".join(
        f"Grade {grade}: {description}" for grade, description in rule.get("grade_definitions", {}).items()
    )
    feedback_text = f"\nCORRECTION FEEDBACK:\n{feedback}\n" if feedback else ""
    return f"""
CLINICAL NOTE
-------------
{note}

TASK
----
Grade ONE sickle cell disease outcome using ONLY the SCOGS rubric below.
Use the most severe documented point in the event. Do not assume missing facts are negative.
Evidence items must be short, verbatim quotes from the clinical note.
If the outcome is absent, use grade -1.
If it appears present but the exact SCOGS grade cannot be supported, use grade 0.
{feedback_text}

TARGET OUTCOME: {target_outcome}
DEFINITION: {rule.get('definition', '')}
DIAGNOSTIC CRITERIA: {rule.get('diagnostic_criteria', '')}
SCOGS GRADES:
{grade_text}

Return ONLY JSON:
{{
  "outcome_present": true,
  "grade": 0,
  "confidence": 0,
  "evidence": ["short exact quote"],
  "reasoning": "brief rubric-based explanation"
}}
""".strip()


def _verify_generic(note: str, obj: Dict[str, Any]) -> List[str]:
    problems = []
    evidence = obj.get("evidence", [])
    if not isinstance(evidence, list):
        return ["evidence must be a list"]
    for i, quote in enumerate(evidence):
        if not quote_is_in_note(note, quote):
            problems.append(f"evidence[{i}] is not a verbatim/whitespace-normalized quote: {quote!r}")
    return problems


def grade_generic(
    note: str,
    rule: Dict[str, Any],
    target_outcome: str,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = 240,
    retry: bool = True,
) -> Dict[str, Any]:
    obj, raw = ollama_json(_generic_prompt(note, rule, target_outcome), model=model, host=host, timeout=timeout)
    problems = _verify_generic(note, obj)
    attempts = [{"raw": raw, "problems": problems}]

    if retry and (problems or obj.get("parse_error")):
        feedback = "The previous answer used unsupported evidence. Use only exact quotes. Problems:\n- " + "\n- ".join(problems)
        obj2, raw2 = ollama_json(
            _generic_prompt(note, rule, target_outcome, feedback=feedback),
            model=model,
            host=host,
            timeout=timeout,
        )
        problems2 = _verify_generic(note, obj2)
        attempts.append({"raw": raw2, "problems": problems2})
        if not problems2 and not obj2.get("parse_error"):
            obj, problems = obj2, problems2

    if problems:
        return {
            "outcome": rule.get("outcome", target_outcome),
            "outcome_present": obj.get("outcome_present"),
            "grade": None,
            "confidence": obj.get("confidence"),
            "evidence": [],
            "reasoning": obj.get("reasoning", ""),
            "status": "cannot_grade",
            "reason": "Model evidence failed exact-quote verification.",
            "grading_method": "model_rubric_with_quote_verification",
            "model": model,
            "attempts": attempts,
        }

    return {
        "outcome": rule.get("outcome", target_outcome),
        "outcome_present": obj.get("outcome_present"),
        "grade": obj.get("grade"),
        "confidence": obj.get("confidence"),
        "evidence": obj.get("evidence", []),
        "reasoning": obj.get("reasoning", ""),
        "status": "graded" if obj.get("grade") is not None else "cannot_grade",
        "reason": "Generic SCOGS rubric grading with verified note quotes.",
        "grading_method": "model_rubric_with_quote_verification",
        "model": model,
        "attempts": attempts,
    }


def grade_note_outcome(
    note: str,
    target_outcome: str,
    rules_path: str | Path,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: int = 240,
    retry: bool = True,
) -> Dict[str, Any]:
    raw_rules = load_rules(rules_path)
    matched_name, rule = find_rule(target_outcome, raw_rules)
    if rule is None:
        raise ValueError(f"No SCOGS rule matched outcome: {target_outcome}")

    if int(rule.get("id", -999)) == 48 or normalize_outcome_name(matched_name or "") == "acute chest syndrome":
        return grade_acs(note, rule, model=model, host=host, timeout=timeout, retry=retry)
    return grade_generic(
        note,
        rule,
        target_outcome=matched_name or target_outcome,
        model=model,
        host=host,
        timeout=timeout,
        retry=retry,
    )


if __name__ == "__main__":
    print("This module is intended to be imported by demo_acs.py or grade_synthetic.py.")
