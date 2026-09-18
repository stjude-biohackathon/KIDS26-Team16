
from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from conformal_utils import apply_conformal, normalize_probabilities

GATEWAY_URL = "https://bifrost.ai-application.stjude.org/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B-FP8"

RESULT_FIELDS = [
    "case_id",
    "true_outcomes",
    "outcome",
    "outcome_present",
    "grade",
    "status",
    "predicted_class",
    "model_confidence_pct",
    "reasoning",
    "evidence_json",
    "class_probabilities_json",
    "conformal_prediction_set",
    "conformal_target_coverage_pct",
    "conformal_predicted_class_p_value_pct",
    "conformal_confidence_pct",
    "conformal_credibility_pct",
    "conformal_calibration_n",
    "conformal_source",
    "validation_warning",
    "grader_model",
]

REVIEW_FIELDS = [
    "case_id",
    "outcome",
    "model_status",
    "model_grade",
    "reviewed_status",
    "reviewed_grade",
    "review_note",
    "reviewed_at",
]


def load_rules(path: str | Path = "scogs_14_outcomes.json") -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rules = payload.get("outcomes", [])
    if len(rules) != 14:
        raise ValueError(f"Expected 14 SCOGS outcomes, found {len(rules)}.")
    return rules


def valid_classes(rule: dict) -> list[str]:
    labels = ["absent", "insufficient_information"]
    for grade, definition in rule.get("grade_definitions", {}).items():
        if str(definition).strip().upper() != "N/A":
            labels.append(str(grade))
    return labels


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def quote_in_text(text: str, quote: str) -> bool:
    q = normalize_ws(quote).casefold()
    t = normalize_ws(text).casefold()
    return bool(q) and q in t


def extract_text(message) -> str:
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
            else:
                txt = getattr(item, "text", None)
                if isinstance(txt, str):
                    pieces.append(txt)
        if pieces:
            return "\n".join(pieces).strip()

    try:
        dumped = message.model_dump()
    except Exception:
        dumped = {}

    for key in ("content", "output_text", "final", "final_answer"):
        value = dumped.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def clean_json(raw: str):
    text = (raw or "").strip()
    if not text:
        raise ValueError("Model returned empty final content.")

    text = (
        text.replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"No JSON object found. Response began: {text[:200]!r}")


def build_prompt(summary: str, batch_rules: list[dict]) -> str:
    rules_for_prompt = []
    for rule in batch_rules:
        rules_for_prompt.append(
            {
                "outcome": rule["outcome"],
                "diagnostic_criteria": rule.get("diagnostic_criteria", ""),
                "grade_definitions": rule.get("grade_definitions", {}),
                "allowed_classes": valid_classes(rule),
            }
        )

    return f"""
You are a clinical research grader applying the supplied SCOGS severity rubrics.

Evaluate EXACTLY the outcomes supplied in SCOGS_RUBRICS.

RULES
- Use ONLY the clinical summary and the supplied rubric.
- Do not use hidden/reference labels.
- Do not invent missing facts.
- Distinguish current acute events from historical diagnoses.
- A chronic diagnosis can establish presence, but do not force a grade if required grading variables are absent.
- Never assign a grade whose definition is N/A.
- Do not convert oxygen L/min to FiO2.
- Choose the highest exact SCOGS grade supported by the documented evidence.
- Evidence must be copied verbatim from the clinical summary.
- Keep reasoning concise and rubric-based.
- model_confidence_pct is an uncalibrated model certainty score from 0 to 100.
- Return class_scores_pct for every allowed class; values should sum approximately to 100.
- Output valid JSON only. No markdown.

STATUS
If outcome is absent:
  outcome_present=false
  grade=null
  status="absent"

If outcome is present but exact grade cannot be determined:
  outcome_present=true
  grade=null
  status="insufficient_information"

If outcome is present and exactly gradable:
  outcome_present=true
  grade=<integer>
  status="graded"

CLINICAL SUMMARY
{summary}

SCOGS_RUBRICS
{json.dumps(rules_for_prompt, ensure_ascii=False)}

RETURN ONLY:
{{
  "results": [
    {{
      "outcome": "exact canonical outcome name",
      "outcome_present": true,
      "grade": 3,
      "status": "graded",
      "model_confidence_pct": 90,
      "reasoning": "brief rubric-based explanation",
      "evidence": ["exact phrase copied from the clinical summary"],
      "class_scores_pct": {{
        "absent": 1,
        "insufficient_information": 2,
        "1": 2,
        "2": 5,
        "3": 90
      }}
    }}
  ]
}}
""".strip()


def create_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.environ.get("PCAI_API_KEY")
    if not key:
        raise RuntimeError(
            "PCAI_API_KEY is not set in this PowerShell session."
        )

    return OpenAI(
        base_url=GATEWAY_URL,
        api_key=key,
        default_headers={"x-bf-vk": key},
        timeout=600,
    )


def is_timeout(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(
        marker in s
        for marker in ("504", "timeout", "timed out", "request_timed_out")
    )


def call_model(
    client: OpenAI,
    model: str,
    summary: str,
    batch_rules: list[dict],
    max_attempts: int = 2,
) -> list[dict]:
    prompt = build_prompt(summary, batch_rules)
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Apply the supplied SCOGS rubric exactly. "
                            "Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max(1800, 1300 * len(batch_rules)),
            }

            response = client.chat.completions.create(**kwargs)

            raw = extract_text(response.choices[0].message)
            parsed = clean_json(raw)

            if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                return parsed["results"]

            if isinstance(parsed, dict) and "outcome" in parsed:
                return [parsed]

            raise ValueError("JSON response did not contain a results list.")

        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(1.5 * attempt)
                continue

    raise last_exc


def request_resilient(
    client: OpenAI,
    model: str,
    summary: str,
    batch_rules: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Retry a batch, then recursively split it. Never crash on one outcome."""
    try:
        return call_model(client, model, summary, batch_rules), []
    except Exception as exc:
        if len(batch_rules) > 1:
            mid = len(batch_rules) // 2
            left_results, left_errors = request_resilient(
                client, model, summary, batch_rules[:mid]
            )
            right_results, right_errors = request_resilient(
                client, model, summary, batch_rules[mid:]
            )
            return left_results + right_results, left_errors + right_errors

        return [], [
            {
                "outcome": batch_rules[0]["outcome"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]


def selected_class(present: bool, grade, status: str) -> str:
    if not present:
        return "absent"
    if grade is None or status == "insufficient_information":
        return "insufficient_information"
    return str(grade)


def load_calibration(path: str | Path = "conformal_calibration.json"):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def calibration_for_outcome(calibration, outcome: str):
    if not calibration:
        return None

    specific = calibration.get("per_outcome", {}).get(outcome)
    if specific and specific.get("usable"):
        return {
            "scores": specific.get("scores", []),
            "qhat": float(specific["qhat"]),
            "n": int(specific["n"]),
            "source": "outcome_specific",
        }

    global_cal = calibration.get("global", {})
    if global_cal.get("usable"):
        return {
            "scores": global_cal.get("scores", []),
            "qhat": float(global_cal["qhat"]),
            "n": int(global_cal["n"]),
            "source": "global_fallback",
        }

    return None


def normalize_result(
    case_id: str,
    true_outcomes: str,
    summary: str,
    rule: dict,
    item: dict,
    model: str,
    calibration=None,
) -> dict:
    outcome = rule["outcome"]
    labels = valid_classes(rule)
    warnings = []

    present = bool(item.get("outcome_present", False))
    status = str(item.get("status", "")).strip()
    raw_grade = item.get("grade")

    try:
        grade = int(raw_grade) if raw_grade is not None else None
    except Exception:
        grade = None

    allowed_grades = {
        int(x) for x in labels
        if x not in {"absent", "insufficient_information"}
    }

    if not present:
        grade = None
        status = "absent"
    elif grade is None:
        status = "insufficient_information"
    elif grade not in allowed_grades:
        warnings.append(f"Grade {grade} is not valid for this outcome.")
        grade = None
        status = "insufficient_information"
    else:
        status = "graded"

    predicted = selected_class(present, grade, status)

    raw_scores = item.get("class_scores_pct", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    probabilities = normalize_probabilities(raw_scores, labels)

    try:
        confidence = float(item.get("model_confidence_pct"))
        confidence = max(0.0, min(100.0, confidence))
    except Exception:
        confidence = 100.0 * probabilities.get(predicted, 0.0)

    evidence = item.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    verified_evidence = [
        str(q).strip()
        for q in evidence
        if str(q).strip() and quote_in_text(summary, str(q))
    ]
    if len(verified_evidence) < len(evidence):
        warnings.append("Non-verbatim evidence quote(s) removed.")

    conformal_set = []
    target_coverage = ""
    predicted_p = ""
    conformal_confidence = ""
    conformal_credibility = ""
    calibration_n = ""
    conformal_source = ""

    cal_item = calibration_for_outcome(calibration, outcome)
    if cal_item:
        c = apply_conformal(
            probabilities,
            cal_item["scores"],
            cal_item["qhat"],
        )
        conformal_set = c["prediction_set"]
        target_coverage = calibration.get("target_coverage_pct", "")
        predicted_p = round(
            100.0 * c["p_values"].get(predicted, 0.0), 1
        )
        conformal_confidence = round(100.0 * c["confidence"], 1)
        conformal_credibility = round(100.0 * c["credibility"], 1)
        calibration_n = cal_item["n"]
        conformal_source = cal_item["source"]

    return {
        "case_id": case_id,
        "true_outcomes": true_outcomes,
        "outcome": outcome,
        "outcome_present": present,
        "grade": grade if grade is not None else "",
        "status": status,
        "predicted_class": predicted,
        "model_confidence_pct": round(confidence, 1),
        "reasoning": str(item.get("reasoning", "")).strip(),
        "evidence_json": json.dumps(verified_evidence, ensure_ascii=False),
        "class_probabilities_json": json.dumps(
            {k: round(v, 6) for k, v in probabilities.items()},
            ensure_ascii=False,
        ),
        "conformal_prediction_set": json.dumps(conformal_set, ensure_ascii=False),
        "conformal_target_coverage_pct": target_coverage,
        "conformal_predicted_class_p_value_pct": predicted_p,
        "conformal_confidence_pct": conformal_confidence,
        "conformal_credibility_pct": conformal_credibility,
        "conformal_calibration_n": calibration_n,
        "conformal_source": conformal_source,
        "validation_warning": " ".join(warnings),
        "grader_model": model,
    }


def append_rows(path: str | Path, rows: list[dict]):
    if not rows:
        return

    p = Path(path)
    first = not p.exists()
    with p.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if first:
            writer.writeheader()
        writer.writerows(rows)


def replace_case_rows(
    path: str | Path,
    case_id: str,
    replacement_rows: list[dict],
):
    """Replace a case atomically enough for a local research dashboard."""
    p = Path(path)
    existing = []

    if p.exists():
        with p.open("r", newline="", encoding="utf-8-sig") as f:
            existing = [
                row for row in csv.DictReader(f)
                if str(row.get("case_id", "")).strip() != case_id
            ]

    combined = existing + replacement_rows
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(combined)


def read_results(path: str | Path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_reviews(path: str | Path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_review(path: str | Path, row: dict):
    p = Path(path)
    existing = read_reviews(p)
    key = (row["case_id"], row["outcome"])

    retained = [
        r for r in existing
        if (r.get("case_id"), r.get("outcome")) != key
    ]
    retained.append(row)

    with p.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(retained)
