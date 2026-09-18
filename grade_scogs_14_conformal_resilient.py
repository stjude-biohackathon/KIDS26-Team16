from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

from conformal_utils import apply_conformal, normalize_probabilities

GATEWAY_URL = "https://bifrost.ai-application.stjude.org/v1"
DEFAULT_MODEL = "gpt-oss-120b"

FIELDS = [
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

ERROR_FIELDS = [
    "case_id", "outcome", "attempts", "last_error"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="SCD_summaries.csv")
    p.add_argument("--rules", default="scogs_14_outcomes.json")
    p.add_argument("--output-csv", default="SCD_grades_14.csv")
    p.add_argument("--output-json", default="SCD_grades_14.json")
    p.add_argument("--errors-csv", default="SCD_grader_errors.csv")
    p.add_argument("--calibration", default="conformal_calibration.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--restart", action="store_true")
    return p.parse_args()


def normalize_ws(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


def quote_in_text(text, quote):
    q = normalize_ws(quote).casefold()
    t = normalize_ws(text).casefold()
    return bool(q) and q in t


def valid_classes(rule):
    labels = ["absent", "insufficient_information"]
    for grade, definition in rule.get("grade_definitions", {}).items():
        if str(definition).strip().upper() != "N/A":
            labels.append(str(grade))
    return labels


def selected_class(present, grade, status):
    if status == "model_error":
        return ""
    if not present:
        return "absent"
    if grade is None or status == "insufficient_information":
        return "insufficient_information"
    return str(grade)


def extract_text(message):
    """Get final text across common OpenAI-compatible message shapes."""
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    pieces.append(item["text"])
            else:
                txt = getattr(item, "text", None)
                if isinstance(txt, str):
                    pieces.append(txt)
        joined = "\n".join(pieces).strip()
        if joined:
            return joined

    # Some OpenAI-compatible providers expose final text in extra fields.
    for attr in ("output_text", "final", "final_answer"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    try:
        dumped = message.model_dump()
    except Exception:
        dumped = {}

    for key in ("content", "output_text", "final", "final_answer"):
        value = dumped.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def clean_json(raw):
    text = (raw or "").strip()
    if not text:
        raise ValueError("Model returned empty final content.")

    text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    # First try direct parse.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Then take the largest JSON-looking object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError(f"No JSON object found. Raw response begins: {text[:300]!r}")


def build_prompt(summary, batch_rules):
    compact_rules = []
    for rule in batch_rules:
        compact_rules.append({
            "outcome": rule["outcome"],
            "diagnostic_criteria": rule.get("diagnostic_criteria", ""),
            "grade_definitions": rule.get("grade_definitions", {}),
            "allowed_classes": valid_classes(rule),
        })

    return f"""
You are grading sickle cell disease outcomes using the supplied SCOGS rubrics.

Evaluate EXACTLY the outcomes in SCOGS_RUBRICS. Do not evaluate any others.

Important:
- Use only the clinical summary and supplied rubric.
- Do not invent missing facts.
- Distinguish current events from historical diagnoses.
- A chronic diagnosis can establish presence, but do not force a grade when grade-defining details are absent.
- Never assign a grade whose rubric is N/A.
- Evidence must be an exact quote from the clinical summary.
- model_confidence_pct is uncalibrated certainty from 0 to 100.
- Return scores for every allowed class. They should sum to approximately 100.
- Keep reasoning brief.
- Output JSON only. No markdown and no text before or after JSON.

Status rules:
absent:
  outcome_present=false, grade=null, status="absent"

present but exact SCOGS grade cannot be determined:
  outcome_present=true, grade=null, status="insufficient_information"

present and exactly gradable:
  outcome_present=true, grade=<integer>, status="graded"

CLINICAL_SUMMARY:
{summary}

SCOGS_RUBRICS:
{json.dumps(compact_rules, ensure_ascii=False)}

Return:
{{
  "results": [
    {{
      "outcome": "exact canonical outcome name",
      "outcome_present": true,
      "grade": 3,
      "status": "graded",
      "model_confidence_pct": 90,
      "reasoning": "brief rubric-based explanation",
      "evidence": ["exact quote from clinical summary"],
      "class_scores_pct": {{
        "absent": 1,
        "insufficient_information": 2,
        "1": 1,
        "2": 2,
        "3": 94
      }}
    }}
  ]
}}
""".strip()


def is_timeout(exc):
    s = str(exc).lower()
    return any(k in s for k in ("504", "timeout", "timed out", "request_timed_out"))


def call_model(client, model, summary, batch_rules, max_attempts=2):
    """Call model. Returns list of results or raises after retries."""
    prompt = build_prompt(summary, batch_rules)
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Apply the supplied SCOGS rubric exactly. "
                            "Return only a valid JSON object."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                # Give gpt-oss room to reason and still emit final JSON.
                max_tokens=max(2500, 1400 * len(batch_rules)),
            )

            # gpt-oss supports lower reasoning effort on compatible gateways.
            # If a gateway rejects this parameter, retry without it.
            try:
                response = client.chat.completions.create(
                    **kwargs,
                    reasoning_effort="low",
                )
            except TypeError:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                # If the provider rejects reasoning_effort, retry same request without it.
                if "reasoning_effort" in str(e).lower() and not is_timeout(e):
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            message = response.choices[0].message
            raw = extract_text(message)
            parsed = clean_json(raw)

            if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
                return parsed["results"]

            # Permit a single result object for one-outcome requests.
            if isinstance(parsed, dict) and "outcome" in parsed:
                return [parsed]

            raise ValueError("JSON was valid but did not contain a results list.")

        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                reason = "timeout" if is_timeout(exc) else "empty/malformed response"
                print(f"    {reason}; retrying ({attempt}/{max_attempts})", flush=True)
                time.sleep(2 * attempt)

    raise last_exc


def request_resilient(client, model, summary, batch_rules):
    """
    Try a batch. If it still fails after retry, recursively split.
    Single-outcome failures are returned separately instead of crashing the run.
    """
    names = [r["outcome"] for r in batch_rules]

    try:
        return call_model(client, model, summary, batch_rules), []
    except Exception as exc:
        if len(batch_rules) > 1:
            mid = len(batch_rules) // 2
            print(
                f"    batch failed for {names}; splitting into "
                f"{mid} + {len(batch_rules) - mid}",
                flush=True,
            )
            left_results, left_errors = request_resilient(
                client, model, summary, batch_rules[:mid]
            )
            right_results, right_errors = request_resilient(
                client, model, summary, batch_rules[mid:]
            )
            return left_results + right_results, left_errors + right_errors

        return [], [{
            "outcome": names[0],
            "error": f"{type(exc).__name__}: {exc}",
        }]


def append_csv(path, fieldnames, rows):
    if not rows:
        return
    first = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if first:
            w.writeheader()
        w.writerows(rows)


def completed_pairs(path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            c = str(row.get("case_id", "")).strip()
            o = str(row.get("outcome", "")).strip()
            if c and o:
                done.add((c, o))
    return done


def write_json_output(csv_path, json_path):
    if not csv_path.exists():
        json_path.write_text("[]", encoding="utf-8")
        return

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    cases = {}
    for row in rows:
        cid = row["case_id"]
        cases.setdefault(
            cid,
            {"case_id": cid, "true_outcomes": row.get("true_outcomes", ""), "results": []},
        )

        def j(name, default):
            try:
                return json.loads(row.get(name, "") or json.dumps(default))
            except Exception:
                return default

        cases[cid]["results"].append({
            "outcome": row.get("outcome", ""),
            "outcome_present": row.get("outcome_present", "").lower() == "true",
            "grade": int(row["grade"]) if str(row.get("grade", "")).isdigit() else None,
            "status": row.get("status", ""),
            "predicted_class": row.get("predicted_class", ""),
            "model_confidence_pct": (
                float(row["model_confidence_pct"])
                if row.get("model_confidence_pct") not in ("", None) else None
            ),
            "reasoning": row.get("reasoning", ""),
            "evidence": j("evidence_json", []),
            "class_probabilities": j("class_probabilities_json", {}),
            "conformal_prediction_set": j("conformal_prediction_set", []),
            "validation_warning": row.get("validation_warning", ""),
        })

    json_path.write_text(
        json.dumps(list(cases.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_calibration(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def calibration_for_outcome(calibration, outcome):
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

    glob = calibration.get("global", {})
    if glob.get("usable"):
        return {
            "scores": glob.get("scores", []),
            "qhat": float(glob["qhat"]),
            "n": int(glob["n"]),
            "source": "global_fallback",
        }

    return None


def result_to_row(case_id, true_outcomes, summary, rule, item, model, calibration):
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
        int(x) for x in labels if x not in {"absent", "insufficient_information"}
    }

    if not present:
        grade = None
        status = "absent"
    elif grade is None:
        status = "insufficient_information"
    elif grade not in allowed_grades:
        warnings.append(f"Returned Grade {grade} is not valid for this outcome.")
        grade = None
        status = "insufficient_information"
    else:
        status = "graded"

    pred = selected_class(present, grade, status)

    raw_scores = item.get("class_scores_pct", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    probs = normalize_probabilities(raw_scores, labels)

    try:
        conf = float(item.get("model_confidence_pct", ""))
        conf = max(0.0, min(100.0, conf))
    except Exception:
        conf = 100.0 * probs.get(pred, 0.0)

    evidence = item.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    verified = [
        str(q).strip()
        for q in evidence
        if str(q).strip() and quote_in_text(summary, q)
    ]
    if len(verified) < len(evidence):
        warnings.append("Non-verbatim evidence quote(s) removed.")

    cset = []
    target = ""
    pred_p = ""
    cconf = ""
    ccred = ""
    cn = ""
    csource = ""

    citem = calibration_for_outcome(calibration, outcome)
    if citem:
        cres = apply_conformal(probs, citem["scores"], citem["qhat"])
        cset = cres["prediction_set"]
        target = calibration.get("target_coverage_pct", "")
        pred_p = round(100 * cres["p_values"].get(pred, 0.0), 1)
        cconf = round(100 * cres["confidence"], 1)
        ccred = round(100 * cres["credibility"], 1)
        cn = citem["n"]
        csource = citem["source"]

    return {
        "case_id": case_id,
        "true_outcomes": true_outcomes,
        "outcome": outcome,
        "outcome_present": present,
        "grade": grade if grade is not None else "",
        "status": status,
        "predicted_class": pred,
        "model_confidence_pct": round(conf, 1),
        "reasoning": str(item.get("reasoning", "")).strip(),
        "evidence_json": json.dumps(verified, ensure_ascii=False),
        "class_probabilities_json": json.dumps(
            {k: round(v, 6) for k, v in probs.items()},
            ensure_ascii=False,
        ),
        "conformal_prediction_set": json.dumps(cset, ensure_ascii=False),
        "conformal_target_coverage_pct": target,
        "conformal_predicted_class_p_value_pct": pred_p,
        "conformal_confidence_pct": cconf,
        "conformal_credibility_pct": ccred,
        "conformal_calibration_n": cn,
        "conformal_source": csource,
        "validation_warning": " ".join(warnings),
        "grader_model": model,
    }


def main():
    args = parse_args()

    key = os.environ.get("PCAI_API_KEY")
    if not key:
        raise RuntimeError(
            "PCAI_API_KEY is not set. In PowerShell set it with:\n"
            '$env:PCAI_API_KEY="YOUR_VIRTUAL_KEY"'
        )

    inp = Path(args.input)
    rules_path = Path(args.rules)
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    errors_csv = Path(args.errors_csv)
    cal_path = Path(args.calibration)

    if not inp.exists():
        raise FileNotFoundError(f"Could not find {inp}")
    if not rules_path.exists():
        raise FileNotFoundError(f"Could not find {rules_path}")

    with inp.open("r", newline="", encoding="utf-8-sig") as f:
        cases = list(csv.DictReader(f))

    if not cases:
        raise ValueError("Input CSV is empty.")

    required = {"case_id", "summary"}
    missing = required - set(cases[0].keys())
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    with rules_path.open("r", encoding="utf-8-sig") as f:
        rule_file = json.load(f)
    rules = rule_file.get("outcomes", [])
    if len(rules) != 14:
        raise ValueError(f"Expected 14 outcomes, found {len(rules)}.")

    if args.limit is not None:
        cases = cases[:args.limit]

    if args.restart:
        for p in (out_csv, out_json, errors_csv):
            if p.exists():
                p.unlink()

    calibration = load_calibration(cal_path)
    if calibration:
        print(
            f"Conformal calibration loaded. "
            f"Target coverage: {calibration.get('target_coverage_pct')}%"
        )
    else:
        print(
            "No conformal_calibration.json found. Raw model scores will be saved "
            "and conformal fields will remain blank."
        )

    client = OpenAI(
        base_url=GATEWAY_URL,
        api_key=key,
        default_headers={"x-bf-vk": key},
        timeout=600,
    )

    done = completed_pairs(out_csv)
    rule_map = {r["outcome"]: r for r in rules}

    print(f"Cases in this run: {len(cases)}")
    print(f"Grader model: {args.model}")
    print(
        f"Initial outcomes/request: {args.batch_size}; "
        "failed batches are automatically split."
    )
    print("Checkpointing is per case + outcome, so successful work is preserved.")

    for case_i, case in enumerate(cases, 1):
        cid = str(case.get("case_id", "")).strip() or f"CASE{case_i:04d}"
        true_outcomes = str(case.get("true_outcomes", "")).strip()
        summary = str(case.get("summary", "")).strip()

        remaining = [r for r in rules if (cid, r["outcome"]) not in done]

        if not remaining:
            print(f"[{case_i}/{len(cases)}] {cid}: all 14 already done")
            continue

        print(
            f"[{case_i}/{len(cases)}] {cid}: "
            f"{len(remaining)} outcome(s) remaining",
            flush=True,
        )

        bs = max(1, args.batch_size)
        for start in range(0, len(remaining), bs):
            batch = remaining[start:start + bs]
            print(
                "  grading: " + ", ".join(r["outcome"] for r in batch),
                flush=True,
            )

            results, errors = request_resilient(
                client, args.model, summary, batch
            )

            returned = {
                str(item.get("outcome", "")).strip(): item
                for item in results
                if isinstance(item, dict)
            }

            rows = []
            for rule in batch:
                outcome = rule["outcome"]
                item = returned.get(outcome)
                if item is None:
                    continue

                rows.append(
                    result_to_row(
                        cid,
                        true_outcomes,
                        summary,
                        rule,
                        item,
                        args.model,
                        calibration,
                    )
                )

            append_csv(out_csv, FIELDS, rows)
            for row in rows:
                done.add((row["case_id"], row["outcome"]))

            error_rows = []
            for e in errors:
                error_rows.append({
                    "case_id": cid,
                    "outcome": e["outcome"],
                    "attempts": 2,
                    "last_error": e["error"],
                })
                print(
                    f"    SKIPPED for now: {e['outcome']} -> {e['error'][:180]}",
                    flush=True,
                )
            append_csv(errors_csv, ERROR_FIELDS, error_rows)

            write_json_output(out_csv, out_json)

    print("Done.")
    print(f"CSV:    {out_csv}")
    print(f"JSON:   {out_json}")
    if errors_csv.exists():
        print(
            f"Errors: {errors_csv}\n"
            "Any missing case/outcome pairs can be retried by running the same "
            "command again WITHOUT --restart."
        )


if __name__ == "__main__":
    main()
