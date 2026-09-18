from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
VALIDATION_PATH = BASE_DIR / "validation_answers.json"
LIVE_RESULTS_DIR = REPO_ROOT / "results" / "live"


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def normalize_name(value: Any) -> str:
    return normalize_text(value)


def normalize_grade(value: Any):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=8)
def _load_answers_cached(path_str: str, mtime_ns: int):
    p = Path(path_str)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"validation_info": {}, "cases": []}
    if not isinstance(data, dict):
        return {"validation_info": {}, "cases": []}
    if not isinstance(data.get("cases"), list):
        data["cases"] = []
    return data


def load_validation_answers(path: str | Path = VALIDATION_PATH):
    p = Path(path)
    if not p.exists():
        return {"validation_info": {}, "cases": []}
    return _load_answers_cached(str(p.resolve()), p.stat().st_mtime_ns)


def find_validation_case(note: str, answers: dict[str, Any]):
    target = normalize_text(note)
    if not target:
        return None
    for case in answers.get("cases", []) or []:
        if normalize_text(case.get("note", "")) == target:
            return case
    return None


def expected_outcomes(case: dict[str, Any]):
    out = {}
    for item in case.get("expected_outcomes", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("outcome")
        if not name:
            continue
        out[normalize_name(name)] = {
            "grade": normalize_grade(item.get("grade")),
            "expected_evidence": item.get("expected_evidence", []) or [],
        }
    return out


def grade_from_result(result: dict[str, Any]):
    grade = result.get("grade")
    if grade is not None:
        return normalize_grade(grade)
    grade_result = result.get("grade_result")
    if isinstance(grade_result, dict):
        return normalize_grade(grade_result.get("grade"))
    return None


def result_name(result: dict[str, Any]) -> str:
    return str(result.get("outcome") or result.get("outcome_name") or "").strip()


def current_result_map(results):
    out = {}
    iterable = results.values() if isinstance(results, dict) else (results or [])
    for result in iterable:
        if not isinstance(result, dict):
            continue
        name = result_name(result)
        if name:
            out[normalize_name(name)] = result
    return out


def verified_evidence_text(result: dict[str, Any]) -> str:
    texts = []

    raw = result.get("raw_reply")
    if isinstance(raw, dict):
        evidence = raw.get("evidence")
        if isinstance(evidence, list):
            texts.extend(str(x) for x in evidence if x)
        elif evidence:
            texts.append(str(evidence))

    for finding in result.get("accepted_findings", []) or []:
        if isinstance(finding, dict) and finding.get("quote"):
            texts.append(str(finding["quote"]))

    return " ".join(texts).lower()


def individual_evidence_score(result: dict[str, Any], criteria: list[Any]):
    if not criteria:
        return None

    evidence = verified_evidence_text(result)
    supported = 0
    for criterion in criteria:
        c = str(criterion).strip().lower()
        if c and c in evidence:
            supported += 1

    total = len(criteria)
    return {
        "score": round(supported / total * 100, 1),
        "supported": supported,
        "total": total,
    }


def _saved_run(path: Path):
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    detailed = doc.get("detailed_records") or []
    if not detailed or not isinstance(detailed[0], dict):
        return None

    record = detailed[0]
    model = str((doc.get("provenance") or {}).get("model", ""))
    timestamp = str((doc.get("provenance") or {}).get("timestamp", ""))
    note = record.get("patient_note", "")
    outcomes = record.get("outcomes") or {}

    results = []
    for item in outcomes.values():
        if isinstance(item, dict):
            results.append(item)

    return {
        "model": model,
        "timestamp": timestamp,
        "note": note,
        "results": results,
    }


def _snapshot(results_dir: Path):
    if not results_dir.exists():
        return ()
    return tuple(sorted(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size)
        for p in results_dir.glob("*.json")
    ))


@lru_cache(maxsize=8)
def _per_outcome_model_stats_cached(
    answers_path_str: str,
    answers_mtime_ns: int,
    results_dir_str: str,
    snapshot,
    model: str,
):
    answers = load_validation_answers(answers_path_str)
    results_dir = Path(results_dir_str)

    # Deduplicate repeated validation runs: latest run per labeled note.
    latest = {}
    for filename, _, _ in snapshot:
        info = _saved_run(results_dir / filename)
        if not info:
            continue
        if model and info["model"] != model:
            continue

        case = find_validation_case(info["note"], answers)
        if case is None:
            continue

        note_key = normalize_text(info["note"])
        prev = latest.get(note_key)
        if prev is None or info["timestamp"] >= prev["timestamp"]:
            latest[note_key] = info

    stats = {}
    for info in latest.values():
        case = find_validation_case(info["note"], answers)
        if case is None:
            continue

        expected = expected_outcomes(case)
        predicted = current_result_map(info["results"])

        for name, exp in expected.items():
            result = predicted.get(name)
            if result is None:
                continue

            expected_grade = exp.get("grade")
            predicted_grade = grade_from_result(result)
            if expected_grade is None or predicted_grade is None:
                continue

            bucket = stats.setdefault(name, {"correct": 0, "n": 0})
            bucket["n"] += 1
            if predicted_grade == expected_grade:
                bucket["correct"] += 1

    for bucket in stats.values():
        bucket["score"] = (
            round(bucket["correct"] / bucket["n"] * 100, 1)
            if bucket["n"] else None
        )
    return stats


def per_outcome_confidence(
    note: str,
    current_results,
    model: str,
    answers_path: str | Path = VALIDATION_PATH,
    results_dir: str | Path = LIVE_RESULTS_DIR,
):
    """
    Returns validation-derived confidence per outcome.

    Model-level score:
      Grade accuracy for this specific outcome across labeled validation cases.

    Individual-level score:
      Expected-evidence support for the current note, only when that exact note
      exists in the validation answer key. New/unlabeled patients return N/A.
    """
    answers_path = Path(answers_path)
    results_dir = Path(results_dir)
    answers = load_validation_answers(answers_path)

    if not answers.get("cases"):
        return {}

    model_stats = _per_outcome_model_stats_cached(
        str(answers_path.resolve()),
        answers_path.stat().st_mtime_ns if answers_path.exists() else 0,
        str(results_dir.resolve()),
        _snapshot(results_dir),
        str(model or ""),
    )

    matched_case = find_validation_case(note, answers)
    expected_current = expected_outcomes(matched_case) if matched_case else {}
    current_map = current_result_map(current_results)

    all_names = set(current_map) | set(model_stats) | set(expected_current)
    output = {}

    for name in all_names:
        model_bucket = model_stats.get(name, {})
        current_result = current_map.get(name)
        individual = None

        if (
            matched_case
            and current_result is not None
            and name in expected_current
        ):
            individual = individual_evidence_score(
                current_result,
                expected_current[name].get("expected_evidence", []),
            )

        output[name] = {
            "model_score": model_bucket.get("score"),
            "model_n": int(model_bucket.get("n", 0) or 0),
            "individual_score": individual.get("score") if individual else None,
            "individual_supported": individual.get("supported") if individual else None,
            "individual_total": individual.get("total") if individual else None,
            "validation_case": matched_case is not None,
        }

    return output
