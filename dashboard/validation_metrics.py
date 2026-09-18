from __future__ import annotations

import json
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


def load_validation_answers(path: str | Path = VALIDATION_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"validation_info": {}, "cases": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"validation_info": {}, "cases": []}
    if not isinstance(data, dict):
        return {"validation_info": {}, "cases": []}
    if not isinstance(data.get("cases"), list):
        data["cases"] = []
    return data


def _result_grade(result: dict[str, Any]):
    if not isinstance(result, dict):
        return None
    if result.get("grade") is not None:
        return normalize_grade(result.get("grade"))
    grade_result = result.get("grade_result")
    if isinstance(grade_result, dict):
        return normalize_grade(grade_result.get("grade"))
    return None


def _result_name(result: dict[str, Any]) -> str:
    return str(
        result.get("outcome")
        or result.get("outcome_name")
        or ""
    ).strip()


def predicted_outcome_map(results) -> dict[str, int | None]:
    output: dict[str, int | None] = {}

    if isinstance(results, dict):
        iterable = []
        for outcome_id, result in results.items():
            if not isinstance(result, dict):
                continue
            item = dict(result)
            item.setdefault("outcome_id", str(outcome_id))
            iterable.append(item)
    else:
        iterable = results or []

    for result in iterable:
        if not isinstance(result, dict):
            continue

        # Validation compares outcomes that the grader called present.
        present = result.get("present")
        if present is False:
            continue

        name = _result_name(result)
        if not name:
            continue

        output[normalize_name(name)] = _result_grade(result)

    return output


def expected_outcome_map(case: dict[str, Any]) -> dict[str, int | None]:
    output: dict[str, int | None] = {}
    for item in case.get("expected_outcomes", []) or []:
        if not isinstance(item, dict):
            continue
        outcome = item.get("outcome")
        if not outcome:
            continue
        output[normalize_name(outcome)] = normalize_grade(item.get("grade"))
    return output


def compare_case(expected_case: dict[str, Any], predicted_results) -> dict[str, Any]:
    expected = expected_outcome_map(expected_case)
    predicted = predicted_outcome_map(predicted_results)

    expected_names = set(expected)
    predicted_names = set(predicted)
    outcome_exact = expected_names == predicted_names

    common = expected_names & predicted_names
    correct_grades = 0
    undergrades = 0
    overgrades = 0
    grade_comparisons = 0

    for outcome in common:
        expected_grade = expected[outcome]
        predicted_grade = predicted[outcome]
        if expected_grade is None or predicted_grade is None:
            continue

        grade_comparisons += 1
        if predicted_grade == expected_grade:
            correct_grades += 1
        elif predicted_grade < expected_grade:
            undergrades += 1
        else:
            overgrades += 1

    grade_exact = outcome_exact and all(
        expected.get(outcome) == predicted.get(outcome)
        for outcome in expected_names
    )
    exact_match = outcome_exact and grade_exact

    return {
        "outcome_exact": outcome_exact,
        "grade_exact": grade_exact,
        "exact_match": exact_match,
        "correct_grades": correct_grades,
        "grade_comparisons": grade_comparisons,
        "undergrades": undergrades,
        "overgrades": overgrades,
    }


def calculate_validation_metrics(validation_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(validation_results)
    if total_cases == 0:
        return {
            "total_cases": 0,
            "outcome_accuracy": None,
            "grade_accuracy": None,
            "exact_match_accuracy": None,
            "undergrade_rate": None,
            "overgrade_rate": None,
            "average_latency": None,
        }

    outcome_correct = 0
    exact_correct = 0
    correct_grades = 0
    grade_comparisons = 0
    undergrades = 0
    overgrades = 0
    latencies: list[float] = []

    for item in validation_results:
        comparison = item.get("comparison", {}) or {}
        if comparison.get("outcome_exact"):
            outcome_correct += 1
        if comparison.get("exact_match"):
            exact_correct += 1

        correct_grades += int(comparison.get("correct_grades", 0) or 0)
        grade_comparisons += int(comparison.get("grade_comparisons", 0) or 0)
        undergrades += int(comparison.get("undergrades", 0) or 0)
        overgrades += int(comparison.get("overgrades", 0) or 0)

        latency = item.get("total_latency")
        if latency is not None:
            try:
                latencies.append(float(latency))
            except (TypeError, ValueError):
                pass

    outcome_accuracy = outcome_correct / total_cases * 100
    exact_match_accuracy = exact_correct / total_cases * 100

    if grade_comparisons:
        grade_accuracy = correct_grades / grade_comparisons * 100
        undergrade_rate = undergrades / grade_comparisons * 100
        overgrade_rate = overgrades / grade_comparisons * 100
    else:
        grade_accuracy = None
        undergrade_rate = None
        overgrade_rate = None

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies else None
    )

    return {
        "total_cases": total_cases,
        "outcome_accuracy": round(outcome_accuracy, 1),
        "grade_accuracy": round(grade_accuracy, 1) if grade_accuracy is not None else None,
        "exact_match_accuracy": round(exact_match_accuracy, 1),
        "undergrade_rate": round(undergrade_rate, 1) if undergrade_rate is not None else None,
        "overgrade_rate": round(overgrade_rate, 1) if overgrade_rate is not None else None,
        "average_latency": round(average_latency, 2) if average_latency is not None else None,
    }


def collect_verified_evidence(result: dict[str, Any]) -> str:
    texts: list[str] = []

    evidence = result.get("evidence")
    if evidence is None:
        raw_reply = result.get("raw_reply")
        if isinstance(raw_reply, dict):
            evidence = raw_reply.get("evidence")

    if isinstance(evidence, dict):
        for value in evidence.values():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                texts.extend(str(x) for x in value if x)
            else:
                texts.append(str(value))
    elif isinstance(evidence, list):
        texts.extend(str(x) for x in evidence if x)
    elif evidence:
        texts.append(str(evidence))

    # Dashboard-native verified quotes.
    for finding in result.get("accepted_findings", []) or []:
        if isinstance(finding, dict) and finding.get("quote"):
            texts.append(str(finding["quote"]))

    return " ".join(texts).lower()


def calculate_individual_support(
    result: dict[str, Any],
    expected_criteria: list[Any],
):
    if not expected_criteria:
        return None

    evidence_text = collect_verified_evidence(result)
    supported = 0

    for criterion in expected_criteria:
        criterion_text = str(criterion).strip().lower()
        if criterion_text and criterion_text in evidence_text:
            supported += 1

    total = len(expected_criteria)
    return {
        "score": round(supported / total * 100, 1),
        "supported": supported,
        "total": total,
    }


def find_validation_case(
    note: str,
    answers: dict[str, Any] | None = None,
):
    answers = answers if answers is not None else load_validation_answers()
    note_normalized = normalize_text(note)

    for case in answers.get("cases", []) or []:
        if normalize_text(case.get("note", "")) == note_normalized and note_normalized:
            return case

    return None


def _results_iter(graded_results):
    if isinstance(graded_results, dict):
        return [
            dict(result, outcome_id=str(outcome_id))
            for outcome_id, result in graded_results.items()
            if isinstance(result, dict)
        ]
    return [r for r in (graded_results or []) if isinstance(r, dict)]


def calculate_current_patient_support(
    note: str,
    graded_results,
    answers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = answers if answers is not None else load_validation_answers()
    matched_case = find_validation_case(note, answers)

    # This is deliberately N/A for a new patient. The supplied method requires
    # an answer-key case and must not fabricate "ground truth" for unseen notes.
    if matched_case is None:
        return {
            "score": None,
            "supported": None,
            "total": None,
            "validation_case": False,
        }

    results = _results_iter(graded_results)
    supported_total = 0
    criteria_total = 0

    for expected in matched_case.get("expected_outcomes", []) or []:
        expected_name = normalize_name(expected.get("outcome"))
        result_match = None

        for result in results:
            if normalize_name(_result_name(result)) == expected_name:
                result_match = result
                break

        criteria = expected.get("expected_evidence", []) or []

        if result_match is None:
            criteria_total += len(criteria)
            continue

        support = calculate_individual_support(result_match, criteria)
        if support is not None:
            supported_total += support["supported"]
            criteria_total += support["total"]

    score = (
        round(supported_total / criteria_total * 100, 1)
        if criteria_total else None
    )

    return {
        "score": score,
        "supported": supported_total,
        "total": criteria_total,
        "validation_case": True,
    }


def _saved_run_to_predicted_results(run_doc: dict[str, Any]):
    detailed = run_doc.get("detailed_records") or []
    if not detailed or not isinstance(detailed[0], dict):
        return []

    outcomes = detailed[0].get("outcomes") or {}
    results = []

    for outcome_id, item in outcomes.items():
        if not isinstance(item, dict):
            continue
        results.append({
            "outcome": item.get("outcome_name", ""),
            "outcome_id": str(outcome_id),
            "present": bool(item.get("present", False)),
            "grade_result": item.get("grade_result", {}),
            "raw_reply": item.get("raw_reply"),
            "accepted_findings": item.get("accepted_findings", []),
        })

    return results


def collect_saved_validation_results(
    answers: dict[str, Any] | None = None,
    results_dir: str | Path = LIVE_RESULTS_DIR,
    model: str | None = None,
) -> list[dict[str, Any]]:
    answers = answers if answers is not None else load_validation_answers()
    cases = answers.get("cases", []) or []
    if not cases:
        return []

    p = Path(results_dir)
    if not p.exists():
        return []

    # Keep the latest run for each answer-key note to avoid counting repeated
    # reruns of the same validation case as independent labeled cases.
    latest_by_note: dict[str, tuple[str, dict[str, Any]]] = {}

    for file_path in sorted(p.glob("*.json")):
        try:
            run_doc = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        provenance = run_doc.get("provenance", {}) or {}
        if model and str(provenance.get("model", "")) != str(model):
            continue

        detailed = run_doc.get("detailed_records") or []
        if not detailed or not isinstance(detailed[0], dict):
            continue

        note = detailed[0].get("patient_note", "")
        matched_case = find_validation_case(note, answers)
        if matched_case is None:
            continue

        norm_note = normalize_text(note)
        timestamp = str(provenance.get("timestamp") or file_path.stat().st_mtime_ns)
        current = latest_by_note.get(norm_note)
        if current is None or timestamp >= current[0]:
            latest_by_note[norm_note] = (timestamp, run_doc)

    output: list[dict[str, Any]] = []
    for _, run_doc in latest_by_note.values():
        detailed = run_doc.get("detailed_records") or []
        note = detailed[0].get("patient_note", "")
        expected_case = find_validation_case(note, answers)
        if expected_case is None:
            continue

        predicted = _saved_run_to_predicted_results(run_doc)
        profiling = run_doc.get("profiling", {}) or {}
        latency = profiling.get("total_latency_seconds")

        output.append({
            "case_id": expected_case.get("case_id"),
            "comparison": compare_case(expected_case, predicted),
            "total_latency": latency,
        })

    return output


def confidence_summary(
    note: str,
    graded_results,
    model: str | None = None,
    answers_path: str | Path = VALIDATION_PATH,
) -> dict[str, Any]:
    answers = load_validation_answers(answers_path)
    validation_results = collect_saved_validation_results(
        answers=answers,
        model=model,
    )
    model_metrics = calculate_validation_metrics(validation_results)
    individual = calculate_current_patient_support(
        note,
        graded_results,
        answers=answers,
    )

    return {
        "answer_key_available": bool(answers.get("cases")),
        "model": model_metrics,
        "individual": individual,
    }
