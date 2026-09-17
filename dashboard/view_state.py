"""What the dashboard cards render, as plain dicts - no Shiny objects, so it is unit-tested.

The server turns the current inputs into one "view state" dict (explore mode:
a case from a saved results file; live mode: the last analysis) and every
renderer reads only that dict.
"""
from __future__ import annotations

from typing import Any

from dashboard.evaluation import FOCUS_OUTCOMES
from scogs import GradeResult

PENDING_REASON = "Click 'Analyze & Grade Note' to compute deterministic SCOGS grade."


def explore_view_state(run_data: dict | None, uid: str, outcome_num: str) -> dict[str, Any]:
    """-> the view state for one case of a loaded results file; {} before a file is loaded."""
    if not run_data:
        return {}
    record = run_data["records_by_uid"].get(uid, {})
    outcome = record.get("outcomes", {}).get(outcome_num, {})
    grade_result = outcome.get("grade_result", {})
    # PMC-Patients stores age as [[value, unit], ...].
    ages = record.get("age", [])
    age = ages[0][0] if (isinstance(ages, list) and ages and isinstance(ages[0], list)) else None
    return {
        "mode": "explore",
        "patient_uid": uid,
        "title": record.get("title", ""),
        "note_text": record.get("patient_note", ""),
        "outcome_num": outcome_num,
        "outcome_name": outcome.get("outcome_name") or FOCUS_OUTCOMES.get(str(outcome_num), {}).get("name", ""),
        "present": outcome.get("present", False),
        # Results files written before `status` existed fall back to presence.
        "status": grade_result.get("status") or ("graded" if outcome.get("present") else "absent"),
        "grade_result": grade_result,
        "accepted_findings": outcome.get("accepted_findings", []),
        "extracted_features": outcome.get("extracted_features", {}),
        "patient_age": age,
        "patient_sex": record.get("gender"),
    }


def live_view_state(results: dict | None, outcome_id: str, note_text: str,
                    patient_age: Any, patient_sex: str | None) -> dict[str, Any]:
    """-> the view state for the live evaluator: the analysed outcome, or a pending card."""
    item = (results or {}).get(outcome_id)
    if item is None:
        return {
            "mode": "live",
            "patient_uid": "LIVE-CASE",
            "title": "Interactive Live Case",
            "note_text": note_text,
            "outcome_num": outcome_id,
            "outcome_name": FOCUS_OUTCOMES.get(outcome_id, {}).get("name", ""),
            "present": True,
            "status": "pending",
            "grade_result": None,
            "accepted_findings": [],
            "extracted_features": {},
            "patient_age": patient_age,
            "patient_sex": patient_sex,
        }
    return {
        "mode": "live",
        "patient_uid": "LIVE-CASE",
        "title": "Live Note Evaluation",
        "note_text": note_text,
        "outcome_num": outcome_id,
        "outcome_name": item.get("outcome_name", ""),
        "present": item.get("present", True),
        "status": item["status"],
        "grade_result": item.get("grade_result"),
        "accepted_findings": item.get("accepted_findings", []),
        "extracted_features": item.get("extracted_features", {}),
        "patient_age": patient_age,
        "patient_sex": patient_sex,
    }


def grade_details(grade_result: GradeResult | dict | None) -> dict[str, Any]:
    """-> the grade card's detail fields from a live GradeResult, a saved record, or nothing yet.

    Saved records hold only status, grade and reason (experiments/grading.py), so
    their matched row and missing features are empty.
    """
    if isinstance(grade_result, GradeResult):
        return {"grade": grade_result.grade, "grades": grade_result.grades,
                "matched": grade_result.matched, "reason": grade_result.reason,
                "missing": grade_result.missing, "undecided": grade_result.undecided,
                "needs_review": grade_result.needs_review}
    if isinstance(grade_result, dict):
        return {"grade": grade_result.get("grade"), "grades": grade_result.get("grades", ()),
                "matched": grade_result.get("matched"), "reason": grade_result.get("reason"),
                "missing": grade_result.get("missing", ()), "undecided": grade_result.get("undecided", ()),
                "needs_review": grade_result.get("status") in ("grade_set", "cannot_grade")}
    return {"grade": None, "grades": (), "matched": None, "reason": PENDING_REASON,
            "missing": (), "undecided": (), "needs_review": False}
