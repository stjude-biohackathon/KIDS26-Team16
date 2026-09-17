"""Grade a note in the dashboard - through the same code path as the CLI.

Two modes, one grading function (`experiments.grading.grade_outcome`):
  * live: the note goes through `experiments.medgemma_extraction.run()` - the
    CLI's prompts, retries and verification - so only findings whose quote is in
    the note are graded;
  * manual: a clinician types feature values; nothing is extracted or verified.

Everything here is plain Python with no Shiny objects, so it is unit-tested
directly (tests/test_dashboard_evaluation.py).
"""
from __future__ import annotations

from typing import Any

from experiments.grading import OutcomeGrade, grade_outcome
from experiments.medgemma_extraction import DEFAULT_PROMPT_STAGE, run, stage
from experiments.ollama_backend import DEFAULT_HOST, DEFAULT_MODEL, preflight
from experiments.verification import Tally
from scogs import ABSENT, CANNOT_GRADE, GRADE_SET, GRADED, NOT_APPLICABLE, TABLES, GradeResult
from scogs.features import FEATURES
from scogs.predicates import UNKNOWN

# 14 Focus Outcomes supported natively
FOCUS_OUTCOMES: dict[str, dict[str, str]] = {
    "10": {"name": "Chronic Sickle Pain", "organ_system": "Pain / Neurological", "acuity": "Chronic"},
    "11": {"name": "Cognitive Dysfunction", "organ_system": "Neurological", "acuity": "Chronic"},
    "12": {"name": "Elevated Transcranial Doppler (TCD) Velocity", "organ_system": "Neurological", "acuity": "Chronic / Screening"},
    "15": {"name": "Stroke", "organ_system": "Neurological", "acuity": "Acute"},
    "17": {"name": "Sickle Cell Retinopathy", "organ_system": "Ophthalmologic", "acuity": "Chronic"},
    "21": {"name": "Chronic Kidney Disease (CKD)", "organ_system": "Renal", "acuity": "Chronic"},
    "24": {"name": "Acute Ischemic Priapism", "organ_system": "Genitourinary (Male)", "acuity": "Acute"},
    "28": {"name": "Acute Sickle Cell Pain Episode (VOC)", "organ_system": "Pain / Vascular", "acuity": "Acute"},
    "29": {"name": "Splenic Sequestration", "organ_system": "Hematologic", "acuity": "Acute"},
    "39": {"name": "Avascular Necrosis (AVN)", "organ_system": "Musculoskeletal", "acuity": "Chronic"},
    "40": {"name": "Chronic Leg Ulcer", "organ_system": "Dermatologic", "acuity": "Chronic"},
    "47": {"name": "Depression", "organ_system": "Psychiatric", "acuity": "Chronic"},
    "48": {"name": "Acute Chest Syndrome (ACS)", "organ_system": "Pulmonary", "acuity": "Acute"},
    "49": {"name": "Asthma", "organ_system": "Pulmonary", "acuity": "Chronic"},
}

# The patient id a pasted note gets inside the harness; it never leaves this module.
LIVE_NOTE_UID = "LIVE-CASE"

# Dashboard and CSV sex codes -> the schema's `patient_sex` values (scogs/features.py).
# Anything else ("unknown", blank) is left out so the rules see it as UNKNOWN.
SEX_TO_SCHEMA = {"m": "male", "male": "male", "f": "female", "female": "female"}


def normalize_outcome_id(raw: str | int) -> str:
    """-> a two-digit outcome id ("5" -> "05"); non-numeric ids pass through."""
    return f"{int(raw):02d}" if str(raw).isdigit() else str(raw)


def outcome_display_name(outcome: str) -> str:
    """-> the dashboard's name for an outcome, falling back to the rubric's."""
    table = TABLES.get(outcome)
    return FOCUS_OUTCOMES.get(outcome, {}).get("name") or getattr(table, "name", f"Outcome {outcome}")


def clinician_context(patient_sex: str | None, patient_age: Any) -> dict[str, Any]:
    """-> the age and sex a clinician entered, as schema features; unknowns are omitted.

    Omitting a value (rather than passing "unknown" or None) keeps it UNKNOWN in
    the rule engine, so it can never silently exclude an outcome.
    """
    context: dict[str, Any] = {}
    sex = SEX_TO_SCHEMA.get(str(patient_sex or "").strip().lower())
    if sex:
        context["patient_sex"] = sex
    try:
        context["patient_age"] = float(patient_age)
    except (TypeError, ValueError):
        pass
    return context


def is_derived(feature: str) -> bool:
    """True when the rules compute this feature from others instead of reading it from the note."""
    spec = FEATURES.get(feature, {})
    return bool(spec.get("derived") or spec.get("computed_from"))


def evaluate_clinical_features(outcome_num: str | int, feature_dict: dict[str, Any],
                               patient_sex: str = "unknown", patient_age: Any = None,
                               present: bool = True) -> OutcomeGrade:
    """-> the grade for feature values typed in by a clinician (no model involved)."""
    outcome = normalize_outcome_id(outcome_num)
    features = {**feature_dict, **clinician_context(patient_sex, patient_age)}
    try:
        return grade_outcome(outcome, features, present)
    except KeyError:
        return cannot_grade(outcome, f"Unknown outcome number {outcome}")
    except ValueError as exc:
        return cannot_grade(outcome, f"Evaluation error: {exc}")


def cannot_grade(outcome: str, reason: str) -> OutcomeGrade:
    """-> a CANNOT_GRADE result carrying `reason`, for input the rules could not evaluate."""
    result = GradeResult(outcome=outcome, status=CANNOT_GRADE, reason=reason)
    return OutcomeGrade(result, CANNOT_GRADE, UNKNOWN, UNKNOWN, None)


def extract_with_harness(note_text: str, outcomes: list[str], *, backend: str = "ollama",
                         model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                         prompt_stage: str = DEFAULT_PROMPT_STAGE, patient_sex: str = "unknown",
                         patient_age: Any = None) -> dict[str, dict[str, Any]]:
    """-> {outcome: result item} for one pasted note, extracted exactly as the CLI does.

    Runs the note as a one-note batch through `run()`: the CLI's prompt stage,
    constrained decoding, retries, quote grounding, unit guards and conflict
    reconciliation. A finding whose quote is not in the note never reaches
    grading. Raises RuntimeError when the backend fails.
    """
    outcome_ids = [normalize_outcome_id(outcome) for outcome in outcomes]
    note = {"patient_uid": LIVE_NOTE_UID, "patient": note_text}
    results, _ = run([note], outcome_ids, backend, model, host, Tally(), st=stage(prompt_stage))
    context = clinician_context(patient_sex, patient_age)

    items = {}
    for outcome, (features, present, reply, detail) in results[LIVE_NOTE_UID].items():
        # Age and sex entered by the clinician override anything extracted.
        features = {**features, **context}
        graded = grade_outcome(outcome, features, present)
        items[outcome] = {
            "outcome_name": outcome_display_name(outcome),
            "present": present,
            "extracted_features": features,
            "accepted_findings": detail["accepted"],
            "conflicts": detail["conflicts"],
            "status": graded.status,
            "grade_result": graded.result,
            "raw_reply": reply,
        }
    return items


def extract_and_grade_note(note_text: str, outcomes: list[str],
                           patient_context: dict[str, Any] | None = None, use_ollama: bool = False,
                           manual_features: dict[str, dict[str, Any]] | None = None,
                           model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> dict[str, dict[str, Any]]:
    """-> {outcome: result item} for the live evaluator, from the model or from typed values.

    Every item has the same keys in both modes, so the UI renders them identically.
    """
    context = patient_context or {}
    sex, age = context.get("patient_sex", "unknown"), context.get("patient_age")
    outcome_ids = [normalize_outcome_id(outcome) for outcome in outcomes]
    if use_ollama:
        try:
            return extract_with_harness(note_text, outcome_ids, model=model, host=host,
                                        patient_sex=sex, patient_age=age)
        except Exception as exc:
            # Any backend failure is shown on the card; it must not crash the session.
            return {outcome: failed_item(outcome, exc) for outcome in outcome_ids}
    typed = manual_features or {}
    return {outcome: manual_item(outcome, typed.get(outcome, {}), sex, age) for outcome in outcome_ids}


def manual_item(outcome: str, typed: dict[str, Any], patient_sex: str, patient_age: Any) -> dict[str, Any]:
    """-> a result item for clinician-typed feature values (`present` may be among them)."""
    features = dict(typed)
    present = features.pop("present", True)
    graded = evaluate_clinical_features(outcome, features, patient_sex, patient_age, present)
    findings = [{"feature": name, "value": value, "quote": None,
                 "unit": FEATURES.get(name, {}).get("unit"), "source": "manual"}
                for name, value in features.items()]
    return {
        "outcome_name": outcome_display_name(outcome),
        "present": present,
        "extracted_features": features,
        "accepted_findings": findings,
        "conflicts": {},
        "status": graded.status,
        "grade_result": graded.result,
    }


def failed_item(outcome: str, error: Exception) -> dict[str, Any]:
    """-> a CANNOT_GRADE result item explaining why extraction failed."""
    return {
        "outcome_name": outcome_display_name(outcome),
        "present": False,
        "extracted_features": {},
        "accepted_findings": [],
        "conflicts": {},
        "status": CANNOT_GRADE,
        "grade_result": cannot_grade(outcome, f"Inference error: {error}").result,
        "error": str(error),
    }


def is_ollama_available(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> bool:
    """True when the local Ollama server answers the preflight for `model`."""
    try:
        preflight(model=model, host=host, timeout=3)
        return True
    except Exception:
        return False


def grade_badge(status: str, grade: int | float | None, grades: tuple = ()) -> tuple[str, str]:
    """-> (CSS class, label) for the grade pill. `status` is the harness status or "pending"."""
    if status == GRADED:
        css = {1: "badge-grade-1", 2: "badge-grade-2"}.get(grade, "badge-grade-3")
        return css, f"GRADE {int(grade)}" if grade is not None else "GRADED"
    if status == GRADE_SET:
        return "badge-grade-set", f"GRADE SET: {grades}" if grades else "GRADE SET"
    labels = {
        CANNOT_GRADE: ("badge-cannot-grade", "CANNOT GRADE"),
        NOT_APPLICABLE: ("badge-not-applicable", "NOT APPLICABLE"),
        ABSENT: ("badge-absent", "ABSENT"),
        "refuted": ("badge-refuted", "REFUTED (CONTRADICTED)"),
        # The model said absent but objective criteria are met: a reviewer must look.
        "missed_presence": ("badge-cannot-grade", "MISSED PRESENCE (REVIEW)"),
    }
    return labels.get(status, ("badge-not-applicable", status.upper()))
