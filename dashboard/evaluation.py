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

from pathlib import Path
import os
import sys
from typing import Any

from experiments.grading import OutcomeGrade, grade_outcome
from experiments.medgemma_extraction import DEFAULT_PROMPT_STAGE, run, stage
from experiments.ollama_backend import DEFAULT_HOST, preflight
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

# Supported models for live extraction
ALLOWED_MODELS: dict[str, str] = {
    "medgemma-1.5-4b-it": "MedGemma 1.5 4B-IT",
    "gemma4:21b": "Gemma 4 21B",
    "medgemma-27b-f16": "MedGemma 27B",
}
DEFAULT_LIVE_MODEL = "medgemma-1.5-4b-it"


def get_installed_models(host: str = DEFAULT_HOST) -> list[str]:
    """-> list of model names/tags installed on the Ollama instance at `host`."""
    from experiments.ollama_backend import request_json
    try:
        data = request_json(host, "/api/tags", timeout=3)
        installed = []
        for item in data.get("models", []):
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                installed.append(name)
        return installed
    except Exception:
        return []


def is_model_installed(model: str, installed_models: list[str]) -> bool:
    """True if `model` matches an entry in `installed_models` (handling :latest canonicalization)."""
    if not model or not installed_models:
        return False
    canonical = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"
    for item in installed_models:
        item_canonical = item if ":" in item.rsplit("/", 1)[-1] else f"{item}:latest"
        if item == model or item == canonical or item_canonical == canonical or item_canonical == model:
            return True
        if item.rstrip(":latest") == model.rstrip(":latest"):
            return True
    return False


def check_ollama_status(model: str = DEFAULT_LIVE_MODEL, host: str = DEFAULT_HOST) -> tuple[str, str]:
    """-> (status, display_message).

    Status is one of:
      - 'ready': Ollama server is reachable and `model` is installed and ready.
      - 'not_installed': Ollama server is reachable, but `model` is not installed.
      - 'offline': Ollama server is unreachable.
    """
    installed = get_installed_models(host)
    if not installed:
        from experiments.ollama_backend import request_json
        try:
            request_json(host, "/api/tags", timeout=3)
            # Server responded but no models installed
            return ("not_installed", f"Model '{model}' not installed (Ollama has 0 models)")
        except Exception:
            return ("offline", f"Ollama Server Offline ({host})")

    if is_model_installed(model, installed):
        return ("ready", f"Ollama Online - '{model}' Ready")
    else:
        return ("not_installed", f"Model '{model}' not installed in Ollama")


def get_model_choices(host: str = DEFAULT_HOST, grouped: bool = False) -> dict[str, Any]:
    """-> choices mapping for live evaluation.

    Categorizes models into installed and not-installed, clearly labeled.
    If grouped=True, returns {group_name: {model_tag: label}}.
    If grouped=False, returns flat {model_tag: label}.
    """
    installed_tags = get_installed_models(host)
    ollama_online = bool(installed_tags)
    if not ollama_online:
        from experiments.ollama_backend import request_json
        try:
            request_json(host, "/api/tags", timeout=3)
            ollama_online = True
        except Exception:
            ollama_online = False

    installed_group: dict[str, str] = {}
    uninstalled_group: dict[str, str] = {}

    if not ollama_online:
        for tag, label in ALLOWED_MODELS.items():
            uninstalled_group[tag] = f"{label} (Ollama Offline)"
        if grouped:
            return {"Supported Models (Ollama Offline)": uninstalled_group}
        return uninstalled_group

    for tag, label in ALLOWED_MODELS.items():
        if is_model_installed(tag, installed_tags):
            installed_group[tag] = f"{label} (Installed)"
        else:
            uninstalled_group[tag] = f"{label} (Not Installed)"

    # Include any additional models detected in Ollama
    for tag in installed_tags:
        if not is_model_installed(tag, list(ALLOWED_MODELS.keys())):
            installed_group[tag] = f"{tag} (Installed)"

    if grouped:
        res: dict[str, dict[str, str]] = {}
        if installed_group:
            res["Installed in Ollama"] = installed_group
        if uninstalled_group:
            res["Not Installed (Pull Required)"] = uninstalled_group
        return res

    return {**installed_group, **uninstalled_group}


# The patient id a pasted note gets inside the harness; it never leaves this module.
LIVE_NOTE_UID = "LIVE-CASE"

#: Seconds the availability check waits. It runs a real generation, so a cold
#: model has to load inside it; 3s was enough for a resident 27B and is not
#: enough for a first call after `ollama run` has unloaded the weights.
PREFLIGHT_TIMEOUT = 60

#: How many of the live evaluator's outcome calls run at once. Grading one note
#: against all 14 focus outcomes is 14 generations, and serially that is a long
#: wait. Batching only confounds *repeated* runs of the same outcome
#: (experiments/run_output.py); the dashboard grades each outcome once.
#: LOCAL: 2, not 4 for large models - at 3 parallel requests this machine's Ollama dies with
#: "llama-server process no longer running" (three 16k contexts on top of 27B weights).
#: For lightweight 4B models (medgemma-1.5-4b-it), memory footprint is small (~2.4 GB),
#: allowing higher concurrency (4 workers) to process the 14 focus outcomes much faster.
LIVE_CONCURRENCY = 2


def detect_system_hardware() -> dict[str, Any]:
    """-> system hardware profile including total RAM (GB), CPU cores, and OS platform."""
    cpu_count = os.cpu_count() or 4
    total_ram_gb = 16.0
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            total_ram_gb = round((page_size * pages) / (1024 ** 3), 1)
        elif sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        pass

    return {
        "total_ram_gb": total_ram_gb,
        "cpu_count": cpu_count,
        "platform": sys.platform,
    }


def get_concurrency_assessment(
    model: str = DEFAULT_LIVE_MODEL,
    concurrency: int = 1,
    ram_gb: float | None = None,
) -> dict[str, Any]:
    """Assess user-chosen concurrency against system hardware for the selected model.

    Returns:
      - recommended: default recommended concurrency for this model & RAM
      - max_safe: maximum safe concurrency before high risk of swap thrashing or OOM
      - estimated_ram_gb: estimated memory needed (weights + KV cache per slot)
      - total_ram_gb: detected system RAM
      - status: 'safe' | 'caution' | 'danger'
      - message: human-readable advisory message
      - model_tier: detected model tier string (e.g. '4B', '12B', '21B', '27B')
    """
    if ram_gb is None:
        hw = detect_system_hardware()
        ram = float(hw.get("total_ram_gb", 16.0))
    else:
        ram = float(ram_gb)

    model_lower = (model or "").lower()
    if "4b" in model_lower:
        weight_gb = 2.4
        kv_slot_gb = 1.0
        rec = 4 if ram >= 12 else 2
        max_safe = 4 if ram < 32 else 8
        model_tier = "4B"
    elif any(k in model_lower for k in ("12b", "14b")):
        weight_gb = 8.0
        kv_slot_gb = 2.0
        rec = 1 if ram <= 16 else 2
        max_safe = 1 if ram <= 16 else 3
        model_tier = "12B"
    elif any(k in model_lower for k in ("20b", "21b")):
        weight_gb = 14.0
        kv_slot_gb = 2.5
        rec = 1 if ram <= 24 else 2
        max_safe = 1 if ram <= 24 else 2
        model_tier = "21B"
    else:
        # Default / 27B models
        weight_gb = 27.0
        kv_slot_gb = 3.5
        rec = 1 if ram <= 32 else 2
        max_safe = 1 if ram <= 32 else 2
        model_tier = "27B"

    conc = max(1, int(concurrency))
    est_footprint = round(weight_gb + (conc * kv_slot_gb), 1)
    safe_ceiling = ram * 0.75

    if est_footprint <= safe_ceiling and conc <= max_safe:
        status = "safe"
        if conc == rec:
            msg = f"✓ Optimal: {conc} worker(s) recommended for {model_tier} model on {ram:.0f} GB RAM (~{est_footprint} GB footprint)."
        else:
            msg = f"✓ Safe: {conc} worker(s) well within hardware capacity ({ram:.0f} GB RAM, ~{est_footprint} GB footprint)."
    elif est_footprint <= ram and conc <= max_safe + 1:
        status = "caution"
        msg = f"⚠️ Caution: {conc} worker(s) will use ~{est_footprint} GB of {ram:.0f} GB RAM. May reduce per-worker token speed."
    else:
        status = "danger"
        msg = f"🚨 High Risk: {conc} worker(s) requires ~{est_footprint} GB (detected {ram:.0f} GB RAM). High risk of Out-of-Memory crash!"

    return {
        "recommended": rec,
        "max_safe": max_safe,
        "estimated_ram_gb": est_footprint,
        "total_ram_gb": ram,
        "status": status,
        "message": msg,
        "model_tier": model_tier,
    }


def get_live_concurrency(model: str = DEFAULT_LIVE_MODEL) -> int:
    """-> concurrency level for live evaluation based on model size.

    For 4B models (e.g. medgemma-1.5-4b-it), memory footprint is small (~2.4 GB),
    enabling 4 parallel workers. For 27B+ models, concurrency is capped at 2.
    """
    model_lower = (model or "").lower()
    if "4b" in model_lower:
        return 4
    return LIVE_CONCURRENCY

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
                         model: str = DEFAULT_LIVE_MODEL, host: str = DEFAULT_HOST,
                         prompt_stage: str = DEFAULT_PROMPT_STAGE, patient_sex: str = "unknown",
                         patient_age: Any = None, concurrency: int = 1,
                         num_ctx: int = 16384) -> dict[str, dict[str, Any]]:
    """-> {outcome: result item} for one pasted note, extracted exactly as the CLI does.

    Runs the note as a one-note batch through `run()`: the CLI's prompt stage,
    constrained decoding, retries, quote grounding, unit guards and conflict
    reconciliation. A finding whose quote is not in the note never reaches
    grading. Raises RuntimeError when the backend fails.
    """
    outcome_ids = [normalize_outcome_id(outcome) for outcome in outcomes]
    note = {"patient_uid": LIVE_NOTE_UID, "patient": note_text}
    results, _ = run([note], outcome_ids, backend, model, host, Tally(),
                     concurrency=concurrency, st=stage(prompt_stage), num_ctx=num_ctx)
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
                           model: str = DEFAULT_LIVE_MODEL, host: str = DEFAULT_HOST,
                           concurrency: int | None = None,
                           num_ctx: int = 16384) -> dict[str, dict[str, Any]]:
    """-> {outcome: result item} for the live evaluator, from the model or from typed values.

    Every item has the same keys in both modes, so the UI renders them identically.
    """
    context = patient_context or {}
    sex, age = context.get("patient_sex", "unknown"), context.get("patient_age")
    outcome_ids = [normalize_outcome_id(outcome) for outcome in outcomes]
    active_concurrency = concurrency if concurrency is not None else get_live_concurrency(model)
    if use_ollama:
        try:
            return extract_with_harness(note_text, outcome_ids, model=model, host=host,
                                        patient_sex=sex, patient_age=age,
                                        concurrency=min(len(outcome_ids), active_concurrency),
                                        num_ctx=num_ctx)
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


def is_ollama_available(model: str = DEFAULT_LIVE_MODEL, host: str = DEFAULT_HOST) -> bool:
    """True when the local Ollama server is running and answers preflight for `model`."""
    status, _ = check_ollama_status(model=model, host=host)
    if status != "ready":
        return False
    try:
        preflight(model=model, host=host, timeout=PREFLIGHT_TIMEOUT, strict=False)
        return True
    except Exception:
        return False


def save_live_run_results(
    note_text: str,
    outcomes: list[str],
    results: dict[str, Any],
    model: str = DEFAULT_LIVE_MODEL,
    backend: str = "ollama",
    patient_age: Any = None,
    patient_sex: str | None = None,
    output_dir: str | Path = "results/live",
    num_ctx: int = 16384,
) -> Path:
    """Save live note evaluation results to a JSON file compatible with the run explorer.

    -> Path to the saved file.
    """
    import json
    from dataclasses import asdict
    from datetime import datetime, timezone

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"live-{timestamp}"
    filename = f"live_run_{timestamp}.json"
    file_path = out_dir / filename

    outcome_ids = [normalize_outcome_id(o) for o in outcomes]
    status_counts: dict[str, int] = {}
    status_by_outcome: dict[str, dict[str, int]] = {}

    per_outcome_details = {}
    for num in outcome_ids:
        item = results.get(num, {})
        status = item.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        status_by_outcome[num] = {status: 1}

        grade_res = item.get("grade_result")
        if isinstance(grade_res, GradeResult):
            grade_dict = asdict(grade_res)
            grade_dict["grades"] = list(grade_dict.get("grades", []))
            grade_dict["missing"] = list(grade_dict.get("missing", []))
            grade_dict["undecided"] = [list(u) for u in grade_dict.get("undecided", [])]
        elif isinstance(grade_res, dict):
            grade_dict = dict(grade_res)
        else:
            grade_dict = {"status": status, "grade": None, "reason": None}

        per_outcome_details[num] = {
            "outcome_name": item.get("outcome_name", outcome_display_name(num)),
            "present": item.get("present", False),
            "status": status,
            "extracted_features": item.get("extracted_features", {}),
            "accepted_findings": item.get("accepted_findings", []),
            "conflicts": item.get("conflicts", []),
            "grade_result": grade_dict,
            "raw_reply": item.get("raw_reply", ""),
        }

    age_float = None
    try:
        if patient_age is not None and str(patient_age).strip():
            age_float = float(patient_age)
    except (ValueError, TypeError):
        pass

    sex_str = str(patient_sex).strip().lower() if patient_sex else None
    if sex_str not in ("male", "female"):
        sex_str = None

    record = {
        "patient_uid": LIVE_NOTE_UID,
        "title": f"Live Evaluation ({model})",
        "age": [[age_float, "year"]] if age_float is not None else None,
        "patient_age": age_float,
        "gender": sex_str,
        "patient_sex": sex_str,
        "patient_note": note_text,
        "outcomes": per_outcome_details,
    }

    run_doc = {
        "provenance": {
            "timestamp": timestamp,
            "run_id": run_id,
            "model": model,
            "weights": model,
            "served_as": model,
            "backend": backend,
            "num_ctx": num_ctx,
            "notes_count": 1,
            "outcomes": outcome_ids,
            "source": "live",
        },
        "profiling": {
            "total_outcomes": len(outcome_ids),
        },
        "automated_metrics": {
            "outcomes_graded": len(results),
            "accepted_findings_total": sum(len(item.get("accepted_findings", [])) for item in results.values()),
        },
        "grade_status": status_counts,
        "grade_status_by_outcome": status_by_outcome,
        "detailed_records": [record],
    }

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(run_doc, f, indent=2)

    return file_path


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
