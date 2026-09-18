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

import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
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
    "gemma4:12b": "Gemma 4 12B",
    "medgemma-27b-f16": "MedGemma 27B",
}
DEFAULT_LIVE_MODEL = "medgemma-1.5-4b-it"
OLLAMA_NUM_CTX = 16384



def get_installed_models(host: str = DEFAULT_HOST) -> list[str]:
    """-> list of model names/tags installed on the Ollama instance at `host`."""
    from experiments.ollama_backend import request_json
    try:
        data = request_json(host, "/api/tags", timeout=3)
        installed = []
        for item in (data.get("models") or []):
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
        if item.removesuffix(":latest") == model.removesuffix(":latest"):
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


def get_model_choices(host: str = DEFAULT_HOST, grouped: bool = False, include_custom: bool = True) -> dict[str, Any]:
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
            res_offline: dict[str, dict[str, str]] = {"Supported Models (Ollama Offline)": uninstalled_group}
            if include_custom:
                res_offline["Custom Model"] = {"__custom__": "Custom / Other Ollama Model..."}
            return res_offline
        res_offline_flat = dict(uninstalled_group)
        if include_custom:
            res_offline_flat["__custom__"] = "Custom / Other Ollama Model..."
        return res_offline_flat

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
        if include_custom:
            res["Custom Model"] = {"__custom__": "Custom / Other Ollama Model..."}
        return res

    res_flat = {**installed_group, **uninstalled_group}
    if include_custom:
        res_flat["__custom__"] = "Custom / Other Ollama Model..."
    return res_flat


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
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_lower)
    param_b = float(m.group(1)) if m else None

    if "4b" in model_lower or (param_b is not None and param_b <= 5.0):
        weight_gb = 2.4
        kv_slot_gb = 1.0
        rec = 4 if ram >= 12 else 2
        max_safe = 4 if ram < 32 else 8
        model_tier = f"{int(param_b) if param_b else 4}B"
    elif any(k in model_lower for k in ("7b", "8b", "9b")) or (param_b is not None and 5.0 < param_b <= 10.0):
        weight_gb = 5.5
        kv_slot_gb = 1.5
        rec = 2 if ram >= 16 else 1
        max_safe = 2 if ram <= 16 else 4
        model_tier = f"{int(param_b) if param_b else 8}B"
    elif any(k in model_lower for k in ("12b", "14b")) or (param_b is not None and 10.0 < param_b <= 16.0):
        weight_gb = 8.0
        kv_slot_gb = 2.0
        rec = 1 if ram <= 16 else 2
        max_safe = 1 if ram <= 16 else 3
        model_tier = "12B"
    elif any(k in model_lower for k in ("20b", "21b")) or (param_b is not None and 16.0 < param_b <= 24.0):
        weight_gb = 14.0
        kv_slot_gb = 2.5
        rec = 1 if ram <= 24 else 2
        max_safe = 1 if ram <= 24 else 2
        model_tier = "21B"
    elif param_b is not None and param_b > 45.0:
        weight_gb = 45.0
        kv_slot_gb = 5.0
        rec = 1
        max_safe = 1 if ram < 64 else 2
        model_tier = f"{int(param_b)}B"
    else:
        # Default / 27B-35B models
        weight_gb = 27.0
        kv_slot_gb = 3.5
        rec = 1 if ram <= 32 else 2
        max_safe = 1 if ram <= 32 else 2
        model_tier = f"{int(param_b)}B" if param_b else "27B"

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


OUTCOME_ORGAN_SYSTEMS: dict[str, str] = {
    "01": "Cardiovascular",
    "02": "Cardiovascular",
    "03": "Cardiovascular",
    "04": "Cardiovascular",
    "05": "Cardiovascular",
    "06": "Cardiovascular",
    "07": "Cardiovascular",
    "08": "Cardiovascular",
    "09": "Central Nervous System",
    "10": "Pain / Neurological",
    "11": "Neurological",
    "12": "Neurological / Screening",
    "13": "Central Nervous System",
    "14": "Central Nervous System",
    "15": "Neurological",
    "16": "Eyes & ENT",
    "17": "Ophthalmologic",
    "18": "Gastrointestinal",
    "19": "Renal",
    "20": "Renal",
    "21": "Renal",
    "22": "Genitourinary (Female)",
    "23": "Genitourinary (Male)",
    "24": "Genitourinary (Male)",
    "25": "Growth & Development",
    "26": "Growth & Development",
    "27": "Growth & Development",
    "28": "Pain / Vascular",
    "29": "Hematologic",
    "30": "Hematologic / Transfusion",
    "31": "Hematologic",
    "32": "Hematologic / Hepatic",
    "33": "Hematologic",
    "34": "Hematologic / Iron",
    "35": "Hematologic / Infection",
    "36": "Infectious Disease",
    "37": "Infectious Disease",
    "38": "Malignancies",
    "39": "Musculoskeletal",
    "40": "Dermatologic",
    "41": "Musculoskeletal",
    "42": "Musculoskeletal",
    "43": "Multiorgan",
    "44": "Pregnancy & Perinatal",
    "45": "Pregnancy & Perinatal",
    "46": "Pregnancy & Perinatal",
    "47": "Psychiatric",
    "48": "Pulmonary",
    "49": "Pulmonary",
    "50": "Pulmonary",
    "51": "Pulmonary",
    "52": "Pulmonary",
    "53": "Pulmonary / Sleep",
}


def get_all_scogs_outcomes() -> dict[str, dict[str, Any]]:
    """-> metadata for all 53 SCOGS decision tables."""
    outcomes: dict[str, dict[str, Any]] = {}
    for num in sorted(TABLES.keys()):
        table = TABLES[num]
        norm = normalize_outcome_id(num)
        int_id = str(int(norm))
        is_focus = int_id in FOCUS_OUTCOMES or norm in FOCUS_OUTCOMES
        focus_meta = FOCUS_OUTCOMES.get(int_id) or FOCUS_OUTCOMES.get(norm)
        name = focus_meta["name"] if focus_meta else table.name
        organ = focus_meta.get("organ_system") if focus_meta else OUTCOME_ORGAN_SYSTEMS.get(norm, "General")
        outcomes[norm] = {
            "id": norm,
            "int_id": int_id,
            "name": name,
            "table_name": table.name,
            "organ_system": organ,
            "is_focus": is_focus,
        }
    return outcomes


def get_outcome_choices(grouped: bool = True) -> dict[str, Any]:
    """-> choice dictionary for selectize input containing all 53 outcomes."""
    all_outcomes = get_all_scogs_outcomes()
    focus_group: dict[str, str] = {}
    other_group: dict[str, str] = {}

    for norm, meta in all_outcomes.items():
        choice_key = meta["int_id"] if meta["is_focus"] else norm
        label = f"#{norm} {meta['name']}"
        if meta["is_focus"]:
            focus_group[choice_key] = f"{label} (Focus • {meta['organ_system']})"
        else:
            other_group[choice_key] = f"{label} ({meta['organ_system']})"

    if grouped:
        return {
            "14 Core Focus Outcomes (Delphi Consensus)": focus_group,
            "Additional SCOGS Decision Tables (39 Outcomes)": other_group,
        }
    return {**focus_group, **other_group}


def normalize_outcome_id(raw: str | int) -> str:
    """-> a two-digit outcome id ("5" -> "05"); non-numeric ids pass through."""
    return f"{int(raw):02d}" if str(raw).isdigit() else str(raw)


def outcome_display_name(outcome: str) -> str:
    """-> the dashboard's name for an outcome, falling back to the rubric's."""
    norm = normalize_outcome_id(outcome)
    int_id = str(int(norm)) if norm.isdigit() else norm
    table = TABLES.get(norm) or TABLES.get(outcome)
    return (
        FOCUS_OUTCOMES.get(int_id, {}).get("name")
        or FOCUS_OUTCOMES.get(outcome, {}).get("name")
        or getattr(table, "name", f"Outcome {outcome}")
    )


def outcome_meta(outcome: str) -> dict[str, str | None]:
    """-> {name, organ_system, acuity} for any of the 53 outcomes.

    Only the 14 focus outcomes carry an acuity; it is None for the rest rather
    than a made-up default.
    """
    norm = normalize_outcome_id(outcome)
    focus = FOCUS_OUTCOMES.get(str(int(norm)) if norm.isdigit() else norm, {})
    return {
        "name": outcome_display_name(outcome),
        "organ_system": focus.get("organ_system") or OUTCOME_ORGAN_SYSTEMS.get(norm),
        "acuity": focus.get("acuity"),
    }


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
                         num_ctx: int = OLLAMA_NUM_CTX) -> dict[str, dict[str, Any]]:
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
                           num_ctx: int = OLLAMA_NUM_CTX) -> dict[str, dict[str, Any]]:
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
        preflight(model=model, host=host, timeout=PREFLIGHT_TIMEOUT, strict=False, num_ctx=OLLAMA_NUM_CTX)
        return True
    except Exception:
        return False


def live_patient_key(patient_id: str | None, note_text: str) -> str:
    """-> the file stem that holds one patient's live results.

    A CSV encounter files under its patient id, so every visit of that patient
    lands in one file. A pasted note has no id, so it files under a hash of its
    text: grading the same note again finds the same file.
    """
    if patient_id and patient_id.strip():
        return "patient_" + (re.sub(r"[^A-Za-z0-9_-]+", "_", patient_id.strip()).strip("_") or "unknown")
    return "note_" + hashlib.sha1(note_text.strip().encode("utf-8")).hexdigest()[:12]


def _saved_grade_result(item: dict[str, Any]) -> dict[str, Any]:
    """-> an item's grade result as JSON-ready data; live results hold the `GradeResult` itself."""
    grade_res = item.get("grade_result")
    if isinstance(grade_res, GradeResult):
        saved = asdict(grade_res)
        saved["grades"] = list(saved.get("grades") or [])
        saved["missing"] = list(saved.get("missing") or [])
        saved["undecided"] = [list(u) for u in (saved.get("undecided") or [])]
        return saved
    if isinstance(grade_res, dict):
        return dict(grade_res)
    return {"status": item.get("status", "unknown"), "grade": None, "reason": None}


def save_live_patient_results(
    note_text: str,
    results: dict[str, Any],
    *,
    patient_id: str | None = None,
    visit: str | None = None,
    title: str | None = None,
    model: str = DEFAULT_LIVE_MODEL,
    backend: str = "ollama",
    patient_age: Any = None,
    patient_sex: str | None = None,
    output_dir: str | Path = "results/live",
    num_ctx: int = OLLAMA_NUM_CTX,
) -> Path:
    """Write a live evaluation into its patient's results file; -> that file's path.

    One file per patient (`live_patient_key`), one record per visit inside it,
    in the same shape as a CLI results file so Explore Saved Runs opens it.
    Grading a visit again updates its record rather than adding a file:
    outcomes graded before are kept and the re-graded ones replaced. If the
    note text changed, the old outcomes describe a different note, so the
    record starts over.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = live_patient_key(patient_id, note_text)
    path = out_dir / f"{key}.json"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if patient_id and patient_id.strip():
        uid = f"{patient_id.strip()} @ {visit}" if visit else patient_id.strip()
    else:
        uid = "NOTE-" + key.removeprefix("note_")[:8].upper()

    try:
        doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    records = [r for r in (doc.get("detailed_records") or []) if isinstance(r, dict)]

    graded_by = {"model": model, "backend": backend, "num_ctx": num_ctx, "graded_at": now}
    new_outcomes = {
        num: {
            "outcome_name": item.get("outcome_name", outcome_display_name(num)),
            "present": item.get("present", False),
            "status": item.get("status", "unknown"),
            "extracted_features": item.get("extracted_features", {}),
            "accepted_findings": item.get("accepted_findings", []),
            "conflicts": item.get("conflicts", {}),
            "grade_result": _saved_grade_result(item),
            "raw_reply": item.get("raw_reply", ""),
            "graded_by": graded_by,
        }
        for num, item in ((normalize_outcome_id(k), v) for k, v in results.items())
    }

    age = None
    try:
        if patient_age is not None and str(patient_age).strip():
            age = float(patient_age)
    except (TypeError, ValueError):
        pass
    sex = str(patient_sex or "").strip().lower()
    sex = sex if sex in ("male", "female") else None

    old = next((r for r in records if r.get("patient_uid") == uid), None)
    kept = old.get("outcomes", {}) if old and old.get("patient_note") == note_text else {}
    record = {
        "patient_uid": uid,
        "title": title or "Pasted clinical note",
        "visit_datetime": visit,
        "age": [[age, "year"]] if age is not None else None,
        "patient_age": age,
        "gender": sex,
        "patient_sex": sex,
        "patient_note": note_text,
        "updated_at": now,
        "outcomes": {**kept, **new_outcomes},
    }
    records = [record if r is old else r for r in records] if old else [*records, record]

    status_counts: dict[str, int] = {}
    status_by_outcome: dict[str, dict[str, int]] = {}
    for rec in records:
        for num, outcome in (rec.get("outcomes") or {}).items():
            status = outcome.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            per_outcome = status_by_outcome.setdefault(num, {})
            per_outcome[status] = per_outcome.get(status, 0) + 1

    doc = {
        "provenance": {
            **(doc.get("provenance") or {}),
            "source": "live",
            "patient_key": key,
            "patient_id": patient_id.strip() if patient_id and patient_id.strip() else None,
            "updated_at": now,
            "model": model,
            "weights": model,
            "backend": backend,
            "num_ctx": num_ctx,
            "notes_count": len(records),
        },
        "profiling": doc.get("profiling") or {},
        "automated_metrics": {
            "outcomes_graded": sum(len(r.get("outcomes") or {}) for r in records),
            "accepted_findings_total": sum(len(o.get("accepted_findings") or [])
                                           for r in records for o in (r.get("outcomes") or {}).values()),
        },
        "grade_status": status_counts,
        "grade_status_by_outcome": status_by_outcome,
        "detailed_records": records,
    }

    # Written beside the target and swapped in, so a crash mid-write never
    # leaves a patient's file half-written.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def grade_badge(status: str, grade: int | float | None, grades: tuple = ()) -> tuple[str, str]:
    """-> (CSS class, label) for the grade pill. `status` is the harness status or "pending"."""
    if status == GRADED:
        css = {1: "badge-grade-1", 2: "badge-grade-2"}.get(grade, "badge-grade-3")
        return css, f"GRADE {int(grade)}" if grade is not None else "GRADED"
    if status == GRADE_SET:
        # Same wording as GradeResult.__str__ ("Grade 2 or 3"), not a Python tuple.
        return "badge-grade-set", f"GRADE {' OR '.join(str(int(g)) for g in grades)}" if grades else "GRADE SET"
    labels = {
        CANNOT_GRADE: ("badge-cannot-grade", "CANNOT GRADE"),
        NOT_APPLICABLE: ("badge-not-applicable", "NOT APPLICABLE"),
        ABSENT: ("badge-absent", "ABSENT"),
        "refuted": ("badge-refuted", "REFUTED (CONTRADICTED)"),
        # The model said absent but objective criteria are met: a reviewer must look.
        "missed_presence": ("badge-cannot-grade", "MISSED PRESENCE (REVIEW)"),
    }
    return labels.get(status, ("badge-not-applicable", status.upper()))
