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
    import re
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


def save_live_run_results(
    note_text: str,
    outcomes: list[str],
    results: dict[str, Any],
    model: str = DEFAULT_LIVE_MODEL,
    backend: str = "ollama",
    patient_age: Any = None,
    patient_sex: str | None = None,
    output_dir: str | Path = "results/live",
    num_ctx: int = OLLAMA_NUM_CTX,
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
            grade_dict["grades"] = list(grade_dict.get("grades") or [])
            grade_dict["missing"] = list(grade_dict.get("missing") or [])
            grade_dict["undecided"] = [list(u) for u in (grade_dict.get("undecided") or [])]
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


# ==============================================================================
# PCAI / GPT-OSS LIVE INFERENCE OVERRIDES
# ==============================================================================
# These definitions intentionally appear at the end of this module.  Python uses
# the final definition of each name, so the saved-run/deterministic infrastructure
# above remains intact while the live dashboard routes inference to St. Jude PCAI.

import json as _pcai_json
import time as _pcai_time
import re as _pcai_re
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor, as_completed as _as_completed

from openai import OpenAI as _OpenAI
from scogs.criteria import PRESENCE_CRITERIA as _PRESENCE_CRITERIA

from dashboard.conformal_utils import (
    apply_conformal as _apply_conformal,
    normalize_probabilities as _normalize_probabilities,
)

PCAI_GATEWAY_URL = "https://bifrost.ai-application.stjude.org/v1"
PCAI_MODEL = "gpt-oss-120b"
PCAI_QWEN_MODEL = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_LIVE_MODEL = PCAI_MODEL
ALLOWED_MODELS = {
    PCAI_MODEL: "GPT-OSS 120B",
    PCAI_QWEN_MODEL: "Qwen3.8 27B FP8",
}

# Keep these IDs aligned with the 14 PI-finalized SCOGS domains.
_PCAI_ID_TO_RULE_NAME = {
    "10": "Chronic pain",
    "11": "Cognitive dysfunction",
    "12": "Elevated TCD ultrasonography velocity",
    "15": "Stroke (hemorrhagic or ischemic)",
    "17": "Sickle cell retinopathy (SCR)",
    "21": "Chronic kidney disease (CKD)",
    "24": "Priapism",
    "28": "Acute sickle cell pain episode",
    "29": "Acute splenic sequestration",
    "39": "Avascular necrosis of joints (AVN)",
    "40": "Leg ulcer",
    "47": "Depression",
    "48": "Acute chest syndrome (ACS)",
    "49": "Asthma exacerbation",
}

_PCAI_RULES_PATH = Path(__file__).resolve().parent / "scogs_14_outcomes.json"
_PCAI_CALIBRATION_PATH = Path(__file__).resolve().parent / "conformal_calibration.json"


def _pcai_load_rules() -> dict[str, dict[str, Any]]:
    payload = _pcai_json.loads(_PCAI_RULES_PATH.read_text(encoding="utf-8-sig"))
    return {r["outcome"]: r for r in payload.get("outcomes", [])}


def _pcai_valid_classes(rule: dict[str, Any]) -> list[str]:
    labels = ["absent", "insufficient_information"]
    for grade, definition in rule.get("grade_definitions", {}).items():
        if str(definition).strip().upper() != "N/A":
            labels.append(str(grade))
    return labels


def _pcai_extract_text(message: Any) -> str:
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


def _pcai_clean_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("GPT-OSS returned empty final content.")
    text = (
        text.replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )
    try:
        parsed = _pcai_json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        parsed = _pcai_json.loads(text[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("GPT-OSS response did not contain a valid JSON object.")


def _pcai_quote_is_exact(note_text: str, quote: str) -> bool:
    q = " ".join(str(quote or "").split()).casefold()
    n = " ".join(str(note_text or "").split()).casefold()
    return bool(q) and q in n


def _pcai_load_calibration():
    if not _PCAI_CALIBRATION_PATH.exists():
        return None
    try:
        return _pcai_json.loads(_PCAI_CALIBRATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pcai_calibration_for_outcome(calibration, outcome_name: str):
    if not calibration:
        return None
    specific = calibration.get("per_outcome", {}).get(outcome_name)
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


def _pcai_prompt(note_text: str, rule: dict[str, Any]) -> str:
    allowed = _pcai_valid_classes(rule)
    return f"""
Apply the supplied SCOGS severity rubric to this clinical note.

Evaluate ONLY this outcome:
{rule["outcome"]}

Rules:
- Use only the supplied note and SCOGS rubric.
- Do not use hidden/reference labels.
- Do not invent missing facts or undocumented negatives.
- Distinguish current acute events from historical diagnoses.
- A chronic diagnosis may establish presence, but do not force an exact grade
  when the required grade-defining variables are missing.
- Never assign a grade whose definition is N/A.
- Do not convert oxygen L/min to FiO2.
- Evidence must be copied verbatim from the note.
- model_confidence_pct is uncalibrated model certainty from 0 to 100.
- Return a score for every allowed class and make scores sum approximately to 100.
- Output JSON only.

If absent:
  outcome_present=false, grade=null, status="absent"

If present but exact grade cannot be determined:
  outcome_present=true, grade=null, status="insufficient_information"

If exactly gradable:
  outcome_present=true, grade=<integer>, status="graded"

CLINICAL NOTE
-------------
{note_text}

SCOGS RUBRIC
------------
Diagnostic criteria:
{rule.get("diagnostic_criteria", "")}

Grade definitions:
{_pcai_json.dumps(rule.get("grade_definitions", {}), ensure_ascii=False)}

Allowed classes:
{_pcai_json.dumps(allowed)}

Return exactly one JSON object:
{{
  "outcome": "{rule["outcome"]}",
  "outcome_present": true,
  "grade": 3,
  "status": "graded",
  "model_confidence_pct": 90,
  "reasoning": "brief rubric-based explanation",
  "evidence": ["exact quote from note"],
  "class_scores_pct": {{
    "absent": 1,
    "insufficient_information": 2,
    "1": 2,
    "2": 5,
    "3": 90
  }}
}}
""".strip()


def _pcai_call_one(note_text: str, outcome_id: str, model: str = PCAI_MODEL, patient_context: dict[str, Any] | None = None) -> dict[str, Any]:
    key = os.environ.get("PCAI_API_KEY")
    if not key:
        raise RuntimeError("PCAI_API_KEY is not set in the terminal that launched Shiny.")

    norm = normalize_outcome_id(outcome_id)
    rule_name = _PCAI_ID_TO_RULE_NAME.get(str(int(norm)))
    if not rule_name:
        return _pcai_extract_and_grade_table(
            note_text, norm, model=model, patient_context=patient_context
        )

    rules = _pcai_load_rules()
    rule = rules.get(rule_name)
    if rule is None:
        raise KeyError(f"No PCAI rubric found for {rule_name}")

    client = _OpenAI(
        base_url=PCAI_GATEWAY_URL,
        api_key=key,
        default_headers={"x-bf-vk": key},
        timeout=600,
    )

    last_error = None
    # GPT-OSS may use a substantial part of the completion budget for reasoning.
    # Use the full 16,384-token completion allowance for every single-outcome
    # request so the model has room to reason and still emit the final JSON.
    token_budgets = (16384, 16384, 16384)
    for attempt in range(1, 4):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Apply the SCOGS rubric strictly. "
                            "Return the final answer immediately as one valid JSON object only. "
                            "Do not include markdown."
                        ),
                    },
                    {"role": "user", "content": _pcai_prompt(note_text + ("\n\nCLINICIAN CONTEXT: " + _pcai_json.dumps(patient_context, ensure_ascii=False) if patient_context else ""), rule)},
                ],
                temperature=0,
                max_tokens=token_budgets[attempt - 1],
            )
            try:
                response = client.chat.completions.create(
                    **kwargs,
                    reasoning_effort="low",
                )
            except TypeError:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "reasoning_effort" in str(exc).lower() and "timeout" not in str(exc).lower():
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            choice = response.choices[0]
            raw_final = _pcai_extract_text(choice.message)
            if not raw_final:
                finish_reason = getattr(choice, "finish_reason", None)
                raise ValueError(
                    f"GPT-OSS returned empty final content "
                    f"(finish_reason={finish_reason!r}, max_tokens={token_budgets[attempt - 1]})."
                )
            item = _pcai_clean_json(raw_final)
            break
        except Exception as exc:
            last_error = exc
            if attempt >= 3:
                raise
            _pcai_time.sleep(1.5 * attempt)
    else:
        raise last_error or RuntimeError("Unknown PCAI inference failure.")

    present = bool(item.get("outcome_present", False))
    status_raw = str(item.get("status", "")).strip()
    raw_grade = item.get("grade")
    try:
        grade = int(raw_grade) if raw_grade is not None else None
    except Exception:
        grade = None

    valid_labels = _pcai_valid_classes(rule)
    valid_grades = {int(x) for x in valid_labels if x not in {"absent", "insufficient_information"}}

    if not present:
        grade = None
        harness_status = ABSENT
        model_status = "absent"
    elif grade is None or grade not in valid_grades:
        grade = None
        harness_status = CANNOT_GRADE
        model_status = "insufficient_information"
    else:
        harness_status = GRADED
        model_status = "graded"

    raw_scores = item.get("class_scores_pct", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    probs = _normalize_probabilities(raw_scores, valid_labels)

    if not present:
        predicted_class = "absent"
    elif grade is None:
        predicted_class = "insufficient_information"
    else:
        predicted_class = str(grade)

    try:
        confidence = max(0.0, min(100.0, float(item.get("model_confidence_pct"))))
    except Exception:
        confidence = 100.0 * probs.get(predicted_class, 0.0)

    evidence = item.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    evidence = [
        str(q).strip()
        for q in evidence
        if str(q).strip() and _pcai_quote_is_exact(note_text, str(q))
    ]

    conformal_set = []
    conformal_target = None
    conformal_p = None
    calibration = _pcai_load_calibration()
    cal_item = _pcai_calibration_for_outcome(calibration, rule_name)
    if cal_item:
        c = _apply_conformal(probs, cal_item["scores"], cal_item["qhat"])
        conformal_set = c["prediction_set"]
        conformal_target = calibration.get("target_coverage_pct")
        conformal_p = round(100.0 * c["p_values"].get(predicted_class, 0.0), 1)

    accepted = [
        {
            "feature": "GPT-OSS evidence",
            "value": "supports model decision",
            "quote": q,
            "unit": None,
            "source": "gptoss",
        }
        for q in evidence
    ]

    reason = str(item.get("reasoning", "")).strip()
    missing = ["Exact SCOGS grade-defining information"] if harness_status == CANNOT_GRADE else []

    grade_result = {
        "outcome": normalize_outcome_id(outcome_id),
        "status": harness_status,
        "grade": grade,
        "grades": [],
        "matched": None,
        "reason": reason,
        "missing": missing,
        "undecided": [],
    }

    return {
        "outcome_name": FOCUS_OUTCOMES.get(str(int(outcome_id)), {}).get("name", rule_name),
        "present": present,
        "extracted_features": {
            "model_status": model_status,
            "model_confidence_pct": round(confidence, 1),
            "class_probabilities": probs,
            "conformal_prediction_set": conformal_set,
            "conformal_target_coverage_pct": conformal_target,
            "conformal_predicted_class_p_value_pct": conformal_p,
        },
        "accepted_findings": accepted,
        "conflicts": {},
        "status": harness_status,
        "grade_result": grade_result,
        "raw_reply": item,
    }




def _pcai_table_feature_names(outcome_id: str) -> list[str]:
    """Features referenced by one deterministic SCOGS table and presence rule."""
    norm = normalize_outcome_id(outcome_id)
    table = TABLES.get(norm)
    if table is None:
        return []
    expressions: list[str] = []
    for _, pred in table.all_rows():
        expressions.append(str(pred))
    if _PRESENCE_CRITERIA.get(norm):
        expressions.append(str(_PRESENCE_CRITERIA[norm]))

    names: set[str] = set()
    for expr in expressions:
        for token in _pcai_re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr):
            if token in FEATURES:
                names.add(token)
    return sorted(names)


def _pcai_feature_spec(outcome_id: str, feature_name: str) -> dict[str, Any]:
    spec = dict(FEATURES.get(feature_name, {}))
    per_outcome = spec.get("per_outcome") or {}
    tailored = per_outcome.get(normalize_outcome_id(outcome_id))
    if tailored:
        spec["outcome_specific_definition"] = tailored
    return {
        "type": spec.get("type"),
        "values": spec.get("values"),
        "unit": spec.get("unit"),
        "definition": spec.get("definition"),
        "outcome_specific_definition": spec.get("outcome_specific_definition"),
        "derived": spec.get("derived"),
    }


def _pcai_table_prompt(note_text: str, outcome_id: str, patient_context: dict[str, Any] | None = None) -> str:
    norm = normalize_outcome_id(outcome_id)
    table = TABLES[norm]
    feature_names = _pcai_table_feature_names(norm)
    feature_specs = {name: _pcai_feature_spec(norm, name) for name in feature_names}
    rows = [{"grade": grade, "predicate": pred} for grade, pred in table.rows]
    strata = {key: [{"grade": g, "predicate": p} for g, p in vals] for key, vals in table.strata.items()}
    axes = {key: [{"grade": g, "predicate": p} for g, p in vals] for key, vals in table.axes.items()}
    ctx = patient_context or {}

    return f"""
You are the evidence-extraction stage for a deterministic SCOGS severity grader.
Do NOT choose the final grade yourself. Extract only structured facts that are
explicitly supported by this clinical note. A deterministic rule engine will
apply the decision table after your response.

OUTCOME #{norm}: {table.name}
Presence criterion: {_PRESENCE_CRITERIA.get(norm, 'No separate numeric presence predicate is defined; use explicit diagnosis or clear outcome-specific evidence.')}
Evaluation mode: {table.eval}
Decision rows (highest grade first when applicable):
{_pcai_json.dumps(rows, ensure_ascii=False)}
Strata:
{_pcai_json.dumps(strata, ensure_ascii=False)}
Axes:
{_pcai_json.dumps(axes, ensure_ascii=False)}
Rubric notes:
{_pcai_json.dumps(table.notes, ensure_ascii=False)}

FEATURE CONTRACT
{_pcai_json.dumps(feature_specs, ensure_ascii=False)}

CLINICIAN CONTEXT (trusted if supplied)
{_pcai_json.dumps(ctx, ensure_ascii=False)}

CLINICAL NOTE
{note_text}

Rules:
- Use only facts in the note plus clinician context above.
- Do not invent normal findings or negative findings.
- If the outcome is not documented and there is no supporting evidence, set outcome_present=false.
- Only emit a feature when an exact quote in the note supports that feature.
- Use categorical values exactly as listed in FEATURE CONTRACT.
- Use JSON booleans for bool features and JSON numbers for numeric features.
- Evidence quotes must be copied verbatim from the note.
- model_confidence_pct is an uncalibrated confidence in the extraction/presence assessment.
- Output JSON only.

Return:
{{
  "outcome_present": true,
  "model_confidence_pct": 90,
  "features": {{"feature_name": "value"}},
  "evidence": [
    {{"feature": "feature_name", "quote": "exact quote from note"}}
  ],
  "reasoning": "brief evidence-extraction explanation"
}}
""".strip()



def _pcai_coerce_feature_value(feature_name: str, value: Any):
    """Coerce GPT-OSS JSON values to the canonical SCOGS feature types."""
    spec = FEATURES.get(feature_name, {})
    typ = spec.get("type")
    if value is None:
        return None
    if typ == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "yes", "1", "present"}:
            return True
        if text in {"false", "no", "0", "absent"}:
            return False
        raise ValueError(f"{feature_name}: invalid boolean {value!r}")
    if typ == "num":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        text = str(value).strip().replace(",", "")
        match = _pcai_re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            raise ValueError(f"{feature_name}: invalid numeric value {value!r}")
        return float(match.group(0))
    if typ in {"cat", "ord"}:
        text = str(value).strip()
        allowed = spec.get("values") or []
        if text not in allowed:
            raise ValueError(f"{feature_name}: {text!r} not in allowed values {allowed}")
        return text
    return value


def _pcai_extract_and_grade_table(
    note_text: str,
    outcome_id: str,
    model: str = PCAI_MODEL,
    patient_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GPT-OSS evidence extraction -> original deterministic SCOGS rule engine."""
    key = os.environ.get("PCAI_API_KEY")
    if not key:
        raise RuntimeError("PCAI_API_KEY is not set in the terminal that launched Shiny.")

    norm = normalize_outcome_id(outcome_id)
    client = _OpenAI(
        base_url=PCAI_GATEWAY_URL,
        api_key=key,
        default_headers={"x-bf-vk": key},
        timeout=600,
    )
    prompt = _pcai_table_prompt(note_text, norm, patient_context)
    last_error = None
    for attempt in range(1, 4):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": "Extract grounded SCOGS evidence. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=2200,
            )
            try:
                response = client.chat.completions.create(**kwargs, reasoning_effort="low")
            except TypeError:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "reasoning_effort" in str(exc).lower() and "timeout" not in str(exc).lower():
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise
            item = _pcai_clean_json(_pcai_extract_text(response.choices[0].message))
            break
        except Exception as exc:
            last_error = exc
            if attempt >= 3:
                raise
            _pcai_time.sleep(1.5 * attempt)
    else:
        raise last_error or RuntimeError("Unknown PCAI extraction failure.")

    present = bool(item.get("outcome_present", False))
    requested_features = item.get("features") if isinstance(item.get("features"), dict) else {}
    evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
    allowed_features = set(_pcai_table_feature_names(norm))

    accepted_findings = []
    grounded_features: dict[str, Any] = {}
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        feature = str(ev.get("feature", "")).strip()
        quote = str(ev.get("quote", "")).strip()
        if feature not in allowed_features or feature not in requested_features:
            continue
        if not _pcai_quote_is_exact(note_text, quote):
            continue
        try:
            value = _pcai_coerce_feature_value(feature, requested_features[feature])
        except Exception:
            continue
        grounded_features[feature] = value
        accepted_findings.append({
            "feature": feature,
            "value": value,
            "quote": quote,
            "unit": FEATURES.get(feature, {}).get("unit"),
            "source": "gptoss",
        })

    # Clinician-entered age/sex are trusted context and may satisfy applicability.
    ctx = patient_context or {}
    grounded_features.update(clinician_context(ctx.get("patient_sex"), ctx.get("patient_age")))

    graded = grade_outcome(norm, grounded_features, present)
    try:
        confidence = max(0.0, min(100.0, float(item.get("model_confidence_pct"))))
    except Exception:
        confidence = 0.0

    display_features = dict(grounded_features)
    display_features["model_confidence_pct"] = round(confidence, 1)
    display_features["pcaI_grading_mode"] = "gptoss_evidence_plus_deterministic_table"

    return {
        "outcome_name": outcome_display_name(norm),
        "present": present,
        "extracted_features": display_features,
        "accepted_findings": accepted_findings,
        "conflicts": {},
        "status": graded.status,
        "grade_result": graded.result,
        "raw_reply": item,
    }


def screen_outcomes_pcai(note_text: str, outcomes: list[str], model: str = PCAI_MODEL) -> list[str]:
    """Experimental high-recall screen used only to speed the optional 53 mode."""
    key = os.environ.get("PCAI_API_KEY")
    if not key:
        raise RuntimeError("PCAI_API_KEY is not set.")
    outcome_ids = [normalize_outcome_id(o) for o in outcomes]
    directory = [{"id": o, "name": outcome_display_name(o)} for o in outcome_ids]
    client = _OpenAI(
        base_url=PCAI_GATEWAY_URL,
        api_key=key,
        default_headers={"x-bf-vk": key},
        timeout=600,
    )
    prompt = f"""
Perform a HIGH-RECALL screening pass over this sickle-cell clinical note.
Return every SCOGS outcome that is explicitly present, historically present,
possibly present, or has any note evidence that could plausibly support it.
Err strongly toward inclusion. Do not grade severity. Exclude an outcome only
when the note provides no meaningful evidence for it.

OUTCOME DIRECTORY
{_pcai_json.dumps(directory, ensure_ascii=False)}

NOTE
{note_text}

Return JSON only:
{{"candidate_ids": ["01", "15", "48"], "reasoning": "brief"}}
""".strip()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "High-recall clinical outcome screening. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=1400,
    )
    parsed = _pcai_clean_json(_pcai_extract_text(response.choices[0].message))
    candidates = parsed.get("candidate_ids") if isinstance(parsed, dict) else []
    if not isinstance(candidates, list):
        raise ValueError("Fast 53 screen did not return candidate_ids.")
    valid = set(outcome_ids)
    result = []
    for item in candidates:
        try:
            norm = normalize_outcome_id(item)
        except Exception:
            continue
        if norm in valid and norm not in result:
            result.append(norm)
    return result


def screened_out_item(outcome: str) -> dict[str, Any]:
    """Explicitly preserves uncertainty for outcomes skipped by Fast 53 screening."""
    norm = normalize_outcome_id(outcome)
    result = GradeResult(
        outcome=norm,
        status=CANNOT_GRADE,
        reason="Experimental Fast 53 pre-screen did not select this outcome for detailed grading.",
    )
    return {
        "outcome_name": outcome_display_name(norm),
        "present": False,
        "extracted_features": {"screened_out_fast53": True},
        "accepted_findings": [],
        "conflicts": {},
        "status": CANNOT_GRADE,
        "grade_result": result,
        "raw_reply": {"screened_out_fast53": True},
    }


def get_installed_models(host: str = DEFAULT_HOST) -> list[str]:
    """Compatibility shim used by the existing Shiny sidebar."""
    return list(ALLOWED_MODELS) if os.environ.get("PCAI_API_KEY") else []


def is_model_installed(model: str, installed_models: list[str]) -> bool:
    return model in ALLOWED_MODELS and model in installed_models


def check_ollama_status(model: str = DEFAULT_LIVE_MODEL, host: str = DEFAULT_HOST) -> tuple[str, str]:
    """Compatibility shim: dashboard status now represents St. Jude PCAI."""
    if not os.environ.get("PCAI_API_KEY"):
        return ("offline", "PCAI_API_KEY not set")
    if model not in ALLOWED_MODELS:
        return ("not_installed", f"Unsupported PCAI model: {model}")
    return ("ready", f"St. Jude PCAI ready — {ALLOWED_MODELS[model]}")


def get_model_choices(host: str = DEFAULT_HOST, grouped: bool = False, include_custom: bool = True) -> dict[str, Any]:
    choices = {
        PCAI_MODEL: "GPT-OSS 120B — grader",
        PCAI_QWEN_MODEL: "Qwen3.8 27B FP8 — comparison model",
    }
    return {"St. Jude PCAI": choices} if grouped else choices


def get_concurrency_assessment(
    model: str = DEFAULT_LIVE_MODEL,
    concurrency: int = 1,
    ram_gb: float | None = None,
) -> dict[str, Any]:
    conc = max(1, int(concurrency))
    status = "safe" if conc <= 2 else "caution"
    display = ALLOWED_MODELS.get(model, model)
    if conc == 1:
        msg = f"✓ {display}: one outcome per PCAI request."
    elif conc == 2:
        msg = f"⚠️ {display}: parallel requests are disabled in the simplified dashboard."
    else:
        msg = (
            f"⚠️ {display}: {conc} parallel requests is experimental and may "
            "increase provider timeouts or rate limiting."
        )
    return {
        "recommended": 1,
        "max_safe": 1,
        "estimated_ram_gb": 0.0,
        "total_ram_gb": 0.0,
        "status": status,
        "message": msg,
        "model_tier": display,
    }


def get_live_concurrency(model: str = DEFAULT_LIVE_MODEL) -> int:
    return 1


def is_ollama_available(model: str = DEFAULT_LIVE_MODEL, host: str = DEFAULT_HOST) -> bool:
    """Compatibility name retained for server.py; now means PCAI is configured."""
    return bool(os.environ.get("PCAI_API_KEY")) and model in ALLOWED_MODELS


def extract_and_grade_note(
    note_text: str,
    outcomes: list[str],
    patient_context: dict[str, Any] | None = None,
    use_ollama: bool = False,
    manual_features: dict[str, dict[str, Any]] | None = None,
    model: str = DEFAULT_LIVE_MODEL,
    host: str = DEFAULT_HOST,
    concurrency: int | None = None,
    num_ctx: int = OLLAMA_NUM_CTX,
) -> dict[str, dict[str, Any]]:
    """Live dashboard inference through St. Jude PCAI using the selected model."""
    outcome_ids = [normalize_outcome_id(o) for o in outcomes]

    if not use_ollama:
        err = RuntimeError(
            "PCAI is not configured. Set PCAI_API_KEY before launching the dashboard."
        )
        return {outcome: failed_item(outcome, err) for outcome in outcome_ids}

    items: dict[str, dict[str, Any]] = {}
    context = patient_context or {}
    workers = max(1, min(int(concurrency or get_live_concurrency(model)), len(outcome_ids), 4))

    if workers == 1 or len(outcome_ids) == 1:
        for outcome in outcome_ids:
            try:
                items[outcome] = _pcai_call_one(
                    note_text, outcome, model=model, patient_context=context
                )
            except Exception as exc:
                items[outcome] = failed_item(outcome, exc)
        return items

    with _ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                _pcai_call_one, note_text, outcome, model, context
            ): outcome
            for outcome in outcome_ids
        }
        for future in _as_completed(future_map):
            outcome = future_map[future]
            try:
                items[outcome] = future.result()
            except Exception as exc:
                items[outcome] = failed_item(outcome, exc)
    return items

