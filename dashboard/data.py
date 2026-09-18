"""Loads saved results files and the legacy clinical-notes CSV.

File paths are relative to the repository root, so the app must be started
from there.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd


def is_run_file(filepath: str | Path) -> bool:
    """Returns True if the file exists, is JSON, and contains detailed_records."""
    path = Path(filepath)
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and "detailed_records" in data
    except Exception:
        return False


def load_run_file(filepath: str | Path) -> dict[str, Any]:
    """Reads structured run JSON and formats records indexed by patient_uid."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Run file not found: {filepath}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Run file is not a valid JSON object: {filepath}")

    detailed = data.get("detailed_records") or []
    records_by_uid: dict[str, dict] = {}
    for r in detailed:
        if isinstance(r, dict) and "patient_uid" in r:
            records_by_uid[str(r["patient_uid"])] = r

    return {
        "provenance": data.get("provenance", {}),
        "profiling": data.get("profiling", {}),
        "automated_metrics": data.get("automated_metrics", {}),
        "grade_status": data.get("grade_status", {}),
        "grade_status_by_outcome": data.get("grade_status_by_outcome", {}),
        "grade_status_by_selection": data.get("grade_status_by_selection", {}),
        "features_extracted": data.get("features_extracted", {}),
        "records_by_uid": records_by_uid,
        "detailed_records": detailed,
    }


def load_csv_notes(filepath: str | Path = "data/clinical_notes.csv") -> pd.DataFrame:
    """Loads and standardizes clinical notes CSV into expected schema."""
    path = Path(filepath)
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "patient_id",
                "visit_datetime",
                "facility_type",
                "clinical_note",
                "note_text",
                "gender",
                "patient_sex",
                "patient_age",
            ]
        )

    df = pd.read_csv(path)

    # SCOGS-Scribe summarized-case schema.
    # This lets the existing dashboard browse dashboard/SCD_summaries.csv
    # without changing the rest of the UI data contract.
    if "case_id" in df.columns and "summary" in df.columns:
        df["patient_id"] = df["case_id"].astype(str)
        df["clinical_note"] = df["summary"].fillna("").astype(str)
        df["note_text"] = df["clinical_note"]
        df["patient_sex"] = "unknown"
        df["patient_age"] = None
        if "true_outcomes" not in df.columns:
            if "true_outcome" in df.columns:
                df["true_outcomes"] = df["true_outcome"].fillna("").astype(str)
            else:
                df["true_outcomes"] = ""
        return df

    # Note text mapping
    if "clinical_note" in df.columns:
        df["clinical_note"] = df["clinical_note"].fillna("")
        df["note_text"] = df["clinical_note"]
    elif "note_text" in df.columns:
        df["note_text"] = df["note_text"].fillna("")
        df["clinical_note"] = df["note_text"]
    else:
        df["note_text"] = ""
        df["clinical_note"] = ""

    # Gender mapping
    if "gender" in df.columns:
        df["patient_sex"] = (
            df["gender"].astype(str).str.strip().str.lower()
            .map({"female": "female", "male": "male"})
            .fillna("unknown")
        )
    else:
        df["patient_sex"] = "unknown"

    # Age calculation from DOB and visit datetime (handling 2-digit century rollover)
    if "visit_datetime" in df.columns and "dob" in df.columns:
        try:
            visit_dt = pd.to_datetime(df["visit_datetime"], format="mixed", errors="coerce")
            dob_dt = pd.to_datetime(df["dob"], format="mixed", errors="coerce")
            future_dob = dob_dt > visit_dt
            dob_dt.loc[future_dob] = dob_dt.loc[future_dob] - pd.DateOffset(years=100)
            df["patient_age"] = (visit_dt - dob_dt).dt.days / 365.25
        except Exception:
            df["patient_age"] = None
    else:
        df["patient_age"] = None

    return df


def get_available_run_files() -> list[tuple[str, str]]:
    """Discovers available run JSON files or defaults to bundled test fixtures."""
    files: list[tuple[str, str]] = []
    results_dir = Path("results")
    if results_dir.is_dir():
        for p in sorted(results_dir.rglob("*.json")):
            if is_run_file(p):
                posix_path = p.as_posix()
                files.append((posix_path, posix_path))
    for p in sorted(glob.glob("tests/fixtures/*.json")):
        if is_run_file(p):
            files.append((p, f"fixture: {Path(p).name}"))
    if not files:
        files.append(("tests/fixtures/sample_run.json", "tests/fixtures/sample_run.json (Sample)"))
    return files
