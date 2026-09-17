"""Export review worksheets from one saved extraction run, without re-running a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.medgemma_extraction import feature_brief, prompt_features, stage

COMMON = ["run_id", "source_sha256", "uid", "selection", "outcome", "outcome_name"]
FIELDS = {
    "handcheck": COMMON + [
        "feature", "value", "quote", "unit_note", "status", "grade",
        "withheld_conflict", "supports_value", "reviewer_note",
    ],
    "conflicts": COMMON + ["feature", "values", "quotes", "reviewer_note"],
    "absence_audit": COMMON + [
        "model_said_present", "truly_absent", "reviewer_note", "look_for", "title", "note_text",
    ],
    "refuted_audit": COMMON + [
        "model_said_present", "truly_absent", "reviewer_note", "look_for", "title", "note_text",
        "extracted_features", "rule_reason",
    ],
}


def summarize(data: dict) -> dict:
    return {
        "provenance": data["provenance"],
        "automated_metrics": data["automated_metrics"],
        "grade_status_by_outcome": data["grade_status_by_outcome"],
        "profiling": data["profiling"],
    }


def export_reviews(
    source: pathlib.Path | str,
    output_dir: pathlib.Path | str | None = None,
    handcheck_limit: int = 100,
    absence_limit: int = 50,
) -> dict:
    """Sample grounded proposals, keep every conflict/refutation, and sample absences.

    CSVs use a UTF-8 BOM for Windows spreadsheet compatibility. Existing sheets
    are never overwritten because they may already contain a clinician's review.
    """
    source = pathlib.Path(source)
    raw = source.read_bytes()
    data = json.loads(raw)
    source_hash = hashlib.sha256(raw).hexdigest()
    run_id = data["provenance"].get("run_id") or source_hash[:16]
    directory = pathlib.Path(output_dir) if output_dir is not None else source.parent / source.stem
    paths = {name: directory / f"{name}.csv" for name in FIELDS}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Review sheet already exists: {path}; choose a new output directory")
    rows = {name: [] for name in FIELDS}
    prompt_stage = stage(data["provenance"].get("prompt_stage", "0"))
    for rec in data["detailed_records"]:
        for num, outcome in rec["outcomes"].items():
            common = {
                "run_id": run_id, "source_sha256": source_hash, "uid": rec["patient_uid"],
                "selection": rec.get("selection", ""), "outcome": num,
                "outcome_name": outcome["outcome_name"],
            }
            grade_result = outcome["grade_result"]
            accepted = outcome["accepted_findings"]
            conflicts = outcome["conflicts"]
            for finding in accepted:
                rows["handcheck"].append({
                    **common, "feature": finding["feature"], "value": finding["value"],
                    "quote": finding["quote"], "unit_note": finding.get("unit"),
                    "status": grade_result["status"], "grade": grade_result.get("grade"),
                    "withheld_conflict": finding["feature"] in conflicts,
                    "supports_value": "", "reviewer_note": "",
                })
            for feature, values in conflicts.items():
                rows["conflicts"].append({
                    **common, "feature": feature, "values": json.dumps(values),
                    "quotes": json.dumps([f["quote"] for f in accepted if f["feature"] == feature]),
                    "reviewer_note": "",
                })
            status = grade_result["status"]
            if status in {"absent", "refuted", "missed_presence"}:
                row = {
                    **common, "model_said_present": outcome["present"], "truly_absent": "",
                    "reviewer_note": "",
                    "look_for": " | ".join(feature_brief(n, num, prompt_stage)
                                          for n in prompt_features(num, prompt_stage)),
                    "title": rec.get("title", ""), "note_text": rec["patient_note"],
                }
                if status == "refuted":
                    row.update(extracted_features=json.dumps(outcome["extracted_features"]),
                               rule_reason=grade_result.get("reason"))
                audit_key = "refuted_audit" if status == "refuted" else "absence_audit"
                rows[audit_key].append(row)
    rng = random.Random(0)
    for name, limit in (("handcheck", handcheck_limit), ("absence_audit", absence_limit)):
        if limit is not None and limit >= 0:
            rows[name] = rng.sample(rows[name], min(limit, len(rows[name])))
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        with path.open("x", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS[name])
            writer.writeheader()
            writer.writerows(rows[name])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument(
        "--handcheck-limit",
        type=int,
        default=100,
        help="Maximum grounded proposals to sample for handcheck.csv (default: 100; pass -1 for all)",
    )
    parser.add_argument(
        "--absence-limit",
        type=int,
        default=50,
        help="Maximum absent pairs to sample for absence_audit.csv (default: 50; pass -1 for all)",
    )
    args = parser.parse_args()
    try:
        paths = export_reviews(
            args.results,
            args.output_dir,
            handcheck_limit=args.handcheck_limit,
            absence_limit=args.absence_limit,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"Could not export review sheets: {exc}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
