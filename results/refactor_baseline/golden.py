"""Golden-output check for the maintainability refactor (tasks/plan.md, Task 0).

    python results/refactor_baseline/golden.py capture   # once, before Task 1
    python results/refactor_baseline/golden.py compare   # after every refactor task

capture runs the mock extraction CLI and the booklet audits and saves their
output next to this file; compare runs them again and fails on any difference.
Only values that change on every run are removed (timestamps, run ids,
wall-clock timings). Everything else - including JSON key order - must match,
because notebooks and review sheets read these files.
"""
from __future__ import annotations

import difflib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE = pathlib.Path(__file__).resolve().parent
EXTRACTION = ROOT / "scripts" / "experiments" / "medgemma_extraction.py"
AUDIT = ROOT / "scripts" / "audit"

MOCK_RUNS = {
    # default prompt stage, both optional passes, and repeats (consistency path)
    "stage3_context_feedback": ["--backend", "mock", "--notes", "3", "--repeat", "2",
                                "--patient-context", "--feedback-retry"],
    # the frozen baseline prompt over every outcome, unstratified selection
    "stage0_all_outcomes": ["--backend", "mock", "--prompt-stage", "0", "--no-stratify",
                            "--outcomes", "all", "--notes", "2"],
}
TIMING_LINE = re.compile(r"^(Run \d+ completed in |Wrote full test results)")


def run_python(args: list[str]) -> str:
    """-> stdout of `python <args>` run from the repository root; exits on failure."""
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"golden: command failed ({proc.returncode}): {args}\n{proc.stderr}")
    return proc.stdout


def scrub_results(data: dict) -> dict:
    """Drop the fields that differ on every run; the rest keeps its key order."""
    data["provenance"].pop("timestamp")
    data["provenance"].pop("run_id")
    data.pop("profiling")
    for tally in data["runs"]:
        tally.pop("wall_clock_sec", None)
    if data.get("patient_context_tally"):
        data["patient_context_tally"].pop("wall_clock_sec", None)
    return data


def extraction_outputs() -> dict[str, str]:
    outputs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, args in MOCK_RUNS.items():
            out = pathlib.Path(tmp) / f"{name}.json"
            stdout = run_python([str(EXTRACTION), *args, "--out", str(out)])
            kept = [line for line in stdout.splitlines() if not TIMING_LINE.match(line)]
            outputs[f"{name}.stdout.txt"] = "\n".join(kept) + "\n"
            data = scrub_results(json.loads(out.read_text(encoding="utf-8")))
            outputs[f"{name}.json"] = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    return outputs


def audit_outputs() -> dict[str, str]:
    """Booklet audits (need Poppler's pdftotext).

    Before Task 9, compare_prose.py re-ran the other two audits on import and
    printed their reports first. Only its own report is the stable contract, so
    that prefix is removed when present.
    """
    if shutil.which("pdftotext") is None:
        print("golden: pdftotext not on PATH - audit outputs SKIPPED")
        return {}
    booklet = run_python([str(AUDIT / "extract_booklet.py")])
    grades = run_python([str(AUDIT / "compare_grades.py")])
    prose = run_python([str(AUDIT / "compare_prose.py")])
    if prose.startswith(booklet + grades):
        prose = prose[len(booklet + grades):]
    return {
        "extract_booklet.stdout.txt": booklet,
        "booklet_grades.json": (AUDIT / "booklet_grades.json").read_text(encoding="utf-8"),
        "compare_grades.stdout.txt": grades,
        "grade_diffs.json": (AUDIT / "grade_diffs.json").read_text(encoding="utf-8"),
        "compare_prose.stdout.txt": prose,
    }


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    if mode not in ("capture", "compare"):
        print(__doc__)
        return 2
    current = {**extraction_outputs(), **audit_outputs()}

    if mode == "capture":
        for name, text in current.items():
            (BASELINE / name).write_text(text, encoding="utf-8")
        print(f"golden: captured {len(current)} outputs in {BASELINE}")
        return 0

    changed = 0
    for name, text in current.items():
        saved = BASELINE / name
        if not saved.exists():
            print(f"golden: no baseline for {name} - run capture first")
            changed += 1
            continue
        expected = saved.read_text(encoding="utf-8")
        if text != expected:
            changed += 1
            diff = difflib.unified_diff(expected.splitlines(), text.splitlines(),
                                        f"baseline/{name}", f"now/{name}", lineterm="", n=2)
            print("\n".join(list(diff)[:60]))
    if changed:
        print(f"golden: {changed} output(s) CHANGED")
        return 1
    print(f"golden: all {len(current)} outputs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
