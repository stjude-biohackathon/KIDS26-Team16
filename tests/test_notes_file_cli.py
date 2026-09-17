"""--notes-file: running an extraction over a labelled CSV instead of the cache.

The failure this guards against is silent: a notes file is an evaluation set
whose denominator matters, and every default in the CLI was chosen for sampling
a 978-case corpus. Left alone, `--notes 20` truncates it and `--stratify` seeds
it away. These tests pin that the whole file runs, and that anything less is
said out loud.
"""
import csv
import json
import sys

import pytest

from experiments import medgemma_extraction as extraction

HEADER = ["case_id", "true_outcome", "original_case_text", "summary"]
ROWS = [
    ["C1", "Leg Ulcer", "A leg ulcer in a patient with sickle cell disease.", "Leg ulcer."],
    ["C2", "Fever", "A fever in a patient with sickle cell disease.", "Fever."],
    ["C3", "Leg Ulcer", "Another leg ulcer, sickle cell disease.", "Leg ulcer again."],
]


@pytest.fixture
def notes_csv(tmp_path):
    path = tmp_path / "cases.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(ROWS)
    return path


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", [str(extraction.__file__), *map(str, args)])
    return extraction.main()


def test_the_whole_file_runs_when_notes_is_not_given(notes_csv, tmp_path, monkeypatch):
    # --notes defaults to 20 for corpus sampling. Applied to a notes file it would
    # look like a deliberate sample size while silently truncating the eval set.
    out = tmp_path / "run.json"
    assert run_cli(monkeypatch, "--backend", "mock", "--notes-file", notes_csv,
                   "--no-stratify", "--outcomes", "40", "--out", out) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {r["patient_uid"] for r in data["detailed_records"]} == {"C1", "C2", "C3"}


def test_an_explicit_notes_limit_is_still_honoured(notes_csv, tmp_path, monkeypatch):
    out = tmp_path / "run.json"
    assert run_cli(monkeypatch, "--backend", "mock", "--notes-file", notes_csv,
                   "--notes", "2", "--no-stratify", "--outcomes", "40", "--out", out) == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))["detailed_records"]) == 2


def test_stratified_selection_over_a_small_file_warns_that_rows_were_dropped(
    notes_csv, tmp_path, monkeypatch, capsys,
):
    # The default stratifier seeds on outcome vocabulary and holds part back. When
    # the file's text does not carry an outcome's vocabulary its seeded slots find
    # nothing, and the run silently shrinks to the holdout - the single easiest way
    # to read a run as "the model found nothing".
    out = tmp_path / "run.json"
    assert run_cli(monkeypatch, "--backend", "mock", "--notes-file", notes_csv,
                   "--outcomes", "17,24", "--out", out) == 0
    printed = capsys.readouterr().out
    assert "--no-stratify" in printed
    assert len(json.loads(out.read_text(encoding="utf-8"))["detailed_records"]) < 3


def test_the_note_column_chooses_which_text_is_extracted(notes_csv, tmp_path, monkeypatch):
    out = tmp_path / "run.json"
    assert run_cli(monkeypatch, "--backend", "mock", "--notes-file", notes_csv,
                   "--note-column", "summary", "--no-stratify",
                   "--outcomes", "40", "--out", out) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    notes = {r["patient_uid"]: r["patient_note"] for r in data["detailed_records"]}
    assert notes["C1"] == "Leg ulcer."


def test_provenance_records_the_file_and_column_so_two_arms_can_be_told_apart(
    notes_csv, tmp_path, monkeypatch,
):
    out = tmp_path / "run.json"
    run_cli(monkeypatch, "--backend", "mock", "--notes-file", notes_csv,
            "--note-column", "summary", "--no-stratify", "--outcomes", "40", "--out", out)
    provenance = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert provenance["notes_file"] == str(notes_csv)
    assert provenance["note_column"] == "summary"
    # The cohort gate did not run, so claiming a cohort would misdescribe the run.
    assert provenance["cohort"] is None


def test_the_two_arms_cover_the_same_cases(notes_csv, tmp_path, monkeypatch):
    # Comparability is the whole point: differing note sets make the diff meaningless.
    ids = []
    for column in ("original_case_text", "summary"):
        out = tmp_path / f"{column}.json"
        run_cli(monkeypatch, "--backend", "mock", "--notes-file", notes_csv,
                "--note-column", column, "--no-stratify", "--outcomes", "40", "--out", out)
        data = json.loads(out.read_text(encoding="utf-8"))
        ids.append(sorted(r["patient_uid"] for r in data["detailed_records"]))
    assert ids[0] == ids[1] == ["C1", "C2", "C3"]


def test_cohort_is_rejected_because_the_gate_does_not_run(notes_csv, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "--backend", "mock", "--notes-file", notes_csv,
                "--cohort", "scd_primary", "--no-stratify", "--outcomes", "40")


def test_a_missing_notes_file_exits_before_any_model_call(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "--backend", "mock", "--notes-file", tmp_path / "absent.csv",
                "--no-stratify", "--outcomes", "40")


def test_a_bad_note_column_exits_with_a_usage_error(notes_csv, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "--backend", "mock", "--notes-file", notes_csv,
                "--note-column", "nope", "--no-stratify", "--outcomes", "40")


def test_the_bundled_cache_is_still_the_default_source(tmp_path, monkeypatch):
    out = tmp_path / "run.json"
    assert run_cli(monkeypatch, "--backend", "mock", "--notes", "1",
                   "--no-stratify", "--outcomes", "40", "--out", out) == 0
    provenance = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert provenance["notes_file"] is None and provenance["cohort"] == "loose"
