"""cohort.load_csv_notes: a labelled CSV as the note source.

The bundled cache is a corpus to sample from; a notes file is an evaluation set
where every row was chosen deliberately. These tests pin the difference: rows
are never filtered by the cohort gate, a row that would silently vanish either
raises or is reported, and the column choice is what makes the same file run
twice as two comparable arms.
"""
import csv
import json

import pytest

from experiments.cohort import load_csv_notes

HEADER = ["case_id", "true_outcome", "original_case_text", "summary"]
ROWS = [
    ["C1", "Leg Ulcer", "Full text for case one.", "Summary one."],
    ["C2", "Fever", "Full text for case two.", "Summary two."],
]


def write_csv(path, rows=ROWS, header=HEADER, *, bom=False):
    encoding = "utf-8-sig" if bom else "utf-8"
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_every_row_becomes_a_note_in_file_order(tmp_path):
    notes = load_csv_notes(write_csv(tmp_path / "n.csv"))
    assert [n["patient_uid"] for n in notes] == ["C1", "C2"]
    assert notes[0]["patient"] == "Full text for case one."


def test_the_column_choice_selects_which_text_is_the_note(tmp_path):
    src = write_csv(tmp_path / "n.csv")
    full = load_csv_notes(src, "original_case_text")
    brief = load_csv_notes(src, "summary")
    assert [n["patient"] for n in brief] == ["Summary one.", "Summary two."]
    # Same ids in the same order: that is what makes the two runs comparable.
    assert [n["patient_uid"] for n in full] == [n["patient_uid"] for n in brief]
    assert brief[0]["source_column"] == "summary"


def test_the_cohort_gate_is_not_applied(tmp_path):
    # No row mentions sickle cell. The gate would drop both; a curated file must not.
    rows = [["C1", "Fever", "A patient with a fever.", "Fever."]]
    assert len(load_csv_notes(write_csv(tmp_path / "n.csv", rows))) == 1


def test_a_row_whose_chosen_column_is_empty_is_dropped_and_reported(tmp_path, capsys):
    rows = [["C1", "Fever", "Full text.", ""], ["C2", "Fever", "Full text.", "Summary."]]
    notes = load_csv_notes(write_csv(tmp_path / "n.csv", rows), "summary")
    assert [n["patient_uid"] for n in notes] == ["C2"]
    assert "C1" in capsys.readouterr().out


def test_duplicate_ids_raise_instead_of_overwriting_each_other(tmp_path):
    # Results are keyed by patient_uid; a duplicate would shrink the run silently.
    rows = [["C1", "Fever", "One.", "One."], ["C1", "Fever", "Two.", "Two."]]
    with pytest.raises(ValueError, match="duplicate"):
        load_csv_notes(write_csv(tmp_path / "n.csv", rows))


def test_a_missing_column_names_the_columns_that_do_exist(tmp_path):
    with pytest.raises(ValueError, match="original_case_text"):
        load_csv_notes(write_csv(tmp_path / "n.csv"), "note_text")


def test_a_bom_written_by_excel_does_not_corrupt_the_first_header(tmp_path):
    notes = load_csv_notes(write_csv(tmp_path / "n.csv", bom=True))
    assert notes[0]["patient_uid"] == "C1"


def test_a_note_longer_than_the_default_field_cap_is_read(tmp_path):
    # csv's default limit is 128 KiB; a long case report would abort the read.
    long_text = "sickle cell. " * 20_000
    rows = [["C1", "Fever", long_text, "Summary."]]
    notes = load_csv_notes(write_csv(tmp_path / "n.csv", rows))
    assert notes[0]["patient"].startswith("sickle cell.")
    assert len(notes[0]["patient"]) > 128 * 1024


def test_age_and_sex_are_absent_rather_than_invented(tmp_path):
    # A CSV of note text carries no patient metadata. Filling these in would put
    # ungrounded values into the results file that nothing could trace back.
    notes = load_csv_notes(write_csv(tmp_path / "n.csv"))
    assert notes[0]["age"] is None and notes[0]["gender"] is None


def test_records_carry_the_keys_the_rest_of_the_pipeline_reads(tmp_path):
    notes = load_csv_notes(write_csv(tmp_path / "n.csv"))
    assert {"patient_uid", "patient", "title", "age", "gender"} <= set(notes[0])
    json.dumps(notes)                      # records land in the results file as-is


def test_a_missing_file_raises_rather_than_returning_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv_notes(tmp_path / "absent.csv")


def test_a_header_only_file_raises(tmp_path):
    with pytest.raises(ValueError, match="no data rows"):
        load_csv_notes(write_csv(tmp_path / "n.csv", []))
