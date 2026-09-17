"""score_presence: presence accuracy against a labelled CSV, and the two-arm diff.

The scorer's job is to be honest about what the labels can and cannot support.
These tests pin the three ways it could quietly lie: counting a note it has no
label for, dropping a label it failed to recognise, and letting the grade
figures read as accuracy when they are only coverage.
"""
import csv
import json

import pytest

from experiments.score_presence import (
    Counts, disagreements, load_labels, main, score_run,
)

HEADER = ["case_id", "true_outcome", "original_case_text", "summary"]


def write_labels(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def results(records, *, note_column="summary", outcomes=("28", "40"), grade_status=None):
    return {
        "provenance": {"run_id": "r1", "note_column": note_column,
                       "outcomes": list(outcomes)},
        "grade_status": grade_status or {},
        "detailed_records": records,
    }


def record(case, calls):
    return {"patient_uid": case,
            "outcomes": {num: {"present": present} for num, present in calls.items()}}


# ------------------------------------------------------------------- labels

def test_labels_are_parsed_into_outcome_ids(tmp_path):
    src = write_labels(tmp_path / "l.csv",
                       [["C1", "Leg Ulcer|Fever", "full", "brief"]])
    assert load_labels(src) == {"C1": {"40", "36"}}


def test_an_unrecognised_label_raises_instead_of_being_dropped(tmp_path):
    # A silently dropped label turns a real presence into a false negative, which
    # is indistinguishable from the model having missed it.
    src = write_labels(tmp_path / "l.csv", [["C1", "Leg Ulcer|Sickle Cell Wobble", "f", "b"]])
    with pytest.raises(ValueError, match="Sickle Cell Wobble"):
        load_labels(src)


def test_a_row_with_no_labels_is_kept_as_an_all_absent_case(tmp_path):
    src = write_labels(tmp_path / "l.csv", [["C1", "", "f", "b"]])
    assert load_labels(src) == {"C1": set()}


def test_every_label_in_the_bundled_csv_matches_a_decision_table():
    # The file this whole runbook targets. If a label stops matching, scoring it
    # would silently under-report recall rather than fail.
    labels = load_labels("data/SCD_summaries.csv")
    assert len(labels) == 3
    assert "28" in labels["PMC10166222_01"] and "40" in labels["PMC10166222_01"]


# -------------------------------------------------------------------- scoring

def test_the_confusion_matrix_counts_all_four_cells(tmp_path):
    labels = {"C1": {"28"}}
    scored = score_run(results([record("C1", {"28": True, "40": False})]), labels)
    assert (scored.overall.tp, scored.overall.fp) == (1, 0)
    assert (scored.overall.fn, scored.overall.tn) == (0, 1)


def test_an_unlabelled_outcome_the_model_called_is_a_false_positive():
    scored = score_run(results([record("C1", {"28": True, "40": True})]), {"C1": {"28"}})
    assert (scored.overall.tp, scored.overall.fp) == (1, 1)


def test_a_labelled_outcome_the_model_missed_is_a_false_negative():
    scored = score_run(results([record("C1", {"28": False, "40": False})]), {"C1": {"28", "40"}})
    assert scored.overall.fn == 2 and scored.overall.tp == 0


def test_a_pair_with_no_presence_call_is_counted_separately_not_as_a_miss():
    # An unparseable reply is not the model saying "absent"; pooling the two
    # would charge a transport failure to recall.
    scored = score_run(results([record("C1", {"28": None, "40": False})]), {"C1": {"28"}})
    assert scored.overall.unknown == 1
    assert scored.overall.fn == 0 and scored.overall.tn == 1


def test_a_note_without_a_label_row_is_excluded_and_named():
    scored = score_run(results([record("C1", {"28": True}), record("C9", {"28": True})]),
                       {"C1": {"28"}})
    assert scored.missing_labels == ["C9"]
    assert scored.cases == ["C1"] and scored.overall.tp == 1


def test_no_overlap_at_all_raises_rather_than_reporting_an_empty_score():
    with pytest.raises(ValueError, match="matched a label row"):
        score_run(results([record("C9", {"28": True})]), {"C1": {"28"}})


def test_a_run_without_detailed_records_raises():
    with pytest.raises(ValueError, match="detailed_records"):
        score_run(results([]), {"C1": {"28"}})


def test_per_outcome_counts_are_kept_alongside_the_pooled_ones():
    scored = score_run(results([record("C1", {"28": True, "40": True})]), {"C1": {"28"}})
    assert scored.by_outcome["28"].tp == 1
    assert scored.by_outcome["40"].fp == 1


def test_grade_status_is_carried_through_untouched():
    scored = score_run(results([record("C1", {"28": True})],
                               grade_status={"cannot_grade": 14, "graded": 0}),
                       {"C1": {"28"}})
    assert scored.grade_status == {"cannot_grade": 14, "graded": 0}


# ------------------------------------------------------------------- metrics

def test_precision_recall_and_f1():
    counts = Counts(tp=3, fp=1, fn=1, tn=5)
    assert counts.precision == 0.75 and counts.recall == 0.75 and counts.f1 == 0.75


def test_metrics_are_undefined_rather_than_zero_when_nothing_was_predicted():
    # 0/0 is not 0%. Printing 0.0 here would read as "the model was wrong".
    counts = Counts(tp=0, fp=0, fn=0, tn=4)
    assert counts.precision is None and counts.recall is None


# ---------------------------------------------------------------- comparison

def test_the_diff_lists_only_pairs_whose_call_changed():
    first = results([record("C1", {"28": True, "40": False})])
    second = results([record("C1", {"28": True, "40": True})])
    assert disagreements(first, second) == [("C1", "40", "False", "True")]


def test_the_diff_ignores_pairs_only_one_run_covered():
    first = results([record("C1", {"28": True})])
    second = results([record("C1", {"28": True, "40": True})])
    assert disagreements(first, second) == []


# --------------------------------------------------------------------- CLI

def test_the_cli_scores_two_runs_and_writes_json(tmp_path, capsys):
    labels = write_labels(tmp_path / "l.csv", [["C1", "Acute Sickle Cell Pain Episode", "f", "b"]])
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps(results([record("C1", {"28": True, "40": False})],
                                        note_column="original_case_text")), encoding="utf-8")
    second.write_text(json.dumps(results([record("C1", {"28": False, "40": False})],
                                         note_column="summary")), encoding="utf-8")
    out = tmp_path / "scores.json"
    assert main([str(first), str(second), "--labels", str(labels),
                 "--json-out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "COMPARISON" in printed
    assert "outcome 28" in printed            # the pair that changed is named
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [r["note_column"] for r in payload["runs"]] == ["original_case_text", "summary"]
    assert payload["runs"][0]["overall"]["tp"] == 1
    assert payload["runs"][1]["overall"]["fn"] == 1


def test_the_cli_refuses_to_overwrite_an_existing_score_file(tmp_path):
    labels = write_labels(tmp_path / "l.csv", [["C1", "Leg Ulcer", "f", "b"]])
    run = tmp_path / "a.json"
    run.write_text(json.dumps(results([record("C1", {"40": True})])), encoding="utf-8")
    out = tmp_path / "scores.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(run), "--labels", str(labels), "--json-out", str(out)])


def test_the_cli_refuses_more_than_two_runs(tmp_path):
    labels = write_labels(tmp_path / "l.csv", [["C1", "Leg Ulcer", "f", "b"]])
    with pytest.raises(SystemExit):
        main(["a.json", "b.json", "c.json", "--labels", str(labels)])
