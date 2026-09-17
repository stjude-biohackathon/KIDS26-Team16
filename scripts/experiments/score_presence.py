"""Score a run's presence calls against a labelled CSV, and diff two runs.

This scores PRESENCE ONLY - whether the model said an outcome is evidenced in
the note - because that is the only thing the labels support. A `true_outcome`
column names the outcomes a case has; it does not say how severe they were, so
no severity grade in a results file has anything to be checked against. A run
whose grades are all `cannot_grade` and a run whose grades are all correct score
identically here. Read `grade status` in the output as a coverage figure, never
as accuracy.

Labels are read closed-world: an outcome the run evaluated and the row does not
list is counted as a true absence. That is what makes recall meaningful, and it
is only sound because the label column enumerates every outcome the case has.

    python scripts/experiments/score_presence.py results/original.json \\
        --labels data/SCD_summaries.csv
    python scripts/experiments/score_presence.py results/original.json \\
        results/summary.json --labels data/SCD_summaries.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scogs.tables import TABLES

LABEL_SEPARATOR = "|"


def _name_to_id() -> dict[str, str]:
    """-> decision-table name -> outcome id, for reading the label column."""
    return {table.name: num for num, table in TABLES.items()}


def load_labels(path: str | pathlib.Path, *, id_column: str = "case_id",
                label_column: str = "true_outcome") -> dict[str, set[str]]:
    """-> {case_id: {outcome id, ...}} from the labelled CSV.

    Raises ValueError naming every label that does not match a decision table,
    rather than dropping it: a silently unmatched label deflates recall for an
    outcome that was in fact present, which looks exactly like a model miss.
    """
    src = pathlib.Path(path)
    if not src.exists():
        raise FileNotFoundError(f"labels file not found: {src}")
    with src.open(newline="", encoding="utf-8-sig") as handle:
        limit = csv.field_size_limit()
        csv.field_size_limit(2**31 - 1)
        try:
            rows = list(csv.DictReader(handle))
        finally:
            csv.field_size_limit(limit)
    if not rows:
        raise ValueError(f"{src} has no data rows")
    for needed in (id_column, label_column):
        if needed not in rows[0].keys():
            raise ValueError(f"{src}: no column {needed!r}")

    by_name = _name_to_id()
    labels: dict[str, set[str]] = {}
    unknown: set[str] = set()
    for row in rows:
        case = (row.get(id_column) or "").strip()
        if not case:
            continue
        ids = set()
        for raw in (row.get(label_column) or "").split(LABEL_SEPARATOR):
            name = raw.strip()
            if not name:
                continue
            if name in by_name:
                ids.add(by_name[name])
            else:
                unknown.add(name)
        labels[case] = ids
    if unknown:
        raise ValueError(
            f"{src}: {len(unknown)} label(s) match no decision table: "
            + ", ".join(sorted(unknown))
            + ". Fix the label text or add the outcome before scoring.")
    return labels


@dataclass
class Counts:
    """One confusion matrix. `unknown` is a pair the run left without a call."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    unknown: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "unknown": self.unknown, "precision": self.precision,
                "recall": self.recall, "f1": self.f1}


@dataclass
class Scored:
    """A scored run: overall and per-outcome confusion matrices, plus context."""

    path: str
    run_id: str | None
    note_column: str | None
    outcomes: list[str]
    cases: list[str]
    overall: Counts = field(default_factory=Counts)
    by_outcome: dict[str, Counts] = field(default_factory=dict)
    grade_status: dict[str, int] = field(default_factory=dict)
    missing_labels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "run_id": self.run_id,
            "note_column": self.note_column,
            "notes_scored": len(self.cases),
            "outcomes": self.outcomes,
            "overall": self.overall.as_dict(),
            "by_outcome": {num: c.as_dict() for num, c in sorted(self.by_outcome.items())},
            "grade_status": self.grade_status,
            "notes_without_labels": self.missing_labels,
        }


def score_run(results: dict, labels: dict[str, set[str]], *, path: str = "") -> Scored:
    """-> the run's presence confusion matrix against `labels`.

    Notes the label file does not cover are excluded from every count and listed
    in `missing_labels`; scoring them would mean guessing their ground truth.
    """
    provenance = results.get("provenance", {})
    records = results.get("detailed_records", [])
    if not records:
        raise ValueError(f"{path or 'results'}: no detailed_records; "
                         "re-run with --out so the notes and calls are saved")

    scored = Scored(
        path=path,
        run_id=provenance.get("run_id"),
        note_column=provenance.get("note_column"),
        outcomes=list(provenance.get("outcomes") or []),
        cases=[],
        grade_status=dict(results.get("grade_status") or {}),
    )
    for rec in records:
        case = rec.get("patient_uid")
        if case not in labels:
            scored.missing_labels.append(str(case))
            continue
        scored.cases.append(case)
        truth = labels[case]
        for num, detail in (rec.get("outcomes") or {}).items():
            counts = scored.by_outcome.setdefault(num, Counts())
            present = detail.get("present")
            is_true = num in truth
            for bucket in (counts, scored.overall):
                if present is None:
                    bucket.unknown += 1
                elif present and is_true:
                    bucket.tp += 1
                elif present and not is_true:
                    bucket.fp += 1
                elif not present and is_true:
                    bucket.fn += 1
                else:
                    bucket.tn += 1
    if not scored.cases:
        raise ValueError(f"{path or 'results'}: no note in the run matched a label row. "
                         "Check that --id-column matched the CSV's case_id column.")
    return scored


def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}"


def print_run(scored: Scored) -> None:
    """Print one run's presence scores, per outcome and overall."""
    label = scored.note_column or "?"
    print(f"\n{'=' * 78}")
    print(f"{pathlib.Path(scored.path).name}   note_column={label}   "
          f"run_id={scored.run_id or 'n/a'}")
    print(f"{'=' * 78}")
    if scored.missing_labels:
        print(f"  {len(scored.missing_labels)} note(s) had no label row and were excluded: "
              f"{', '.join(scored.missing_labels)}")
    print(f"  {len(scored.cases)} note(s) x {len(scored.outcomes)} outcome(s) scored\n")

    print(f"  {'id':>3} {'outcome':34} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} "
          f"{'prec%':>6} {'rec%':>6} {'F1%':>6}")
    for num, counts in sorted(scored.by_outcome.items()):
        if not (counts.tp or counts.fp or counts.fn):
            continue                      # an all-true-negative outcome says nothing
        print(f"  {num:>3} {TABLES[num].name[:34]:34} {counts.tp:>3} {counts.fp:>3} "
              f"{counts.fn:>3} {counts.tn:>3} {_pct(counts.precision):>6} "
              f"{_pct(counts.recall):>6} {_pct(counts.f1):>6}")
    silent = sum(1 for c in scored.by_outcome.values() if not (c.tp or c.fp or c.fn))
    if silent:
        print(f"  ({silent} outcome(s) omitted: never labelled and never called, "
              f"so they only add true negatives)")

    o = scored.overall
    print(f"\n  {'OVERALL':38} {o.tp:>3} {o.fp:>3} {o.fn:>3} {o.tn:>3} "
          f"{_pct(o.precision):>6} {_pct(o.recall):>6} {_pct(o.f1):>6}")
    if o.unknown:
        print(f"  {o.unknown} pair(s) had no presence call at all (unparseable reply).")

    if scored.grade_status:
        total = sum(scored.grade_status.values())
        shown = ", ".join(f"{k}={v}" for k, v in sorted(scored.grade_status.items()))
        graded = scored.grade_status.get("graded", 0)
        print(f"\n  grade status ({total} pairs): {shown}")
        print(f"  gradeable: {graded}/{total} ({graded / total * 100:.1f}%) "
              f"- coverage, NOT accuracy: these labels carry no severity grades.")


def print_comparison(runs: list[Scored],
                     moved: list[tuple[str, str, str, str]] | None = None) -> None:
    """Print the arm-to-arm diff: the point of running one file twice."""
    print(f"\n{'=' * 78}")
    print("COMPARISON")
    print(f"{'=' * 78}")
    cases = [set(r.cases) for r in runs]
    if len(set(map(frozenset, cases))) > 1:
        print("  WARNING: the runs do not cover the same notes; the diff below is "
              "not like-for-like.")
    if len({tuple(r.outcomes) for r in runs}) > 1:
        print("  WARNING: the runs do not cover the same outcomes; the diff below is "
              "not like-for-like.")

    head = f"  {'metric':22}" + "".join(f"{(r.note_column or pathlib.Path(r.path).stem)[:14]:>15}"
                                        for r in runs)
    print(head)
    rows = [
        ("presence precision %", lambda r: r.overall.precision),
        ("presence recall %", lambda r: r.overall.recall),
        ("presence F1 %", lambda r: r.overall.f1),
    ]
    for name, get in rows:
        line = f"  {name:22}" + "".join(f"{_pct(get(r)):>15}" for r in runs)
        if len(runs) == 2:
            a, b = get(runs[0]), get(runs[1])
            if a is not None and b is not None:
                line += f"   delta {(b - a) * 100:+.1f}"
        print(line)

    def graded_pct(r: Scored) -> float | None:
        total = sum(r.grade_status.values())
        return r.grade_status.get("graded", 0) / total if total else None

    line = f"  {'gradeable %':22}" + "".join(f"{_pct(graded_pct(r)):>15}" for r in runs)
    if len(runs) == 2:
        a, b = graded_pct(runs[0]), graded_pct(runs[1])
        if a is not None and b is not None:
            line += f"   delta {(b - a) * 100:+.1f}"
    print(line)

    if len(runs) == 2:
        moved = moved or []
        if moved:
            print(f"\n  {len(moved)} pair(s) where the presence call differs:")
            for case, num, first, second in moved[:40]:
                print(f"    {case} outcome {num} ({TABLES[num].name[:28]}): "
                      f"{first} -> {second}")
            if len(moved) > 40:
                print(f"    ... and {len(moved) - 40} more")
        else:
            print("\n  The two runs made identical presence calls on every shared pair.")


def presence_calls(results: dict) -> dict[tuple[str, str], object]:
    """-> {(case, outcome): present} for every pair a results file recorded."""
    return {(rec["patient_uid"], num): detail.get("present")
            for rec in results.get("detailed_records", [])
            for num, detail in (rec.get("outcomes") or {}).items()}


def disagreements(first: dict, second: dict) -> list[tuple[str, str, str, str]]:
    """-> pairs present in both runs whose presence call changed.

    Which pairs moved is the part of a summarised-versus-original comparison that
    a rate cannot show: two runs can share an F1 and still disagree everywhere.
    """
    a, b = presence_calls(first), presence_calls(second)
    return [(case, num, str(a[(case, num)]), str(b[(case, num)]))
            for case, num in sorted(set(a) & set(b)) if a[(case, num)] != b[(case, num)]]


def compare_runs(runs: list[Scored], raw: list[dict]) -> None:
    """Print the arm-to-arm comparison, including the pair-level presence diff."""
    moved = disagreements(raw[0], raw[1]) if len(runs) == 2 else []
    print_comparison(runs, moved)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="one or two results JSON files")
    ap.add_argument("--labels", required=True,
                    help="CSV with the ground-truth outcome column")
    ap.add_argument("--id-column", default="case_id")
    ap.add_argument("--label-column", default="true_outcome")
    ap.add_argument("--json-out", default=None,
                    help="also write the scores as JSON (never overwrites)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if len(args.results) > 2:
        ap.error("score at most two runs at a time; the comparison is pairwise")
    if args.json_out and pathlib.Path(args.json_out).exists():
        ap.error(f"Output already exists: {args.json_out}. Choose a new path.")

    try:
        labels = load_labels(args.labels, id_column=args.id_column,
                             label_column=args.label_column)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))

    raw, scored_runs = [], []
    for path in args.results:
        src = pathlib.Path(path)
        if not src.exists():
            ap.error(f"results file not found: {path}")
        try:
            results = json.loads(src.read_text(encoding="utf-8"))
            scored_runs.append(score_run(results, labels, path=path))
        except (json.JSONDecodeError, ValueError) as exc:
            ap.error(str(exc))
        raw.append(results)

    for scored in scored_runs:
        print_run(scored)
    if len(scored_runs) == 2:
        compare_runs(scored_runs, raw)

    print("\nPresence only. These labels carry no severity grades, so nothing here "
          "\nchecks whether a grade is right - only whether the outcome was spotted.")

    if args.json_out:
        payload = {"labels": str(args.labels),
                   "runs": [s.as_dict() for s in scored_runs]}
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
        print(f"Wrote scores to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
