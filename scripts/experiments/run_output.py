"""What a finished extraction run prints and saves.

Turns the verified results from `medgemma_extraction.run()` into the console
report, the grade summary, and the results-file dictionary that notebooks, the
dashboard and review_results.py read. No model calls and no verification: every
function is plain computation over results that already exist, plus printing.

The results-file layout (keys and their order) is a contract with those
readers. Change it deliberately, never as a side effect of tidying.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field

from experiments.cohort import is_scd_primary
from experiments.grading import grade_outcome
from experiments.ollama_backend import WEIGHTS
from experiments.verification import Tally
from scogs.tables import TABLES


@dataclass
class ServedModel:
    """What Ollama reported serving after preflight. Empty for the mock backend."""

    model_info: dict = field(default_factory=dict)
    digest: str | None = None
    quant: str | None = None


@dataclass
class Repeats:
    """One entry per --repeat.

    runs[i] is {patient_uid: {outcome: (features, present, raw_reply, detail)}};
    context_runs[i] is {patient_uid: (features, raw_reply, detail)} when the
    prompt stage extracts patient context.
    """

    runs: list[dict] = field(default_factory=list)
    tallies: list[Tally] = field(default_factory=list)
    context_runs: list[dict] = field(default_factory=list)
    context_tallies: list[Tally] = field(default_factory=list)


@dataclass
class GradeSummary:
    """Grade statuses for the first repeat, counted three ways, plus each pair's record."""

    statuses: Counter
    by_outcome: dict[str, Counter]
    by_selection: dict[str, Counter]
    records: dict[str, dict[str, dict]]     # patient_uid -> outcome -> results-file `grade_result`


def summarize_grades(first_run: dict, outcomes: list[str], selection: dict[str, str],
                     tally: Tally) -> GradeSummary:
    """-> grade statuses for every (note, outcome) pair of the first repeat.

    Side effect: each `missed_presence` is added to `tally.presence_contradicted`,
    so read `tally.report()` only after calling this.
    """
    summary = GradeSummary(
        statuses=Counter(),
        by_outcome={num: Counter() for num in outcomes},
        by_selection={"seeded": Counter(), "holdout": Counter(), "random": Counter()},
        records={},
    )
    for uid, per_outcome in first_run.items():
        summary.records[uid] = {}
        for num, (features, present, *_) in per_outcome.items():
            graded = grade_outcome(num, features, present)
            if graded.status == "missed_presence":
                tally.presence_contradicted += 1
            summary.statuses[graded.status] += 1
            summary.by_outcome[num][graded.status] += 1
            # selection values are "seeded:<outcome>", "holdout" or "random"
            summary.by_selection[selection[uid].split(":")[0]][graded.status] += 1
            summary.records[uid][num] = graded.to_record(features, present)
    return summary


def measure_consistency(runs: list[dict], outcomes: list[str]) -> tuple[int, int]:
    """-> (pairs identical in every repeat, pairs compared).

    A pair is identical when its features AND presence match in every repeat;
    the raw reply text may differ.
    """
    same = total = 0
    for uid in runs[0]:
        for num in outcomes:
            total += 1
            same += all(runs[0][uid][num][:2] == other[uid][num][:2] for other in runs[1:])
    return same, total


def quality_band(value: float, good: float, workable: float) -> str:
    """-> "GOOD (≥good%)", "WORKABLE (workable-good%)" or "CONCERNING (<workable%)"."""
    if value >= good:
        return f"GOOD (≥{good}%)"
    if value >= workable:
        return f"WORKABLE ({workable}-{good}%)"
    return f"CONCERNING (<{workable}%)"


def print_run_banner(args: argparse.Namespace, notes: list[dict], outcomes: list[str]) -> None:
    """Print the run banner describing configuration, backend, and notes count."""
    print("=" * 70)
    print("P11 MedGemma Extraction Test")
    print(f"weights={WEIGHTS if args.backend == 'ollama' else 'none (mock)'}  "
          f"served-as={args.model}  backend={args.backend}")
    print(f"notes={len(notes)}  outcomes={','.join(outcomes)}  repeat={args.repeat}  "
          f"concurrency={args.concurrency}")
    print("=" * 70)


def print_concurrency_warning(concurrency: int) -> None:
    """Print warning when concurrency > 1 is combined with repeat > 1."""
    print()
    print("!! CONCURRENCY WARNING - run-to-run consistency is confounded.")
    print(f"   At --concurrency {concurrency} the server batches requests, and a batch's")
    print("   composition depends on timing, so it differs between repeats. Batched float")
    print("   reductions are not bit-identical, so a token can flip at temperature 0 for")
    print("   reasons that have nothing to do with the model. Mismatches below are then")
    print("   'model nondeterminism OR batching', and you cannot tell which.")
    print("   Measure consistency with --concurrency 1. Use >1 for throughput and cost.")
    print()


def print_run_tally(run_number: int, tally: Tally, context_tally: Tally | None, n_notes: int) -> None:
    """Print the tally report for a completed repeat."""
    report = tally.report()
    sec_per_note = tally.wall_clock_sec / n_notes if n_notes else 0
    tok_per_sec = tally.completion_tokens / tally.wall_clock_sec if tally.wall_clock_sec > 0 else 0
    print(f"\nRun {run_number} completed in {tally.wall_clock_sec:.1f}s ({sec_per_note:.2f}s/note, {tok_per_sec:.1f} tok/s):")
    if context_tally is not None:
        context_report = context_tally.report()
        print(f"  Patient context:      {context_report['accepted']} accepted / {context_report['proposed']} proposed "
              f"({context_report['quote_verified']} verified quotes, {context_report['quote_unfound']} unfound)")
    print(f"  Proposed findings:    {report['proposed']}")
    print(f"    null placeholders:  {report['null_placeholder']:4d}  {report['null_placeholder_pct']:5.1f}%  (no quote -> prompt not followed)")
    print(f"    quote verified:     {tally.quote_ok:4d}  {report['quote_verified_pct_of_quoted']:5.1f}% of quoted | {report['quote_verified_pct']:.1f}% of all")
    print(f"    quote not in note:  {tally.quote_unfound:4d}  {report['hallucinated_pct_of_quoted']:5.1f}% of quoted | {report['hallucinated_quote_pct']:.1f}% of all")
    print(f"  Accepted findings:    {report['accepted']}")
    print("     ^ a verified quote means the words are in the note, NOT that they")
    print("       support the value. Precision needs the hand-check sheet.")
    print(f"  Invalid values:       {report['invalid_value']}")
    if report['unit_converted'] or report['unit_mismatch'] or report['unit_ambiguous']:
        print(f"  Unit guard:           {report['unit_converted']} converted into the "
              f"schema's unit, {report['unit_mismatch']} rejected as unconvertible, "
              f"{report['unit_ambiguous']} left alone (quote carried several units)")
    if report['quote_value_mismatch']:
        print(f"  Number not in quote:  {report['quote_value_mismatch']} rejected - the "
              f"value is neither the number its quote carries nor its conversion")
    if report['present_true']:
        print(f"    present=true:       {report['present_true']:4d}  of which quoted "
              f"{report['present_quoted']} ({report['present_quoted_pct']:.1f}%), "
              f"unfound {report['present_quote_unfound']}, unquoted {report['present_unquoted']}")
    if report['content_retries'] or report['unusable_replies']:
        print(f"    content retries:    {report['content_retries']:4d}  "
              f"still unusable after retrying: {report['unusable_replies']}")
    if report.get('feedback_retries'):
        print(f"    feedback retries:   {report['feedback_retries']:4d}  "
              f"re-prompts with error feedback")
    if report['value_conflicts']:
        print(f"  !! VALUE CONFLICTS:   {report['value_conflicts']} feature(s) had several "
              f"verified values and no aggregation rule.")
        print("     Withheld from grading rather than guessed at; listed per note in --out.")
    if report['tokenizer_artifacts']:
        print(f"  !! TOKENIZER ARTIFACTS: {report['tokenizer_artifacts']} quotes carry corrupt GGUF byte tokens.")
        print("     The served weights are broken; these numbers are not a clean measurement.")
    print(f"  Unparseable replies:  {report['unparseable_replies']}")
    print(f"  Prompt tokens:        {report['prompt_tokens']} (~{report['prompt_tokens']//n_notes} tok/note)")
    print(f"  Completion tokens:    {report['completion_tokens']} (~{report['completion_tokens']//n_notes} tok/note)")


def print_grade_summary(summary: GradeSummary, n_notes: int, outcomes: list[str]) -> None:
    """Print the grade summary table and breakdown across outcomes and selection cohorts."""
    print(f"\nGrade status over {n_notes}x{len(outcomes)} note-outcome pairs:")
    for k, v in summary.statuses.most_common():
        print(f"   {k:15s} {v:3d} ({100*v/(n_notes*len(outcomes)):.1f}%)")

    cols = ["graded", "grade_set", "cannot_grade", "absent", "refuted", "missed_presence", "not_applicable"]
    print(f"\nPer outcome (n={n_notes} each) - a pooled number hides this shape:")
    print(f"   {'':>3s} {'outcome':30s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for num in outcomes:
        c = summary.by_outcome[num]
        print(f"   {num:>3s} {TABLES[num].name[:30]:30s} "
              + " ".join(f"{c.get(col, 0):>12d}" for col in cols))

    print("\nSeeded vs unstratified holdout - the size of the selection bias:")
    for k, c in summary.by_selection.items():
        if sum(c.values()):
            print(f"   {k:8s} n={sum(c.values()):3d}  " + "  ".join(
                f"{col}={c.get(col, 0)}" for col in cols if c.get(col)))


def report_consistency(runs: list[dict], outcomes: list[str], repeat: int, concurrency: int) -> float | None:
    """Report run-to-run consistency across repeats; returns consistency_pct or None."""
    if repeat <= 1:
        return None
    same, tot = measure_consistency(runs, outcomes)
    consistency_pct = round(100 * same / tot, 1)
    print(f"\nTemperature-0 consistency across runs 1-{repeat}: {consistency_pct}% "
          f"({same}/{tot} pairs have identical features and presence in every repeat)")
    if concurrency == 1:
        print("   At --concurrency 1 with greedy decoding this is close to a tautology:")
        print("   100% is the expected result and evidences nothing about the model.")
        print("   It is a smoke test for a nondeterministic serving stack, not a metric.")
    return consistency_pct


def print_metrics_assessment(report: dict, consistency_pct: float | None, concurrency: int) -> float:
    """Print the threshold assessment for automated metrics and return invalid_value_pct."""
    print("\n" + "=" * 70)
    print("Automated Metrics Evaluation (docs/research/extraction_protocol.md):")
    qv_quoted = report["quote_verified_pct_of_quoted"]
    qv_status = quality_band(qv_quoted, 95, 85)
    if consistency_pct is None:
        cs_status = "NOT MEASURED (needs --repeat 2)"
    elif consistency_pct < 90:
        cs_status = "CONCERNING (<90%)"
    elif consistency_pct < 98:
        cs_status = "WORKABLE (90-98%)"
    elif concurrency == 1:
        cs_status = "EXPECTED - greedy and unbatched; a near-tautology, not evidence"
    else:
        cs_status = "GOOD (≥98%) and meaningful - it held under batching"
    iv_pct = 100 * report["invalid_value"] / (report["proposed"] or 1)
    iv_status = "GOOD (≤2%)" if iv_pct <= 2 else ("WORKABLE (2-10%)" if iv_pct <= 10 else "CONCERNING (>10%)")
    print(f"  - Quote-verified % (of quoted proposals):  {qv_quoted}% -> {qv_status}")
    print(f"  - Quote-verified % (of ALL proposals):     {report['quote_verified_pct']}%")
    print("      NB: quote-verified is a GROUNDING check, not precision. It asks only")
    print("      whether the quoted words appear in the note - a quote that does not")
    print("      support its value passes it. Precision comes from the hand-check sheet.")
    print(f"  - Null-placeholder rate:   {report['null_placeholder_pct']}% -> "
          f"{'GOOD (≤5%)' if report['null_placeholder_pct'] <= 5 else 'CONCERNING - the prompt omission rule is being ignored'}")
    cs_value = "  n/a" if consistency_pct is None else f"{consistency_pct}%"
    print(f"  - Run-to-run consistency:  {cs_value} -> {cs_status}")
    print(f"  - Invalid-value rate:      {iv_pct:.1f}% -> {iv_status}")
    print(f"  - Unit conversions:        {report['unit_converted']} applied, "
          f"{report['unit_mismatch']} unconvertible, "
          f"{report['quote_value_mismatch']} value/quote mismatches")
    print(f"  - Value conflicts:         {report['value_conflicts']} withheld "
          f"(several verified values, no aggregation rule)")
    print(f"  - Unparseable replies:     {report['unparseable_replies']} -> {'GOOD (0)' if report['unparseable_replies']==0 else 'CONCERNING'}")
    print("=" * 70)
    return iv_pct


def build_results(args: argparse.Namespace, st, served: ServedModel, notes: list[dict],
                  selection: dict[str, str], outcomes: list[str], repeats: Repeats,
                  summary: GradeSummary, first_report: dict, consistency_pct: float | None,
                  invalid_value_pct: float) -> dict:
    """Build the final structured results dictionary for JSON output."""
    detailed_records = []
    for rec in notes:
        uid = rec["patient_uid"]
        per_outcome_details = {}
        for num in outcomes:
            feats, present, reply, detail = repeats.runs[0][uid][num]
            per_outcome_details[num] = {
                "outcome_name": TABLES[num].name,
                "present": present,
                "extracted_features": feats,
                "accepted_findings": detail["accepted"],
                "conflicts": detail["conflicts"],
                "grade_result": summary.records[uid][num],
                "raw_reply": reply,
            }
        rec_dict = {
            "patient_uid": uid,
            "selection": selection[uid],          # seeded:<outcome> | holdout | random
            "scd_primary": is_scd_primary(rec),   # a label now, not a filter
            "title": rec.get("title", ""),
            "age": rec.get("age"),
            "gender": rec.get("gender"),
            "patient_note": rec["patient"],
            "outcomes": per_outcome_details,
        }
        if st.patient_context and repeats.context_runs:
            ctx_feats, ctx_reply, ctx_det = repeats.context_runs[0].get(uid, ({}, "", {}))
            rec_dict["patient_context"] = {
                "extracted_features": ctx_feats,
                "accepted_findings": ctx_det.get("accepted", []),
                "conflicts": ctx_det.get("conflicts", []),
                "raw_reply": ctx_reply,
            }
        detailed_records.append(rec_dict)

    provenance = {
        "tier": "full" if args.backend == "ollama" else "mock",
        "weights": WEIGHTS if args.backend == "ollama" else None,
        "served_as": args.model if args.backend == "ollama" else None,
        "backend": args.backend,
        "quant": served.quant,
        "model_digest": served.digest,
        "parameter_count": served.model_info.get("model_info", {}).get("general.parameter_count"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes_count": len(notes),
        "cohort": args.cohort,
        "stratified": args.stratify,
        "holdout_frac": args.holdout_frac if args.stratify else None,
        "outcomes": outcomes,
        "repeat": args.repeat,
        "concurrency": args.concurrency,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict or (2048 if st.cot else 1024),
        "temperature": 0.0,
        "initial_seed": 0,
        "repeat_penalty": 1.0 if st.flat_penalty else 1.1,
        "consistency_definition": "features and presence identical across all repeats",
        "prompt_stage": st.name,
        "prompt_stage_flags": {k: v for k, v in vars(st).items()
                               if k != "name" and v},
        "consistency_confounded_by_batching": args.concurrency > 1 and args.repeat > 1,
    }
    provenance["run_id"] = "{}-{:08x}".format(
        provenance["timestamp"].replace("-", "").replace(":", "").rstrip("Z"),
        zlib.crc32(json.dumps(provenance, sort_keys=True).encode("utf-8")))

    return {
        "provenance": provenance,
        "profiling": {
            "total_wall_clock_sec": round(repeats.tallies[0].wall_clock_sec, 2),
            "sec_per_note": round(repeats.tallies[0].wall_clock_sec / len(notes), 2),
            "total_prompt_tokens": repeats.tallies[0].prompt_tokens,
            "total_completion_tokens": repeats.tallies[0].completion_tokens,
            "prompt_tokens_per_note": repeats.tallies[0].prompt_tokens // len(notes),
            "completion_tokens_per_note": repeats.tallies[0].completion_tokens // len(notes),
            "completion_tokens_per_sec": round(repeats.tallies[0].completion_tokens / repeats.tallies[0].wall_clock_sec, 1) if repeats.tallies[0].wall_clock_sec > 0 else 0,
        },
        "automated_metrics": {
            "proposed": first_report["proposed"],
            "quoted": first_report["quoted"],
            "quote_verified": first_report["quote_verified"],
            "quote_unfound": first_report["quote_unfound"],
            "null_placeholder": first_report["null_placeholder"],
            "null_placeholder_pct": first_report["null_placeholder_pct"],
            "quote_verified_pct": first_report["quote_verified_pct"],
            "quote_verified_pct_of_quoted": first_report["quote_verified_pct_of_quoted"],
            "hallucinated_quote_pct": first_report["hallucinated_quote_pct"],
            "hallucinated_pct_of_quoted": first_report["hallucinated_pct_of_quoted"],
            "tokenizer_artifacts": first_report["tokenizer_artifacts"],
            "invalid_value_count": first_report["invalid_value"],
            "invalid_value_pct": round(invalid_value_pct, 1),
            "unit_converted": first_report["unit_converted"],
            "unit_ambiguous": first_report["unit_ambiguous"],
            "unit_mismatch": first_report["unit_mismatch"],
            "quote_value_mismatch": first_report["quote_value_mismatch"],
            "value_conflicts": first_report["value_conflicts"],
            "accepted": first_report["accepted"],
            "unknown_feature": first_report["unknown_feature"],
            "present_true": first_report["present_true"],
            "present_quoted": first_report["present_quoted"],
            "present_quote_unfound": first_report["present_quote_unfound"],
            "present_unquoted": first_report["present_unquoted"],
            "presence_contradicted": first_report["presence_contradicted"],
            "present_quoted_pct": first_report["present_quoted_pct"],
            "content_retries": first_report["content_retries"],
            "unusable_replies": first_report["unusable_replies"],
            "feedback_retries": first_report.get("feedback_retries", 0),
            "quote_verified_measures": "quote presence in the note, not value support",
            "unparseable_replies": first_report["unparseable_replies"],
            "run_to_run_consistency_pct": consistency_pct,
        },
        "runs": [t.report() for t in repeats.tallies],
        "patient_context_tally": repeats.context_tallies[0].report() if (st.patient_context and repeats.context_tallies) else None,
        "grade_status": dict(summary.statuses),
        "grade_status_by_outcome": {k: dict(v) for k, v in summary.by_outcome.items()},
        "grade_status_by_selection": {k: dict(v) for k, v in summary.by_selection.items() if sum(v.values())},
        "features_extracted": dict(repeats.tallies[0].per_feature),
        "detailed_records": detailed_records,
    }


def write_results(path: str, results: dict) -> None:
    """Write results dict to path in 'x' mode (refuses overwrite)."""
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(results, indent=2))
    print(f"\nWrote full test results and note texts to {path}")
