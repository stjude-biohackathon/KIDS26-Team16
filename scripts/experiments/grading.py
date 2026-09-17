"""From verified features to a grade status: the one grading path.

`grade_outcome()` is called by the CLI for every results file and by the
dashboard for live and manual grading, so a fix here reaches both. It extracts
and verifies nothing: its features have already passed
experiments/verification.py (or were typed in by a clinician).
"""
from __future__ import annotations

from dataclasses import dataclass

from scogs.applicability import applicability
from scogs.criteria import criteria_met
from scogs.evaluate import GradeResult, grade
from scogs.predicates import UNKNOWN


def harness_status(rule_status: str, present, criteria=None) -> str:
    """Reconcile model presence and objective criteria against rule engine outcome status.

    Clinical intent:
        Separates genuine absences from rule refutations and flagged presence contradictions.
        Without this check, when a model claims a patient has fever with 36.5 °C, the rule
        engine marks it absent, polluting the negative-control absence audit. Similarly,
        if objective criteria are met (e.g. TRV >= 2.5 m/s) but the model failed to recognize
        the condition, it must be flagged for clinical review rather than silently ignored.

    Checks performed:
        1. Contradicted Presence Check ('missed_presence'):
           If objective diagnostic criteria are met (`criteria is True`) but the model said
           the condition was not present (`present is not True`), returns 'missed_presence'.
        2. Rule Refutation Check ('refuted'):
           If the model claimed presence (`present is True`) but deterministic tables graded
           it as absent (`rule_status == "absent"`), returns 'refuted'.
        3. Pass-Through Status:
           Otherwise passes through rule engine status:
           - 'graded': Evaluated to a definitive single grade (1-5).
           - 'grade_set': Multiple candidate grades possible due to missing non-essential data.
           - 'cannot_grade': Missing essential prerequisite (e.g. patient age).
           - 'not_applicable': Demographic exclusion (e.g. sex restriction, infant age < 60d).
           - 'absent': Both model and rules agree outcome is absent.

    Returns:
        str: Reconciled harness status string.
    """
    if present is not True and criteria is True:
        return "missed_presence"
    return "refuted" if rule_status == "absent" and present else rule_status


@dataclass(frozen=True)
class OutcomeGrade:
    """One (note, outcome) pair, graded the way every saved run grades it."""

    result: GradeResult            # the rule engine's verdict: grade, matched row, missing features
    status: str                    # harness_status(): the rule status, or `refuted` / `missed_presence`
    applicability: object          # True | False | UNKNOWN - rules.md sex and age restrictions
    criteria: object               # True | False | UNKNOWN - objective presence criteria
    grade_if_present: int | float | None   # only for `missed_presence`: the grade had presence been called

    def to_record(self, features: dict, present) -> dict:
        """-> the `grade_result` object stored in results files.

        The keys and their order are the results-file format that review sheets
        and the dashboard read; change them deliberately, never while tidying.
        """
        return {
            "status": self.status,
            "rule_status": self.result.status,
            "grade": self.result.grade,
            "features": features,
            "present": present,
            "applicability": "unknown" if self.applicability is UNKNOWN else bool(self.applicability),
            "criteria_met": "unknown" if self.criteria is UNKNOWN else bool(self.criteria),
            "grade_if_present": self.grade_if_present,
            "reason": self.result.reason,
        }


def grade_outcome(outcome: str, features: dict, present) -> OutcomeGrade:
    """-> the grade for one outcome, from features that already passed verification.

    `present` is the presence call (True, False or None). Raises KeyError for an
    outcome id that has no decision table.
    """
    applies = applicability(outcome, features)
    criteria = criteria_met(outcome, features)
    # Only a definite False excludes an outcome: an unstated sex or age is
    # UNKNOWN, and unknown must never be read as "does not apply".
    applicable = applies is not False
    result = grade(outcome, features, present=bool(present), applicable=applicable)
    # "the model never saw this outcome" and "the model called it and the
    # tables overruled the call" are different questions. Pooled as one
    # `absent` they send a reviewer to confirm an absence the rule engine
    # produced, on a note where the model actually said present.
    status = harness_status(result.status, present, criteria=criteria is True)
    grade_if_present = (grade(outcome, features, present=True, applicable=applicable).grade
                        if status == "missed_presence" else None)
    return OutcomeGrade(result, status, applies, criteria, grade_if_present)
