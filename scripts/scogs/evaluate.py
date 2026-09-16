"""Layer 4 evaluation: (outcome, features) -> grade. Deterministic, auditable.

Three-valued throughout. A feature the note does not mention is UNKNOWN, never
False - because `not treated` must not silently become true just because nobody
wrote "treated" down. UNKNOWN propagates through the rows, and a row that cannot
be decided keeps its grade in the answer as a candidate instead of falling
through to a lower one.

That yields the four answers the project distinguishes:

    graded          exactly one grade, every clause decided
    grade_set       more than one grade still possible; says which and why
    absent          every row decided, none matched
    cannot_grade    a prerequisite is missing (e.g. age for a stratified outcome)

Two of the five answers are the caller's to supply, because the rubric does not
encode either judgement:

    present=False       Layer 1/2 found no evidence of this outcome  -> absent
    applicable=False    the outcome cannot apply to this patient
                        (pregnancy outcomes in a male patient)       -> not_applicable

This matters. Most Grade 4 cells read "<outcome> resulting in a life-threatening
complication where the need for life-supporting treatment indicated", and the
tables encode only the second half - so a patient on vasopressors for acute chest
syndrome would otherwise come back Grade 4 for papillary necrosis too. Layer 4
grades an outcome it is *told* is present; deciding presence is Layer 1/2's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import FEATURES, ordinal_rank
from .predicates import UNKNOWN, parse

# ------------------------------------------------------------------ result type

GRADED, GRADE_SET, ABSENT, CANNOT_GRADE, NOT_APPLICABLE = (
    "graded", "grade_set", "absent", "cannot_grade", "not_applicable")

@dataclass
class GradeResult:
    outcome: str
    status: str
    grade: int | None = None
    grades: tuple[int, ...] = ()
    reason: str | None = None
    matched: str | None = None                 # the rubric clause that fired
    undecided: tuple[tuple[int, str], ...] = ()  # (grade, clause) rows left UNKNOWN
    missing: tuple[str, ...] = ()              # features that would have decided them

    def __str__(self):
        if self.status == GRADED:        return f"{self.outcome}: Grade {self.grade}"
        if self.status == GRADE_SET:     return f"{self.outcome}: Grade {' or '.join(map(str, self.grades))}"
        if self.status == ABSENT:        return f"{self.outcome}: absent"
        if self.status == NOT_APPLICABLE:return f"{self.outcome}: not applicable"
        return f"{self.outcome}: cannot grade - {self.reason}"

    @property
    def needs_review(self) -> bool:
        return self.status in (GRADE_SET, CANNOT_GRADE)

# ------------------------------------------------------------------- comparison

class Context:
    """Ordinal-aware comparison against the feature registry."""

    def compare(self, name, left, op, right, left_is_value=False):
        spec = FEATURES.get(name)
        if spec is None:
            raise KeyError(f"undeclared feature {name!r}")

        if spec["type"] == "ord":
            a, b = ordinal_rank(name, left), ordinal_rank(name, right)
            if a is None or b is None:
                off = left if a is None else right
                raise ValueError(f"{name}: {off!r} is not one of {spec['values']}")
            return self._apply(op, a, b)

        if spec["type"] == "bool":
            if op not in ("==", "!="):
                raise ValueError(f"{name} is boolean; {op} is not defined on it")
            want = right if isinstance(right, bool) else str(right).lower() == "true"
            return self._apply(op, bool(left), want)

        # numeric / categorical
        if spec["type"] == "cat":
            if op not in ("==", "!="):
                raise ValueError(f"{name} is categorical; {op} is not defined on it")
            return self._apply(op, str(left), str(right))

        try:
            a, b = float(left), float(right)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: expected a number, got {left!r} / {right!r}")
        return self._apply(op, a, b)

    @staticmethod
    def _apply(op, a, b):
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b,
                "==": a == b, "!=": a != b}[op]

CTX = Context()

# --------------------------------------------------------------------- derived

def _truthy(env, key):
    v = env.get(key, UNKNOWN)
    return UNKNOWN if v is UNKNOWN or v is None else bool(v)

def resolve_derived(features: dict) -> dict:
    """Fill in features the tables use but the extractor does not emit directly.

    Each is computed only when absent, so an explicit value from the extractor or
    from a human annotator always wins.
    """
    env = dict(features)

    if "age_stratum" not in env:
        age = env.get("patient_age")
        if age is not None and age is not UNKNOWN:
            env["age_stratum"] = "pediatric" if float(age) < 18 else "adult"

    if "life_support" not in env:
        parts = [
            (env.get("resp_support") == "invasive_ventilation")
            if env.get("resp_support") not in (None, UNKNOWN) else UNKNOWN,
            _truthy(env, "vasopressors"),
            _truthy(env, "renal_replacement"),
            _truthy(env, "life_support_other"),
        ]
        if any(p is True for p in parts):
            env["life_support"] = True
        elif all(p is False for p in parts):
            env["life_support"] = False
        # otherwise leave UNKNOWN - some component is undocumented

    if "pro_severe_count" not in env:
        checks = []
        for key, op, thr in (("pain_hurt_score", "<", 60),
                             ("promis_interference_t", ">", 64),
                             ("promis_behavior_t", ">", 57)):
            v = env.get(key)
            if v is None or v is UNKNOWN: checks.append(UNKNOWN)
            else: checks.append(v < thr if op == "<" else v > thr)
        if UNKNOWN not in checks:
            env["pro_severe_count"] = sum(1 for c in checks if c)

    return env

# ----------------------------------------------------------------- table walking

_CACHE: dict[str, object] = {}

def _ast(pred: str):
    if pred not in _CACHE: _CACHE[pred] = parse(pred)
    return _CACHE[pred]

def _walk(rows, env):
    """-> (matched_grade, matched_clause, [(grade, clause, missing) left UNKNOWN above it])"""
    undecided = []
    for g, pred in rows:
        node = _ast(pred)
        v = node.evaluate(env, CTX)
        if v is True:
            return g, pred, undecided
        if v is UNKNOWN:
            missing = tuple(sorted(n for n in node.names()
                                   if env.get(n, UNKNOWN) is UNKNOWN or env.get(n) is None))
            undecided.append((g, pred, missing))
    return None, None, undecided

def _result(outcome, matched, clause, undecided):
    if matched is not None and not undecided:
        return GradeResult(outcome, GRADED, grade=matched, matched=clause)
    if matched is not None:
        gs = tuple(sorted({matched} | {g for g, _, _ in undecided}, reverse=True))
        miss = tuple(sorted({m for _, _, ms in undecided for m in ms}))
        return GradeResult(outcome, GRADE_SET, grade=None, grades=gs, matched=clause,
                           undecided=tuple((g, c) for g, c, _ in undecided), missing=miss,
                           reason="undetermined evidence could support a higher grade")
    if undecided:
        gs = tuple(sorted({g for g, _, _ in undecided}, reverse=True))
        miss = tuple(sorted({m for _, _, ms in undecided for m in ms}))
        return GradeResult(outcome, CANNOT_GRADE, grades=gs,
                           undecided=tuple((g, c) for g, c, _ in undecided), missing=miss,
                           reason="no rule decided; missing " + ", ".join(miss))
    return GradeResult(outcome, ABSENT)

def grade(outcome: str, features: dict, *, present: bool = True,
          applicable: bool = True) -> GradeResult:
    """Grade one outcome, on the assumption that it is present.

    `features` maps feature name -> value; omit what the note does not say. Do
    NOT pass False for an undocumented finding - that is the difference between
    "the note says no" and "the note does not say", and the whole result type
    exists to keep them apart.

    Pass `present=False` when Layer 1/2 found no evidence of the outcome, and
    `applicable=False` when it cannot apply to this patient at all.
    """
    from .tables import TABLES
    if outcome not in TABLES:
        raise KeyError(f"unknown outcome {outcome!r}")
    if not applicable:
        return GradeResult(outcome, NOT_APPLICABLE, reason="outcome does not apply to this patient")
    if not present:
        return GradeResult(outcome, ABSENT, reason="no evidence of this outcome in the note")

    table = TABLES[outcome]
    env = resolve_derived(features)

    if table.eval == "stratified":
        key = env.get(table.on)
        if key is None or key is UNKNOWN:
            why = ("cannot grade: age unknown" if table.on == "age_stratum"
                   else f"cannot grade: {table.on} unknown")
            return GradeResult(outcome, CANNOT_GRADE, reason=why, missing=(table.on,))
        skey = str(key).lower() if isinstance(key, bool) else str(key)
        if skey not in table.strata:
            raise ValueError(f"{outcome}: {table.on}={key!r} is not one of {sorted(table.strata)}")
        return _result(outcome, *_walk(table.strata[skey], env))

    if table.eval == "max_of":
        best, best_clause, undecided = None, None, []
        for axis, rows in table.axes.items():
            g, clause, und = _walk(rows, env)
            if g is not None and (best is None or g > best):
                best, best_clause = g, f"[{axis}] {clause}"
            undecided += [(gg, f"[{axis}] {cc}", ms) for gg, cc, ms in und]
        # only undetermined rows that could BEAT the best grade still matter
        undecided = [u for u in undecided if best is None or u[0] > best]
        return _result(outcome, best, best_clause, undecided)

    return _result(outcome, *_walk(table.rows, env))

def grade_all(features: dict, *, present: dict[str, bool] | None = None,
              applicable: dict[str, bool] | None = None) -> dict[str, GradeResult]:
    """Grade every outcome. `present` defaults to True for all 53, which is only
    right when the caller really has run detection over all of them."""
    from .tables import TABLES
    present, applicable = present or {}, applicable or {}
    return {num: grade(num, features, present=present.get(num, True),
                       applicable=applicable.get(num, True))
            for num in sorted(TABLES)}
