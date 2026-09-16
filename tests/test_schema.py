"""The schema is the contract: every table identifier is declared, nothing is
orphaned, and the reachable grade set of every table matches the rubric exactly."""
import re
import pathlib

import pytest

from scogs.features import FEATURES
from scogs.predicates import parse
from scogs.tables import TABLES

ROOT = pathlib.Path(__file__).resolve().parent.parent


def table_identifiers():
    used = set()
    for t in TABLES.values():
        if t.on: used.add(t.on)
        for _, pred in t.all_rows(): used |= parse(pred).names()
    return used


def derived_inputs():
    used = set()
    for spec in FEATURES.values():
        if spec["derived"]:
            for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", spec["derived"]):
                if w in FEATURES: used.add(w)
    return used


def test_every_table_identifier_is_a_declared_feature():
    assert sorted(table_identifiers() - set(FEATURES)) == []


def test_no_orphan_features():
    """A feature nobody grades on is schema drift; extraction-only vocabulary
    belongs to the Layer 1/2 lexicon, not to the grading contract."""
    assert sorted(set(FEATURES) - table_identifiers() - derived_inputs()) == []


def test_every_feature_has_a_definition_and_a_type():
    for name, spec in FEATURES.items():
        assert spec["type"] in {"bool", "num", "ord", "cat"}, name
        assert spec["definition"].strip(), name
        if spec["type"] in {"ord", "cat"}:
            assert spec["values"], f"{name} is {spec['type']} but declares no values"
        if spec["type"] == "num":
            assert spec["unit"], f"{name} is numeric but declares no unit"


def test_ordinal_values_are_unique():
    for name, spec in FEATURES.items():
        if spec["type"] == "ord":
            assert len(spec["values"]) == len(set(spec["values"])), name


def test_enum_comparisons_use_declared_values():
    """Catches the class of bug where a table compares against a value the
    feature never declares (e.g. invasive_procedure == surgical_shunt)."""
    from scogs.predicates import Cmp, In

    def walk(node):
        yield node
        for f in ("child",):
            if hasattr(node, f): yield from walk(getattr(node, f))
        if hasattr(node, "children"):
            for c in node.children: yield from walk(c)

    for num, t in TABLES.items():
        for _, pred in t.all_rows():
            for node in walk(parse(pred)):
                spec = FEATURES[node.name] if hasattr(node, "name") else None
                if spec is None or spec["type"] not in {"ord", "cat"}: continue
                operands = ([node.operand] if isinstance(node, Cmp)
                            else list(node.options) if isinstance(node, In) else [])
                for o in operands:
                    if isinstance(o, float):
                        o = str(int(o)) if o.is_integer() else str(o)
                    assert str(o) in spec["values"], (
                        f"outcome {num}: {node.name} == {o!r} is not one of {spec['values']}")


# -------------------------------------------------- cross-check against rules.md

def rubric_cells():
    """Every physical grade cell in rules.md as (outcome, table_index, grade, is_na).

    Outcomes 26 and 53 print two parallel tables, so the rubric has 275 physical
    cells across 265 (outcome, grade) pairs.
    """
    text = ROOT.joinpath("rules.md").read_text()
    secs = re.split(r"\n### (\d{2})\. ", text)
    cells = []
    for i in range(1, len(secs), 2):
        num, body = secs[i], secs[i + 1].split("#### Methodology")[0]
        seen, ti = set(), 0
        for g, crit in re.findall(r"^\|\s*\*\*Grade ([1-5])\*\*\s*\|[^|]*\|(.+?)\|\s*$",
                                  body, re.M):
            if g in seen: ti += 1; seen = set()
            seen.add(g)
            na = crit.strip().strip("*").strip().lower() == "n/a"
            cells.append((num, ti, int(g), na))
    return cells


def rubric_grades():
    """Non-N/A grades per outcome, unioned across parallel tables."""
    out = {}
    for num, _ti, g, na in rubric_cells():
        out.setdefault(num, set())
        if not na: out[num].add(g)
    return out


def test_all_53_outcomes_present():
    assert sorted(TABLES) == sorted(rubric_grades()) == [f"{i:02d}" for i in range(1, 54)]


@pytest.mark.parametrize("num", sorted(TABLES))
def test_table_grade_set_matches_the_rubric(num):
    """The §9 validation: a table must offer exactly the grades the rubric marks
    non-N/A - no extra rows, and no rubric grade left unimplemented."""
    assert TABLES[num].grades() == rubric_grades()[num], (
        f"outcome {num} {TABLES[num].name}: table has {sorted(TABLES[num].grades())}, "
        f"rubric has {sorted(rubric_grades()[num])}")


def test_rows_are_ordered_highest_grade_first():
    """Bottom-up evaluation silently under-grades 19, 34 and 48, whose Grade 3 and
    Grade 4 cells share a trigger set."""
    for num, t in TABLES.items():
        for rows in [t.rows] + list(t.strata.values()) + list(t.axes.values()):
            grades = [g for g, _ in rows]
            assert grades == sorted(grades, reverse=True), f"outcome {num}: {grades}"


def test_na_grade_count_matches_the_plan():
    """275 physical cells, 217 real, 58 N/A - the figures §5 of the plan is built on."""
    cells = rubric_cells()
    real = sum(1 for *_, na in cells if not na)
    assert (len(cells), real, len(cells) - real) == (275, 217, 58)


def test_only_26_and_53_have_parallel_tables():
    multi = sorted({num for num, ti, *_ in rubric_cells() if ti > 0})
    assert multi == ["26", "53"]
