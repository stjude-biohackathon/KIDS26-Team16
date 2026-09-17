"""The booklet audit scripts: importable without side effects, and the text rules they compare with."""
import importlib
import json
import sys

import pytest

from audit import compare_grades
from audit.compare_grades import cell_words, load_rules_grade_tables, normalize_cell


def test_importing_the_audits_runs_nothing(capsys):
    # compare_prose.py used to import the other two scripts and so re-ran both:
    # re-parsed the PDF, rewrote both JSON files and printed their reports first.
    for name in ("audit.extract_booklet", "audit.compare_grades", "audit.compare_prose"):
        sys.modules.pop(name, None)
        importlib.import_module(name)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("raw, expected", [
    ("Hb ≥ 2.5 g/dL[3]", "hb >= 2.5 g/dl"),
    ("requiring treatments3 or more", "requiring treatments or more"),
    ("hospitalised.3 Then", "hospitalised then"),
    ("Pain – “severe” N/A", "pain severe na"),
    # Known limitation, kept on purpose: a digit glued to a word reads as a footnote marker.
    ("SpO₂ < 90% for **2** days", "spo < 90% for 2 days"),
])
def test_normalize_cell(raw, expected):
    assert normalize_cell(raw) == expected


def test_cell_words_keep_decimals_and_units():
    assert cell_words("Grade 3: TRV ≥ 3.0 m/s") == ["grade", "3", "trv", ">=", "3.0", "m/s"]


def test_a_repeated_grade_starts_a_new_table_and_methodology_is_ignored():
    rules = ("intro\n### 01. Test Outcome\n"
             "| **Grade 1** | x | mild |\n| **Grade 2** | x | worse |\n"
             "| **Grade 1** | x | child mild |\n"
             "#### Methodology\n| **Grade 5** | x | ignored |\n")
    assert load_rules_grade_tables(rules) == {"01": [{"1": "mild", "2": "worse"}, {"1": "child mild"}]}


def use_audit_files(monkeypatch, tmp_path, booklet_tables, rules_md):
    booklet = {"01": {"name": "Test Outcome", "tables": booklet_tables}}
    (tmp_path / "booklet_grades.json").write_text(json.dumps(booklet), encoding="utf-8")
    (tmp_path / "rules.md").write_text(rules_md, encoding="utf-8")
    monkeypatch.setattr(compare_grades, "BOOKLET_GRADES", tmp_path / "booklet_grades.json")
    monkeypatch.setattr(compare_grades, "RULES_MD", tmp_path / "rules.md")
    monkeypatch.setattr(compare_grades, "GRADE_DIFFS", tmp_path / "grade_diffs.json")


def test_matching_tables_exit_0_and_save_no_differences(tmp_path, monkeypatch):
    use_audit_files(monkeypatch, tmp_path, [{"grades": {"1": "Mild pain."}}],
                    "x\n### 01. Test Outcome\n| **Grade 1** | x | mild pain |\n")
    assert compare_grades.main([]) == 0
    assert json.loads((tmp_path / "grade_diffs.json").read_text(encoding="utf-8")) == []


def test_a_table_count_mismatch_exits_1(tmp_path, monkeypatch, capsys):
    use_audit_files(monkeypatch, tmp_path, [{"grades": {"1": "mild"}}],
                    "x\n### 01. Test Outcome\n| **Grade 1** | x | mild |\n| **Grade 1** | x | child |\n")
    assert compare_grades.main([]) == 1
    assert "STRUCTURAL: ('01', 'Test Outcome', 1, 2)" in capsys.readouterr().out


def test_a_missing_booklet_extract_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_grades, "BOOKLET_GRADES", tmp_path / "missing.json")
    assert compare_grades.main([]) == 2
