"""Check every grade cell in rules.md against SCOGS_Booklet.pdf - step 2 of the rubric audit.

    python scripts/audit/extract_booklet.py    # first: PDF -> booklet_grades.json
    python scripts/audit/compare_grades.py     # this script

Both sources are split into per-outcome grading tables (an outcome can have
several, e.g. adults and children) and compared cell by cell after
`normalize_cell()`, which erases differences between PDF text and Markdown that
do not change clinical meaning. Differing cells are printed least-similar first
as word-level edits, and saved to grade_diffs.json beside this file.

A wording difference is a finding, not a failure: the known, intentional ones
are logged in docs/reference/rules_vs_booklet_discrepancies.md.

Exit codes:
    0  every outcome has the same number of tables in both sources
    1  some outcome's table count differs (listed as STRUCTURAL; its cells were not compared)
    2  booklet_grades.json is missing - run extract_booklet.py first
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import sys
import unicodedata
from dataclasses import dataclass, field

AUDIT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
RULES_MD = REPO_ROOT / "rules.md"
BOOKLET_GRADES = AUDIT_DIR / "booklet_grades.json"
GRADE_DIFFS = AUDIT_DIR / "grade_diffs.json"

# Characters the PDF and Markdown encode differently, applied in this order after
# NFKC. Some entries are already handled by NFKC; they are kept so the audit's
# output stays identical to earlier runs.
UNICODE_TO_ASCII = {
    "≥": ">=", "≤": "<=", "–": "-", "—": "-",
    "’": "'", "“": '"', "”": '"', "‘": "'",
    "°": "deg", "º": "deg", " ": " ", "−": "-",
    "µ": "u", "⁄": "/", "²": "2", "*": " ",
}


def normalize_cell(text: str) -> str:
    """-> lower-case text with PDF/Markdown formatting differences removed.

    Removes Unicode variants, Markdown emphasis, `<br>`, rules.md footnote refs
    such as `[3]`, booklet superscript footnote digits glued to a word
    ("treatments3"), sentence periods (decimals survive) and punctuation.

    Known limitation: a digit glued to a word cannot be told apart from a
    footnote marker, so "SpO2" becomes "spo". Both sources are normalized the
    same way, so the comparison stays symmetric.
    """
    text = unicodedata.normalize("NFKC", text)
    for variant, ascii_form in UNICODE_TO_ASCII.items():
        text = text.replace(variant, ascii_form)
    text = text.replace("<br>", " ").replace("<br/>", " ")
    text = re.sub(r"[*`_]", " ", text)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", text)                # rules.md footnote refs
    text = text.lower()
    # Booklet superscript footnote markers: "treatments3", "treatments.3".
    # Never after a digit, so decimals like "2.5" survive.
    text = re.sub(r"(?<=[a-z\)])\d{1,2}(?:,\d{1,2})*(?=\s|$)", " ", text)
    text = re.sub(r"(?<=[a-z]\.)\d{1,2}(?:,\d{1,2})*(?=\s|$)", " ", text)
    text = text.replace("n/a", "na")
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)                    # sentence periods, keep decimals
    text = re.sub(r"[^a-z0-9<>=%/\.]+", " ", text)                  # drop punctuation, including '-'
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cell_words(text: str) -> list[str]:
    """-> the normalized words of a cell (the unit the word diff works on)."""
    return normalize_cell(text).split()


def word_diff(booklet_text: str, rules_text: str) -> tuple[float, list[tuple[str, str, str]]]:
    """-> (similarity 0-1 rounded to 3 places, [(edit, booklet words, rules.md words), ...]).

    `edit` is a difflib opcode: "replace", "delete" (words only in the booklet)
    or "insert" (words only in rules.md). Equal stretches are left out.
    """
    booklet_words, rules_words = cell_words(booklet_text), cell_words(rules_text)
    matcher = difflib.SequenceMatcher(None, booklet_words, rules_words)
    edits = [(tag, " ".join(booklet_words[b_start:b_end]), " ".join(rules_words[r_start:r_end]))
             for tag, b_start, b_end, r_start, r_end in matcher.get_opcodes() if tag != "equal"]
    return round(matcher.ratio(), 3), edits


def load_rules_grade_tables(rules_text: str) -> dict[str, list[dict[str, str]]]:
    """-> {"01": [{"1": criterion, ..., "5": criterion}, ...], ...} from rules.md.

    Grade rows look like `| **Grade 3** | **Severe** | criterion |`. A grade number
    seen again starts the outcome's next table - the rule extract_booklet.py uses
    too, so both sources split multi-table outcomes the same way. Everything after
    "#### Methodology" is prose, not a grading table.
    """
    sections = re.split(r"\n### (\d{2})\. ", rules_text)
    tables_by_outcome = {}
    for i in range(1, len(sections), 2):
        number, body = sections[i], sections[i + 1]
        body = body.split("#### Methodology")[0]
        rows = re.findall(r"^\|\s*\*\*Grade ([1-5])\*\*\s*\|[^|]*\|(.+?)\|\s*$", body, re.M)
        tables, table = [], {}
        for grade, criterion in rows:
            if grade in table:
                tables.append(table)
                table = {}
            table[grade] = criterion.strip()
        if table:
            tables.append(table)
        tables_by_outcome[number] = tables
    return tables_by_outcome


@dataclass
class GradeComparison:
    cells_compared: int = 0
    identical: int = 0
    # One dict per differing cell; saved as grade_diffs.json (keys are that file's format).
    differences: list[dict] = field(default_factory=list)
    # (outcome, name, booklet table count, rules.md table count)
    structural: list[tuple[str, str, int, int]] = field(default_factory=list)


def compare_grade_tables(booklet: dict, rules_tables: dict) -> GradeComparison:
    """-> every grade cell compared. Outcomes whose table counts differ are listed, not compared."""
    comparison = GradeComparison()
    for number in sorted(booklet):
        name = booklet[number]["name"]
        booklet_tables, rules_md_tables = booklet[number]["tables"], rules_tables.get(number, [])
        if len(booklet_tables) != len(rules_md_tables):
            comparison.structural.append((number, name, len(booklet_tables), len(rules_md_tables)))
            continue
        for table_index, (booklet_table, rules_table) in enumerate(zip(booklet_tables, rules_md_tables)):
            for grade in "12345":
                booklet_cell = booklet_table["grades"].get(grade, "")
                rules_cell = rules_table.get(grade, "")
                comparison.cells_compared += 1
                if normalize_cell(booklet_cell) == normalize_cell(rules_cell):
                    comparison.identical += 1
                    continue
                similarity, edits = word_diff(booklet_cell, rules_cell)
                comparison.differences.append(dict(
                    num=number, name=name, table=table_index, grade=grade, ratio=similarity,
                    ops=edits, booklet=booklet_cell, rules=rules_cell))
    return comparison


def print_report(comparison: GradeComparison, outcome_count: int) -> None:
    print(f"outcomes: {outcome_count}   table-count mismatches: {len(comparison.structural)}")
    for mismatch in comparison.structural:
        print("   STRUCTURAL:", mismatch)
    print(f"grade cells compared: {comparison.cells_compared}")
    print(f"identical after normalization: {comparison.identical}")
    print(f"differing: {len(comparison.differences)}\n")
    for cell in sorted(comparison.differences, key=lambda cell: cell["ratio"]):
        print(f"=== {cell['num']} {cell['name'][:38]} table{cell['table']} "
              f"Grade {cell['grade']}  sim {cell['ratio']}")
        for edit, booklet_words, rules_words in cell["ops"]:
            if edit == "delete":
                print(f"    BOOKLET ONLY : {booklet_words}")
            elif edit == "insert":
                print(f"    RULES.MD ONLY: {rules_words}")
            else:
                print(f"    BOOKLET      : {booklet_words}\n    RULES.MD     : {rules_words}")
        print()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    if not BOOKLET_GRADES.exists():
        print(f"{BOOKLET_GRADES} not found - run scripts/audit/extract_booklet.py first",
              file=sys.stderr)
        return 2
    booklet = json.loads(BOOKLET_GRADES.read_text(encoding="utf-8"))
    rules_tables = load_rules_grade_tables(RULES_MD.read_text(encoding="utf-8"))
    comparison = compare_grade_tables(booklet, rules_tables)
    print_report(comparison, len(booklet))
    with GRADE_DIFFS.open("w", encoding="utf-8") as output:
        json.dump(comparison.differences, output, indent=1)
    return 1 if comparison.structural else 0


if __name__ == "__main__":
    sys.exit(main())
