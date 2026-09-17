"""Parse the grading tables out of SCOGS_Booklet.pdf - step 1 of the rubric audit.

    python scripts/audit/extract_booklet.py    # this script: PDF -> booklet_grades.json
    python scripts/audit/compare_grades.py     # grade cells: booklet vs rules.md
    python scripts/audit/compare_prose.py      # definitions, criteria, frequency labels

Converts the PDF to text once with Poppler's `pdftotext -layout` (cached as
booklet_layout.txt beside this file; delete it after replacing the PDF), then
reads each outcome's pages, located by the table of contents below, into
booklet_grades.json:

    {"01": {"name": ..., "pages": [first, last],
            "tables": [{"stratum": label or null, "grades": {"1": text, ...}}]}}

An outcome has more than one table when the booklet grades groups separately,
e.g. adults and children (the "stratum").

Exit codes: 0 parsed and written; 2 the PDF text could not be produced
(PDF missing, or `pdftotext` not installed).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

AUDIT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[1]
BOOKLET_PDF = REPO_ROOT / "docs" / "reference" / "SCOGS_Booklet.pdf"
LAYOUT_TEXT = AUDIT_DIR / "booklet_layout.txt"
BOOKLET_GRADES = AUDIT_DIR / "booklet_grades.json"
# The last outcome's section ends on this page.
LAST_OUTCOME_PAGE = 131

# (first page, outcome name) for all 53 outcomes in booklet order, from the
# booklet's table of contents. Outcome number = position + 1.
TOC = [
 (13,"Arrhythmia"),(14,"Deep Vein Thrombosis (DVT)"),(15,"Diastolic Dysfunction"),
 (17,"Heart Failure Exacerbation"),(19,"Myocardial Infarction"),
 (22,"Systemic Arterial Hypertension"),(24,"Systolic Dysfunction"),
 (26,"TRV Elevation on Echocardiogram"),
 (29,"Cerebral Vasculopathy"),(31,"Chronic Pain"),(34,"Cognitive Dysfunction"),
 (36,"Elevated TCD Ultrasonography Velocity"),
 (37,"Posterior Reversible Encephalopathy Syndrome (PRES)"),
 (39,"Silent Cerebral Infarct"),(41,"Stroke (hemorrhagic or ischemic)"),
 (45,"Hearing Loss (in at least one ear)"),(46,"Sickle Cell Retinopathy (SCR)"),
 (49,"Cholecystitis/Cholelithiasis (gallstones)"),
 (53,"Acute Kidney Injury (AKI)"),(55,"Acute Papillary Necrosis"),
 (57,"Chronic Kidney Disease (CKD)"),(59,"Female Ovarian Dysfunction"),
 (61,"Male Impairments"),(63,"Priapism"),
 (65,"Delayed puberty"),(66,"Malnutrition Leading to Stunting (Decreased Height Velocity)"),
 (68,"Underweight"),
 (71,"Acute Sickle Cell Pain Episode"),(73,"Acute Splenic Sequestration"),
 (75,"Alloimmunization/ Delayed Hemolytic Transfusion Reaction"),
 (77,"Chronic Hypersplenism"),(79,"Hepatopathy"),(81,"Splenic Infarction"),
 (83,"Transfusional Iron Overload (Hemochromatosis or Hemosiderosis)"),
 (85,"Transient Aplastic Crisis Secondary to Parvovirus B19 Infection"),
 (89,"Fever"),(90,"Sepsis"),
 (93,"Malignant Neoplasms"),
 (97,"Avascular Necrosis of Joints (AVN)"),(99,"Leg Ulcer"),(101,"Osteomyelitis"),
 (103,"Osteoporosis"),
 (107,"Acute Multiorgan Failure"),
 (111,"Fetal Growth Restriction"),(112,"Pregnancy Loss"),(113,"Premature Delivery"),
 (115,"Depression"),
 (119,"Acute Chest Syndrome (ACS)"),(121,"Asthma Exacerbation"),
 (123,"Chronic Restrictive Lung Physiology"),(125,"Pulmonary Embolism (PE)"),
 (127,"Pulmonary Hypertension"),(129,"Sleep Apnea (obstructive or central)"),
]
assert len(TOC) == 53

# Page furniture printed on every page; never table content.
FOOTER = re.compile(r"St\. Jude Global \| Sickle Cell Outcome Grading System|^\s*\d{1,3}\s*$")
HEADER = re.compile(r"^\s*Health Outcomes by Organ/System")
# Prose headings. An outcome's grading table always comes before the first one.
SECTION = re.compile(r"^\s{0,3}(Definition|Diagnostic Criteria|Methodology|References)\b")
# A table row label. Groups: indentation, grade number, cell text on the same line.
GRADE = re.compile(r"^(\s{0,14})Grade\s+([1-5])\b\s*(.*)$")
# Sub-headings that split one outcome into parallel tables (e.g. adults / children).
STRATUM = re.compile(r"^\s*(Adults?\s*\(age|Children\s*\(age|If there (is|are))", re.I)


def load_pages() -> list[str]:
    """-> the booklet as text, one string per PDF page (index 0 is page 1).

    Runs `pdftotext` only when the cached layout text is missing. Raises
    FileNotFoundError when `pdftotext` is not installed and CalledProcessError
    when it fails (for example, the PDF is missing).
    """
    if not LAYOUT_TEXT.exists():
        subprocess.run(["pdftotext", "-layout", str(BOOKLET_PDF), str(LAYOUT_TEXT)], check=True)
    return LAYOUT_TEXT.read_text(encoding="utf-8").split("\f")


def page_table_lines(pages: list[str], page_number: int) -> tuple[list[str], bool]:
    """-> (table lines on one page, whether a prose heading ended the table there).

    Grading tables always precede Definition/Diagnostic Criteria/Methodology/
    References on a page, and a table that overflows continues at the top of the
    next page - so truncating per page keeps continuations and drops prose.
    """
    if page_number - 1 >= len(pages):
        return [], True                   # past the end of the document: stop reading
    lines = []
    for line in pages[page_number - 1].split("\n"):
        if FOOTER.search(line) or HEADER.search(line):
            continue
        if SECTION.match(line):
            return lines, True
        lines.append(line.rstrip())
    return lines, False


def parse_tables(lines: list[str]) -> list[dict]:
    """-> [{"stratum": label or None, "grades": {"1": text, ...}}] in document order.

    Cells are blank-line-delimited blocks. The "Grade N" label is vertically
    centred in its row, so a tall cell can be split by an internal blank line
    leaving an orphan block above the label - those attach to the next
    grade-bearing block. Lines flush at column 0 that are not Grade labels are
    outcome titles / running text, not cell content.
    """
    # Keep grade labels, indented cell text (8+ spaces) and the blank separators.
    kept = [line for line in lines
            if GRADE.match(line) or (line.strip() and len(line) - len(line.lstrip()) >= 8)
            or not line.strip()]

    blocks, block = [], []
    for line in kept:
        if line.strip():
            block.append(line)
        elif block:
            blocks.append(block)
            block = []
    if block:
        blocks.append(block)

    labelled = []                         # (grade number or None, [cell text lines])
    for block in blocks:
        grade, text = None, []
        for line in block:
            label = GRADE.match(line)
            if label and grade is None:
                grade = label.group(2)
                if label.group(3).strip():
                    text.append(label.group(3).strip())
            elif STRATUM.match(line):
                continue
            elif line.strip():
                text.append(line.strip())
        labelled.append((grade, text))

    # Orphan blocks (no Grade label) belong to the next graded block; if none
    # follows, to the previous one.
    merged, orphan_text = [], []
    for grade, text in labelled:
        if grade is None:
            orphan_text += text
        else:
            merged.append((grade, orphan_text + text))
            orphan_text = []
    if orphan_text and merged:
        merged[-1] = (merged[-1][0], merged[-1][1] + orphan_text)

    # A grade number seen a second time starts the outcome's next table.
    tables, table = [], {}
    for grade, text in merged:
        if grade in table:
            tables.append(table)
            table = {}
        table[grade] = " ".join(text).strip()
    if table:
        tables.append(table)

    strata = [line.strip() for line in lines if STRATUM.match(line)]
    return [dict(stratum=strata[i] if i < len(strata) else None, grades=grades)
            for i, grades in enumerate(tables)]


def extract_outcomes(pages: list[str]) -> dict[str, dict]:
    """-> {"01": {"name", "pages", "tables"}, ...} for all 53 outcomes."""
    outcomes = {}
    for index, (first_page, name) in enumerate(TOC):
        last_page = TOC[index + 1][0] - 1 if index + 1 < len(TOC) else LAST_OUTCOME_PAGE
        # The grading table always precedes Definition and never resumes after it,
        # so stop consuming pages at the first section heading in the outcome.
        lines = []
        for page_number in range(first_page, last_page + 1):
            page_lines, table_ended = page_table_lines(pages, page_number)
            lines += page_lines
            if table_ended:
                break
        outcomes[f"{index + 1:02d}"] = dict(name=name, pages=[first_page, last_page],
                                            tables=parse_tables(lines))
    return outcomes


def print_summary(outcomes: dict[str, dict]) -> None:
    """One line per outcome: pages, table count, grades found, and any missing grade."""
    print(f"parsed {len(outcomes)} outcomes")
    for number, outcome in sorted(outcomes.items()):
        tables = outcome["tables"]
        flag = "  <-- MULTI-TABLE" if len(tables) > 1 else ""
        grades = "/".join("".join(sorted(table["grades"])) for table in tables)
        missing = [f"{number}G{g}" for table in tables for g in "12345" if g not in table["grades"]]
        first_page, last_page = outcome["pages"]
        print(f"  {number} {outcome['name'][:42]:44s} p{first_page:3d}-{last_page:3d} "
              f"tables={len(tables)} grades={grades}{flag}"
              + (f"  MISSING {missing}" if missing else ""))
        for table in tables:
            if table["stratum"]:
                print(f"        stratum: {table['stratum']}")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    try:
        pages = load_pages()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"could not convert {BOOKLET_PDF.name} to text ({exc}). "
              "Install Poppler's pdftotext (macOS: brew install poppler).", file=sys.stderr)
        return 2
    outcomes = extract_outcomes(pages)
    BOOKLET_GRADES.write_text(json.dumps(outcomes, indent=1), encoding="utf-8")
    print_summary(outcomes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
