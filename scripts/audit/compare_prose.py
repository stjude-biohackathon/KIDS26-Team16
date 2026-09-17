"""Check rules.md prose and frequency labels against SCOGS_Booklet.pdf - step 3 of the rubric audit.

    python scripts/audit/extract_booklet.py    # first: caches the PDF as text
    python scripts/audit/compare_prose.py      # this script

For every outcome, compares the frequency classification (Acute, Chronic, ...)
and the Definition, Diagnostic Criteria and Methodology sections, using the
normalization and word diff from compare_grades.py. A section that is nearly
identical (similarity above 0.97 and no edit longer than two words) prints on
one line marked [minor]. Prints a report; writes no files.

Exit codes: 0 report printed (differences are findings, not failures);
2 the booklet text could not be produced (PDF missing or `pdftotext` not installed).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

# Run as a script, Python puts only this file's folder on sys.path. Add
# `scripts/` so the shared packages import by their one canonical name
# (`scogs.…`, `experiments.…`, `audit.…`). Tests get the same path from pyproject.toml.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from audit.compare_grades import RULES_MD, normalize_cell, word_diff
from audit.extract_booklet import FOOTER, HEADER, LAST_OUTCOME_PAGE, TOC, load_pages

# A prose heading in the booklet (the section key used on both sides).
PROSE_HEADING = re.compile(r"^\s{0,3}(Definition|Diagnostic Criteria|Methodology|References)\b")
# The frequency label that ends an outcome's title line.
FREQUENCY = re.compile(r"(Acute \+ Chronic|Chronic with [Ee]xacerbations?|Chronic|Acute)\s*$")
COMPARED_SECTIONS = ["Definition", "Diagnostic Criteria", "Methodology"]
# Booklet section name -> the matching `#### ` heading in rules.md.
RULES_HEADINGS = {
    "Definition": "Definition",
    "Diagnostic Criteria": "Diagnostic Criteria",
    "Methodology": "Methodology & Operational Notes",
    "References": "References",
}


def outcome_page_lines(pages: list[str], first_page: int, last_page: int, title: str = "",
                       strip_running_header: bool = True) -> list[str]:
    """-> an outcome's page text, optionally with the running title/frequency block dropped.

    The title and frequency label are reprinted at the top of every continuation
    page; left in, they inject phantom text into whichever prose section spans
    the page break. Only the first few lines of a *continuation* page can be a
    running header - prose legitimately starts with the outcome name (e.g.
    "Osteomyelitis: Inflammation of bone...").
    """
    title_words = normalize_cell(title)
    lines = []
    for page_number in range(first_page, last_page + 1):
        if page_number - 1 >= len(pages):
            break
        body = [line.rstrip() for line in pages[page_number - 1].split("\n")
                if not FOOTER.search(line) and not HEADER.search(line)]
        if strip_running_header and page_number > first_page:
            removed = 0
            while body and removed < 4:
                line = body[0]
                if not line.strip():
                    body.pop(0)
                    continue
                if line.startswith(" "):
                    break                  # indented: body text, not the running header
                probe = normalize_cell(FREQUENCY.sub("", line.strip()))
                is_title = bool(probe) and (probe in title_words or title_words in probe)
                is_bare_frequency = not probe and bool(FREQUENCY.search(line.strip()))
                if is_title or is_bare_frequency:
                    body.pop(0)
                    removed += 1
                    continue
                break
        lines += body
    return lines


def booklet_prose(pages: list[str]) -> dict[str, dict]:
    """-> {"01": {"name", "sections": {heading: text}, "freq": label or None}, ...} from the PDF text."""
    outcomes = {}
    for index, (first_page, name) in enumerate(TOC):
        last_page = TOC[index + 1][0] - 1 if index + 1 < len(TOC) else LAST_OUTCOME_PAGE
        lines = outcome_page_lines(pages, first_page, last_page, name)
        first_page_lines = outcome_page_lines(pages, first_page, first_page, name,
                                              strip_running_header=False)

        sections, current = {}, None
        for line in lines:
            heading = PROSE_HEADING.match(line)
            if heading:
                current = heading.group(1)
                sections.setdefault(current, [])
            elif current:
                sections[current].append(line.strip())

        # The label ends a title line near the top of the first page. Grade rows
        # can end with the same words, so they are skipped.
        frequency = None
        for line in first_page_lines[:12]:
            label = FREQUENCY.search(line.strip())
            if label and not line.strip().startswith("Grade"):
                frequency = label.group(1)
                break
        outcomes[f"{index + 1:02d}"] = dict(
            name=name,
            sections={heading: " ".join(text).strip() for heading, text in sections.items()},
            freq=frequency)
    return outcomes


def rules_prose(rules_text: str) -> dict[str, dict]:
    """-> {"01": {"Definition": text, ..., "References": text, "freq": label or None}, ...} from rules.md."""
    parts = re.split(r"\n### (\d{2})\. ", rules_text)
    outcomes = {}
    for i in range(1, len(parts), 2):
        number, body = parts[i], parts[i + 1]
        entry = {}
        for section, heading in RULES_HEADINGS.items():
            match = re.search(rf"#### {heading}\n(.*?)(?=\n#### |\n---|\Z)", body, re.S)
            entry[section] = match.group(1).strip() if match else ""
        frequency = re.search(r"\|\s*\*\*Frequency Classification\*\*\s*\|\s*(.+?)\s*\|", body)
        entry["freq"] = re.sub(r"[*]", "", frequency.group(1)).strip() if frequency else None
        outcomes[number] = entry
    return outcomes


def is_minor(similarity: float, edits: list[tuple[str, str, str]]) -> bool:
    """A near-identical section: similarity above 0.97 and no edit longer than two words."""
    return similarity > 0.97 and all(
        len(booklet_words.split()) <= 2 and len(rules_words.split()) <= 2
        for _, booklet_words, rules_words in edits)


def print_frequency_report(booklet: dict, rules: dict) -> None:
    print("=== FREQUENCY CLASSIFICATION ===")
    mismatches = []
    for number in sorted(booklet):
        booklet_label, rules_label = booklet[number]["freq"] or "", rules[number]["freq"] or ""
        if normalize_cell(booklet_label) != normalize_cell(rules_label):
            mismatches.append((number, booklet[number]["name"], booklet_label, rules_label))
    print(f"match: {len(booklet) - len(mismatches)}/{len(booklet)}")
    for number, name, booklet_label, rules_label in mismatches:
        print(f"   {number} {name[:38]:40s} booklet={booklet_label!r}  rules={rules_label!r}")


def print_section_report(booklet: dict, rules: dict) -> None:
    print("\n=== PROSE SECTIONS ===")
    for section in COMPARED_SECTIONS:
        differing = []
        for number in sorted(booklet):
            booklet_text = booklet[number]["sections"].get(section, "")
            rules_text = rules[number].get(section, "")
            if not booklet_text and not rules_text:
                continue
            if normalize_cell(booklet_text) == normalize_cell(rules_text):
                continue
            similarity, edits = word_diff(booklet_text, rules_text)
            differing.append((number, booklet[number]["name"], similarity, edits))

        print(f"\n--- {section}: {len(booklet) - len(differing)}/{len(booklet)} identical, "
              f"{len(differing)} differ")
        for number, name, similarity, edits in sorted(differing, key=lambda item: item[2]):
            if is_minor(similarity, edits):
                summary = "; ".join(f"{booklet_words!r}->{rules_words!r}"
                                    for _, booklet_words, rules_words in edits)
                print(f"   {number} {name[:34]:36s} sim {similarity}  [minor] " + summary[:110])
                continue
            print(f"   {number} {name[:34]:36s} sim {similarity}")
            for edit, booklet_words, rules_words in edits[:6]:
                if edit == "delete":
                    print(f"        BOOKLET ONLY : {booklet_words[:200]}")
                elif edit == "insert":
                    print(f"        RULES.MD ONLY: {rules_words[:200]}")
                else:
                    print(f"        BOOKLET  : {booklet_words[:150]}\n        RULES.MD : {rules_words[:150]}")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    try:
        pages = load_pages()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"could not read the booklet text ({exc}); run scripts/audit/extract_booklet.py",
              file=sys.stderr)
        return 2
    booklet = booklet_prose(pages)
    rules = rules_prose(RULES_MD.read_text(encoding="utf-8"))
    print_frequency_report(booklet, rules)
    print_section_report(booklet, rules)
    return 0


if __name__ == "__main__":
    sys.exit(main())
