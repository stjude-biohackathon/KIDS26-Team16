"""Outcome definitions and diagnostic criteria, read from `rules.md`.

`tables.py` encodes how to GRADE an outcome once you are told it is present.
Nothing in this repo encoded what makes it present in the first place - that
judgement was left entirely to the caller (see `evaluate.py`'s module docstring:
"present=False ... Layer 1/2 found no evidence of this outcome").

When the caller is a language model, "decide presence" is a question it was
being asked to answer from the outcome's NAME alone. The rubric states the
answer for all 53 outcomes, in two sections this module extracts:

    #### Definition            what the outcome is
    #### Diagnostic Criteria   what must be observed to call it

This is a reader, not a second source of truth. `rules.md` stays canonical; if a
section is missing here the caller gets None and prompts without it, rather than
this file carrying a copy that can drift.
"""
from __future__ import annotations

import functools
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RULES = ROOT / "rules.md"

# "### 48. Acute Chest Syndrome (ACS)"
_HEADING = re.compile(r"^### (\d{2})\. (.+?)\s*$", re.M)
_CITATION = re.compile(r"\s*\[\d+(?:,\s*\d+)*\]")


def _section(body: str, title: str) -> str | None:
    """The text under `#### <title>`, up to the next `####` or the end."""
    m = re.search(rf"^#### {re.escape(title)}\s*\n(.*?)(?=^#### |\Z)", body, re.M | re.S)
    return m.group(1).strip() if m else None


def _clean(text: str) -> str:
    """Rubric prose -> prompt prose.

    Drops the numeric citation markers (`[1]`) that make a quote-copying model
    reach for a bracket, and unwraps the rubric's backticked `OR`/`AND`, which
    are markdown emphasis rather than part of the criteria.
    """
    text = _CITATION.sub("", text)
    text = re.sub(r"`(AND|OR|NOT)`", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@functools.lru_cache(maxsize=1)
def _parse() -> dict[str, dict]:
    if not RULES.exists():
        return {}
    txt = RULES.read_text(encoding="utf-8")
    hits = list(_HEADING.finditer(txt))
    out: dict[str, dict] = {}
    for i, m in enumerate(hits):
        body = txt[m.end(): hits[i + 1].start() if i + 1 < len(hits) else len(txt)]
        definition = _section(body, "Definition")
        criteria = _section(body, "Diagnostic Criteria")
        if not definition and not criteria:
            continue
        out[m.group(1)] = {
            "name": m.group(2).strip(),
            "definition": _clean(definition) if definition else None,
            "criteria": _clean(criteria) if criteria else None,
        }
    return out


def outcome_definition(outcome: str) -> dict | None:
    """-> {"name", "definition", "criteria"} for one outcome, or None."""
    return _parse().get(outcome)


def presence_brief(outcome: str) -> str | None:
    """The two rubric sections as one prompt block, or None if neither parses."""
    d = outcome_definition(outcome)
    if not d:
        return None
    parts = []
    if d["definition"]:
        parts.append(d["definition"])
    if d["criteria"]:
        parts.append("Diagnostic criteria:\n" + d["criteria"])
    return "\n\n".join(parts) or None
