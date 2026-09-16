"""Regenerate data/scogs_feature_schema.json from the canonical registry.

Run: python3 scripts/scogs/build_schema.py
The build fails if any decision table references an undeclared feature, or if any
declared feature is graded on by nothing.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scogs.features import FEATURES
from scogs.predicates import parse
from scogs.tables import TABLES

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "scogs_feature_schema.json"


def main() -> int:
    used, per_outcome = set(), {}
    for num, t in TABLES.items():
        names = set()
        if t.on: names.add(t.on)
        for _, pred in t.all_rows():
            names |= parse(pred).names()          # raises on a malformed predicate
        per_outcome[num] = sorted(names)
        used |= names

    derived_in = set()
    for spec in FEATURES.values():
        if spec["derived"]:
            derived_in |= {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", spec["derived"])
                           if w in FEATURES}

    undeclared = sorted(used - set(FEATURES))
    orphans = sorted(set(FEATURES) - used - derived_in)
    if undeclared or orphans:
        for u in undeclared: print(f"ERROR undeclared feature referenced by a table: {u}")
        for o in orphans:    print(f"ERROR feature declared but graded on by nothing: {o}")
        return 1

    consumers = {name: sorted(n for n, ns in per_outcome.items() if name in ns)
                 for name in FEATURES}

    schema = {
        "version": 2,
        "source": "rules.md, verified against SCOGS_Booklet.pdf 2026-08-28",
        "counts": {
            "features": len(FEATURES),
            "outcomes": len(TABLES),
            "rules": sum(1 for t in TABLES.values() for _ in t.all_rows()),
        },
        "features": {
            name: {k: v for k, v in spec.items() if v not in (None, [], {}, False)}
                  | {"used_by_outcomes": consumers[name]}
            for name, spec in sorted(FEATURES.items())
        },
        "outcomes": {
            num: {
                "name": t.name,
                "eval": t.eval,
                **({"stratified_on": t.on} if t.on else {}),
                "grades": sorted(t.grades()),
                "features_required": per_outcome[num],
                "rules": ([{"grade": g, "when": p} for g, p in t.rows]
                          or {k: [{"grade": g, "when": p} for g, p in rs]
                              for k, rs in (t.strata or t.axes).items()}),
                **({"notes": t.notes} if t.notes else {}),
            }
            for num, t in sorted(TABLES.items())
        },
        "review_queue": sorted(n for n, s in FEATURES.items() if s["review"]),
    }

    OUT.write_text(json.dumps(schema, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(FEATURES)} features, {len(TABLES)} outcomes, "
          f"{schema['counts']['rules']} rules")
    print(f"  {len(schema['review_queue'])} features flagged for clinical review")
    print(f"  0 undeclared identifiers, 0 orphans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
