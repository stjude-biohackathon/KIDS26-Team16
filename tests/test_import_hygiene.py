"""Every module in this repo is imported under exactly one name.

`scripts/` and the repository root are the import roots, so code says
`scogs.tables`, `experiments.verification`, `audit.compare_grades` or
`dashboard.evaluation`. A file imported under two names (for example
`scripts.scogs.predicates` and `scogs.predicates`) is loaded twice, and the two
copies share nothing: `UNKNOWN` from one `is not` `UNKNOWN` from the other, so a
three-valued check silently takes the wrong branch.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Modules that live inside a package (scripts/experiments, scripts/audit,
# dashboard) and therefore must always be imported with the package prefix.
# Add a name here whenever you create such a module.
PACKAGE_MEMBERS = ["medgemma_extraction", "ollama_backend", "review_results"]

MEMBERS = "|".join(PACKAGE_MEMBERS)
FORBIDDEN_IMPORT = re.compile(
    rf"^[ \t]*(?:from|import)[ \t]+(?:scripts(?=[.\s]|$)|(?:{MEMBERS})\b)", re.MULTILINE)
FORBIDDEN_PATCH_TARGET = re.compile(rf"""setattr\(\s*["'](?:scripts\.|(?:{MEMBERS})\.)""")


def python_sources():
    for folder in ("scripts", "dashboard", "tests"):
        yield from sorted((ROOT / folder).rglob("*.py"))


def test_modules_are_imported_by_their_canonical_name():
    offenders = []
    for path in python_sources():
        text = path.read_text(encoding="utf-8")
        for match in FORBIDDEN_IMPORT.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0).strip()}")
    assert not offenders, "import these by their package name:\n" + "\n".join(offenders)


def test_monkeypatch_string_targets_use_canonical_names():
    # monkeypatch.setattr("medgemma_extraction.X", ...) patches a second copy of the
    # module, not the one the code under test uses.
    offenders = [str(path.relative_to(ROOT)) for path in python_sources()
                 if path.name != "test_import_hygiene.py" and FORBIDDEN_PATCH_TARGET.search(path.read_text(encoding="utf-8"))]
    assert not offenders, offenders
