# Custom Datasets Implementation Plan

> **Status (2026-09-17): not started; partly stale.** Written before the maintainability
> refactor (tasks/plan.md). `load_notes` and the cohort constants now live in
> `scripts/experiments/cohort.py`, not `medgemma_extraction.py`; its line numbers and
> the "1,960-line" figure are out of date, and the "uncommitted changes" precondition
> no longer applies. Find code by function name before executing it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `medgemma_extraction.py` read notes from a user-supplied `.json` / `.jsonl` / `.csv` file or a folder of `.txt` files, alongside the default PMC-Patients SCD notes.

**Architecture:** A new stdlib-only module, `scripts/experiments/note_sources.py`, turns any supported input into the record shape the harness already uses (`patient_uid`, `patient`, `title`, `age`, `gender`). `load_notes()` gains a `data_path` argument and an `all` cohort; `main()` gains `--data-path` and records the input's path and SHA-256 in provenance. Nothing downstream of `load_notes()` changes.

**Tech Stack:** Python 3 standard library only (`csv`, `json`, `hashlib`, `pathlib`), pytest.

**Spec:** The "Support Custom Datasets" proposal from the 2026-09-16 session, as amended by "Changes from the original proposal" below. This plan wins where they differ.

**Audience:** the LLM implementing this. Line numbers are approximate; search by the quoted code. Every code block below was run on a scratch copy of the repo: full suite **504 passed** (483 baseline + 21 new).

## Changes from the original proposal (and why)

1. **Duplicate IDs are numbered, not trusted.** `data/clinical_notes.csv` has 75,016 rows but only 5,000 unique `patient_id`s (one row per visit). The harness keys every result by `patient_uid` (`results[uid]`, `selection[uid]`), so a repeated ID silently overwrites earlier notes. Repeats become `<id>#1`, `<id>#2`, … in file order, and the original is kept as `source_id`.
2. **`patient` is the *last* text alias.** PMC-Patients uses `patient` for the note text, but in other data a `patient` column is as likely to be a name. The text precedence is `clinical_note, note, text, content, patient`.
3. **The loader is its own module**, not more code in the 1,960-line `medgemma_extraction.py`. That also lets it be unit-tested without the harness.
4. **Parsing tests go in a new `tests/test_note_sources.py`.** `tests/test_ollama_workflow.py` only gets the CLI-level tests, since that file holds the subprocess/CLI checks.
5. **Provenance records the input.** `data_source` and `data_sha256` go in `provenance`. This repo treats a results file that can't name its inputs as unusable (see the `run_id` comments in `main()`).
6. **Cohorts filter any source the same way:** `all` = no filter, `loose` = mentions SCD, `scd_primary` = `is_scd_primary()`. On the bundled cache, which is already the SCD-mention subset, `all` and `loose` give the same pool. The default stays `loose` without `--data-path` and becomes `all` with it.
7. **Documented footgun, behaviour unchanged:** `--stratify` (on by default) seeds notes by outcome regex, so a small or non-SCD dataset can return fewer notes than `--notes`. The README tells users to pass `--no-stratify`.

## Preconditions

- [ ] The working tree already has **uncommitted, unrelated changes** in `scripts/experiments/medgemma_extraction.py` and `tests/test_ollama_workflow.py` (CLI default prompt stage `2b` → `3`). **Stop and ask the user to commit them first.** Do not include them in this work's commits.
- [ ] Baseline: `python3 -m pytest -q` → **483 passed**. If it doesn't, stop and report.

## Global Constraints

- Standard library only; do not edit `requirements.txt`.
- Must work on Windows: open files with explicit `encoding=`, CSV with `newline=""`, `csv.field_size_limit(2**31 - 1)` (not `sys.maxsize`, which overflows there).
- Keep note text verbatim (no strip or normalisation). Quote verification matches against it.
- Without `--data-path`, behaviour is unchanged: same pool, same `cohort=loose` default, same stdout.
- Every invalid input exits non-zero **without** writing `--out`, and before any note is sent to a model.
- Do not touch `select_notes()`, prompts, grading, or `review_results.py`.
- Do not change the README's `--prompt-stage` row (it belongs to the pending uncommitted change).
- Commit messages follow repo style: `feat: …`, `docs: …`.

## File Map

| File | Change | Responsibility |
| --- | --- | --- |
| `scripts/experiments/note_sources.py` | Create | Read JSON/JSONL/CSV/txt folder → harness records; content hash |
| `tests/test_note_sources.py` | Create | Unit tests for the loader |
| `scripts/experiments/medgemma_extraction.py` | Modify | `COHORTS`, `SCD_MENTION`, `load_notes(data_path=)`, `--data-path`, provenance |
| `tests/test_ollama_workflow.py` | Modify | CLI tests: custom run, loose filter, default unchanged, bad input |
| `README.md` | Modify | Options table rows + "Using custom datasets" section |

---

### Task 1: `note_sources` loader

**Files:**
- Create: `scripts/experiments/note_sources.py`
- Test: `tests/test_note_sources.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_dataset(path: str | pathlib.Path) -> list[dict]`. Each dict has `patient_uid: str`, `patient: str`, `title: str` (`""` if absent), `age` (raw value or `None`), `gender` (raw value or `None`), plus `source_id: str` only when the ID was repeated. Raises `ValueError` with a readable message on any unusable input.
  - `dataset_sha256(path: str | pathlib.Path) -> str`: hex SHA-256 of the file's bytes, or for a folder, of its sorted `*.txt` names and contents.

Note: `tests/conftest.py` puts `scripts/` on `sys.path`, and `experiments` is a namespace package (no `__init__.py`), so `from experiments import note_sources` works.

- [ ] **Step 1: Write the failing tests** in `tests/test_note_sources.py`:

```python
"""note_sources: user-supplied notes become records the extraction harness can read."""
import csv
import hashlib
import json
import re

import pytest

from experiments import note_sources


def write_csv(path, header, rows, encoding="utf-8"):
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_json_array_maps_aliases_into_harness_keys(tmp_path):
    path = tmp_path / "notes.json"
    path.write_text(json.dumps([{"patient_id": 7, "clinical_note": "Painful crisis treated.",
                                 "title": "Case", "age": 12, "sex": "F"}]), encoding="utf-8")
    assert note_sources.load_dataset(path) == [
        {"patient_uid": "7", "patient": "Painful crisis treated.", "title": "Case",
         "age": 12, "gender": "F"},
    ]


def test_pmc_patients_records_keep_their_uid_and_note(tmp_path):
    rec = {"patient_id": "1", "patient_uid": "123-1", "PMID": "123", "title": "T",
           "patient": "A 9-year-old with HbSS.", "age": [[9.0, "year"]], "gender": "M"}
    path = tmp_path / "pmc.json"
    path.write_text(json.dumps([rec]), encoding="utf-8")
    assert note_sources.load_dataset(path) == [
        {"patient_uid": "123-1", "patient": "A 9-year-old with HbSS.", "title": "T",
         "age": [[9.0, "year"]], "gender": "M"},
    ]


def test_jsonl_skips_blank_lines_and_falls_back_to_line_number(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text('{"text": "first note"}\n\n{"uid": "x", "text": "second note"}\n',
                    encoding="utf-8")
    assert [(r["patient_uid"], r["patient"]) for r in note_sources.load_dataset(path)] == [
        ("row-1", "first note"), ("x", "second note")]


def test_csv_headers_match_case_insensitively_and_a_bom_is_ignored(tmp_path):
    path = tmp_path / "notes.csv"
    write_csv(path, ["ID", "Note", "Gender"], [["a", "note a", "Female"]], encoding="utf-8-sig")
    assert note_sources.load_dataset(path) == [
        {"patient_uid": "a", "patient": "note a", "title": "", "age": None, "gender": "Female"},
    ]


def test_a_specific_text_column_beats_a_column_named_patient(tmp_path):
    path = tmp_path / "notes.csv"
    write_csv(path, ["patient", "note"], [["Jane Doe", "the actual note"]])
    assert note_sources.load_dataset(path)[0]["patient"] == "the actual note"


def test_repeated_ids_are_numbered_in_file_order(tmp_path):
    # data/clinical_notes.csv lists each patient once per visit.
    path = tmp_path / "visits.csv"
    write_csv(path, ["patient_id", "clinical_note"],
              [["P1", "visit one"], ["P2", "only visit"], ["P1", "visit two"]])
    recs = note_sources.load_dataset(path)
    assert [r["patient_uid"] for r in recs] == ["P1#1", "P2", "P1#2"]
    assert [r.get("source_id") for r in recs] == ["P1", None, "P1"]


def test_empty_text_is_skipped_and_other_text_is_kept_verbatim(tmp_path):
    path = tmp_path / "notes.csv"
    write_csv(path, ["id", "text"], [["a", "  "], ["b", " kept \n"]])
    assert [(r["patient_uid"], r["patient"]) for r in note_sources.load_dataset(path)] == [
        ("b", " kept \n")]


def test_txt_folder_uses_file_stems_in_name_order(tmp_path):
    (tmp_path / "b.txt").write_text("note b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("note a", encoding="utf-8")
    (tmp_path / "readme.md").write_text("not a note", encoding="utf-8")
    assert [(r["patient_uid"], r["patient"]) for r in note_sources.load_dataset(tmp_path)] == [
        ("a", "note a"), ("b", "note b")]


@pytest.mark.parametrize("name,content,message", [
    ("notes.xlsx", "x", "unsupported format"),
    ("notes.json", '{"text": "not a list"}', "JSON array of objects"),
    ("notes.jsonl", '{"text": "ok"}\nnot json\n', "notes.jsonl:2"),
    ("notes.csv", "id,body\n1,hello\n", "no note text field"),
    ("notes.csv", "id,text\n1,\n", "no records with note text"),
])
def test_unusable_inputs_raise_a_clear_error(tmp_path, name, content, message):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        note_sources.load_dataset(path)


def test_missing_path_and_empty_folder_raise(tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        note_sources.load_dataset(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="no .txt files"):
        note_sources.load_dataset(tmp_path)


def test_dataset_hash_tracks_content_and_file_names(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text('{"text": "a"}\n', encoding="utf-8")
    first = note_sources.dataset_sha256(path)
    assert first == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('{"text": "b"}\n', encoding="utf-8")
    assert note_sources.dataset_sha256(path) != first

    folder = tmp_path / "txt"
    folder.mkdir()
    (folder / "a.txt").write_text("x", encoding="utf-8")
    before = note_sources.dataset_sha256(folder)
    (folder / "a.txt").rename(folder / "b.txt")
    assert note_sources.dataset_sha256(folder) != before
```

- [ ] **Step 2: Run and confirm it fails**

Run: `python3 -m pytest tests/test_note_sources.py -q`
Expected: collection error, `ImportError: cannot import name 'note_sources'`.

- [ ] **Step 3: Implement** `scripts/experiments/note_sources.py`:

```python
"""Load clinical notes from a user-supplied file or folder into the harness's record shape.

A record is {"patient_uid", "patient", "title", "age", "gender"} - the keys the
extraction harness already reads from PMC-Patients. Only the note text is required.
The text is kept verbatim because quotes are verified against it; age and gender
are passed through unchanged and only ever written to the results file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter

# The first alias a record carries wins, matched case-insensitively. `patient` is
# last: PMC-Patients names its note text that, but elsewhere a column called
# "patient" is as likely to hold a name.
ID_ALIASES = ("patient_uid", "patient_id", "uid", "id")
TEXT_ALIASES = ("clinical_note", "note", "text", "content", "patient")
GENDER_ALIASES = ("gender", "sex")


def _pick(record: dict, aliases: tuple[str, ...]):
    """-> (key, value) for the first alias the record carries, else (None, None)."""
    lowered = {str(k).strip().lower(): k for k in record}
    for alias in aliases:
        if alias in lowered:
            key = lowered[alias]
            return key, record[key]
    return None, None


def normalize(raw: list[tuple[str, dict]], source: str) -> list[dict]:
    """(fallback uid, raw record) pairs -> harness records, in input order.

    Records with empty text are skipped. A uid that repeats (a visit-level CSV lists
    a patient once per visit) becomes `<uid>#<n>` in file order, because the harness
    keys every result by uid and a repeat would silently overwrite the first.
    """
    records, skipped = [], 0
    for fallback, rec in raw:
        text_key, text = _pick(rec, TEXT_ALIASES)
        if text_key is None:
            raise ValueError(f"{source}: record {fallback} has no note text field; expected one "
                             f"of {', '.join(TEXT_ALIASES)}; found {', '.join(map(str, rec)) or 'none'}")
        if text is None or not str(text).strip():
            skipped += 1
            continue
        _, uid = _pick(rec, ID_ALIASES)
        uid = str(uid).strip() if uid is not None and str(uid).strip() else fallback
        _, title = _pick(rec, ("title",))
        _, age = _pick(rec, ("age",))
        _, gender = _pick(rec, GENDER_ALIASES)
        records.append({"patient_uid": uid, "patient": str(text),
                        "title": "" if title is None else str(title),
                        "age": age, "gender": gender})
    if skipped:
        print(f"{source}: skipped {skipped} record(s) with empty note text")
    if not records:
        raise ValueError(f"{source}: no records with note text")

    repeated = {uid for uid, n in Counter(r["patient_uid"] for r in records).items() if n > 1}
    if repeated:
        seen = Counter()
        for r in records:
            if r["patient_uid"] in repeated:
                seen[r["patient_uid"]] += 1
                r["source_id"] = r["patient_uid"]
                r["patient_uid"] = f"{r['source_id']}#{seen[r['source_id']]}"
        print(f"{source}: {len(repeated)} id(s) repeat across records; numbered <id>#<n> in file order")
    return records


def _read_json(path: pathlib.Path) -> list[tuple[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise ValueError(f"{path}: expected a JSON array of objects")
    return [(f"row-{i}", r) for i, r in enumerate(data, 1)]


def _read_jsonl(path: pathlib.Path) -> list[tuple[str, dict]]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: not valid JSON ({exc.msg})") from exc
            if not isinstance(rec, dict):
                raise ValueError(f"{path}:{n}: expected a JSON object")
            rows.append((f"row-{n}", rec))
    return rows


def _read_csv(path: pathlib.Path) -> list[tuple[str, dict]]:
    # Long notes exceed the csv module's 131 KB field limit; sys.maxsize overflows on Windows.
    csv.field_size_limit(2**31 - 1)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [(f"row-{i}", row) for i, row in enumerate(csv.DictReader(f), 1)]


def _txt_files(folder: pathlib.Path) -> list[pathlib.Path]:
    # Sorted by name, not Path, so the order is the same on case-insensitive Windows.
    return sorted((p for p in folder.glob("*.txt") if p.is_file()), key=lambda p: p.name)


def _read_txt_dir(folder: pathlib.Path) -> list[tuple[str, dict]]:
    files = _txt_files(folder)
    if not files:
        raise ValueError(f"{folder}: folder has no .txt files")
    return [(p.stem, {"id": p.stem, "text": p.read_text(encoding="utf-8-sig")}) for p in files]


READERS = {".json": _read_json, ".jsonl": _read_jsonl, ".csv": _read_csv}


def load_dataset(path: str | pathlib.Path) -> list[dict]:
    """A .json array, .jsonl or .csv file, or a folder of .txt files -> harness records."""
    path = pathlib.Path(path)
    if path.is_dir():
        raw = _read_txt_dir(path)
    elif not path.is_file():
        raise ValueError(f"{path}: no such file or folder")
    elif path.suffix.lower() in READERS:
        raw = READERS[path.suffix.lower()](path)
    else:
        raise ValueError(f"{path}: unsupported format {path.suffix!r}; "
                         f"use .json, .jsonl, .csv or a folder of .txt files")
    return normalize(raw, str(path))


def dataset_sha256(path: str | pathlib.Path) -> str:
    """Content hash of what load_dataset reads, so a results file names its input."""
    path = pathlib.Path(path)
    if not path.is_dir():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    h = hashlib.sha256()
    for p in _txt_files(path):   # a renamed file is a different uid, so names count
        h.update(p.name.encode("utf-8") + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
```

- [ ] **Step 4: Run and confirm it passes**

Run: `python3 -m pytest tests/test_note_sources.py -q`
Expected: `15 passed`.

Sanity check against real data:
```bash
python3 -c "import sys; sys.path.insert(0, 'scripts'); from experiments.note_sources import load_dataset as L; r = L('data/clinical_notes.csv'); print(len(r), r[0]['patient_uid'])"
```
Expected: a "5000 id(s) repeat" line, then `75016 P00000#1` (takes about 3 s).

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/note_sources.py tests/test_note_sources.py
git commit -m "feat: add note_sources loader for JSON, JSONL, CSV and txt-folder datasets"
```

---

### Task 2: Wire `--data-path` and the `all` cohort into the harness

**Files:**
- Modify: `scripts/experiments/medgemma_extraction.py` (module docstring; imports ~line 36; `load_notes` ~line 1235; `main()` argparse ~line 1538; validation ~line 1589; `load_notes` call ~line 1617; `rec_dict` ~line 1836; `provenance` ~line 1866)
- Test: `tests/test_ollama_workflow.py`

**Interfaces:**
- Consumes: `load_dataset(path) -> list[dict]` and `dataset_sha256(path) -> str` from Task 1.
- Produces:
  - `COHORTS = ("all", "loose", "scd_primary")`, `SCD_MENTION: re.Pattern`
  - `load_notes(cohort: str = "loose", data_path: str | pathlib.Path | None = None) -> list[dict]` (raises `ValueError` from the loader)
  - CLI `--data-path PATH`; `--cohort {all,loose,scd_primary}` defaulting to `all` with `--data-path`, else `loose`
  - `provenance["data_source"]` (`"pmc_patients"` or the path string as given), `provenance["data_sha256"]` (`None` for the default source)
  - `detailed_records[i]["source_id"]`, present only for renumbered IDs

- [ ] **Step 1: Write the failing tests** in `tests/test_ollama_workflow.py`.

1a. Add `import hashlib` after `import csv` at the top.

1b. In the `test_invalid_cli_input_is_rejected_before_running` parametrize list, replace
```python
    ["--num-predict", "0"], ["--num-ctx", "0"], ["--tier", "local"],
])
```
with
```python
    ["--num-predict", "0"], ["--num-ctx", "0"], ["--tier", "local"],
    ["--cohort", "everything"], ["--data-path", "no/such/notes.jsonl"],
])
```

1c. Append to the end of the file:

```python
CUSTOM_NOTES = [
    {"id": "A1", "note": "Sickle cell anemia with priapism requiring aspiration.", "sex": "M"},
    {"id": "B2", "note": "Routine asthma review; inhaler technique discussed.", "sex": "F"},
]


def run_custom(tmp_path, *flags):
    data = tmp_path / "notes.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in CUSTOM_NOTES), encoding="utf-8")
    out = tmp_path / "custom.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mock", "--data-path", str(data),
         "--notes", "2", "--no-stratify", "--outcomes", "24", "--out", str(out), *flags],
        capture_output=True, text=True, encoding="utf-8",
    )
    return data, out, result


def test_custom_dataset_runs_every_note_and_records_its_source(tmp_path):
    data, out, result = run_custom(tmp_path)
    assert result.returncode == 0, result.stderr
    saved = json.loads(out.read_text(encoding="utf-8"))
    provenance = saved["provenance"]
    assert provenance["cohort"] == "all"
    assert provenance["data_source"] == str(data)
    assert provenance["data_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    records = sorted(saved["detailed_records"], key=lambda r: r["patient_uid"])
    assert [(r["patient_uid"], r["gender"]) for r in records] == [("A1", "M"), ("B2", "F")]
    review = importlib.import_module("experiments.review_results")
    assert review.export_reviews(out)["handcheck"].exists()


def test_loose_cohort_keeps_only_custom_notes_that_mention_scd(tmp_path):
    _, out, result = run_custom(tmp_path, "--cohort", "loose")
    assert result.returncode == 0, result.stderr
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["provenance"]["cohort"] == "loose"
    assert [r["patient_uid"] for r in saved["detailed_records"]] == ["A1"]


def test_default_source_is_still_the_pmc_loose_pool(tmp_path):
    out = tmp_path / "default_source.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mock", "--notes", "1",
         "--outcomes", "36", "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    provenance = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert (provenance["cohort"], provenance["data_source"], provenance["data_sha256"]) == (
        "loose", "pmc_patients", None)


def test_unreadable_custom_dataset_fails_without_writing_output(tmp_path):
    data = tmp_path / "notes.xlsx"
    data.write_bytes(b"not a supported format")
    out = tmp_path / "bad.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mock", "--data-path", str(data),
         "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "unsupported format" in result.stderr
    assert not out.exists()
```

- [ ] **Step 2: Run and confirm the new tests fail**

Run: `python3 -m pytest tests/test_ollama_workflow.py -q -k "custom or default_source or everything"`
Expected: the new tests FAIL (argparse rejects `--data-path` as an unrecognised argument; the default-source test fails on a `KeyError` for `data_source`). `["--cohort", "everything"]` and `["--data-path", …]` already pass because argparse rejects them. That's fine: they guard against regressions.

- [ ] **Step 3: Implement.** Make each replacement exactly once.

3a. In the module docstring's `Usage examples:`, after the `--backend mock --notes 2` line, add:
```
    python scripts/experiments/medgemma_extraction.py --data-path my_notes.csv --no-stratify --notes 20 --out results/mine.json
```

3b. After the `from experiments.ollama_backend import (...)` block, add:
```python
from experiments.note_sources import dataset_sha256, load_dataset
```

3c. Replace the whole `def load_notes(cohort: str = "loose") -> list[dict]:` function with:
```python
COHORTS = ("all", "loose", "scd_primary")
SCD_MENTION = re.compile(r"sickle cell|\bSCD\b|HbSS|HbSC", re.I)


def load_notes(cohort: str = "loose", data_path: str | pathlib.Path | None = None) -> list[dict]:
    """The candidate pool. Selection happens in select_notes().

    The source is `data_path` (see note_sources.load_dataset) or, when None, the
    bundled PMC-Patients SCD cache. `all` keeps every note, `loose` keeps notes that
    mention SCD, `scd_primary` keeps notes about SCD. The cache is already the
    SCD-mention subset, so on it `all` and `loose` are the same pool.
    """
    if data_path is not None:
        pool = load_dataset(data_path)
    else:
        cache = ROOT / "data" / "pmc_patients" / "scd_cache.json"
        if cache.exists():
            pool = json.loads(cache.read_text(encoding="utf-8"))
        else:
            data = json.loads((cache.parent / "PMC-Patients-V2.json").read_text(encoding="utf-8"))
            pool = [r for r in data if SCD_MENTION.search(r.get("patient", ""))]
            pool.sort(key=lambda r: r["patient_uid"])          # deterministic before sampling
            try:
                cache.write_text(json.dumps(pool), encoding="utf-8")
            except Exception:
                pass
    if cohort == "scd_primary":
        kept = [r for r in pool if is_scd_primary(r)]
        print(f"cohort=scd_primary: {len(kept)}/{len(pool)} kept "
              f"({len(pool) - len(kept)} dropped as mention-only)")
        return kept
    if cohort == "loose" and data_path is not None:
        kept = [r for r in pool if SCD_MENTION.search(r["patient"])]
        print(f"cohort=loose: {len(kept)}/{len(pool)} kept "
              f"({len(pool) - len(kept)} dropped for no SCD mention)")
        return kept
    return pool
```

3d. In `main()`, replace the `--cohort` argument:
```python
    ap.add_argument("--cohort", choices=["scd_primary", "loose"], default="loose",
                    help="pool definition. loose (default): any SCD mention - keeps notes "
                         "where outcomes are genuinely absent, which the absence audit needs. "
                         "scd_primary: SCD-primary notes only")
```
with
```python
    ap.add_argument("--data-path", default=None,
                    help="notes to extract from: a .json array, .jsonl or .csv file, or a "
                         "folder of .txt files (README: Using custom datasets). "
                         "Default: the bundled PMC-Patients SCD notes")
    ap.add_argument("--cohort", choices=list(COHORTS), default=None,
                    help="pool filter. all: every note (default with --data-path). "
                         "loose: any SCD mention (default otherwise) - keeps notes where "
                         "outcomes are genuinely absent, which the absence audit needs. "
                         "scd_primary: SCD-primary notes only")
```

3e. Directly after `a = ap.parse_args()`, add:
```python
    if a.cohort is None:
        a.cohort = "all" if a.data_path else "loose"
```

3f. Directly before `if a.check_model and a.backend != "ollama":`, add (this runs before preflight, so a bad path never reaches a model):
```python
    if a.data_path and not pathlib.Path(a.data_path).exists():
        ap.error(f"--data-path not found: {a.data_path}")
```

3g. Replace `    pool = load_notes(cohort=a.cohort)` with:
```python
    try:
        pool = load_notes(cohort=a.cohort, data_path=a.data_path)
    except ValueError as exc:
        ap.error(str(exc))
    data_sha256 = dataset_sha256(a.data_path) if a.data_path else None
```

3h. In the `--out` block, right after the `rec_dict = {...}` literal closes (the line after `"outcomes": per_outcome_details,` and `}`), add:
```python
            if "source_id" in rec:                    # the file's own id, before <id>#<n>
                rec_dict["source_id"] = rec["source_id"]
```

3i. In `provenance`, replace `            "cohort": a.cohort,` with:
```python
            "cohort": a.cohort,
            "data_source": str(a.data_path) if a.data_path else "pmc_patients",
            "data_sha256": data_sha256,
```

- [ ] **Step 4: Run and confirm everything passes**

Run: `python3 -m pytest tests/test_ollama_workflow.py -q`
Expected: all pass.

Run: `python3 -m pytest -q`
Expected: **504 passed** (483 + 15 from Task 1 + 6 from this task).

Manual smoke on the real visit-level CSV:
```bash
python3 scripts/experiments/medgemma_extraction.py --backend mock --data-path data/clinical_notes.csv --no-stratify --notes 3 --outcomes 10
```
Expected: the "5000 id(s) repeat" line, `notes=3`, and a clean finish.

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/medgemma_extraction.py tests/test_ollama_workflow.py
git commit -m "feat: add --data-path for custom note datasets and an 'all' cohort"
```

---

### Task 3: README documentation

**Files:**
- Modify: `README.md` ("Important run options" table and the section after it)

**Interfaces:** Consumes the CLI behaviour from Task 2. Produces nothing code relies on.

- [ ] **Step 1: Update the options table.** Replace the `--cohort` row:
```markdown
| `--cohort` | `loose` | `scd_primary` for disease-focused reporting; `loose` includes mention-only cases |
```
with these two rows:
```markdown
| `--data-path` | bundled PMC-Patients SCD notes | `.json`, `.jsonl`, `.csv` file or folder of `.txt` notes; see [Using custom datasets](#using-custom-datasets) |
| `--cohort` | `loose`, or `all` with `--data-path` | `scd_primary` for disease-focused reporting; `loose` includes mention-only cases; `all` applies no filter |
```

- [ ] **Step 2: Add the section.** Insert it right before `## Outputs and human review`, after the paragraph that ends with `[the evaluation protocol](docs/research/extraction_protocol.md).`:

````markdown
### Using custom datasets

By default notes come from the bundled PMC-Patients SCD subset
(`data/pmc_patients/scd_cache.json`). Pass `--data-path` to extract from other notes.

| Input | Shape | Note ID when no ID field |
| --- | --- | --- |
| `.json` | An array of objects | `row-<n>` |
| `.jsonl` | One object per line; blank lines skipped | `row-<line>` |
| `.csv` | Header row, UTF-8 (a BOM is fine) | `row-<n>` |
| Folder | Each `*.txt` file directly inside it is one note | File name without `.txt` |

Field names are matched case-insensitively, and the first one present wins:

| Field | Accepted names | Required |
| --- | --- | --- |
| Note ID | `patient_uid`, `patient_id`, `uid`, `id` | No |
| Note text | `clinical_note`, `note`, `text`, `content`, `patient` | Yes |
| Title | `title` | No |
| Age | `age` | No |
| Gender | `gender`, `sex` | No |

- Text is used verbatim, since quotes are verified against it. Records with empty text are skipped.
- The harness evaluates notes, not patients. A repeated ID, such as one row per visit in `data/clinical_notes.csv`, becomes `<id>#1`, `<id>#2`, … in file order. The original ID is kept as `source_id` in the results.
- With `--data-path`, `--cohort` defaults to `all`. `loose` keeps notes that mention SCD, and `scd_primary` keeps notes about SCD.
- `--stratify` (the default) picks notes whose text matches each outcome, so a small or non-SCD dataset can return fewer notes than `--notes`. Add `--no-stratify` for a plain random sample.
- Provenance records `data_source` (the path you passed) and `data_sha256` (a hash of its content).
- Results JSON contains full note text. `results/` is git-ignored; keep identifiable notes out of commits.

```bash
# Model-free check that a file loads and runs end to end.
python scripts/experiments/medgemma_extraction.py --backend mock --data-path my_notes.csv --no-stratify --notes 2

# Visit-level notes shipped with the repository.
python scripts/experiments/medgemma_extraction.py --data-path data/clinical_notes.csv --no-stratify --notes 20 --out results/clinical_notes.json

# A folder of .txt notes, SCD-primary only.
python scripts/experiments/medgemma_extraction.py --data-path notes/ --cohort scd_primary --notes 20 --out results/txt_notes.json
```
````

- [ ] **Step 3: Verify the documented command works**

Run: `python3 scripts/experiments/medgemma_extraction.py --backend mock --data-path data/clinical_notes.csv --no-stratify --notes 2 --outcomes 10`
Expected: exit 0.

Also run `python3 -m pytest -q` again (a notebook/README test may read this file). Expected: **504 passed**.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document --data-path and custom dataset formats"
```

---

### Task 4: Final verification (no code)

- [ ] `python3 -m pytest -q` → **504 passed**, 0 failed.
- [ ] `git status` shows no stray files. The user's own untracked `tasks/todo.md` and `tasks/plan.md` should be untouched.
- [ ] `git log --oneline -3` shows the three commits above and nothing else from this work.
- [ ] Report to the user: the commits, the test count, and the out-of-scope notes below.

## Out of scope (report, don't do)

- Merging a patient's visit notes into one note, or using `clinical_notes.csv`'s outcome columns as gold labels.
- `--id-field` / `--text-field` override flags (add only if alias matching proves insufficient).
- `.tsv`, `.xlsx`, nested JSON (`{"notes": [...]}`), or recursive `.txt` folders.
- Hashing the default PMC cache into `data_sha256`.
- The README's `--prompt-stage` default row still says `2b`. It is part of the user's pending uncommitted change.
