"""Automated unit tests for SCOGS Interactive Dashboard and Backend Bridges."""
import json
from pathlib import Path
import pytest

from dashboard.data import (
    get_available_run_files,
    is_run_file,
    load_csv_notes,
    load_run_file,
)
from dashboard.highlight import find_quote_spans, highlight_note_quotes
from dashboard.interactive_dashboard import app


def test_load_run_file_with_fixture():
    """Test loading and indexing of run JSON fixture."""
    fixture_path = Path("tests/fixtures/sample_run.json")
    assert fixture_path.exists(), "Sample run fixture must exist"

    run_data = load_run_file(fixture_path)
    assert "provenance" in run_data
    assert "profiling" in run_data
    assert "automated_metrics" in run_data
    assert "grade_status" in run_data
    assert "records_by_uid" in run_data
    assert "detailed_records" in run_data

    records_by_uid = run_data["records_by_uid"]
    assert "5793438-1" in records_by_uid
    assert "10010448-1" in records_by_uid
    assert "10010448-2" in records_by_uid

    rec = records_by_uid["5793438-1"]
    assert rec["scd_primary"] is True
    assert "outcomes" in rec
    assert "28" in rec["outcomes"]
    assert rec["outcomes"]["28"]["grade_result"]["grade"] == 3.0


def test_load_run_file_missing():
    """Test error handling when run file is missing."""
    with pytest.raises(FileNotFoundError):
        load_run_file("results/nonexistent_run_file_12345.json")


def test_load_run_file_not_a_dict(tmp_path):
    """Test error handling when JSON root is not a dictionary (e.g. list)."""
    list_file = tmp_path / "list_run.json"
    list_file.write_text(json.dumps([{"item": 1}]), encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid JSON object"):
        load_run_file(list_file)


def test_is_run_file(tmp_path):
    """Test identification of valid extraction run files."""
    valid_run = tmp_path / "valid_run.json"
    valid_run.write_text(json.dumps({"detailed_records": []}), encoding="utf-8")
    assert is_run_file(valid_run) is True

    not_run = tmp_path / "not_run.json"
    not_run.write_text(json.dumps({"grades": {}}), encoding="utf-8")
    assert is_run_file(not_run) is False

    list_json = tmp_path / "list.json"
    list_json.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert is_run_file(list_json) is False

    corrupt = tmp_path / "bad.json"
    corrupt.write_text("invalid json content", encoding="utf-8")
    assert is_run_file(corrupt) is False

    assert is_run_file(tmp_path / "nonexistent.json") is False


def test_get_available_run_files_discovers_nested(tmp_path, monkeypatch):
    """Test discovery of run files nested in subdirectories."""
    res_dir = tmp_path / "results"
    nested_dir = res_dir / "subfolder"
    nested_dir.mkdir(parents=True)

    valid_nested = nested_dir / "run.json"
    valid_nested.write_text(json.dumps({"detailed_records": [{"patient_uid": "p1"}]}), encoding="utf-8")

    # Non-run JSON (e.g. metadata or list)
    (nested_dir / "metadata.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    (nested_dir / "diffs.json").write_text(json.dumps([{"diff": True}]), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    files = get_available_run_files()
    assert any(p == "results/subfolder/run.json" for p, _ in files)
    assert not any("metadata.json" in p for p, _ in files)
    assert not any("diffs.json" in p for p, _ in files)


def test_load_csv_notes_columns():
    """Test loading data/clinical_notes.csv and mapping columns."""
    csv_path = Path("data/clinical_notes.csv")
    if not csv_path.exists():
        pytest.skip("data/clinical_notes.csv does not exist")

    df = load_csv_notes(csv_path)
    assert "clinical_note" in df.columns
    assert "note_text" in df.columns
    assert "gender" in df.columns
    assert "patient_sex" in df.columns
    assert "patient_age" in df.columns
    assert "encounter_id" not in df.columns

    # Verify sex mapping
    valid_sexes = {"female", "male", "unknown"}
    assert set(df["patient_sex"].dropna().unique()).issubset(valid_sexes)

    # Verify age computation is positive
    valid_ages = df["patient_age"].dropna()
    assert (valid_ages >= 0).all()


def test_find_quote_spans():
    """Test quote span detection with overlapping quotes and case insensitivity."""
    text = "A 33-year-old male presented with severe acute chest syndrome and fever."
    quotes = ["severe acute chest syndrome", "acute chest syndrome", "33-year-old male", "nonexistent quote"]
    spans = find_quote_spans(text, quotes)

    assert len(spans) == 2
    # First span: 33-year-old male
    assert spans[0][0] == 2
    assert text[spans[0][0]:spans[0][1]] == "33-year-old male"
    # Second span: severe acute chest syndrome (longest first, preventing collision with 'acute chest syndrome')
    assert text[spans[1][0]:spans[1][1]] == "severe acute chest syndrome"


def test_highlight_note_quotes_html():
    """Test highlight_note_quotes generates valid HTML with mark tags and escaped text."""
    text = "Patient has <abnormal> lab values and acute chest syndrome."
    findings = [
        {"feature": "acs", "value": True, "quote": "acute chest syndrome", "unit": None},
        {"feature": "fever", "value": True, "quote": "nonexistent text", "unit": None},
    ]
    html_out = highlight_note_quotes(text, findings)
    assert "<mark class=\"quote-highlight\"" in html_out
    assert "acute chest syndrome" in html_out
    # Check that <abnormal> was escaped to prevent HTML injection
    assert "&lt;abnormal&gt;" in html_out
    assert "<abnormal>" not in html_out


def test_shiny_app_instance():
    """Test that the Shiny app object is constructed and valid."""
    assert app is not None
    assert hasattr(app, "ui")
    assert hasattr(app, "server")


def test_layout_inlines_the_theme_assets():
    # The CSS and JS moved to dashboard/static/; the page must still carry them inline.
    from dashboard.layout import app_ui
    page = str(app_ui)
    assert "--scogs-canvas" in page                  # dashboard.css
    assert "function applyTheme(theme)" in page      # theme.js


def test_layout_ordering_and_summary_card():
    """Verify that patient outcomes summary is near the top and findings table precedes note context."""
    from dashboard.layout import app_ui
    page = str(app_ui)
    assert "patient_outcomes_summary_ui" in page
    assert "executive_grade_card" in page
    # Findings table must precede note context
    note_pos = page.find("Clinical Note Context &amp; Verified Spans")
    if note_pos == -1:
        note_pos = page.find("Clinical Note Context & Verified Spans")
    findings_pos = page.find("Clinical Findings &amp; Grounding Verification")
    if findings_pos == -1:
        findings_pos = page.find("Clinical Findings & Grounding Verification")
    assert note_pos != -1
    assert findings_pos != -1
    assert findings_pos < note_pos, "Findings table must appear before note context"


def test_dashboard_css_contains_scroll_classes():
    """Verify CSS has rules for findings scroll container and outcome summary buttons."""
    from dashboard.layout import STATIC_DIR
    css_content = (STATIC_DIR / "dashboard.css").read_text(encoding="utf-8")
    assert ".findings-scroll-container" in css_content
    assert ".outcome-summary-btn" in css_content

