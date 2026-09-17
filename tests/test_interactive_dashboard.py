"""Automated unit tests for SCOGS Interactive Dashboard and Backend Bridges."""
from pathlib import Path
import pytest

from dashboard.interactive_dashboard import (
    app,
    find_quote_spans,
    highlight_note_quotes,
    load_csv_notes,
    load_run_file,
)


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
