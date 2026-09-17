"""Automated unit tests for SCOGS Interactive Dashboard and Backend Bridges."""
import json
from pathlib import Path
import pytest
import pandas as pd

from scogs import GradeResult
from dashboard.interactive_dashboard import (
    load_run_file,
    load_csv_notes,
    evaluate_clinical_features,
    is_ollama_available,
    highlight_note_quotes,
    find_quote_spans,
    FOCUS_OUTCOMES,
    app,
)


def test_focus_outcomes_structure():
    """Verify that all 14 focus outcomes are registered with proper metadata."""
    expected_ids = {"10", "11", "12", "15", "17", "21", "24", "28", "29", "39", "40", "47", "48", "49"}
    assert set(FOCUS_OUTCOMES.keys()) == expected_ids
    for outcome_id, meta in FOCUS_OUTCOMES.items():
        assert "name" in meta
        assert "organ_system" in meta
        assert "acuity" in meta


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


def test_evaluate_clinical_features_voc():
    """Test deterministic grading of outcome 28 (VOC) with demographic enrichment."""
    # Outcome 28: care_setting >= inpatient and not pain_co_complication -> Grade 3
    features = {
        "care_setting": "inpatient",
        "pain_co_complication": False,
        "death_attributed": False,
        "life_support": False,
    }
    result = evaluate_clinical_features(
        outcome_num="28",
        feature_dict=features,
        patient_sex="M",
        patient_age=33.0,
        present=True,
    )
    assert isinstance(result, GradeResult)
    assert result.status == "graded"
    assert result.grade == 3
    assert result.matched is not None
    assert "care_setting >= inpatient" in result.matched


def test_evaluate_clinical_features_derived_resolution():
    """Test that evaluate_clinical_features passes features through resolve_derived."""
    # Outcome 19 (AKI): creatinine baseline ratio derived
    features = {
        "creatinine": 3.0,
        "creatinine_baseline": 1.0,
        "creatinine_increase_mg_dl": 2.0,
        "death_attributed": False,
        "renal_replacement_therapy": False,
        "renal_replacement": False,
        "esrd": False,
        "esrd_progression": False,
        "patient_age": 25.0,
    }
    result = evaluate_clinical_features(
        outcome_num="19",
        feature_dict=features,
        patient_sex="female",
        patient_age=25.0,
        present=True,
    )
    assert isinstance(result, GradeResult)
    # With ratio 3.0, AKI stage 3
    assert result.status == "graded"
    assert result.grade == 3


def test_evaluate_clinical_features_unknown_outcome():
    """Test graceful handling of invalid outcome number."""
    result = evaluate_clinical_features("999", {}, patient_sex="unknown")
    assert isinstance(result, GradeResult)
    assert result.status == "cannot_grade"
    assert "Unknown outcome" in (result.reason or "")


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
    valid_sexes = {"F", "M", "unknown"}
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


def test_is_ollama_available_offline():
    """Test is_ollama_available returns False when Ollama server is offline."""
    # When host points to a closed port, is_ollama_available should return False without raising
    assert is_ollama_available(host="http://localhost:59999") is False


def test_shiny_app_instance():
    """Test that the Shiny app object is constructed and valid."""
    assert app is not None
    assert hasattr(app, "ui")
    assert hasattr(app, "server")
