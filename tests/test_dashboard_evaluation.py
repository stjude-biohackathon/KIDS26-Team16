"""dashboard/evaluation.py: the dashboard grades through the CLI's own extraction and grading."""
import json

import pytest

from dashboard import evaluation
from dashboard.evaluation import (
    FOCUS_OUTCOMES, clinician_context, evaluate_clinical_features, extract_and_grade_note,
    extract_with_harness, grade_badge, is_derived, is_ollama_available,
)
from experiments.medgemma_extraction import DEFAULT_OUTCOMES
from scogs import GradeResult

NOTE = ("A 14-year-old with HbSS presented with chest pain. FiO2 was escalated to 60%. "
        "He received a simple transfusion of 2 units and was started on norepinephrine.")
VOC_INPATIENT = {"care_setting": "inpatient", "pain_co_complication": False,
                 "death_attributed": False, "life_support": False}
AKI_TRIPLED_CREATININE = {"creatinine": 3.0, "creatinine_baseline": 1.0,
                          "creatinine_increase_mg_dl": 2.0, "death_attributed": False,
                          "renal_replacement_therapy": False, "renal_replacement": False,
                          "esrd": False, "esrd_progression": False, "patient_age": 25.0}


# ------------------------------------------------------------ live extraction

def test_live_extraction_grades_only_verified_findings():
    # The mock backend returns two findings for one feature: value False quoted
    # from the note ("presented") and value True quoted from a sentence the note
    # does not contain. The old dashboard graded the invented True.
    item = extract_with_harness(NOTE, ["48"], backend="mock", model="mock", host="")["48"]
    assert len(json.loads(item["raw_reply"])["findings"]) == 2
    assert item["accepted_findings"] == [
        {"feature": "death_attributed", "value": False, "quote": "presented", "unit": None}]
    assert item["extracted_features"] == {"death_attributed": False}
    assert item["status"] == "cannot_grade"


def test_a_backend_failure_is_shown_on_the_card_not_raised(monkeypatch):
    def unreachable(*args, **kwargs):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(evaluation, "extract_with_harness", unreachable)
    item = extract_and_grade_note(NOTE, ["28"], use_ollama=True)["28"]
    assert item["status"] == "cannot_grade"
    assert "connection refused" in item["error"]


# ------------------------------------------------------ clinician-entered values

def test_clinician_sex_codes_reach_the_rules_as_schema_values():
    # rules.md §24: priapism applies to male patients only. "M" used to be passed
    # through unchanged, and the rules do not know "M".
    def status(sex):
        return extract_and_grade_note("", ["24"], patient_context={"patient_sex": sex},
                                      manual_features={"24": {}})["24"]["status"]
    assert status("M") == "cannot_grade"
    assert status("female") == "not_applicable"


@pytest.mark.parametrize("sex, age, expected", [
    ("M", 33, {"patient_sex": "male", "patient_age": 33.0}),
    ("female", "12.5", {"patient_sex": "female", "patient_age": 12.5}),
    ("unknown", None, {}),
    ("", "not a number", {}),
])
def test_clinician_context_uses_the_schema_vocabulary(sex, age, expected):
    assert clinician_context(sex, age) == expected


def test_manual_voc_is_graded_with_the_matched_rule():
    graded = evaluate_clinical_features("28", VOC_INPATIENT, patient_sex="M", patient_age=33.0)
    assert isinstance(graded.result, GradeResult)
    assert (graded.status, graded.result.grade) == ("graded", 3)
    assert "care_setting >= inpatient" in graded.result.matched


def test_manual_derived_features_are_resolved():
    graded = evaluate_clinical_features("19", AKI_TRIPLED_CREATININE, patient_sex="female",
                                        patient_age=25.0)
    assert (graded.status, graded.result.grade) == ("graded", 3)


def test_an_unknown_outcome_cannot_be_graded():
    graded = evaluate_clinical_features("999", {}, patient_sex="unknown")
    assert graded.status == "cannot_grade"
    assert "Unknown outcome" in graded.result.reason


# ------------------------------------------------------------------ display helpers

def test_derived_features_come_from_the_schema():
    assert is_derived("bp_stage") and is_derived("life_support")
    assert not is_derived("creatinine")
    assert not is_derived("bmi")          # not a SCOGS feature at all


@pytest.mark.parametrize("status, grade, grades, expected", [
    ("graded", 3, (), ("badge-grade-3", "GRADE 3")),
    ("graded", 1, (), ("badge-grade-1", "GRADE 1")),
    ("grade_set", None, (2, 3), ("badge-grade-set", "GRADE 2 OR 3")),
    ("missed_presence", None, (), ("badge-cannot-grade", "MISSED PRESENCE (REVIEW)")),
    ("refuted", None, (), ("badge-refuted", "REFUTED (CONTRADICTED)")),
    ("pending", None, (), ("badge-not-applicable", "PENDING")),
])
def test_grade_badge(status, grade, grades, expected):
    assert grade_badge(status, grade, grades) == expected


def test_focus_outcomes_are_the_cli_default_and_described():
    assert set(FOCUS_OUTCOMES) == set(DEFAULT_OUTCOMES.split(","))
    for meta in FOCUS_OUTCOMES.values():
        assert {"name", "organ_system", "acuity"} <= set(meta)


def test_is_ollama_available_offline():
    assert is_ollama_available(host="http://localhost:59999") is False


def test_allowed_models_and_model_choices():
    from dashboard.evaluation import ALLOWED_MODELS, OLLAMA_NUM_CTX, get_model_choices
    assert "medgemma-1.5-4b-it" in ALLOWED_MODELS
    assert "gemma4:12b" in ALLOWED_MODELS
    assert "medgemma-27b-f16" in ALLOWED_MODELS
    assert OLLAMA_NUM_CTX == 16384
    choices = get_model_choices(host="http://localhost:59999")
    for k in ("medgemma-1.5-4b-it", "gemma4:12b", "medgemma-27b-f16"):
        assert k in choices
    assert "__custom__" in choices


VOC_ITEM = {
    "outcome_name": "Acute Sickle Cell Pain Episode (VOC)",
    "present": True,
    "extracted_features": {"care_setting": "inpatient"},
    "accepted_findings": [{"feature": "care_setting", "value": "inpatient", "quote": "admitted"}],
    "conflicts": {},
    "status": "graded",
    "grade_result": {"status": "graded", "grade": 3, "reason": "care_setting >= inpatient"},
}


def test_saved_patient_file_opens_in_explore_mode(tmp_path):
    from dashboard.data import is_run_file, load_run_file
    from dashboard.evaluation import save_live_patient_results
    from dashboard.view_state import explore_view_state

    live = evaluate_clinical_features("48", {})    # a real GradeResult, not a dict
    path = save_live_patient_results(
        "Patient admitted with severe pain.",
        {"28": VOC_ITEM, "48": {"status": live.status, "grade_result": live.result}},
        patient_id="P00000", visit="10/8/20", title="ER visit 10/8/20",
        patient_age=31, patient_sex="female", output_dir=tmp_path,
    )
    assert path == tmp_path / "patient_P00000.json"
    assert is_run_file(path)
    run = load_run_file(path)
    assert run["provenance"]["source"] == "live"
    state = explore_view_state(run, "P00000 @ 10/8/20", "28")
    assert (state["status"], state["patient_age"], state["patient_sex"]) == ("graded", 31.0, "female")
    assert run["records_by_uid"]["P00000 @ 10/8/20"]["outcomes"]["48"]["grade_result"]["status"] == live.status


def test_every_visit_of_a_patient_shares_one_file(tmp_path):
    from dashboard.evaluation import save_live_patient_results

    for visit in ("10/8/20", "12/26/20"):
        save_live_patient_results(f"Note from {visit}.", {"28": VOC_ITEM},
                                  patient_id="P00000", visit=visit, output_dir=tmp_path)
    save_live_patient_results("Another patient.", {"28": VOC_ITEM},
                              patient_id="P00001", visit="1/1/21", output_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["patient_P00000.json", "patient_P00001.json"]
    doc = json.loads((tmp_path / "patient_P00000.json").read_text())
    assert [r["patient_uid"] for r in doc["detailed_records"]] == ["P00000 @ 10/8/20", "P00000 @ 12/26/20"]
    assert doc["grade_status"] == {"graded": 2}


def test_regrading_a_visit_updates_its_record_instead_of_adding_one(tmp_path):
    from dashboard.evaluation import save_live_patient_results

    note = "Patient admitted with severe pain."
    save_live_patient_results(note, {"28": VOC_ITEM}, patient_id="P00000", visit="10/8/20", output_dir=tmp_path)
    absent = {**VOC_ITEM, "outcome_name": "Stroke", "present": False, "status": "absent",
              "grade_result": {"status": "absent", "grade": None}}
    save_live_patient_results(note, {"15": absent}, patient_id="P00000", visit="10/8/20", output_dir=tmp_path)

    doc = json.loads((tmp_path / "patient_P00000.json").read_text())
    assert len(doc["detailed_records"]) == 1
    assert set(doc["detailed_records"][0]["outcomes"]) == {"28", "15"}   # earlier outcome kept

    # An edited note is a different note: its old outcomes no longer apply.
    save_live_patient_results(note + " Edited.", {"15": absent}, patient_id="P00000", visit="10/8/20",
                              output_dir=tmp_path)
    doc = json.loads((tmp_path / "patient_P00000.json").read_text())
    assert set(doc["detailed_records"][0]["outcomes"]) == {"15"}
    assert list(tmp_path.iterdir()) == [tmp_path / "patient_P00000.json"]   # no temp file left


def test_a_pasted_note_files_under_a_hash_of_its_text(tmp_path):
    from dashboard.evaluation import live_patient_key, save_live_patient_results

    first = save_live_patient_results("Pasted note.", {"28": VOC_ITEM}, output_dir=tmp_path)
    again = save_live_patient_results("  Pasted note.  ", {"28": VOC_ITEM}, output_dir=tmp_path)
    other = save_live_patient_results("A different note.", {"28": VOC_ITEM}, output_dir=tmp_path)

    assert first == again != other
    assert first.name.startswith("note_") and first.stem == live_patient_key(None, "Pasted note.")
    assert live_patient_key("P 00/1", "x") == "patient_P_00_1"     # safe as a file name


def test_is_model_installed_matching():
    from dashboard.evaluation import is_model_installed
    installed = ["medgemma-1.5-4b-it:latest", "gemma4:12b", "qwen3.6:35b-a3b"]

    assert is_model_installed("medgemma-1.5-4b-it", installed) is True
    assert is_model_installed("medgemma-1.5-4b-it:latest", installed) is True
    assert is_model_installed("gemma4:12b", installed) is True
    assert is_model_installed("qwen3.6:35b-a3b", installed) is True
    # Crucial: gemma4:21b must NOT match gemma4:12b just because of the gemma4 prefix
    assert is_model_installed("gemma4:21b", installed) is False
    assert is_model_installed("medgemma-27b-f16", installed) is False
    assert is_model_installed("", installed) is False
    assert is_model_installed("medgemma-1.5-4b-it", []) is False


def test_get_model_choices_grouped_and_offline():
    from dashboard.evaluation import get_model_choices
    # Offline port
    grouped_offline = get_model_choices(host="http://localhost:59999", grouped=True)
    assert "Supported Models (Ollama Offline)" in grouped_offline
    assert "medgemma-1.5-4b-it" in grouped_offline["Supported Models (Ollama Offline)"]
    assert "Offline" in grouped_offline["Supported Models (Ollama Offline)"]["medgemma-1.5-4b-it"]

    flat_offline = get_model_choices(host="http://localhost:59999", grouped=False)
    assert "medgemma-1.5-4b-it" in flat_offline
    assert "Offline" in flat_offline["medgemma-1.5-4b-it"]


def test_check_ollama_status_offline():
    from dashboard.evaluation import check_ollama_status
    status, msg = check_ollama_status(model="medgemma-1.5-4b-it", host="http://localhost:59999")
    assert status == "offline"
    assert "Offline" in msg


def test_check_ollama_status_and_choices_mocked(monkeypatch):
    from dashboard import evaluation
    monkeypatch.setattr(evaluation, "get_installed_models", lambda host: ["medgemma-1.5-4b-it:latest", "gemma4:12b", "llama3:8b"])

    status_ready, _ = evaluation.check_ollama_status("medgemma-1.5-4b-it")
    assert status_ready == "ready"

    status_missing, _ = evaluation.check_ollama_status("medgemma-27b-f16")
    assert status_missing == "not_installed"

    choices_grouped = evaluation.get_model_choices(grouped=True)
    assert "Installed in Ollama" in choices_grouped
    assert "medgemma-1.5-4b-it" in choices_grouped["Installed in Ollama"]
    assert "(Installed)" in choices_grouped["Installed in Ollama"]["medgemma-1.5-4b-it"]
    assert "gemma4:12b" in choices_grouped["Installed in Ollama"]
    assert "llama3:8b" in choices_grouped["Installed in Ollama"]

    assert "Not Installed (Pull Required)" in choices_grouped
    assert "medgemma-27b-f16" in choices_grouped["Not Installed (Pull Required)"]
    assert "(Not Installed)" in choices_grouped["Not Installed (Pull Required)"]["medgemma-27b-f16"]
    assert "Custom Model" in choices_grouped
    assert "__custom__" in choices_grouped["Custom Model"]

    choices_flat = evaluation.get_model_choices(grouped=False)
    assert "(Installed)" in choices_flat["medgemma-1.5-4b-it"]
    assert "(Installed)" in choices_flat["gemma4:12b"]
    assert "(Installed)" in choices_flat["llama3:8b"]
    assert "(Not Installed)" in choices_flat["medgemma-27b-f16"]
    assert "__custom__" in choices_flat


def test_get_live_concurrency():
    from dashboard.evaluation import get_live_concurrency
    assert get_live_concurrency("medgemma-1.5-4b-it") == 4
    assert get_live_concurrency("medgemma-1.5-4b-it:latest") == 4
    assert get_live_concurrency("medgemma-4b") == 4
    assert get_live_concurrency("medgemma-27b-f16") == 2
    assert get_live_concurrency("gemma4:21b") == 2
    assert get_live_concurrency("") == 2


def test_detect_system_hardware():
    from dashboard.evaluation import detect_system_hardware
    hw = detect_system_hardware()
    assert isinstance(hw["total_ram_gb"], float)
    assert hw["total_ram_gb"] > 0
    assert isinstance(hw["cpu_count"], int)
    assert hw["cpu_count"] > 0
    assert isinstance(hw["platform"], str)


def test_get_concurrency_assessment_model_tiers():
    from dashboard.evaluation import get_concurrency_assessment

    # 4B model on 16 GB system
    res_4b_optimal = get_concurrency_assessment("medgemma-1.5-4b-it", concurrency=4, ram_gb=16.0)
    assert res_4b_optimal["recommended"] == 4
    assert res_4b_optimal["status"] == "safe"
    assert "Optimal" in res_4b_optimal["message"]

    res_4b_high = get_concurrency_assessment("medgemma-1.5-4b-it", concurrency=5, ram_gb=16.0)
    assert res_4b_high["status"] == "caution"

    # 12B model on 16 GB system
    res_12b_1 = get_concurrency_assessment("gemma4:12b", concurrency=1, ram_gb=16.0)
    assert res_12b_1["recommended"] == 1
    assert res_12b_1["status"] == "safe"

    res_12b_2 = get_concurrency_assessment("gemma4:12b", concurrency=2, ram_gb=16.0)
    assert res_12b_2["status"] == "caution"

    res_12b_3 = get_concurrency_assessment("gemma4:12b", concurrency=3, ram_gb=16.0)
    assert res_12b_3["status"] == "danger"
    assert "High Risk" in res_12b_3["message"]

    # 12B model on 64 GB workstation
    res_12b_workstation = get_concurrency_assessment("gemma4:12b", concurrency=2, ram_gb=64.0)
    assert res_12b_workstation["status"] == "safe"


def test_get_all_scogs_outcomes_and_choices():
    from dashboard.evaluation import (
        TABLES,
        get_all_scogs_outcomes,
        get_outcome_choices,
        outcome_display_name,
    )

    outcomes = get_all_scogs_outcomes()
    assert len(outcomes) == 53
    assert len(TABLES) == 53

    focus_count = sum(1 for m in outcomes.values() if m["is_focus"])
    non_focus_count = sum(1 for m in outcomes.values() if not m["is_focus"])
    assert focus_count == 14
    assert non_focus_count == 39

    # Verify specific focus and non-focus outcomes
    assert outcomes["15"]["is_focus"] is True
    assert outcomes["15"]["name"] == "Stroke"
    assert outcomes["01"]["is_focus"] is False
    assert outcomes["01"]["name"] == "Arrhythmia"
    assert outcomes["53"]["is_focus"] is False
    assert outcomes["53"]["name"] == "Sleep Apnea (obstructive or central)"

    # Test grouped choices
    choices_grouped = get_outcome_choices(grouped=True)
    assert "14 Core Focus Outcomes (Delphi Consensus)" in choices_grouped
    assert "Additional SCOGS Decision Tables (39 Outcomes)" in choices_grouped
    assert len(choices_grouped["14 Core Focus Outcomes (Delphi Consensus)"]) == 14
    assert len(choices_grouped["Additional SCOGS Decision Tables (39 Outcomes)"]) == 39

    # Test flat choices
    choices_flat = get_outcome_choices(grouped=False)
    assert len(choices_flat) == 53

    # Test outcome display name resolution
    assert outcome_display_name("1") == "Arrhythmia"
    assert outcome_display_name("01") == "Arrhythmia"
    assert outcome_display_name("15") == "Stroke"
    assert outcome_display_name("28") == "Acute Sickle Cell Pain Episode (VOC)"
    assert outcome_display_name("53") == "Sleep Apnea (obstructive or central)"



