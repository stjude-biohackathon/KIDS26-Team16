"""Cross-platform workflow checks; the HTTP server stands in for Ollama, not a model."""
import csv
import importlib
import json
import pathlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from experiments import medgemma_extraction as extraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiments" / "medgemma_extraction.py"


def model_info(quant="F16", count=27_000_000_000, file_type=1):
    return {
        "details": {"family": "gemma3", "parameter_size": "27.0B",
                    "quantization_level": quant},
        "model_info": {"general.architecture": "gemma3",
                       "general.parameter_count": count,
                       "general.file_type": file_type},
    }


@pytest.fixture
def ollama_server():
    state = {"info": model_info(), "requests": [], "error": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond({"models": [{"name": "medgemma-27b-f16:latest",
                                      "digest": "sha256:test-model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body))
            if self.path == "/api/show":
                self.respond(state["info"])
            elif state["error"]:
                self.respond({"error": "out of memory"}, status=500)
            else:
                reply = (extraction.call_mock(body["prompt"], "", "")
                         if "Health outcome under consideration:" in body["prompt"]
                         else '{"present": false, "findings": []}')
                self.respond({"response": reply, "done": True, "eval_count": 20,
                              "prompt_eval_count": 40})

        def respond(self, data, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["host"] = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_only_ollama_and_mock_backends_remain():
    assert set(extraction.BACKENDS) == {"ollama", "mock"}


@pytest.mark.parametrize("quant,file_type", [("F16", 1), ("BF16", 32)])
def test_full_precision_27b_metadata_is_accepted(quant, file_type):
    backend = importlib.import_module("experiments.ollama_backend")
    assert backend.validate_model(model_info(quant, file_type=file_type)) == quant


@pytest.mark.parametrize("info", [
    model_info("Q4_K_M", file_type=15),
    model_info("F32", file_type=0),
    model_info(count=4_000_000_000),
    model_info(count=70_000_000_000),
    model_info(file_type=15),
    {},
    {"details": {"parameter_size": "27B", "quantization_level": "F16"}},
])
def test_smaller_quantized_or_unverifiable_models_are_rejected(info):
    backend = importlib.import_module("experiments.ollama_backend")
    with pytest.raises(ValueError):
        backend.validate_model(info)


def test_preflight_checks_metadata_digest_and_generation(ollama_server):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--host", ollama_server["host"], "--check-model"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "F16" in result.stdout and "sha256:test-model" in result.stdout
    assert [p for p, _ in ollama_server["requests"]] == ["/api/show", "/api/generate"]


def test_rejected_model_never_receives_patient_notes(ollama_server, tmp_path):
    ollama_server["info"] = model_info("Q4_K_M", file_type=15)
    out = tmp_path / "invalid.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--host", ollama_server["host"],
         "--notes", "1", "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert not out.exists()
    assert [p for p, _ in ollama_server["requests"]] == ["/api/show"]


def test_generation_error_is_not_an_empty_success(ollama_server, monkeypatch):
    ollama_server["error"] = True
    monkeypatch.setattr(extraction.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="out of memory"):
        extraction.call_ollama("synthetic prompt", "medgemma-27b-f16",
                               ollama_server["host"])


@pytest.mark.parametrize("flags", [
    ["--notes", "0"], ["--repeat", "0"], ["--outcomes", ""],
    ["--holdout-frac", "1.5"], ["--timeout", "0"],
    ["--num-predict", "0"], ["--num-ctx", "0"], ["--tier", "local"],
])
def test_invalid_cli_input_is_rejected_before_running(flags, tmp_path):
    out = tmp_path / "invalid.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mock", "--out", str(out), *flags],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert not out.exists()


def test_results_and_review_exports_use_the_real_output_contract(ollama_server, tmp_path):
    out = tmp_path / "run with spaces" / "results.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--host", ollama_server["host"], "--notes", "2",
         "--cohort", "scd_primary", "--repeat", "2", "--prompt-stage", "3",
         "--out", str(out)], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["provenance"]["quant"] == "F16"
    assert data["provenance"]["model_digest"] == "sha256:test-model"
    assert data["provenance"]["num_ctx"] == 16384
    review = importlib.import_module("experiments.review_results")
    paths = review.export_reviews(out)
    with paths["handcheck"].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == data["automated_metrics"]["accepted"]
    assert all(row["quote"] and row["supports_value"] == "" for row in rows)
    assert all(row["run_id"] == data["provenance"]["run_id"] for row in rows)
    assert all(len(row["source_sha256"]) == 64 for row in rows)
    with pytest.raises(FileExistsError):
        review.export_reviews(out)
    rerun = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "mock", "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert rerun.returncode != 0
    assert json.loads(out.read_text(encoding="utf-8")) == data


def test_review_conflicts_and_absences_are_separate(tmp_path):
    review = importlib.import_module("experiments.review_results")
    outcome = {
        "outcome_name": "Fever", "present": True,
        "extracted_features": {},
        "accepted_findings": [
            {"feature": "treated", "value": True, "quote": "treated", "unit": None},
            {"feature": "treated", "value": False, "quote": "not treated", "unit": None},
        ],
        "conflicts": {"treated": [True, False]},
        "grade_result": {"status": "refuted", "grade": None, "reason": "no matching rule"},
    }
    source = tmp_path / "results.json"
    source.write_text(json.dumps({
        "provenance": {"model_digest": "abc", "prompt_stage": "3"},
        "detailed_records": [{
            "patient_uid": "test", "patient_note": "treated; later not treated",
            "outcomes": {"36": outcome},
        }],
    }), encoding="utf-8")
    paths = review.export_reviews(source)
    def rows(name):
        with paths[name].open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    assert len(rows("handcheck")) == 2
    assert len(rows("conflicts")) == 1
    assert rows("absence_audit") == []
    assert len(rows("refuted_audit")) == 1


def test_consistency_includes_presence_and_every_repeat(tmp_path, monkeypatch):
    calls = 0

    def changing_presence(*args, **kwargs):
        nonlocal calls
        calls += 1
        return json.dumps({"present": calls < 3, "findings": []})

    out = tmp_path / "repeats.json"
    monkeypatch.setitem(extraction.BACKENDS, "mock", changing_presence)
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--backend", "mock", "--notes", "1", "--outcomes", "36",
        "--repeat", "3", "--out", str(out),
    ])
    assert extraction.main() == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["automated_metrics"]["run_to_run_consistency_pct"] == 0


@pytest.mark.parametrize("name,expected_runs", [
    ("01_run_extraction.ipynb", 1), ("02_compare_prompts.ipynb", 5),
])
def test_notebooks_execute_in_order_without_colab_or_shell_magics(
    name, expected_runs, ollama_server, tmp_path, monkeypatch,
):
    path = ROOT / "notebooks" / name
    assert path.exists()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    monkeypatch.chdir(ROOT / "notebooks")
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        source = "".join(cell["source"])
        assert len(source) < 2500, f"Cell {index} is too large to follow"
        exec(compile(source, f"{name}:cell{index}", "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(HOST=ollama_server["host"], NOTES=2, REPEAT=1,
                             RUN_DIR=tmp_path / "notebook results")
    outputs = list((tmp_path / "notebook results").glob("*.json"))
    assert len(outputs) == expected_runs
    assert len(list((tmp_path / "notebook results").glob("*/handcheck.csv"))) == expected_runs
    assert all(json.loads(p.read_text(encoding="utf-8"))["provenance"]["quant"] == "F16" for p in outputs)
