# SCOGS: MedGemma 27B feature extraction

Extract clinical evidence from sickle cell case reports with **MedGemma 27B
through Ollama**, then grade it with deterministic SCOGS decision tables.
The model proposes features and supporting quotes; it never assigns the grades.

**Supported weights: full, unquantized F16 or BF16 MedGemma 27B only.**
The runner checks actual model metadata, not just the tag name. Smaller models
and quantized weights are rejected. The `mock` backend is a model-free plumbing
check, not an alternative research model.

**Research use only.** Quote verification checks that words occur in the note,
not that they support the extracted value. Human review is required before
making accuracy claims. This is not a validated clinical decision tool.

## Start here

1. Install **Python 3.10+** and [Ollama](https://ollama.com/download).
2. Clone the repo and create a Python environment using the commands below.
3. Follow [full 27B Ollama setup](docs/ollama_setup.md) to import verified
   F16/BF16 weights and check memory requirements.
4. Open [01_run_extraction.ipynb](notebooks/01_run_extraction.ipynb), edit its
   configuration cell, and run the cells in order.

### Windows PowerShell

```powershell
git clone https://github.com/stjude-biohackathon/KIDS26-Team16.git
cd KIDS26-Team16
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name scogs --display-name "SCOGS"
.\.venv\Scripts\python.exe -m jupyterlab
```

Using the environment's Python directly avoids PowerShell activation-policy
issues. Select the **SCOGS** kernel in Jupyter or VS Code.

### Linux

```bash
git clone https://github.com/stjude-biohackathon/KIDS26-Team16.git
cd KIDS26-Team16
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name scogs --display-name "SCOGS"
python -m jupyterlab
```

If `venv` is unavailable, install your distribution's Python venv package first.
The extraction and CSV-export scripts themselves use only the Python standard
library. Python does not need PyTorch, Transformers, CUDA packages, or pandas;
Ollama manages model inference independently.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [01_run_extraction](notebooks/01_run_extraction.ipynb) | Normal run: configure, verify model, extract, inspect, export review sheets, archive |
| [02_compare_prompts](notebooks/02_compare_prompts.ipynb) | Optional comparison of stages `0`, `1`, `2a`, `2b`, and `3` on the same model and cases |

Both run on Windows/Linux, use the selected kernel's Python interpreter, handle
paths containing spaces, and stop on subprocess failure. Each creates a unique
folder under `results/`; neither downloads models or changes the Ollama service.
The old Colab/Drive recovery notebooks and dashboard have been removed.

## Terminal workflow

The following commands are identical across operating systems when `python`
points to the project environment. On Windows without activation, substitute
`.\.venv\Scripts\python.exe` for `python`.

```bash
# Optional model-free check; deliberately verifies exactly half its quotes.
python scripts/experiments/medgemma_extraction.py --backend mock --notes 2

# Check the imported model, including a synthetic JSON generation.
python scripts/experiments/medgemma_extraction.py --check-model

# Small real run first.
python scripts/experiments/medgemma_extraction.py --cohort scd_primary --notes 2 --out results/smoke.json

# Reporting run, with repeatability measured sequentially.
python scripts/experiments/medgemma_extraction.py --cohort scd_primary --notes 20 --repeat 2 --out results/full.json

# Export worksheets from exactly that saved run; no model call.
python scripts/experiments/review_results.py results/full.json
```

The default local tag is `medgemma-27b-f16`. For an imported BF16 model, pass
`--model medgemma-27b-bf16`. These are **local aliases you create**, not claimed
public Ollama registry tags. No weights are stored in this repository.

### Important run options

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `OLLAMA_HOST` or `http://localhost:11434` | Complete Ollama HTTP(S) URL |
| `--model` | `medgemma-27b-f16` | Local tag; metadata must verify 27B and F16/BF16 |
| `--cohort` | `loose` | `scd_primary` for disease-focused reporting; `loose` includes mention-only cases |
| `--outcomes` | `10,11,12,15,17,21,24,28,29,39,40,47,48,49` | 14 focus outcomes (Chronic Pain, CD, TCD, Stroke, Retinopathy, CKD, Priapism, Pain episode, SS, AVN, Leg Ulcer, Depression, ACS, Asthma); '14', 'focus', 'all', or custom comma-separated IDs supported |
| `--notes` | `20` | Number of notes; evaluated pairs = notes x outcomes |
| `--stratify` / `--no-stratify` | enabled | Outcome-enriched sampling plus a random holdout |
| `--holdout-frac` | `0.25` | Random fraction used to assess sampling bias |
| `--prompt-stage` | `2b` | Prompt stage: `0`, `1`, `2a`, `2b` or `3`. `2b` scored best on the prompt comparison; `0` is the original prompt, kept as the baseline |
| `--repeat` | `1` | Repeated extraction with temperature-zero decoding |
| `--concurrency` | `1` | In-flight requests; keep at one for repeatability comparisons |
| `--patient-context` / `--no-patient-context` | disabled | Extract patient-level context (age, sex) once per note and share across outcomes |
| `--feedback-retry` / `--no-feedback-retry` | disabled | Re-prompt once with specific error feedback if extracted quotes or values fail verification |
| `--num-ctx` | `16384` | Context tokens; increase for long notes, allowing extra memory |
| `--num-predict` | `1024` / `2048` | Completion budget; stages `2b`/`3` need space for evidence |
| `--timeout` | `300` seconds | Request timeout; increase for slow loading or CPU offload |
| `--out` | none | Optional JSON destination; existing files are never overwritten |

Parallel serving needs matching Ollama server configuration and extra memory.
Batching can change floating-point results even at temperature zero; do not
interpret batched variation as model nondeterminism alone. See
[the evaluation protocol](docs/research/extraction_protocol.md).

## Outputs and human review

JSON contains model digest and precision, prompt settings, cohort/selection
metadata, per-run counters, profiling, grade-status distributions, and
`detailed_records` with note text and evidence. Detailed records describe the
**first repeat**; `runs` holds counters for all repeats.

`review_results.py results/full.json` creates `results/full/`:

| File | Review task |
| --- | --- |
| `handcheck.csv` | Up to 100 grounded proposals; mark whether the quote supports the value |
| `conflicts.csv` | All features withheld because verified values disagree |
| `absence_audit.csv` | Up to 50 model-absent pairs, including the note and features to look for |
| `refuted_audit.csv` | All model-present pairs that the rules refute, kept separate from model absences |

CSVs use UTF-8 with a BOM for Windows spreadsheet compatibility. Each carries the
source `run_id` and a SHA-256 of the result file; even an empty sheet has headers. Existing sheets are
protected because they may contain completed reviews. Sample limits default to 100 for `handcheck.csv`
and 50 for `absence_audit.csv`, but can be customized with `--handcheck-limit` and `--absence-limit` (use `-1` for all).
For custom limits or an independent reviewer:

```bash
python scripts/experiments/review_results.py results/full.json --output-dir results/reviewer_2 --handcheck-limit 300 --absence-limit 150
```

`results/`, model weights, raw dataset downloads, and local environments are
ignored by Git. Keep run artifacts together and share only with authorized
collaborators. Clear notebook outputs before committing.

## Repository map

```text
notebooks/                  Two ordered, cross-platform entry points
scripts/experiments/        Extraction, Ollama preflight/client, review exports
scripts/scogs/              Feature definitions, predicates, deterministic tables
scripts/audit/              Optional rubric-to-PDF verification utilities
scripts/download_data.py    Optional full-corpus downloader
data/pmc_patients/          Bundled SCD case cache and original dataset card
data/clinical_notes*.csv    Legacy scaffold data, not SCOGS ground truth
data/scogs_feature_schema.json
docs/ollama_setup.md        Model import, Windows/Linux service setup, troubleshooting
docs/research/              Current architecture and evaluation protocol
docs/reference/             Source booklet, proposal, rubric verification log
rules.md                   Human-readable grading authority
tests/                     Rule, extraction, HTTP, and notebook workflow checks
results/                   Local outputs; ignored by Git
```

The bundled cache contains 978 published case summaries; `scd_primary` is a
stricter subset. Normal runs do not download a dataset. The legacy clinical CSVs
are retained as source material, not used for evaluation or training labels.
See [data notes](data/README.md) and the
[PMC-Patients dataset card](data/pmc_patients/README.md) for provenance/licensing.

For optional full-corpus work:

```bash
python scripts/download_data.py --files v2
```

This downloads into `data/pmc_patients/` and rebuilds the cache. Freeze a copy
before rebuilding if reproducing an earlier experiment.

## Development

```bash
python -m pytest -q
python scripts/scogs/build_schema.py
```

CI runs Python tests and notebook-cell workflows on Windows and Linux against a
local simulated Ollama API. It does not download or validate real model weights.
Real hardware still requires `--check-model` and a small extraction run.

Optional booklet audits require Poppler's `pdftotext` on `PATH`, not a model:

```bash
python scripts/audit/extract_booklet.py
python scripts/audit/compare_grades.py
python scripts/audit/compare_prose.py
```
