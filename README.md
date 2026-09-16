# SCOGS: Sickle Cell Outcome Grading System & MedGemma Extraction

This repository contains the deterministic rule engine for the **Sickle Cell Outcome Grading System (SCOGS)**, the schema definition for clinical features, and the test harness for evaluating **MedGemma** (4B and 27B) standalone clinical feature extraction against the §2 verbatim quote verification contract.

---

## 📦 Datasets & Data Acquisition

### 1. Pre-Packaged Data (Included in Git Repository)

The repository comes pre-packaged with all data required to run the test suite and feature extraction experiments immediately upon cloning:

- **[`PMC-Patients/scd_cache.json`](PMC-Patients/scd_cache.json) (3.6 MB):** 978 curated Sickle Cell Disease (SCD) patient case reports filtered from the PubMed Central patient dataset. This is the primary dataset read by `scripts/experiments/medgemma_extraction.py`.
- **[`data/clincal_notes.csv`](data/clincal_notes.csv) & [`data/clincal_notes_org.csv`](data/clincal_notes_org.csv):** Clinical note datasets with multi-outcome labels.
- **[`data/scogs_feature_schema.json`](data/scogs_feature_schema.json):** The 137 clinical feature schema definitions for the 53 SCOGS health outcomes.

---

### 2. Downloading Full Raw Datasets (~1.3 GB)

If you need the entire raw 250,000-patient case report corpus from PubMed Central (e.g. for broader cohort exploration or training):

#### Automated Download Script (Cross-Platform: Windows, Mac, Linux)
Run the built-in downloader script:

```bash
# Downloads full PMC-Patients-V2.json (~800MB) & PMC-Patients.csv (~520MB) from Hugging Face
python scripts/download_data.py
```

Options:
```bash
python scripts/download_data.py --files v2    # Download only PMC-Patients-V2.json
python scripts/download_data.py --files csv   # Download only PMC-Patients.csv
python scripts/download_data.py --force       # Re-download even if already present
```

#### Manual Download Links (Hugging Face)
You can also download the files directly from the official [Hugging Face dataset repo (`zhengyun21/PMC-Patients`)](https://huggingface.co/datasets/zhengyun21/PMC-Patients):

- **PMC-Patients-V2.json (250,294 patients):**
  [`https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients-V2.json`](https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients-V2.json)
- **PMC-Patients.csv (167k patient summaries):**
  [`https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv`](https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv)

Place the downloaded files inside the `PMC-Patients/` directory at the project root.

---

## Quick Start & Installation

### 1. Clone & Set Up Python Environment

```bash
# Clone repository
git clone https://github.com/Edward-Bae-00/st_jude.git
cd st_jude

# Set up Python virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install testing dependencies
pip install pytest
```

### 2. Verify Rule Engine & Unit Tests

Run the full test suite (313 unit tests covering table evaluation, schema logic, predicates, and extraction verification):

```bash
pytest
```

---

## Running MedGemma Feature Extraction

The test harness evaluates whether MedGemma can extract clinical findings from patient notes and pass the verbatim quote verification gate.

### Tier Overview

| Tier | Model ID | Recommended Hardware | Purpose |
| :--- | :--- | :--- | :--- |
| **`mock`** | N/A (Deterministic dummy) | Any CPU | Validates the harness and verifier pipeline end-to-end |
| **`local`** | `google/medgemma-1.5-4b-it` | Apple Silicon / CPU / GPU | Fast dev iteration loop |
| **`full`** | `google/medgemma-27b-text-it` | NVIDIA GPU (e.g. RTX 4080 16GB + 32GB RAM, RTX 3090/4090 24GB, A100) | Full reporting tier for high-capacity extraction |

> [!NOTE]
> In Google's MedGemma family, the large text model is **MedGemma 27B** (`google/medgemma-27b-text-it`, ~24B–27B parameter class), while the lightweight iteration model is **MedGemma 4B** (`google/medgemma-1.5-4b-it`).

---

### Cohort & Note Selection

The unit of evaluation is the **(note, outcome) pair**, not the note. `absent` is a
first-class answer, so the eval set needs outcomes that are genuinely present *and*
outcomes that are genuinely not.

| Flag | Default | What it controls |
| :--- | :--- | :--- |
| `--cohort` | `loose` | `loose` keeps any note mentioning sickle cell. `scd_primary` keeps only notes *about* the disease — dropping the carrier/trait state, denials ("denied a family history of SCD"), a relative's diagnosis, and cardiology notes where "SCD" means sudden cardiac death. Pool **978 → 362**. |
| `--outcomes` | `28,48,36,19` | Comma-separated SCOGS outcome ids to grade each note against. |
| `--notes` | `20` | Notes drawn from the pool. Pairs = notes × outcomes. |
| `--stratify` / `--no-stratify` | on | Pick notes that hit every target outcome, then add a random holdout. |
| `--holdout-frac` | `0.25` | Fraction drawn at random rather than by seed regex — this is what makes the seeds' bias *measurable* instead of merely disclosed. |
| `--repeat` | `1` | Run N times and report run-to-run consistency at temperature 0. |
| `--concurrency` | `1` | In-flight backend requests. Needs a batching server (`OLLAMA_NUM_PARALLEL>=N`). |

> [!WARNING]
> **`--concurrency > 1` confounds run-to-run consistency.** Batched reductions are not
> bit-identical, so a token can flip at temperature 0 for reasons that have nothing to do
> with the model. Measure consistency at `--concurrency 1`; raise it only for throughput.
> The harness prints this warning and records `consistency_confounded_by_batching` in the
> result file's provenance.

**Which cohort to use depends on what you are measuring.** `loose` is the CLI default
because the absence audit needs notes where an outcome is genuinely absent. Reporting runs
use `scd_primary` — mention-only notes can *only* ever score `absent`, so leaving them in
dilutes every rate with absences the model never had a chance to avoid, which is a
measurement artifact indistinguishable from a model that extracts nothing.

> [!NOTE]
> `scripts/run_windows_27b.ps1` and `.bat` do not expose `--cohort`, so they run the
> `loose` default. Pass `--cohort scd_primary` via the direct Python command for a run
> whose rates are comparable to the Colab reporting runs.

---

## 🖥️ Running on an NVIDIA Windows PC (MedGemma 27B / RTX 4080 & 3090/4090)

This section details how to run the full reporting tier on Windows with NVIDIA GPUs, specifically tuned for configurations like the **RTX 4080 (16 GB VRAM) + 32 GB DDR5 RAM**, as well as 24 GB cards (RTX 3090 / 4090) and multi-GPU setups.

### Hardware & VRAM / RAM Guide

| Hardware Profile | Model Format / Quantization | VRAM & RAM Behavior | Recommended Backend |
| :--- | :--- | :--- | :--- |
| **RTX 4080 (16 GB VRAM) + 32.0 GB DDR5 RAM** *(Target Setup)* | **4-bit Quantized** (GGUF Q4_K_M or `bitsandbytes` 4-bit) | Model weights consume ~14.2 GB. Ollama/HF loads ~95–100% of layers into the 16 GB GDDR6X VRAM, with the **32 GB DDR5 RAM** providing ample safety margin for KV cache & OS desktop overhead without CUDA OOM. | **Ollama** (fastest, ~15–25 tok/s) or **HF 4-bit** |
| **RTX 3090 / RTX 4090 (24 GB VRAM)** | **4-bit or 8-bit Quantized** | Fits 100% inside 24 GB VRAM with substantial headroom. | **Ollama** or **HF 4-bit / 8-bit** |
| **Multi-GPU / Enterprise ($\ge 56\text{ GB}$ VRAM: 2x 3090/4090, A6000, A100)** | **Full Precision (bfloat16)** | Full 16-bit unquantized model weights (~54 GB). | **HF bfloat16** or **vLLM** |

---

### Option A: Using Ollama for Windows (Recommended for RTX 4080 + 32GB RAM)

Ollama is the easiest and most performant way to run MedGemma 27B on an RTX 4080 with 32 GB DDR5 RAM. Its GGUF runtime optimizes memory mapping and layer placement to deliver high throughput (~15–25 tokens/s) while utilizing system DDR5 RAM for any memory buffer headroom.

1. **Install Ollama for Windows:**
   - Download and run the installer from [ollama.com/download/windows](https://ollama.com/download/windows).
   - Ollama automatically detects NVIDIA CUDA drivers on your RTX 4080.

2. **Pull the MedGemma 27B Model:**
   Open PowerShell and run:
   ```powershell
   ollama pull medgemma-27b-text-it
   ```

3. **Run Extraction:**
   - **Using the PowerShell Runner Script:**
     ```powershell
     .\scripts\run_windows_27b.ps1 -Backend ollama -Notes 20 -Repeat 2 -Out results\full.json
     ```
   - **Using the Batch Runner Script (Command Prompt):**
     ```cmd
     scripts\run_windows_27b.bat --notes 20 --repeat 2 --out results\full.json
     ```
   - **Using Direct Python Command:**
     ```powershell
     python scripts\experiments\medgemma_extraction.py --tier full --backend ollama --notes 20 --repeat 2 --out results\full.json
     ```

---

### Option B: Using PyTorch + Hugging Face with CUDA & 4-bit Quantization

If running directly in Python via Hugging Face Transformers:

1. **Install PyTorch with CUDA 12.1+ support:**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Install Transformers & 4-bit Quantization packages:**
   ```powershell
   pip install transformers accelerate bitsandbytes sentencepiece
   ```

3. **Run with 4-bit Quantization (`bitsandbytes`):**
   The harness automatically enables `device_map="auto"` and `load_in_4bit=True`, which fits onto the RTX 4080 16GB VRAM and uses the 32GB DDR5 RAM as a fallback offload buffer if needed:
   ```powershell
   .\scripts\run_windows_27b.ps1 -Backend hf -Quant 4bit -Notes 20 -Repeat 2 -Out results\full.json
   ```
   Or via direct Python command:
   ```powershell
   python scripts\experiments\medgemma_extraction.py --tier full --backend hf --quant 4bit --notes 20 --repeat 2 --out results\full.json
   ```

---

### Option C: Using vLLM or OpenAI-Compatible Local Server

If serving MedGemma 27B via vLLM, SGLang, or llama.cpp server:

```powershell
python scripts\experiments\medgemma_extraction.py --tier full --backend openai --host http://localhost:8000 --notes 20 --repeat 2 --out results\full.json
```

---

### 💡 Windows & RTX 4080 Optimization Tips

- **Free Up Initial VRAM:** Windows Desktop Window Manager (DWM) and hardware-accelerated web browsers typically consume 0.5–1.5 GB of VRAM. Closing heavy background applications (games, 3D apps, GPU-accelerated browser tabs) before starting ensures maximum free VRAM for model layers.
- **DDR5 Bandwidth Advantage:** Your 32.0 GB DDR5 RAM provides substantial memory bandwidth (typically 4800–6000 MT/s), ensuring that any layer spillover or KV cache operations happen with minimal latency impact.
- **PowerShell Execution Policy:** If running `.ps1` scripts for the first time, allow local scripts by running `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell.

---

## ☁️ Running on Google Colab (NVIDIA A100 GPU)

This is the **reporting path** — the runs whose numbers are quoted come from here.

- **Notebook:** [`notebooks/medgemma_27b_a100_16bit.ipynb`](notebooks/medgemma_27b_a100_16bit.ipynb) — 12 steps, A100 80GB, BF16 via Ollama.

### How this notebook actually gets its weights

> [!IMPORTANT]
> **Nothing is pulled from a model registry.** There is no `ollama pull` for MedGemma 27B
> that works here. The notebook merges a BF16 GGUF, builds an Ollama store from it, and
> caches **both** in `MyDrive/scogs_ollama_models`. Later sessions restore from Drive and
> skip the merge entirely. Colab wipes local scratch every session; only Drive survives.

### The steps

| Step | What it does |
| :--- | :--- |
| 1–2 | Verify the A100, clone the repo at `BRANCH`, install only what is missing. |
| 3 | `pytest -q` — the decision tables, schema, predicates and the §2 verifier. Red here means nothing downstream is worth reading. |
| 4–5 | Model choice and the Drive cache; get the model into the local store. `5b`/`5c` are recovery paths, off by default. |
| **6** | **Preflight gate ⛔** — halts the notebook. This is the cell that catches a daemon that answers but cannot `generate`, and quotes carrying corrupt GGUF byte tokens. |
| 7 | Run the harness: `--cohort scd_primary --stratify --holdout-frac 0.25 --notes 20 --repeat 2`. |
| 8–9 | Results, grounding against **both denominators**, and every accepted finding with its quote. |
| **10–11** | **Generate the two review sheets ⭐** — the hand-check and the absence audit. See below; these produce the numbers nothing automated can. |
| 12 | Download artifacts. |

> [!WARNING]
> **The notebook runs from what is _pushed_, never your working copy.** Step 2 clones from
> GitHub and checks out `BRANCH`. Uncommitted fixes are invisible in Colab — commit and push
> before running, or Step 2's harness assertion will fail.

---

## 🍎 Running on Mac / Apple Silicon (MedGemma 4B)

1. **Install and Start Ollama:**
   ```bash
   ollama serve
   ollama pull medgemma-1.5-4b-it
   ```

2. **Run Extraction (20 notes, 2 repeats):**
   ```bash
   python3 scripts/experiments/medgemma_extraction.py --tier local --notes 20 --repeat 2 --out results/local.json
   ```

3. **Run Mock Mode (No GPU / No Model Needed):**
   ```bash
   python3 scripts/experiments/medgemma_extraction.py --backend mock --notes 2
   ```

---

## 📊 Understanding the Output & Metrics

The extraction harness automatically calculates:

- **Quote-verified %:** Measures whether proposed features carry a verbatim quote directly present in the note (catches hallucinations). **This is a grounding check, not precision.** It asks whether the quoted words are in the note; it cannot ask whether they *support* the value. `fio2_pct = 21` quoting *"the patient needed increasing oxygen by nasal cannula"* verifies at 100%. Precision is a human judgement and comes only from the hand-check sheet's `supports_value` column.
- **Unit guard:** For a numeric feature whose schema declares a convertible unit, the number is read out of its own verified quote, in the unit the note wrote it in, and converted. A note using mg/L throughout otherwise puts `7.0` into an mg/dL creatinine field — a 10× error no type check can see. A value matching neither its quote's number nor that number's conversion is rejected and counted.
- **Value conflicts:** One `(note, outcome)` routinely yields several verified values for one feature — a note carries five creatinines across twelve years and one of them belongs to the transplant donor. Where the schema states an aggregation ("Highest level of care this event actually reached") the values collapse by it; where it does not, the disagreement is reported and the feature is withheld from grading rather than resolved by emission order.
- **Run-to-run consistency:** Measures output determinism at temperature 0 across repeated runs. At `--concurrency 1` with greedy decoding, 100% is the *expected* result — it is a smoke test for a nondeterministic serving stack, not evidence about the model.
- **Invalid-value rate:** Detects values outside declared types or enums.
- **Unparseable replies:** Tracks any malformed JSON generations.
- **SCOGS Decision Table Grading:** Evaluates extracted features through the official SCOGS logic (outcomes: `graded`, `absent`, `refuted`, `grade_set`, or `cannot_grade`). `refuted` is split out of `absent`: the model *did* evidence the outcome and the decision tables overruled the call (a 36.5 °C "fever"). Pooled with `absent` it sends a reviewer to confirm an absence the rule engine, not the model, produced.
- **Provenance & `run_id`:** Every result file carries a `run_id`, and every review sheet generated from it stamps that id on each row. A filled-in sheet that has drifted apart from its results file is worse than no sheet, because the review hours land on the wrong run and nothing says so.
- **Throughput & Profiling:** Wall-clock time, tokens/second, and tokens per note.

All detailed extractions and clinical note text are saved to the JSON file specified by `--out` (e.g. `results/full.json`, or `results/a100_27b_ollama.json` from the Colab notebook).

---

## 📝 Review Sheets — the numbers the harness cannot compute

Everything above is behaviour the harness can check by itself. **Neither precision nor the
false-negative rate is in that list**, and neither can be. Both need a human to read the note.
Steps 10 and 11 of the Colab notebook generate the sheets that collect them.

| Sheet | Column to fill | What it measures |
| :--- | :--- | :--- |
| `results/handcheck.csv` | `supports_value` (y/n) | **Precision.** Quote verification says the span is real; it does not say the span *supports the value*. This column is the only thing that does. |
| `results/absence_audit.csv` | `truly_absent` (y/n) | **False-negative rate.** Of the pairs the model called absent, how many really were? Nothing else substitutes for this number. |
| `results/refuted_audit.csv` | `truly_absent`, `reviewer_note` | Pairs where the model *did* evidence the outcome and the decision tables overruled it. Kept separate from `absent` on purpose — pooled, nobody would look. |
| `results/conflicts.csv` | — | Features withheld from grading because one `(note, outcome)` produced several verified, disagreeing values. |

> [!CAUTION]
> **`results/*.csv` is gitignored, and that is deliberate.** Sheets are regenerated per run.
> A stale sheet sitting beside a newer results file silently describes a *different run*, and
> the review hours land on the wrong one. Every sheet stamps the `run_id` of the results file
> it came from on each row — check it matches before filling anything in. If a filled-in sheet
> ever needs committing, narrow the ignore rule first rather than forcing the add.

---

## 📈 Results Dashboard

A self-contained browser dashboard for reading a run: KPI cards, outcome-status breakdown,
a case explorer with **interactive verbatim quote highlighting** over the note text, the
refuted/audit reviewer sheet, and drag-and-drop loading of any results JSON.

```bash
open dashboard/index.html            # no server needed
python3 dashboard/server.py          # or, with live results/ endpoints, on :8080
```

See [`dashboard/README.md`](dashboard/README.md). Note that the run bundled into
`data_bundle.js` is a **stale demonstration run**, not a current result — load your own
results JSON through the Runs tab.

---

## 📚 Reference Documents

| Document | What it is |
| :--- | :--- |
| [`rules.md`](rules.md) | The SCOGS rubric compiled to prose and decision tables. **This is the authority** — there is no external gold standard; the tables in `scripts/scogs/tables.py` implement this file. |
| [`rules_vs_booklet_discrepancies.md`](rules_vs_booklet_discrepancies.md) | Every place `rules.md` departs from the printed booklet, and why. Also records where `tables.py` deliberately deviates from `rules.md` — a different axis, kept in its own section. |
| [`SCOGS_Booklet.pdf`](SCOGS_Booklet.pdf) | The source booklet the rubric was transcribed from. |
| [`tasks/plan.md`](tasks/plan.md) | Implementation plan. §7 defines the ground-truth review protocol. |
| [`tasks/medgemma_extraction_test.md`](tasks/medgemma_extraction_test.md) | The P11 protocol: what is measured, the pass gates, and the limitations to state with any result. |
| [`tasks/handoff.md`](tasks/handoff.md) | **Read this before running anything.** Current state, what to do next in order, and the traps that have already cost a session. |

---

## Repository Structure

```
├── README.md
├── rules.md                          # The SCOGS rubric as prose + tables (the authority)
├── rules_vs_booklet_discrepancies.md # Where rules.md departs from the booklet, and why
├── SCOGS_Booklet.pdf                 # Source booklet
├── dashboard/                        # Self-contained results dashboard
│   ├── index.html                    #   open directly, no server required
│   ├── server.py                     #   optional zero-dependency server w/ results/ API
│   └── data_bundle.js                #   bundled demo run (stale - see dashboard/README.md)
├── docs/
│   └── windows_gpu_setup.md          # Detailed Windows NVIDIA GPU setup guide
├── notebooks/
│   └── medgemma_27b_a100_16bit.ipynb # Colab A100 reporting run, Steps 1-12
├── scripts/
│   ├── download_data.py              # Automated dataset downloader & cache builder
│   ├── audit/                        # Booklet extraction and rules-vs-booklet comparison
│   │   ├── extract_booklet.py
│   │   ├── compare_grades.py
│   │   └── compare_prose.py
│   ├── experiments/
│   │   └── medgemma_extraction.py    # Main MedGemma extraction test harness
│   ├── run_windows_27b.ps1           # PowerShell runner for Windows
│   ├── run_windows_27b.bat           # Batch runner for Windows
│   └── scogs/                        # Core SCOGS rule engine and tables
│       ├── build_schema.py           # Generates data/scogs_feature_schema.json
│       ├── evaluate.py               # Rule evaluator
│       ├── features.py               # 137 feature definitions
│       ├── predicates.py             # Predicate parsing
│       └── tables.py                 # 53 decision tables
├── tasks/
│   ├── plan.md                       # Implementation plan & roadmap
│   ├── medgemma_extraction_test.md   # P11 test protocol & metrics definitions
│   └── handoff.md                    # Current state, next actions, known traps
├── tests/                            # Pytest suite (313 tests)
├── data/
│   ├── scogs_feature_schema.json     # 137 features across 53 outcomes
│   ├── clincal_notes.csv             # Clinical notes with multi-outcome labels
│   └── clincal_notes_org.csv
├── PMC-Patients/
│   ├── scd_cache.json                # 978 pre-filtered sickle cell patient summaries
│   ├── PMC-Patients-V2.json          # Full raw corpus (downloaded via script, gitignored)
│   └── PMC-Patients.csv              # Full summary table (downloaded via script, gitignored)
└── results/                          # Generated run JSON + review sheets (*.csv gitignored)
```
