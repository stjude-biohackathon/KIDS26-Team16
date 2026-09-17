# MedGemma Pipeline Execution & Validation Checklist

This checklist outlines the concise steps to verify the MedGemma extraction pipeline, run the 14 core focus outcomes (including primary SCD base evaluation and negative-control absence auditing), benchmark alternative models (Gemma4-31B and Qwen3.6-35BA3B), and audit clinical extraction accuracy.

---

## 1. Overall Check that the MedGemma Pipeline Works

Before starting long extraction runs, verify dependencies, the deterministic rules engine, the mock harness, and the Ollama model service.

- [ ] **1.1 Check Python environment & dependencies**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **1.2 Verify the deterministic SCOGS rule engine & test suite**
  Run the full test suite (tables, schema contract, predicates, criteria, prompt digests):
  ```bash
  python3 -m pytest -q
  ```
  *Expected:* 481+ tests pass with zero failures.

- [ ] **1.3 Run mock extraction smoke test (no GPU/Ollama required)**
  Validates end-to-end pipeline plumbing, 3-valued grading, quote grounding verification, and summary tallying:
  ```bash
  python3 scripts/experiments/medgemma_extraction.py --backend mock --notes 2
  ```
  *Expected:* Completes cleanly in < 1s; exactly 50% grounding (mock generates 1 true / 1 fake quote by design).

- [ ] **1.4 Check Ollama service & verify MedGemma 27B weights**
  Ensure the Ollama daemon is running (`ollama serve` or macOS app on port 11434). Run the preflight integrity check:
  ```bash
  python3 scripts/experiments/medgemma_extraction.py --model medgemma-27b-f16 --check-model
  ```
  *Expected:* Verifies unquantized F16/BF16 precision, 27B parameter count, SHA-256 weight digest, and synthetic JSON completion.

---

## 2. Extraction Runs on the 14 Focus Outcomes (Base Run & Absence Audit)

The 14 focus outcomes represent the core clinical complications evaluated in the pipeline:
- **10**: Chronic Pain
- **11**: Cognitive Dysfunction
- **12**: Elevated TCD Ultrasonography Velocity
- **15**: Stroke (hemorrhagic or ischemic)
- **17**: Sickle Cell Retinopathy (SCR)
- **21**: Chronic Kidney Disease (CKD)
- **24**: Priapism
- **28**: Acute Sickle Cell Pain Episode
- **29**: Acute Splenic Sequestration
- **39**: Avascular Necrosis of Joints (AVN)
- **40**: Leg Ulcer
- **47**: Depression
- **48**: Acute Chest Syndrome (ACS)
- **49**: Asthma Exacerbation

Evaluating these 14 outcomes requires two complementary runs:
1. **Base Run (`--cohort scd_primary`):** Tests clinical extraction and precision on notes verified to be primary Sickle Cell Disease cases.
2. **Absence Audit (`--cohort loose`):** Evaluates negative controls (notes matching SCD keywords like sickle cell trait, cardiology "sudden cardiac death", or negated family history) to verify the model does not hallucinate false positives when complications are absent, and to audit false negatives.

### Execution Steps:

- [ ] **2.1 Base Run: 14 Focus Outcomes on Primary SCD Cohort (`--cohort scd_primary`)**
  Run default Stage 3 extraction (incorporates precision rules, negation handling, and episode scoping) on 37 notes (28 stratified seeded cases—2 for each of the 14 focus outcomes—plus 9 holdout cases):
  ```bash
  python3 scripts/experiments/medgemma_extraction.py \
    --model medgemma-27b-f16 \
    --cohort scd_primary \
    --notes 37 \
    --prompt-stage 3 \
    --outcomes 14 \
    --stratify \
    --holdout-frac 0.25 \
    --repeat 2 \
    --concurrency 1 \
    --out results/run_14_focus_stage3.json
  ```
  *(Optional: To run alternative prompt stages such as the Stage 2b baseline, set `--prompt-stage 2b` and change `--out results/run_14_focus_stage2b.json`)*

- [ ] **2.2 Absence Audit: 14 Focus Outcomes on Negative-Control Notes (`--cohort loose`)**
  The `loose` cohort keeps notes that mention SCD broadly without meeting primary focus complications (e.g. carrier trait or cardiology SCD), providing genuine negative controls to check for false positives and measure false-negative rates:
  ```bash
  python3 scripts/experiments/medgemma_extraction.py \
    --model medgemma-27b-f16 \
    --cohort loose \
    --notes 20 \
    --prompt-stage 3 \
    --outcomes 14 \
    --stratify \
    --holdout-frac 0.25 \
    --repeat 2 \
    --concurrency 1 \
    --out results/run_absence_loose_stage3.json
  ```

- [ ] **2.3 Export review worksheets for both runs**
  Generate human audit CSVs directly from the run JSON files:
  ```bash
  # Export worksheets for the base primary SCD run:
  python3 scripts/experiments/review_results.py results/run_14_focus_stage3.json --output-dir results/review_14_focus_stage3

  # Export worksheets for the negative-control absence audit:
  python3 scripts/experiments/review_results.py results/run_absence_loose_stage3.json --output-dir results/review_absence_loose_stage3
  ```
  Generated sheets in each output directory:
  - `handcheck.csv`: Grounded findings sample (up to 100 rows).
  - `absence_audit.csv`: Absent pairs sample (up to 50 rows).
  - `conflicts.csv`: Any conflicting grounded feature values.
  - `refuted_audit.csv`: Cases where model claimed presence but rules found no matching grade.

---

## 3. Alternative Model Benchmarking (Gemma4-31B & Qwen3.6-35BA3B)

Evaluate alternative models (**Gemma4-31B** and **Qwen3.6-35BA3B**) against MedGemma 27B to benchmark cross-architecture performance, quote-grounding fidelity, and reasoning consistency on clinical extraction across the 14 focus outcomes.

> [!NOTE]
> The default extraction preflight in `scripts/experiments/ollama_backend.py` checks for `gemma3` architecture and F16/BF16 27B weights. When testing alternative models like `Gemma4-31B` or `Qwen3.6-35BA3B` via Ollama, ensure backend model validation accommodates their respective architecture and parameter specifications.

### 3A. Gemma4-31B Evaluation

- [ ] **3.1 Setup / serve Gemma4-31B in Ollama**
  Verify the model tag (e.g. `gemma4-31b` or `gemma4:31b`) is loaded in Ollama:
  ```bash
  ollama list
  ```

- [ ] **3.2 Run 14 Focus Outcomes on Primary SCD Cohort (`Gemma4-31B`)**
  Run extraction on the 14 core focus outcomes with default Stage 3 prompt formatting:
  ```bash
  python3 scripts/experiments/medgemma_extraction.py \
    --model gemma4-31b \
    --cohort scd_primary \
    --notes 20 \
    --prompt-stage 3 \
    --outcomes 14 \
    --stratify \
    --holdout-frac 0.25 \
    --repeat 2 \
    --concurrency 1 \
    --out results/run_gemma4_31b_14_focus_stage3.json
  ```

- [ ] **3.3 Absence Audit on Negative-Control Notes (`Gemma4-31B`)**
  Evaluate the loose cohort to check if Gemma4-31B hallucinates features on negative controls and measure false-negative rates:
  ```bash
  python3 scripts/experiments/medgemma_extraction.py \
    --model gemma4-31b \
    --cohort loose \
    --notes 20 \
    --prompt-stage 3 \
    --outcomes 14 \
    --stratify \
    --holdout-frac 0.25 \
    --repeat 2 \
    --concurrency 1 \
    --out results/run_gemma4_31b_absence_loose_stage3.json
  ```

- [ ] **3.4 Export Review Worksheets for Gemma4-31B**
  ```bash
  python3 scripts/experiments/review_results.py results/run_gemma4_31b_14_focus_stage3.json --output-dir results/review_gemma4_31b_14_focus_stage3 --handcheck-limit 100 --absence-limit 50
  python3 scripts/experiments/review_results.py results/run_gemma4_31b_absence_loose_stage3.json --output-dir results/review_gemma4_31b_absence_loose_stage3 --handcheck-limit 100 --absence-limit 50
  ```

### 3B. Qwen3.6-35BA3B Evaluation

- [ ] **3.5 Setup / serve Qwen3.6-35BA3B in Ollama**
  Verify the model tag (e.g. `qwen3.6-35ba3b` or `qwen3.6:35b-a3b`) is loaded in Ollama:
  ```bash
  ollama list
  ```

- [ ] **3.6 Run 14 Focus Outcomes on Primary SCD Cohort (`Qwen3.6-35BA3B`)**
  Run extraction on the 14 core focus outcomes with default Stage 3 prompt formatting:
  ```bash
  python3 scripts/experiments/medgemma_extraction.py \
    --model qwen3.6-35ba3b \
    --cohort scd_primary \
    --notes 20 \
    --prompt-stage 3 \
    --outcomes 14 \
    --stratify \
    --holdout-frac 0.25 \
    --repeat 2 \
    --concurrency 1 \
    --out results/run_qwen36_35ba3b_14_focus_stage3.json
  ```

- [ ] **3.7 Absence Audit on Negative-Control Notes (`Qwen3.6-35BA3B`)**
  Evaluate the loose cohort to check if Qwen3.6-35BA3B hallucinates features on negative controls and measure false-negative rates:
  ```bash
  python3 scripts/experiments/medgemma_extraction.py \
    --model qwen3.6-35ba3b \
    --cohort loose \
    --notes 20 \
    --prompt-stage 3 \
    --outcomes 14 \
    --stratify \
    --holdout-frac 0.25 \
    --repeat 2 \
    --concurrency 1 \
    --out results/run_qwen36_35ba3b_absence_loose_stage3.json
  ```

- [ ] **3.8 Export Review Worksheets for Qwen3.6-35BA3B**
  ```bash
  python3 scripts/experiments/review_results.py results/run_qwen36_35ba3b_14_focus_stage3.json --output-dir results/review_qwen36_35ba3b_14_focus_stage3 --handcheck-limit 100 --absence-limit 50
  python3 scripts/experiments/review_results.py results/run_qwen36_35ba3b_absence_loose_stage3.json --output-dir results/review_qwen36_35ba3b_absence_loose_stage3 --handcheck-limit 100 --absence-limit 50
  ```

### 3C. Cross-Model Head-to-Head Comparison

- [ ] **3.9 Cross-Model Synthesis & Benchmark Comparison**
  Compare key metrics across all three model architectures (MedGemma 27B vs. Gemma4-31B vs. Qwen3.6-35BA3B):
  - Quote-verified grounding rate (`accepted / proposed`)
  - Absence specificity & false negative rate (`absence_audit.csv`)
  - Rule refutations (`refuted_audit.csv`) and internal value conflicts (`conflicts.csv`)
  - Structured output compliance (JSON parseability and omission contract adherence)

---

## 4. Key Performance Indicators & Quality Gates

When reviewing run summaries (`automated_metrics`), verify against these protocol benchmarks:

| Metric | Target | Warning / Action |
|:-------|:-------|:-----------------|
| **Quote-Verified Grounding** | ≥ 95% | < 85% is concerning; check note truncation or prompt stage. |
| **Run-to-Run Consistency** | 100% | Must be bit-identical at temperature 0 with `--concurrency 1`. |
| **Null Placeholders** | ≤ 5% | If > 10%, model failed to follow omission contract. |
| **Invalid Values** | ≤ 2% | If > 5%, schema or type constraint was violated. |
| **Unparseable Replies** | 0 | Any unparseable reply requires prompt inspection. |
| **Clinical Precision** (`handcheck.csv`) | ≥ 90% | Proportion of grounded proposals marked `supports_value = y`. |
