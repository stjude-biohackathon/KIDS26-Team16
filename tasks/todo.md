# MedGemma Pipeline Execution & Validation Checklist

This checklist outlines the concise steps to verify the MedGemma extraction pipeline, run the 14 core focus outcomes (including primary SCD base evaluation and negative-control absence auditing), and audit clinical extraction accuracy.

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
  Run default Stage 3 extraction (incorporates precision rules, negation handling, and episode scoping):
  ```bash
  python3 scripts/experiments/medgemma_extraction.py \
    --model medgemma-27b-f16 \
    --cohort scd_primary \
    --notes 20 \
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

## 3. Key Performance Indicators & Quality Gates

When reviewing run summaries (`automated_metrics`), verify against these protocol benchmarks:

| Metric | Target | Warning / Action |
|:-------|:-------|:-----------------|
| **Quote-Verified Grounding** | ≥ 95% | < 85% is concerning; check note truncation or prompt stage. |
| **Run-to-Run Consistency** | 100% | Must be bit-identical at temperature 0 with `--concurrency 1`. |
| **Null Placeholders** | ≤ 5% | If > 10%, model failed to follow omission contract. |
| **Invalid Values** | ≤ 2% | If > 5%, schema or type constraint was violated. |
| **Unparseable Replies** | 0 | Any unparseable reply requires prompt inspection. |
| **Clinical Precision** (`handcheck.csv`) | ≥ 90% | Proportion of grounded proposals marked `supports_value = y`. |
