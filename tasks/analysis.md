# Comparative Analysis: Live Evaluation vs. Benchmark Run (Model Size & Degradation)

## 1. Executive Summary

This document analyzes the performance discrepancy between the live dashboard evaluation run ([`results/live/live_run_20260917T233844Z.json`](file:///Users/edward/Desktop/KIDS26-Team16/results/live/live_run_20260917T233844Z.json)) and the benchmark run ([`results/scd_summaries_gemma4/original.json`](file:///Users/edward/Desktop/KIDS26-Team16/results/scd_summaries_gemma4/original.json)).

Both runs evaluated the **exact same clinical note** (`PMC10253251_01`): a 58-year-old male with sickle cell disease (SCD) admitted with acute chest pain, diagnosed with acute chest syndrome (ACS) requiring endotracheal intubation, who subsequently developed atrial fibrillation with RVR and was treated with amiodarone.

Key findings:
1. **Context Window**: The model was confirmed to run with a context window of **16,384 tokens** (`num_ctx: 16384`). Context truncation was not the cause of failure.
2. **Model Difference**: The benchmark used **`gemma4:12b`** (11.9B parameters, `Q4_K_M`), whereas the live run used **`medgemma-1.5-4b-it`** (3.9B parameters, `Q4_0`).
3. **Core Failure Modes of the 4B Model**:
   - **Instruction & Schema Leakage**: Copied prompt schema descriptions directly as evidence quotes instead of quoting patient text, causing 100% rejection by quote grounding.
   - **Degenerate Repetition Loops**: Trapped in token repetition loops on Outcomes 12 and 28, truncating completion tokens and causing malformed JSON.
   - **Severe False-Positive Hallucination**: Falsely marked 5 completely absent outcomes as `present: true`.
   - **Grading Result**: 0 outcomes successfully graded (8 absent, 6 cannot grade) vs. Grade 4 for both VOC and ACS in `gemma4:12b`.

---

## 2. Context Window Verification (`num_ctx: 16384`)

The live run model was confirmed to use a context window of 16,384 tokens:
- **Provenance Record**: In [`live_run_20260917T233844Z.json`](file:///Users/edward/Desktop/KIDS26-Team16/results/live/live_run_20260917T233844Z.json#L8-L9):
  ```json
  "backend": "ollama",
  "num_ctx": 16384
  ```
- **Harness Default**: In [`dashboard/evaluation.py`](file:///Users/edward/Desktop/KIDS26-Team16/dashboard/evaluation.py#L246):
  ```python
  def extract_with_harness(..., num_ctx: int = 16384)
  ```
- **Ollama Backend Payload**: In [`scripts/experiments/ollama_backend.py`](file:///Users/edward/Desktop/KIDS26-Team16/scripts/experiments/ollama_backend.py#L112):
  ```python
  "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, ...}
  ```
- **Ollama Modelfile**: `medgemma-1.5-4b-it` explicitly defines:
  ```dockerfile
  PARAMETER num_ctx 16384
  ```

---

## 3. Side-by-Side Comparison on Case `PMC10253251_01`

| Dimension | Benchmark Run (`results/scd_summaries_gemma4`) | Live Run (`results/live`) |
| :--- | :--- | :--- |
| **Model** | `gemma4:12b` (`google/medgemma-27b-text-it`) | `medgemma-1.5-4b-it` |
| **Parameter Count** | **11.9 Billion** | **3.9 Billion** (~3x smaller) |
| **Quantization** | `Q4_K_M` (Medium quality block quant) | `Q4_0` (Basic 4-bit quant) |
| **Capabilities** | Completion, vision, audio, tools, thinking | Completion only |
| **Grading Summary** | **Grade 4** for VOC (28), **Grade 4** for ACS (48) | **0 graded** (8 absent, 6 cannot_grade) |
| **Quote Grounding** | High fidelity; quotes matched note text | **100% rejected** (quoted prompt instructions) |
| **Degenerate Loops** | None observed | Infinite repetition on Outcomes 12 and 28 |
| **Negative Discrimination** | Correctly classified all 10 absent conditions as `absent` | Hallucinated presence for 5 absent outcomes |

### Detailed Outcome Comparison:

| Outcome ID | Outcome Name | Benchmark (`gemma4:12b`) | Live Run (`medgemma-1.5-4b-it`) |
| :---: | :--- | :--- | :--- |
| **10** | Chronic Sickle Pain | `absent` (present: false) | `absent` (present: false, but hallucinated 0 values) |
| **11** | Cognitive Dysfunction | `absent` (present: false) | `cannot_grade` (hallucinated present: true, quoted entire note) |
| **12** | Elevated TCD Velocity | `absent` (present: false) | `absent` (infinite loop of schema prompt string) |
| **15** | Stroke | `absent` (present: false) | `absent` (present: false, quoted prompt guidelines) |
| **17** | Sickle Cell Retinopathy | `absent` (present: false) | `cannot_grade` (hallucinated present: true) |
| **21** | Chronic Kidney Disease | `cannot_grade` (present: true, missing eGFR/labs) | `absent` (failed to extract CKD history) |
| **24** | Priapism | `absent` (present: false) | `absent` (present: false) |
| **28** | Acute Sickle Cell Pain | **GRADED: Grade 4** | `absent` (infinite loop, JSON truncated, present: null) |
| **29** | Splenic Sequestration | `absent` (present: false) | `cannot_grade` (hallucinated present: true) |
| **39** | Avascular Necrosis (AVN) | `cannot_grade` (present: true, missing ADL/stage) | `absent` (falsely claimed avn_on_imaging: false) |
| **40** | Leg Ulcer | `absent` (present: false) | `cannot_grade` (hallucinated present: true) |
| **47** | Depression | `absent` (present: false) | `cannot_grade` (hallucinated present: true) |
| **48** | Acute Chest Syndrome | **GRADED: Grade 4** | `cannot_grade` (all quotes rejected from prompt leakage) |
| **49** | Asthma | `absent` (present: false) | `absent` (present: false) |

---

## 4. Root Causes for 4B Model Failure

### 4.1. Instruction Leakage & Prompt Copying as Evidence Quotes
In Stage 3 prompt extraction, the system prompt defines each feature along with a description/unit schema. The 3.9B model struggled with context separation, mistaking schema instructions for text from the patient note.

**Example from Outcome 48 (ACS) in `live_run_20260917T233844Z.json`**:
```json
{
  "feature": "resp_support",
  "value": "room_air",
  "quote": "maximum respiratory support given during the event."
},
{
  "feature": "transfusion_type",
  "value": "none",
  "quote": "Most intensive erythrocyte transfusion given for this event."
},
{
  "feature": "fio2_pct",
  "value": 21,
  "quote": "Highest documented FiO2 as a percentage."
}
```
Because the KIDS26 grounding verification checks whether `quote` is an exact substring of the clinical note, **every one of these extracted features was rejected**. The grading rule engine was left with zero accepted findings (`accepted_findings: []`), causing `cannot_grade`.

In contrast, **`gemma4:12b`** extracted true patient note substrings:
```json
{"feature": "resp_support", "value": "invasive_ventilation", "quote": "He was intubated"},
{"feature": "transfusion_type", "value": "simple", "quote": "transfused with three units of pRBCs"},
{"feature": "death_attributed", "value": false, "quote": "the patient was discharged two days later"}
```
All were accepted, leading to an accurate **Grade 4** determination.

### 4.2. Degenerate Repetition Loops
Small models (especially when quantized with basic Q4_0) have lower attention diversity and tend to get trapped in attractor states during constrained decoding:

- **Outcome 12 (TCD)**: Generated over 50 identical copies of:
  ```json
  {"feature": "tcd_velocity", "value": 0, "quote": "Time-averaged mean velocity on non-imaging transcranial Doppler."}
  ```
- **Outcome 28 (Pain Episode)**: Repeated:
  ```json
  {"feature": "pain_co_complication", "value": true, "quote": "His pain remained uncontrolled, requiring morphine via patient-controlled analgesia."}
  ```
  dozens of times until hitting the completion token limit. The JSON generation was cut off before writing `"present"`, leaving `"present": null`. The evaluation pipeline interpreted `null` as absent, causing the model to miss the patient's primary diagnosis entirely.

### 4.3. Hallucinated Disease Presence
The 3.9B model exhibited almost zero discriminative capability for negative cases. When asked to evaluate Outcome 11 (Cognitive Dysfunction), Outcome 17 (Retinopathy), Outcome 29 (Splenic Sequestration), Outcome 40 (Leg Ulcers), and Outcome 47 (Depression), it answered `present: true` for all of them despite none of these conditions being present in the clinical text.

---

## 5. Is Model Size That Big of a Difference?

**Yes.** Structured clinical extraction under strict schemas is one of the most demanding tasks for generative models:

1. **Parameter Capacity (~4B vs ~12B)**:
   - A ~12B model has approximately 3 times the parameter count and deeper attention layers, allowing it to maintain clear boundaries between system instructions, JSON schemas, few-shot examples, and source clinical text.
   - At ~4B parameters, models frequently suffer from "attention bleeding," confusing metadata/descriptions in the prompt with candidate extraction tokens.
2. **Quantization Impact (Q4_0 vs Q4_K_M)**:
   - `Q4_0` uses scalar 4-bit quantization with uniform scaling per block, significantly degrading attention head sensitivity in small models.
   - `Q4_K_M` uses k-quantization with higher precision for critical attention and feed-forward projection layers.
3. **Negative Reasoning & Grounded Quoting**:
   - Small models struggle to decide that an outcome is absent; they tend to hallucinate presence or populate fields with placeholder values (`0`, `"none"`).
   - Higher-tier models (`gemma4:12b`) possess the reasoning depth needed to state `"no evidence in note"` and return `findings: []`, preserving precision.

---

## 6. Conclusions & Recommendations

1. **Default Live Model Selection**:
   - Use `gemma4:12b` (or larger) as the default model for live dashboard evaluations.
   - The 4B model (`medgemma-1.5-4b-it`) is insufficiently capable for reliable zero-shot structured extraction with quote-grounding constraints.
2. **Inference Guardrails**:
   - If smaller models must be supported for low-resource environments:
     - Apply a repetition penalty (e.g. `repeat_penalty: 1.15 - 1.2`) to mitigate degenerate loops.
     - Constrain output token budget and schema generation with grammar-based decoding.
