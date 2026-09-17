# SCOGS-Scribe local demo update

This update keeps the 53-outcome SCOGS JSON, reconciles Acute Chest Syndrome (ACS) to booklet pages 119-120, and changes the live ACS workflow to:

`clinical note -> MedGemma feature extraction -> exact-quote verification -> deterministic SCOGS rule`

The LLM no longer assigns the ACS grade directly.

## Files

- `scogs_grader.py` - reusable Ollama + SCOGS grading engine
- `demo_acs.py` - single-note ACS demo using the clinical note from the project discussion
- `grade_synthetic.py` - batch grader for `synthetic_patients.csv`
- `scogs_outcomes.json` - original 53 outcomes preserved; ACS corrected/enriched for machine grading

## Copy into the Edward worktree

From PowerShell:

```powershell
cd "C:\Users\jwardell\OneDrive - St. Jude Children's Research Hospital\Biohackathon\KIDS26-Team16-edward-run"
```

Copy the four files into this folder (or into a dedicated `scogs_demo` folder and run from there).

No `openai` Python package is required. The code uses Ollama's native HTTP API.

## Check Ollama

```powershell
ollama list
ollama run medgemma:4B "Return exactly the word OK"
```

## Run the live ACS demo

```powershell
.\.venv\Scripts\python.exe .\demo_acs.py --model medgemma:4B
```

You can also try the newer small model already installed on your computer:

```powershell
.\.venv\Scripts\python.exe .\demo_acs.py --model medgemma1.5:latest
```

To grade a note stored in a text file:

```powershell
.\.venv\Scripts\python.exe .\demo_acs.py --model medgemma:4B --note-file .\my_note.txt
```

## Run the batch grader

Expected columns by default:

- `case_id`
- `true_outcome`
- `summary`

Run:

```powershell
.\.venv\Scripts\python.exe .\grade_synthetic.py --model medgemma:4B
```

If you want to grade `original_case_text` instead of the summary:

```powershell
.\.venv\Scripts\python.exe .\grade_synthetic.py --model medgemma:4B --text-column original_case_text
```

## Important ACS behavior

The official ACS table requires:

- Grade 3: FiO2 >=50%, BiPAP, high-flow O2, or exchange transfusion.
- Grade 4: a Grade 3 trigger **and** critical cardio-respiratory support such as intubation/invasive ventilation/vasopressors.

Therefore the grader does **not** convert `15 L/min` into an FiO2 percentage and does not automatically call an unspecified oxygen device "high-flow." If a note documents intubation but does not establish a Grade 3 trigger, the strict grader returns `cannot_grade` rather than inventing the missing criterion.

For a clean Grade 4 demo, use a note that explicitly documents something like `high-flow oxygen`, `BiPAP`, `FiO2 >=50%`, or `exchange transfusion` in addition to the critical support.


## v3 hybrid ACS fix
The ACS demo now uses deterministic text extraction as a safety net for explicit facts such as intubation, oxygen flow, and RBC transfusion. Verified MedGemma output supplements missing fields but cannot overwrite those anchored facts. The grader never converts L/min to FiO2. For ACS only, explicit intubation/invasive mechanical ventilation is operationalized as respiratory support at least as intensive as the Grade-3 respiratory modalities; this implementation choice should be clinically reviewed before production use.
