# summarizer.py
# Reads SCD_note.csv, summarizes case_text with Qwen3 30B,
# and saves the results to SCD_summaries.csv
#
# Expected columns:
#   case_id
#   outcomes_full
#   case_text
#
# Run:
#   python summarizer.py

from openai import OpenAI
import csv

# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "SCD_note.csv"
OUTPUT_CSV = "SCD_summaries.csv"

CASE_ID_COLUMN = "case_id"
OUTCOME_COLUMN = "outcomes_full"
TEXT_COLUMN = "case_text"

MODEL = "qwen3:30b"

# ============================================================
# OLLAMA CONNECTION
# ============================================================

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ============================================================
# READ SCD_note.csv
# ============================================================

with open(
    INPUT_CSV,
    "r",
    newline="",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)
    rows = list(reader)


if len(rows) == 0:
    raise ValueError(f"{INPUT_CSV} is empty.")


# ============================================================
# CHECK COLUMNS
# ============================================================

columns = list(rows[0].keys())

print("Columns found:", columns)

required_columns = [
    CASE_ID_COLUMN,
    OUTCOME_COLUMN,
    TEXT_COLUMN
]

missing_columns = [
    col
    for col in required_columns
    if col not in columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Columns found: {columns}"
    )


print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

# ============================================================
# SUMMARIZE EACH CASE
# ============================================================

results = []

for i, row in enumerate(rows, start=1):

    # --------------------------------------------------------
    # READ CASE INFORMATION
    # --------------------------------------------------------

    case_id = row.get(CASE_ID_COLUMN, f"CASE{i:04d}")

    true_outcome = row.get(
        OUTCOME_COLUMN,
        ""
    )

    case_text = row.get(
        TEXT_COLUMN,
        ""
    )

    print()
    print(
        f"[{i}/{len(rows)}] "
        f"Case: {case_id} | "
        f"True outcome: {true_outcome}"
    )

    # --------------------------------------------------------
    # SKIP EMPTY CASE TEXT
    # --------------------------------------------------------

    if not case_text.strip():

        print("  Skipping empty case_text")

        results.append({
            "case_id": case_id,
            "true_outcome": true_outcome,
            "original_case_text": case_text,
            "summary": ""
        })

        continue

    # ========================================================
    # SUMMARIZATION PROMPT
    # ========================================================

    prompt = f"""
You are a clinical note summarization assistant specializing in
sickle cell disease.

Your task is to summarize the clinical text below.

The clinical outcome associated with this case is provided only
to help you focus on clinically relevant information.

IMPORTANT RULES:

- Do NOT assign a SCOGS grade.
- Do NOT predict a SCOGS grade.
- Do NOT invent information.
- Do NOT add facts that are not explicitly stated.
- Do NOT infer treatments, symptoms, or complications that are
  not explicitly documented.

Preserve clinically important information including:

- diagnosis
- presenting symptoms
- duration of symptoms
- physical examination findings
- laboratory values
- imaging findings
- oxygen requirement
- respiratory support
- medications
- analgesics
- antibiotics
- transfusions
- simple transfusion versus exchange transfusion
- procedures
- hospitalization
- ICU admission
- ventilation
- vasopressors
- neurologic findings
- renal findings
- disposition
- important negative findings
- exact numerical values and units when clinically relevant

Keep the summary concise but clinically complete.

Focus especially on information that may later be useful for
determining the severity of the sickle cell disease outcome.

TRUE OUTCOME:
{true_outcome}

CLINICAL CASE:
{case_text}

Return only the clinical summary.
"""

    # ========================================================
    # RUN QWEN3 30B
    # ========================================================

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        summary = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        print("  Summary generated.")

    except Exception as e:

        print(f"  ERROR: {e}")

        summary = ""

    # ========================================================
    # STORE RESULT
    # ========================================================

    results.append({
        "case_id": case_id,
        "true_outcome": true_outcome,
        "original_case_text": case_text,
        "summary": summary
    })

# ============================================================
# SAVE OUTPUT
# ============================================================

fieldnames = [
    "case_id",
    "true_outcome",
    "original_case_text",
    "summary"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("Done.")
print(
    f"Saved {len(results)} summaries "
    f"to {OUTPUT_CSV}"
)
print("=" * 60)