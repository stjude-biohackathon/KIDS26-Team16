# grade_csv.py — read notes from a CSV, grade each with Qwen, save answers to JSON
# Run with:  python grade_csv.py

from openai import OpenAI
import json, csv

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ===== SETTINGS (change these if your names are different) =====
CSV_FILE   = "notes.csv"        # your notes file
NOTE_COL   = "note_text"        # the column that holds the note text
OUTCOME_COL = "outcome"         # the column that says which outcome to grade
MODEL      = "qwen3:4b"         # your Qwen model (check with: ollama list)
OUT_FILE   = "grades.json"      # where the answers get saved
# ==============================================================

with open("rules.json", encoding="utf-8") as f:
    RULES = json.load(f)


def build_prompt(note, rule):
    grade_lines = "\n".join(f"- Grade {g}: {d}" for g, d in rule["grades"].items())
    return f"""You are a clinical grading assistant using the official SCOGS rubric.
Grade the note for ONE outcome only: {rule['outcome']}.

DEFINITION: {rule.get('definition','')}
DIAGNOSTIC CRITERIA: {rule.get('diagnostic_criteria','')}
VALIDATION: {rule.get('validation','')}
- If the outcome is NOT present, GRADE is -1.
- If present but not gradable, GRADE is 0.

GRADE SCALE (use ONLY these):
{grade_lines}

READING RULES:
- NEGATION: "no", "denies", "without", "ruled out", "not" mean ABSENT.
- TEMPERATURE UNITS: normal ~37C (98.6F). Fever ~38C (100.4F)+.
  "99" cannot be 99C in a person, so read it as Fahrenheit (99F = normal).

Clinical Note:
{note}

Return ONLY valid JSON, nothing else, in this exact shape:
{{"grade": <number -1 to 5>, "evidence": "<exact words from the note>", "reasoning": "<one sentence>"}}
"""


def parse_json(raw):
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"grade": "?", "evidence": "", "reasoning": raw}


# read all notes from the CSV
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    notes = list(csv.DictReader(f))

results = []
for row in notes:
    outcome = row[OUTCOME_COL]
    rule = RULES.get(outcome)
    if not rule:
        print(f"skip (no rule for outcome): {outcome}")
        continue

    prompt = build_prompt(row[NOTE_COL], rule)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = parse_json(resp.choices[0].message.content.strip())
    answer["note_id"] = row.get("note_id", "")
    answer["outcome"] = outcome
    results.append(answer)
    print(f"{row.get('note_id','')}  ->  grade {answer.get('grade')}")

# save all answers to one JSON file
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} grades to {OUT_FILE}")