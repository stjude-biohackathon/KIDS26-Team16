# ============================================================
#  SCOGS Clinical Note Grader  —  MedGemma edition
#  Grades notes with MedGemma 4B, running locally on your laptop via Ollama.
#  Run with:  streamlit run app.py
# ============================================================

import streamlit as st
from openai import OpenAI          # used to talk to local MedGemma (Ollama)
import json

# ---- CONNECT TO MEDGEMMA (local, via Ollama) ----
# MedGemma 4B runs on YOUR laptop through Ollama.
# Ollama speaks the same language as OpenAI, so we point OpenAI at it.
# No real key needed — "ollama" is just a placeholder Ollama ignores.
medgemma_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ---- LOAD THE OFFICIAL SCOGS RULES ----
# rules.json holds the real booklet criteria for each outcome.
@st.cache_data
def load_rules():
    with open("rules.json", encoding="utf-8") as f:
        return json.load(f)

RULES = load_rules()
OUTCOMES = [k for k in RULES if k != "_source"]


# ---- BUILD THE PROMPT ----
# Works for ANY outcome — it reads the criteria out of rules.json.
def build_prompt(note, rule):
    grade_lines = "\n".join(f"- Grade {g}: {d}" for g, d in rule["grades"].items())
    return f"""You are a clinical grading assistant using the official SCOGS
(Sickle Cell Outcome Grading System) rubric.

Grade the clinical note below for ONE outcome only: {rule['outcome']}.

DEFINITION: {rule.get('definition', '')}
DIAGNOSTIC CRITERIA: {rule.get('diagnostic_criteria', '')}
VALIDATION (read first): {rule.get('validation', '')}
- If the outcome is NOT present in the note, GRADE is -1.
- If it is present but does not meet gradable criteria, GRADE is 0.

GRADE SCALE (use ONLY these; some outcomes skip a number on purpose):
{grade_lines}
GRADING HINTS (common mistakes to avoid — do NOT under-grade):
- "Admitted" / "admission" / "inpatient" / "hospitalized" = Grade 3, NOT a facility visit (Grade 2).
- High-flow oxygen, BiPAP, FiO2 >=50%, or exchange transfusion = Grade 3.
- A facility VISIT or ED visit with NO admission = Grade 2.
- If admission OR high-level respiratory support is present, the grade is AT LEAST 3.
READING RULES:
- NEGATION: "no", "denies", "without", "ruled out", "not" mean a finding is ABSENT.
- TEMPERATURE UNITS: normal is about 37C (98.6F). A fever is about 38C (100.4F)
  or higher. If only a number is given (like "99"), 99C is impossible for a
  person, so read it as Fahrenheit (99F = normal).

Clinical Note:
{note}

Respond in this EXACT format:
GRADE: [number from -1 to 5]
EVIDENCE: [the exact words from the note that decided the grade]
REASONING: [one short sentence explaining why]
"""


# ---- CALL MEDGEMMA ----
def grade_with_medgemma(note, rule):
    prompt = build_prompt(note, rule)
    response = medgemma_client.chat.completions.create(
        model="medgemma:4b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,          # 0 = consistent, best for grading
    )
    return response.choices[0].message.content


# ---- READ THE MODEL'S ANSWER ----
# Splits the AI text into 3 parts: grade, evidence, reasoning.
def parse_result(result):
    grade, evidence, reasoning = "", "", ""
    for line in result.split("\n"):
        if line.startswith("GRADE:"):
            grade = line.replace("GRADE:", "").strip()
        elif line.startswith("EVIDENCE:"):
            evidence = line.replace("EVIDENCE:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
    return grade, evidence, reasoning


# ---- COLORED GRADE BADGE ----
# Green for 1-2, yellow for 3, red for 4-5, blue for anything else.
def show_grade_badge(grade):
    if grade in ["1", "2"]:
        st.success(f"✅ SCOGS Grade: {grade}")
    elif grade == "3":
        st.warning(f"⚠️ SCOGS Grade: {grade}")
    elif grade in ["4", "5"]:
        st.error(f"🚨 SCOGS Grade: {grade}")
    else:
        st.info(f"ℹ️ SCOGS Grade: {grade} (not gradable / outcome not present)")


# ============================================================
#  MAIN APP  —  this is what shows on the screen
# ============================================================
st.set_page_config(page_title="SCOGS Grader", page_icon="🩸")

# ---- TITLE ----
st.title("SCOGS Clinical Note Grader")
st.subheader("Sickle Cell Outcome Grading System (1–5) — powered by MedGemma 4B")

st.markdown("Pick the outcome, paste a clinical note, and MedGemma grades it "
            "using the official SCOGS booklet criteria — running locally on this laptop.")

# ---- OUTCOME PICKER ----
outcome = st.selectbox("Outcome to grade:", OUTCOMES)
rule = RULES[outcome]


# ---- NOTE INPUT BOX ----
note = st.text_area(
    "Paste clinical note here:",
    height=200,
    placeholder="Example: 12-year-old with HbSS, fever 39C, new right lower lobe "
                "infiltrate on chest X-ray, started on high-flow oxygen and exchange transfusion.",
)

# ---- GRADE BUTTON ----
if st.button("Grade Note"):
    if note.strip() == "":
        st.warning("Please paste a clinical note first.")
    else:
        st.markdown("---")
        st.markdown("## 📊 Grading Result — MedGemma 4B")
        with st.spinner("MedGemma grading on your laptop... (first run can take ~20s)"):
            try:
                grade, evidence, reasoning = parse_result(grade_with_medgemma(note, rule))
                show_grade_badge(grade)
                st.markdown("### 🔍 Evidence from the note")
                st.write(evidence if evidence else "—")
                st.markdown("### 💡 Reasoning")
                st.info(reasoning if reasoning else "—")
            except Exception:
                st.error("MedGemma not reachable. Make sure Ollama is running and "
                         "you've run:  ollama pull medgemma:4b")

# ---- FOOTER ----
st.markdown("---")
st.caption("Grades follow the official St. Jude Global SCOGS booklet. "
           "MedGemma 4B runs locally via Ollama. "
           "Validated on clean practice notes; next step is real clinical notes.")