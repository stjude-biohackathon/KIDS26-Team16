from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from scogs_grader import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    grade_note_outcome,
    load_rules,
    ollama_json,
)

BASE_DIR = Path(__file__).resolve().parent
RULES_PATH = BASE_DIR / "scogs_outcomes.json"

st.set_page_config(page_title="SCOGS-Scribe", page_icon="🩸", layout="wide")
st.title("🩸 SCOGS-Scribe Clinical Note Grader")
st.caption("Clinical note → outcome identification → verified extraction/evidence → SCOGS grading")


@st.cache_data
def get_rules(path: str):
    return load_rules(path)


def outcome_catalog(raw_rules: dict) -> list[dict]:
    rows = []
    for item in raw_rules.get("outcomes", []):
        if not isinstance(item, dict) or not item.get("outcome"):
            continue
        rows.append({
            "outcome": item.get("outcome"),
            "definition": item.get("definition", ""),
            "diagnostic_criteria": item.get("diagnostic_criteria", ""),
        })
    return rows


def identify_outcomes(note: str, raw_rules: dict, model: str, host: str, timeout: int):
    catalog = outcome_catalog(raw_rules)
    prompt = f"""
CLINICAL NOTE
-------------
{note}

TASK
----
Identify the SCOGS outcomes that are explicitly supported by this clinical note.
This is outcome identification only; do NOT assign grades.

Rules:
- Choose ONLY exact outcome names from the catalog below.
- Return an outcome only when the note contains evidence that it is present.
- Respect negation: 'no', 'denies', 'without', 'ruled out', and similar wording mean absent.
- Do not infer undocumented diagnoses.
- Historical diagnoses alone should not be treated as a new acute event unless the note describes the event being graded.
- Multiple outcomes may be returned when independently supported.
- If none are supported, return an empty list.

SCOGS OUTCOME CATALOG
---------------------
{json.dumps(catalog, ensure_ascii=False)}

Return ONLY JSON in this exact shape:
{{
  "outcomes": [
    {{"outcome": "Exact catalog outcome name", "evidence": "short exact quote from note"}}
  ]
}}
""".strip()

    obj, raw = ollama_json(prompt, model=model, host=host, timeout=timeout, num_predict=700)
    if obj.get("parse_error"):
        raise RuntimeError("MedGemma returned invalid JSON during outcome identification.")
    items = obj.get("outcomes", [])
    if not isinstance(items, list):
        raise RuntimeError("MedGemma outcome-identification response did not contain an outcomes list.")

    valid_names = {x["outcome"] for x in catalog}
    identified = []
    seen = set()
    for item in items:
        if isinstance(item, str):
            name, evidence = item.strip(), ""
        elif isinstance(item, dict):
            name = str(item.get("outcome", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
        else:
            continue
        if name in valid_names and name not in seen:
            identified.append({"outcome": name, "evidence": evidence})
            seen.add(name)
    return identified, raw


if not RULES_PATH.exists():
    st.error(f"Could not find {RULES_PATH.name}. Keep app.py, scogs_grader.py, and scogs_outcomes.json in the same folder.")
    st.stop()

try:
    RULES = get_rules(str(RULES_PATH))
except Exception as exc:
    st.error(f"Could not load SCOGS rules: {exc}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", value=DEFAULT_MODEL)
    host = st.text_input("Ollama host", value=DEFAULT_HOST)
    timeout = st.number_input("Timeout (seconds)", min_value=30, max_value=1200, value=240, step=30)
    st.write(f"Loaded **{len(RULES.get('outcomes', []))}** SCOGS outcomes.")

note = st.text_area(
    "Clinical note / summary",
    height=330,
    placeholder="Paste one clinical note or patient summary here...",
)

if st.button("Analyze & Grade", type="primary", use_container_width=True):
    if not note.strip():
        st.warning("Paste a clinical note first.")
        st.stop()

    try:
        with st.spinner("Identifying supported SCOGS outcomes..."):
            identified, identification_raw = identify_outcomes(
                note.strip(), RULES, model=model, host=host, timeout=int(timeout)
            )
    except Exception as exc:
        st.error(str(exc))
        st.info("Confirm Ollama is running and that the selected model appears in `ollama list`.")
        st.stop()

    if not identified:
        st.info("No supported SCOGS outcome was identified from this clinical note.")
        with st.expander("Debug: outcome-identification response"):
            st.code(identification_raw)
        st.stop()

    st.success(f"Identified {len(identified)} supported SCOGS outcome(s).")

    for idx, item in enumerate(identified, start=1):
        outcome = item["outcome"]
        try:
            with st.spinner(f"Grading {idx}/{len(identified)}: {outcome}..."):
                result = grade_note_outcome(
                    note=note.strip(),
                    target_outcome=outcome,
                    rules_path=RULES_PATH,
                    model=model,
                    host=host,
                    timeout=int(timeout),
                    retry=True,
                )
        except Exception as exc:
            st.error(f"{outcome}: {exc}")
            continue

        # The grading engine is the final check. Do not display outcomes it rejects as absent.
        if result.get("outcome_present") is False or str(result.get("grade")) == "-1":
            continue

        st.divider()
        st.subheader(result.get("outcome") or outcome)

        c1, c2, c3 = st.columns(3)
        c1.metric("SCOGS Grade", result.get("grade") if result.get("grade") is not None else "N/A")
        c2.metric("Status", result.get("status", ""))
        confidence = result.get("confidence")
        c3.metric("Confidence", f"{confidence}%" if confidence is not None else "N/A")

        st.markdown("**Reason**")
        st.write(result.get("reason") or "—")

        reasoning = result.get("reasoning")
        if reasoning:
            st.markdown("**Reasoning**")
            st.write(reasoning)

        evidence = result.get("evidence", {})
        st.markdown("**Verified evidence**")
        if isinstance(evidence, dict):
            evidence_rows = [
                {"feature": key, "evidence": value}
                for key, value in evidence.items()
                if value not in (None, "", [], {})
            ]
            if evidence_rows:
                st.dataframe(evidence_rows, use_container_width=True, hide_index=True)
            else:
                st.write("No verified evidence returned.")
        elif isinstance(evidence, list):
            if evidence:
                for quote in evidence:
                    st.write(f'• "{quote}"')
            else:
                st.write("No verified evidence returned.")
        else:
            st.write(evidence or "No verified evidence returned.")

        features = result.get("features")
        if features:
            with st.expander("Verified features"):
                st.json(features)

        sources = result.get("feature_sources")
        if sources:
            with st.expander("Feature sources"):
                st.json(sources)

        with st.expander("Technical details"):
            st.write("Grading method:", result.get("grading_method", ""))
            st.write("Model:", result.get("model", model))
            if result.get("implementation_note"):
                st.write("Implementation note:", result.get("implementation_note"))

    with st.expander("Debug: outcome-identification response"):
        st.code(identification_raw)

st.divider()
st.caption("Research/hackathon use only. Human review is required before making clinical claims.")
