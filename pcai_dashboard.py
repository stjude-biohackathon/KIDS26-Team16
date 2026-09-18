
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from pcai_grader_backend import (
    DEFAULT_MODEL,
    create_client,
    load_calibration,
    load_rules,
    normalize_result,
    read_results,
    read_reviews,
    request_resilient,
    replace_case_rows,
    save_review,
    valid_classes,
)

APP_DIR = Path(__file__).resolve().parent
SUMMARIES_PATH = APP_DIR / "SCD_summaries.csv"
RULES_PATH = APP_DIR / "scogs_14_outcomes.json"
RESULTS_PATH = APP_DIR / "SCD_grades_14_gptoss.csv"
REVIEWS_PATH = APP_DIR / "SCD_reviewed_grades.csv"
CALIBRATION_PATH = APP_DIR / "conformal_calibration.json"

st.set_page_config(
    page_title="SCOGS-Scribe",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{
    --navy:#071d35;
    --blue:#1558d6;
    --blue2:#2d6cdf;
    --line:#dce3ec;
    --soft:#f7f9fc;
    --ink:#1b2737;
    --muted:#68778c;
    --green:#18a368;
    --green-soft:#e8f7ef;
    --amber:#b87800;
    --amber-soft:#fff6df;
    --red:#c54646;
    --red-soft:#fff0f0;
}
html,body,[class*="css"]{
    font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                "Segoe UI",sans-serif;
}
.stApp{
    background:linear-gradient(180deg,#f7f9fc 0%,#f3f6fa 100%);
}
.block-container{
    max-width:1500px;
    padding-top:0.7rem;
    padding-bottom:3rem;
}
[data-testid="stSidebar"]{
    background:#ffffff;
    border-right:1px solid var(--line);
}
.topbar{
    background:var(--navy);
    color:white;
    border-radius:0 0 0 0;
    padding:16px 24px 14px 24px;
    margin:0 -1rem 24px -1rem;
    box-shadow:0 8px 24px rgba(5,25,48,.16);
}
.brand{
    display:flex;
    align-items:center;
    gap:10px;
}
.brandmark{
    font-size:1.45rem;
}
.brandtitle{
    font-size:1.25rem;
    font-weight:800;
    letter-spacing:-.02em;
}
.brandsub{
    font-size:.78rem;
    color:rgba(255,255,255,.72);
    margin-top:2px;
}
.panel{
    background:#fff;
    border:1px solid var(--line);
    border-radius:11px;
    box-shadow:0 3px 12px rgba(29,50,82,.05);
    padding:18px 20px;
}
.panel-title{
    font-size:1.05rem;
    font-weight:800;
    color:var(--ink);
    margin-bottom:4px;
}
.panel-sub{
    color:var(--muted);
    font-size:.82rem;
    margin-bottom:14px;
}
.result-header{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:10px;
    border-bottom:1px solid #e6ebf1;
    padding-bottom:11px;
    margin-bottom:16px;
}
.complete-pill{
    background:var(--green-soft);
    color:#128254;
    border:1px solid #c8ead9;
    padding:5px 10px;
    border-radius:999px;
    font-size:.73rem;
    font-weight:800;
}
.outcome-kicker{
    font-size:.75rem;
    font-weight:800;
    color:#56657a;
    margin-bottom:4px;
}
.outcome-title{
    font-size:1.08rem;
    font-weight:800;
    color:var(--ink);
    margin-bottom:12px;
}
.grade-box{
    border:1px solid #b8c7db;
    border-radius:8px;
    padding:12px 12px;
    margin:8px 0 12px 0;
    background:#fbfdff;
}
.grade-line{
    font-size:1.6rem;
    font-weight:850;
    color:var(--ink);
    letter-spacing:-.03em;
}
.grade-line small{
    font-size:.78rem;
    color:#55667b;
    font-weight:750;
    letter-spacing:.02em;
}
.confidence-wrap{
    margin:8px 0 14px 0;
}
.confidence-label{
    display:flex;
    justify-content:space-between;
    font-size:.78rem;
    color:#46576a;
    font-weight:700;
    margin-bottom:5px;
}
.confidence-track{
    height:8px;
    background:#e6ebf1;
    border-radius:999px;
    overflow:hidden;
}
.confidence-fill{
    height:8px;
    background:var(--green);
}
.section-label{
    font-size:.78rem;
    font-weight:800;
    color:#2e3b4e;
    margin:14px 0 6px 0;
}
.reason{
    color:#3b4a5d;
    line-height:1.48;
    font-size:.84rem;
}
.evidence{
    background:#eef3fb;
    border:1px solid #dde7f5;
    border-radius:7px;
    padding:9px 11px;
    font-size:.8rem;
    color:#405066;
    font-style:italic;
    margin:6px 0;
}
.criteria-table{
    border:1px solid #dfe5ee;
    border-radius:7px;
    overflow:hidden;
    margin-top:6px;
}
.criteria-row{
    display:grid;
    grid-template-columns:54px 1fr;
    border-bottom:1px solid #e9edf2;
}
.criteria-row:last-child{border-bottom:none;}
.criteria-grade{
    padding:8px 10px;
    font-size:.78rem;
    font-weight:800;
    background:#f7f9fc;
    color:#34465d;
}
.criteria-text{
    padding:8px 10px;
    font-size:.77rem;
    color:#4a5a6d;
    line-height:1.35;
}
.outcome-card{
    background:white;
    border:1px solid var(--line);
    border-radius:10px;
    padding:12px 13px;
    min-height:126px;
    box-shadow:0 2px 8px rgba(29,50,82,.035);
    margin-bottom:10px;
}
.outcome-card-title{
    font-weight:800;
    font-size:.83rem;
    color:var(--ink);
    line-height:1.2;
    min-height:2.0em;
}
.outcome-card-grade{
    font-weight:850;
    font-size:1.22rem;
    margin-top:7px;
}
.muted{color:#7c8796;}
.good{color:#0f8a5d;}
.warn{color:#a66b00;}
.bad{color:#bf4141;}
.card-conf{
    font-size:.7rem;
    color:#708096;
    margin-top:2px;
}
.review-badge{
    display:inline-block;
    margin-top:6px;
    background:#f0ebff;
    color:#6450b7;
    border:1px solid #dfd5ff;
    border-radius:999px;
    padding:3px 7px;
    font-size:.65rem;
    font-weight:800;
}
.casebar{
    background:#fff;
    border:1px solid var(--line);
    border-radius:10px;
    padding:10px 13px;
    margin-bottom:12px;
}
.caseid{
    font-size:.78rem;
    text-transform:uppercase;
    letter-spacing:.08em;
    color:#718095;
    font-weight:800;
}
.casevalue{
    font-size:1rem;
    font-weight:800;
    color:var(--ink);
}
.stButton>button{
    border-radius:8px !important;
    font-weight:750 !important;
}
div[data-testid="stTextArea"] textarea{
    border-radius:7px !important;
    background:white;
    border-color:#cfd8e5 !important;
    min-height:500px !important;
    font-size:.83rem !important;
    line-height:1.5 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_cases():
    df = pd.read_csv(SUMMARIES_PATH)
    if "true_outcomes" not in df.columns:
        if "true_outcome" in df.columns:
            df["true_outcomes"] = df["true_outcome"].fillna("")
        else:
            df["true_outcomes"] = ""
    return df

@st.cache_data(show_spinner=False)
def get_rules():
    return load_rules(RULES_PATH)

def get_results_df():
    rows = read_results(RESULTS_PATH)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def get_reviews_df():
    rows = read_reviews(REVIEWS_PATH)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def status_label(row):
    status = str(row.get("status",""))
    if status == "graded":
        return f"Grade {row.get('grade','')}"
    if status == "insufficient_information":
        return "Needs info"
    if status == "absent":
        return "Not detected"
    return "Not run"

def grade_color(row):
    status = str(row.get("status",""))
    if status == "graded":
        try:
            g = int(float(row.get("grade",0)))
        except Exception:
            g = 0
        return "bad" if g >= 4 else "good"
    if status == "insufficient_information":
        return "warn"
    return "muted"

df = load_cases()
rules = get_rules()
rule_map = {r["outcome"]: r for r in rules}
case_ids = df["case_id"].astype(str).tolist()

if "case_index" not in st.session_state:
    st.session_state.case_index = 0
if "selected_outcome" not in st.session_state:
    st.session_state.selected_outcome = "Acute chest syndrome (ACS)"

st.markdown(
    """
<div class="topbar">
  <div class="brand">
    <div class="brandmark">✚</div>
    <div>
      <div class="brandtitle">SCOGS-Scribe</div>
      <div class="brandsub">Automated Severity Grading for Sickle Cell Disease Clinical Notes</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar navigation / dataset controls
with st.sidebar:
    st.markdown("### Dataset")
    chosen = st.selectbox(
        "Case",
        case_ids,
        index=min(st.session_state.case_index, len(case_ids)-1),
    )
    st.session_state.case_index = case_ids.index(chosen)

    pcol, ncol = st.columns(2)
    with pcol:
        if st.button("← Prev", use_container_width=True, disabled=st.session_state.case_index == 0):
            st.session_state.case_index -= 1
            st.rerun()
    with ncol:
        if st.button("Next →", use_container_width=True, disabled=st.session_state.case_index >= len(case_ids)-1):
            st.session_state.case_index += 1
            st.rerun()

    st.divider()
    if os.environ.get("PCAI_API_KEY"):
        st.success("PCAI key detected", icon="✅")
    else:
        st.error("PCAI_API_KEY is not set", icon="⚠️")

    model = st.text_input("Model", value=DEFAULT_MODEL)
    batch_size = st.select_slider(
        "Outcomes per request",
        options=[1,2,3],
        value=1,
        help="1 is safest with the current Bifrost timeout behavior.",
    )

    st.divider()
    show_ref = st.toggle("Show reference labels", value=False)
    st.caption("Reference labels are never sent to GPT-OSS.")

case = df.iloc[st.session_state.case_index]
case_id = str(case["case_id"])
summary = str(case.get("summary","") or "")
true_outcomes = str(case.get("true_outcomes","") or "")

all_results = get_results_df()
all_reviews = get_reviews_df()

case_results = (
    all_results[all_results["case_id"].astype(str)==case_id].copy()
    if not all_results.empty else pd.DataFrame()
)
case_reviews = (
    all_reviews[all_reviews["case_id"].astype(str)==case_id].copy()
    if not all_reviews.empty else pd.DataFrame()
)

result_map = {}
if not case_results.empty:
    result_map = {
        str(row["outcome"]): row.to_dict()
        for _,row in case_results.iterrows()
    }
review_map = {}
if not case_reviews.empty:
    review_map = {
        str(row["outcome"]): row.to_dict()
        for _,row in case_reviews.iterrows()
    }

st.markdown(
    f"""
<div class="casebar">
  <div class="caseid">Current case</div>
  <div class="casevalue">{case_id}</div>
</div>
""",
    unsafe_allow_html=True,
)

if show_ref:
    st.info(f"Reference outcome labels (evaluation only): {true_outcomes or 'None supplied'}")

left, right = st.columns([1.05,1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Enter Clinical Note</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-sub">Paste or review the clinical note/summary below. GPT-OSS will apply all 14 SCOGS rubrics.</div>',
        unsafe_allow_html=True,
    )

    note_text = st.text_area(
        "Clinical note",
        value=summary,
        label_visibility="collapsed",
        height=510,
        key=f"note_{case_id}",
    )

    analyze = st.button("Analyze Note", type="primary", use_container_width=True)

    st.markdown(
        '<div style="text-align:center;color:#738197;font-size:.72rem;margin-top:9px;">'
        'Research prototype • Uses GPT-OSS through St. Jude PCAI/Bifrost'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

if analyze:
    if not os.environ.get("PCAI_API_KEY"):
        st.error('Set the key in this terminal first: $env:PCAI_API_KEY="YOUR_VIRTUAL_KEY"')
    else:
        client = create_client()
        calibration = load_calibration(CALIBRATION_PATH)

        progress = st.progress(0, text="Starting GPT-OSS grading…")
        status = st.empty()
        new_rows = []
        errors = []

        total = len(rules)
        for start in range(0,total,batch_size):
            batch = rules[start:start+batch_size]
            status.info("Grading: " + ", ".join(r["outcome"] for r in batch))

            batch_results, batch_errors = request_resilient(client, model, note_text, batch)
            returned = {
                str(item.get("outcome","")).strip(): item
                for item in batch_results if isinstance(item,dict)
            }

            for rule in batch:
                item = returned.get(rule["outcome"])
                if item is not None:
                    new_rows.append(
                        normalize_result(
                            case_id=case_id,
                            true_outcomes=true_outcomes,
                            summary=note_text,
                            rule=rule,
                            item=item,
                            model=model,
                            calibration=calibration,
                        )
                    )

            errors.extend(batch_errors)
            done = min(start+len(batch),total)
            progress.progress(done/total, text=f"Processed {done} of {total} outcomes")

        replace_case_rows(RESULTS_PATH, case_id, new_rows)
        progress.empty()
        status.empty()
        st.cache_data.clear()

        if errors:
            st.warning(f"Saved {len(new_rows)} outcome results. {len(errors)} outcome(s) need retry.")
        else:
            st.success("Analysis complete.")
        st.rerun()

# refresh
all_results = get_results_df()
case_results = (
    all_results[all_results["case_id"].astype(str)==case_id].copy()
    if not all_results.empty else pd.DataFrame()
)
result_map = {
    str(row["outcome"]): row.to_dict()
    for _,row in case_results.iterrows()
} if not case_results.empty else {}

# Choose primary result: highest grade among graded, then needs-info, then absent.
primary = None
if result_map:
    rows = list(result_map.values())
    def rank(r):
        if str(r.get("status","")) == "graded":
            try:
                return (3,int(float(r.get("grade",0))))
            except Exception:
                return (3,0)
        if str(r.get("status","")) == "insufficient_information":
            return (2,0)
        return (1,0)
    primary = sorted(rows, key=rank, reverse=True)[0]

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if not primary:
        st.markdown('<div class="panel-title">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-sub">Click <b>Analyze Note</b> to run GPT-OSS across all 14 finalized SCOGS outcomes.</div>',
            unsafe_allow_html=True,
        )
        st.info("No model output is available for this case yet.")
    else:
        out = str(primary.get("outcome",""))
        rule = rule_map[out]
        conf = float(primary.get("model_confidence_pct",0) or 0)
        label = status_label(primary)

        st.markdown(
            '<div class="result-header">'
            '<div class="panel-title" style="margin:0;">Analysis Results</div>'
            '<div class="complete-pill">✓ Analysis complete</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="outcome-kicker">Primary Outcome</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="outcome-title">{out}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="grade-box"><div class="grade-line">{label} '
            f'<small>SCOGS Severity</small></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="confidence-wrap">
  <div class="confidence-label"><span>Confidence</span><span>{conf:.0f}%</span></div>
  <div class="confidence-track"><div class="confidence-fill" style="width:{max(0,min(conf,100))}%;"></div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Why this grade?</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="reason">{primary.get("reasoning","")}</div>', unsafe_allow_html=True)

        try:
            evidence = json.loads(primary.get("evidence_json","[]") or "[]")
        except Exception:
            evidence = []
        st.markdown('<div class="section-label">Evidence from note</div>', unsafe_allow_html=True)
        if evidence:
            for e in evidence:
                st.markdown(f'<div class="evidence">"{e}"</div>', unsafe_allow_html=True)
        else:
            st.caption("No exact evidence quote survived verification.")

        st.markdown('<div class="section-label">SCOGS Criteria</div>', unsafe_allow_html=True)
        st.markdown('<div class="criteria-table">', unsafe_allow_html=True)
        for grade,definition in rule.get("grade_definitions",{}).items():
            st.markdown(
                f'<div class="criteria-row"><div class="criteria-grade">Grade {grade}</div>'
                f'<div class="criteria-text">{definition}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("New Note", use_container_width=True):
            st.session_state.case_index = min(st.session_state.case_index + 1, len(case_ids)-1)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# All outcomes
st.write("")
st.markdown("### All 14 SCOGS Outcomes")
for start in range(0,len(rules),4):
    cols = st.columns(4)
    for col,rule in zip(cols,rules[start:start+4]):
        out = rule["outcome"]
        row = result_map.get(out)
        review = review_map.get(out)

        if row:
            label = status_label(row)
            conf = row.get("model_confidence_pct","")
            color = grade_color(row)
        else:
            label = "Not run"
            conf = ""
            color = "muted"

        with col:
            st.markdown(
                f"""
<div class="outcome-card">
  <div class="outcome-card-title">{out}</div>
  <div class="outcome-card-grade {color}">{label}</div>
  <div class="card-conf">{(str(conf)+'% confidence') if str(conf) not in ('','nan','None') else ''}</div>
  {'<div class="review-badge">Reviewed</div>' if review else ''}
</div>
""",
                unsafe_allow_html=True,
            )

# Reviewer panel
st.write("")
st.markdown("### Review / Correct a Grade")

selected = st.selectbox(
    "Outcome",
    [r["outcome"] for r in rules],
    index=[r["outcome"] for r in rules].index(st.session_state.selected_outcome)
    if st.session_state.selected_outcome in [r["outcome"] for r in rules] else 0,
)
st.session_state.selected_outcome = selected
rule = rule_map[selected]
row = result_map.get(selected)
review = review_map.get(selected)

r1,r2 = st.columns([1.3,1])

with r1:
    if row:
        st.markdown(
            f"**Model decision:** {status_label(row)}  \n"
            f"**Confidence:** {row.get('model_confidence_pct','')}%  \n"
            f"**Reasoning:** {row.get('reasoning','')}"
        )
    else:
        st.info("This outcome has not been graded yet.")

with r2:
    allowed = valid_classes(rule)
    choices = ["Accept model"] + [
        "Absent" if x=="absent"
        else "Insufficient information" if x=="insufficient_information"
        else f"Grade {x}"
        for x in allowed
    ]

    default = "Accept model"
    if review:
        rs = str(review.get("reviewed_status",""))
        rg = str(review.get("reviewed_grade",""))
        if rs=="absent":
            default="Absent"
        elif rs=="insufficient_information":
            default="Insufficient information"
        elif rs=="graded" and rg:
            default=f"Grade {rg}"

    choice = st.selectbox("Reviewer decision", choices, index=choices.index(default) if default in choices else 0)
    note = st.text_area("Review note", value=str(review.get("review_note","")) if review else "", height=90)

    if st.button("Save review", use_container_width=True):
        if not row:
            st.error("Analyze the note first.")
        else:
            if choice=="Accept model":
                reviewed_status = str(row.get("status",""))
                reviewed_grade = str(row.get("grade",""))
            elif choice=="Absent":
                reviewed_status="absent"; reviewed_grade=""
            elif choice=="Insufficient information":
                reviewed_status="insufficient_information"; reviewed_grade=""
            else:
                reviewed_status="graded"; reviewed_grade=choice.replace("Grade ","").strip()

            save_review(
                REVIEWS_PATH,
                {
                    "case_id":case_id,
                    "outcome":selected,
                    "model_status":str(row.get("status","")),
                    "model_grade":str(row.get("grade","")),
                    "reviewed_status":reviewed_status,
                    "reviewed_grade":reviewed_grade,
                    "review_note":note,
                    "reviewed_at":datetime.now(timezone.utc).isoformat(),
                },
            )
            st.success("Review saved.")
            st.rerun()

with st.expander("View original clinical note"):
    st.text(str(case.get("original_case_text","") or ""))

st.caption(
    "GPT-OSS model confidence is uncalibrated. "
    "Conformal prediction is shown only after a valid held-out expert calibration file is added."
)
