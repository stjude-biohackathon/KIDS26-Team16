"""Parse SCOGS_Booklet.pdf into per-outcome grade criteria.

Extracts SCOGS_Booklet.pdf on first run (needs poppler's `pdftotext`).
"""
import re, json, pathlib

SP = pathlib.Path(__file__).parent
_TXT = SP / "booklet_layout.txt"
if not _TXT.exists():
    import subprocess
    pdf = SP.parent.parent / "SCOGS_Booklet.pdf"
    subprocess.run(["pdftotext", "-layout", str(pdf), str(_TXT)], check=True)
pages = _TXT.read_text().split("\f")

TOC = [
 (13,"Arrhythmia"),(14,"Deep Vein Thrombosis (DVT)"),(15,"Diastolic Dysfunction"),
 (17,"Heart Failure Exacerbation"),(19,"Myocardial Infarction"),
 (22,"Systemic Arterial Hypertension"),(24,"Systolic Dysfunction"),
 (26,"TRV Elevation on Echocardiogram"),
 (29,"Cerebral Vasculopathy"),(31,"Chronic Pain"),(34,"Cognitive Dysfunction"),
 (36,"Elevated TCD Ultrasonography Velocity"),
 (37,"Posterior Reversible Encephalopathy Syndrome (PRES)"),
 (39,"Silent Cerebral Infarct"),(41,"Stroke (hemorrhagic or ischemic)"),
 (45,"Hearing Loss (in at least one ear)"),(46,"Sickle Cell Retinopathy (SCR)"),
 (49,"Cholecystitis/Cholelithiasis (gallstones)"),
 (53,"Acute Kidney Injury (AKI)"),(55,"Acute Papillary Necrosis"),
 (57,"Chronic Kidney Disease (CKD)"),(59,"Female Ovarian Dysfunction"),
 (61,"Male Impairments"),(63,"Priapism"),
 (65,"Delayed puberty"),(66,"Malnutrition Leading to Stunting (Decreased Height Velocity)"),
 (68,"Underweight"),
 (71,"Acute Sickle Cell Pain Episode"),(73,"Acute Splenic Sequestration"),
 (75,"Alloimmunization/ Delayed Hemolytic Transfusion Reaction"),
 (77,"Chronic Hypersplenism"),(79,"Hepatopathy"),(81,"Splenic Infarction"),
 (83,"Transfusional Iron Overload (Hemochromatosis or Hemosiderosis)"),
 (85,"Transient Aplastic Crisis Secondary to Parvovirus B19 Infection"),
 (89,"Fever"),(90,"Sepsis"),
 (93,"Malignant Neoplasms"),
 (97,"Avascular Necrosis of Joints (AVN)"),(99,"Leg Ulcer"),(101,"Osteomyelitis"),
 (103,"Osteoporosis"),
 (107,"Acute Multiorgan Failure"),
 (111,"Fetal Growth Restriction"),(112,"Pregnancy Loss"),(113,"Premature Delivery"),
 (115,"Depression"),
 (119,"Acute Chest Syndrome (ACS)"),(121,"Asthma Exacerbation"),
 (123,"Chronic Restrictive Lung Physiology"),(125,"Pulmonary Embolism (PE)"),
 (127,"Pulmonary Hypertension"),(129,"Sleep Apnea (obstructive or central)"),
]
assert len(TOC) == 53

FOOTER  = re.compile(r"St\. Jude Global \| Sickle Cell Outcome Grading System|^\s*\d{1,3}\s*$")
HEADER  = re.compile(r"^\s*Health Outcomes by Organ/System")
SECTION = re.compile(r"^\s{0,3}(Definition|Diagnostic Criteria|Methodology|References)\b")
GRADE   = re.compile(r"^(\s{0,14})Grade\s+([1-5])\b\s*(.*)$")
# stratum sub-headings that split an outcome into two parallel tables
STRATUM = re.compile(r"^\s*(Adults?\s*\(age|Children\s*\(age|If there (is|are))", re.I)

def page_lines(p):
    """Table region of one page: content lines up to the first section heading.

    Grading tables always precede Definition/Diagnostic Criteria/Methodology/
    References on a page, and a table that overflows continues at the top of the
    next page - so truncating per page keeps continuations and drops prose.
    """
    if p-1 >= len(pages): return []
    out, stopped = [], False
    for ln in pages[p-1].split("\n"):
        if FOOTER.search(ln) or HEADER.search(ln): continue
        if SECTION.match(ln): stopped = True; break
        out.append(ln.rstrip())
    return out, stopped

def parse(lines):
    """-> list of {stratum, grades:{1..5: text}} in document order.

    Cells are blank-line-delimited blocks. The "Grade N" label is vertically
    centred in its row, so a tall cell can be split by an internal blank line
    leaving an orphan block above the label - those attach to the next
    grade-bearing block. Lines flush at column 0 that are not Grade labels are
    outcome titles / running text, not cell content.
    """
    kept = [ln for ln in lines
            if GRADE.match(ln) or (ln.strip() and len(ln) - len(ln.lstrip()) >= 8)
            or not ln.strip()]

    blocks, cur = [], []
    for ln in kept:
        if not ln.strip():
            if cur: blocks.append(cur); cur = []
        else: cur.append(ln)
    if cur: blocks.append(cur)

    parsed = []                       # (grade|None, [text lines], is_stratum)
    for b in blocks:
        g, txt = None, []
        for ln in b:
            m = GRADE.match(ln)
            if m and g is None:
                g = m.group(2)
                if m.group(3).strip(): txt.append(m.group(3).strip())
            elif STRATUM.match(ln):
                continue
            else:
                t = ln.strip()
                if t: txt.append(t)
        parsed.append((g, txt))

    # attach orphan blocks (no Grade label) to the next graded block; if none
    # follows, to the previous one
    merged = []
    pending = []
    for g, txt in parsed:
        if g is None:
            pending += txt
        else:
            merged.append((g, pending + txt)); pending = []
    if pending and merged:
        merged[-1] = (merged[-1][0], merged[-1][1] + pending)

    tables, cur_t = [], {}
    for g, txt in merged:
        if g in cur_t: tables.append(cur_t); cur_t = {}
        cur_t[g] = " ".join(txt).strip()
    if cur_t: tables.append(cur_t)

    labels = [ln.strip() for ln in lines if STRATUM.match(ln)]
    return [dict(stratum=labels[i] if i < len(labels) else None, grades=t)
            for i, t in enumerate(tables)]

data = {}
for i,(start,name) in enumerate(TOC):
    end = TOC[i+1][0]-1 if i+1 < len(TOC) else 131
    # the grading table always precedes Definition and never resumes after it,
    # so stop consuming pages at the first section heading in the outcome
    lines = []
    for p in range(start, end+1):
        chunk, stopped = page_lines(p)
        lines += chunk
        if stopped: break
    data[f"{i+1:02d}"] = dict(name=name, pages=[start,end], tables=parse(lines))

SP.joinpath("booklet_grades.json").write_text(json.dumps(data, indent=1))
print(f"parsed {len(data)} outcomes")
for n,d in sorted(data.items()):
    ts = d["tables"]
    flag = "  <-- MULTI-TABLE" if len(ts) > 1 else ""
    gs = "/".join("".join(sorted(t["grades"])) for t in ts)
    miss = [f"{n}G{g}" for t in ts for g in "12345" if g not in t["grades"]]
    print(f"  {n} {d['name'][:42]:44s} p{d['pages'][0]:3d}-{d['pages'][1]:3d} tables={len(ts)} grades={gs}{flag}"
          + (f"  MISSING {miss}" if miss else ""))
    for t in ts:
        if t["stratum"]: print(f"        stratum: {t['stratum']}")
