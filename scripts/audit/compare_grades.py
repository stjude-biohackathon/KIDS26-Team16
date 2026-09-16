"""Verify rules.md grade cells against SCOGS_Booklet.pdf.

Regenerate the input first:
    pdftotext -layout SCOGS_Booklet.pdf scripts/audit/booklet_layout.txt
"""
import re, json, pathlib, unicodedata, difflib

SP   = pathlib.Path(__file__).parent
ROOT = pathlib.Path("/Users/edward/Desktop/st_jude")
book = json.loads(SP.joinpath("booklet_grades.json").read_text())
rules = ROOT.joinpath("rules.md").read_text()

# ---- rules.md side: grade rows, grouped into tables the same way (grade resets)
secs = re.split(r"\n### (\d{2})\. ", rules)
rm = {}
for i in range(1, len(secs), 2):
    num, body = secs[i], secs[i+1]
    body = body.split("#### Methodology")[0]
    rows = re.findall(r"^\|\s*\*\*Grade ([1-5])\*\*\s*\|[^|]*\|(.+?)\|\s*$", body, re.M)
    tables, cur = [], {}
    for g, crit in rows:
        if g in cur: tables.append(cur); cur = {}
        cur[g] = crit.strip()
    if cur: tables.append(cur)
    rm[num] = tables

UNI = {"≥":">=", "≤":"<=", "–":"-", "—":"-", "’":"'", "“":'"', "”":'"', "‘":"'",
       "°":"deg", "º":"deg", " ":" ", "−":"-", "µ":"u", "⁄":"/", "²":"2", "*":" "}
def norm(s, strip_fn=True):
    s = unicodedata.normalize("NFKC", s)
    for k,v in UNI.items(): s = s.replace(k,v)
    s = s.replace("<br>"," ").replace("<br/>"," ")
    s = re.sub(r"[*`_]", " ", s)
    s = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", s)          # rules.md footnote refs
    s = s.lower()
    if strip_fn:                                         # booklet superscript refs
        # booklet superscript footnote markers: "treatments3", "treatments.3"
        # (never after a digit, so decimals like "2.5" survive)
        s = re.sub(r"(?<=[a-z\)])\d{1,2}(?:,\d{1,2})*(?=\s|$)", " ", s)
        s = re.sub(r"(?<=[a-z]\.)\d{1,2}(?:,\d{1,2})*(?=\s|$)", " ", s)
    s = s.replace("n/a", "na")
    s = re.sub(r"(?<!\d)\.(?!\d)", " ", s)                  # sentence periods, keep decimals
    s = re.sub(r"[^a-z0-9<>=%/\.]+", " ", s)             # drop punctuation incl. '-'
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def words(s): return norm(s).split()

n_cells = n_exact = 0
diffs, structural = [], []
for num in sorted(book):
    bt, rt = book[num]["tables"], rm.get(num, [])
    if len(bt) != len(rt):
        structural.append((num, book[num]["name"], len(bt), len(rt))); continue
    for ti,(btab, rtab) in enumerate(zip(bt, rt)):
        for g in "12345":
            b, r = btab["grades"].get(g,""), rtab.get(g,"")
            n_cells += 1
            if norm(b) == norm(r):
                n_exact += 1; continue
            wb, wr = words(b), words(r)
            sm = difflib.SequenceMatcher(None, wb, wr)
            ops = [(t," ".join(wb[i1:i2])," ".join(wr[j1:j2]))
                   for t,i1,i2,j1,j2 in sm.get_opcodes() if t!="equal"]
            diffs.append(dict(num=num, name=book[num]["name"], table=ti, grade=g,
                              ratio=round(sm.ratio(),3), ops=ops, booklet=b, rules=r))

print(f"outcomes: {len(book)}   table-count mismatches: {len(structural)}")
for s in structural: print("   STRUCTURAL:", s)
print(f"grade cells compared: {n_cells}")
print(f"identical after normalization: {n_exact}")
print(f"differing: {len(diffs)}\n")
for d in sorted(diffs, key=lambda x: x["ratio"]):
    print(f"=== {d['num']} {d['name'][:38]} table{d['table']} Grade {d['grade']}  sim {d['ratio']}")
    for t,b,r in d["ops"]:
        if t=="delete":   print(f"    BOOKLET ONLY : {b}")
        elif t=="insert": print(f"    RULES.MD ONLY: {r}")
        else:             print(f"    BOOKLET      : {b}\n    RULES.MD     : {r}")
    print()
json.dump(diffs, open(SP/"grade_diffs.json","w"), indent=1)
