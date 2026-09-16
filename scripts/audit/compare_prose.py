"""Verify rules.md prose sections + frequency against the booklet.

Regenerate the input first:
    pdftotext -layout SCOGS_Booklet.pdf scripts/audit/booklet_layout.txt
"""
import re, json, pathlib, unicodedata, difflib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from extract_booklet import TOC, pages, FOOTER, HEADER          # reuse page map
from compare_grades import norm, words                                  # reuse normalizer

ROOT = pathlib.Path("/Users/edward/Desktop/st_jude")
rules = ROOT.joinpath("rules.md").read_text()
SECTION = re.compile(r"^\s{0,3}(Definition|Diagnostic Criteria|Methodology|References)\b")
FREQ = re.compile(r"(Acute \+ Chronic|Chronic with [Ee]xacerbations?|Chronic|Acute)\s*$")

def raw_pages(a, b, title="", strip_header=True):
    """Outcome pages, optionally with the running title/frequency block dropped.

    The title and frequency label are reprinted at the top of every continuation
    page; left in, they inject phantom text into whichever prose section spans
    the page break. Only the first few lines of a *continuation* page can be a
    running header - prose legitimately starts with the outcome name (e.g.
    "Osteomyelitis: Inflammation of bone...").
    """
    tnorm = norm(title)
    out = []
    for p in range(a, b+1):
        if p-1 >= len(pages): break
        body = [ln.rstrip() for ln in pages[p-1].split("\n")
                if not FOOTER.search(ln) and not HEADER.search(ln)]
        if strip_header and p > a:
            seen = 0
            while body and seen < 4:
                ln = body[0]
                if not ln.strip(): body.pop(0); continue
                if ln.startswith(" "): break
                probe = norm(FREQ.sub("", ln.strip()))
                if probe and (probe in tnorm or tnorm in probe) or \
                   (not probe and FREQ.search(ln.strip())):
                    body.pop(0); seen += 1; continue
                break
        out += body
    return out

# ---- booklet: prose sections + frequency
bk = {}
for i,(start,name) in enumerate(TOC):
    end = TOC[i+1][0]-1 if i+1 < len(TOC) else 131
    lines = raw_pages(start, end, name)
    head  = raw_pages(start, start, name, strip_header=False)
    secs, cur = {}, None
    for ln in lines:
        m = SECTION.match(ln)
        if m:
            cur = m.group(1); secs.setdefault(cur, [])
        elif cur:
            secs[cur].append(ln.strip())
    freq = None
    for ln in head[:12]:
        m = FREQ.search(ln.strip())
        if m and not ln.strip().startswith("Grade"):
            freq = m.group(1); break
    bk[f"{i+1:02d}"] = dict(name=name,
        sections={k: " ".join(v).strip() for k,v in secs.items()}, freq=freq)

# ---- rules.md: prose sections + frequency attribute
secs = re.split(r"\n### (\d{2})\. ", rules)
rm = {}
for i in range(1, len(secs), 2):
    num, body = secs[i], secs[i+1]
    def grab(head, nxt):
        m = re.search(rf"#### {head}\n(.*?)(?=\n#### |\n---|\Z)", body, re.S)
        return m.group(1).strip() if m else ""
    fm = re.search(r"\|\s*\*\*Frequency Classification\*\*\s*\|\s*(.+?)\s*\|", body)
    rm[num] = dict(
        Definition=grab("Definition", None),
        **{"Diagnostic Criteria": grab("Diagnostic Criteria", None)},
        Methodology=grab(r"Methodology & Operational Notes", None),
        References=grab("References", None),
        freq=re.sub(r"[*]", "", fm.group(1)).strip() if fm else None)

print("=== FREQUENCY CLASSIFICATION ===")
fbad = []
for n in sorted(bk):
    b, r = (bk[n]["freq"] or ""), (rm[n]["freq"] or "")
    if norm(b) != norm(r): fbad.append((n, bk[n]["name"], b, r))
print(f"match: {53-len(fbad)}/53")
for n,name,b,r in fbad: print(f"   {n} {name[:38]:40s} booklet={b!r}  rules={r!r}")

print("\n=== PROSE SECTIONS ===")
for key in ["Definition", "Diagnostic Criteria", "Methodology"]:
    diffs = []
    for n in sorted(bk):
        b = bk[n]["sections"].get(key, "")
        r = rm[n].get(key, "")
        if not b and not r: continue
        if norm(b) == norm(r): continue
        wb, wr = words(b), words(r)
        sm = difflib.SequenceMatcher(None, wb, wr)
        ops = [(t," ".join(wb[i1:i2])," ".join(wr[j1:j2]))
               for t,i1,i2,j1,j2 in sm.get_opcodes() if t!="equal"]
        diffs.append((n, bk[n]["name"], round(sm.ratio(),3), ops))
    print(f"\n--- {key}: {53-len(diffs)}/53 identical, {len(diffs)} differ")
    for n,name,ratio,ops in sorted(diffs, key=lambda x:x[2]):
        if ratio > 0.97 and all(len(b.split())<=2 and len(r.split())<=2 for _,b,r in ops):
            print(f"   {n} {name[:34]:36s} sim {ratio}  [minor] "
                  + "; ".join(f"{b!r}->{r!r}" for _,b,r in ops)[:110])
            continue
        print(f"   {n} {name[:34]:36s} sim {ratio}")
        for t,b,r in ops[:6]:
            if t=="delete":   print(f"        BOOKLET ONLY : {b[:200]}")
            elif t=="insert": print(f"        RULES.MD ONLY: {r[:200]}")
            else:             print(f"        BOOKLET  : {b[:150]}\n        RULES.MD : {r[:150]}")
