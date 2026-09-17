from __future__ import annotations

import argparse
import json
from pathlib import Path

from scogs_grader import DEFAULT_HOST, DEFAULT_MODEL, grade_note_outcome

SAMPLE_NOTE = """
58-year-old male with sickle cell disease (SCD), avascular necrosis (humeral/femoral),
pulmonary hypertension with right heart failure, chronic kidney disease, paroxysmal atrial
fibrillation, and deep vein thrombosis/pulmonary embolism history presented with chest pain
and shortness of breath, diagnosed with sickle cell pain crisis. Chest X-ray showed pulmonary
interstitial prominence (congestion/mild edema). Electrocardiogram revealed sinus tachycardia.
Hemoglobin decreased to 5.8 g/dL, requiring three packed red blood cell (pRBC) transfusions.
Pain uncontrolled with morphine patient-controlled analgesia. Subsequent deterioration:
diaphoresis, tachycardia, tachypnea, increased oxygen requirement to 15 L/min, intubated,
transferred to ICU. Diagnosed with acute chest syndrome (fever, chest pain, bilateral patchy
consolidations on chest X-ray). Developed supraventricular tachycardia/atrial fibrillation with
rapid ventricular response (>160 bpm) and hypotension; treated with intravenous amiodarone
(150 mg bolus, then 1 mg/min drip), later transitioned to oral amiodarone (400 mg twice daily
for 10-gram load, then 200 mg daily). Hemoglobin improved from 6.4 g/dL to 9.4 g/dL
(hematocrit 19.3% to 28.5%) over four days with one additional pRBC unit, correlating with
symptom improvement. Discharged two days after improvement. Continued amiodarone for three
months, then discontinued due to long-term side effect concerns; planned ablation.
""".strip()


def parse_args():
    p = argparse.ArgumentParser(description="Live ACS SCOGS demo")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--rules", default="scogs_outcomes.json")
    p.add_argument("--note-file", help="Optional .txt file containing a clinical note")
    p.add_argument("--no-retry", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    note = SAMPLE_NOTE
    if args.note_file:
        note = Path(args.note_file).read_text(encoding="utf-8")

    print("=" * 78)
    print("SCOGS-SCRIBE LIVE DEMO - ACUTE CHEST SYNDROME")
    print("=" * 78)
    print(f"Model: {args.model}")
    print("Pipeline: deterministic text safety-net + MedGemma extraction -> verification -> SCOGS rule")
    print()

    result = grade_note_outcome(
        note=note,
        target_outcome="Acute Chest Syndrome (ACS)",
        rules_path=Path(args.rules),
        model=args.model,
        host=args.host,
        timeout=args.timeout,
        retry=not args.no_retry,
    )

    print("ACS present:", result.get("outcome_present"))
    print("Presence evidence:", result.get("present_quote"))
    print()
    print("VERIFIED FEATURES")
    print(json.dumps(result.get("features", {}), indent=2, ensure_ascii=False))
    print()
    print("VERIFIED EVIDENCE")
    print(json.dumps(result.get("evidence", {}), indent=2, ensure_ascii=False))
    print()
    print("FEATURE SOURCES")
    print(json.dumps(result.get("feature_sources", {}), indent=2, ensure_ascii=False))
    print()
    print("FINAL SCOGS RESULT")
    print("Grade:", result.get("grade"))
    print("Status:", result.get("status"))
    print("Reason:", result.get("reason"))

    print()
    print("Implementation note:", result.get("implementation_note"))


if __name__ == "__main__":
    main()
