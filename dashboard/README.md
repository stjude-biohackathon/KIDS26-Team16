# 🩸 SCOGS & MedGemma Results Dashboard

An interactive clinical analysis dashboard for evaluating MedGemma extraction runs and SCOGS deterministic rule outcomes.

---

> [!IMPORTANT]
> **The bundled run is stale.** `data_bundle.js` ships run
> `20260831T162426-b9b6eee6`, captured *before* two fixes landed: it was selected
> on the `loose` cohort rather than `scd_primary` (`39c63ee`), and it records the
> BF16 weights as `quant: "file_type_32"` (`af756e9`). Its charts are a working
> demonstration of the dashboard, **not a current result.** For live numbers, load
> a fresh results JSON through the **Runs & File Manager** tab.

## 🚀 How to Run the Dashboard

### Option 1: Direct Browser Opening (No Server Needed)
Simply double-click or open `dashboard/index.html` in any modern web browser (Chrome, Edge, Safari, Firefox).
The dashboard is self-contained with bundled run data and audit records in `data_bundle.js`.

```bash
# On macOS:
open dashboard/index.html
```

### Option 2: Python Local Server (With live API endpoints)
Run the built-in server script:

```bash
python3 dashboard/server.py
```
Then navigate to `http://localhost:8080` (or the port displayed in your terminal).

---

## 📊 Key Features

1. **Overview & Analytics**:
   - Executive KPI cards for model provenance, inference speed (`tok/s`, `sec/note`), quote verification rate (100%), and hallucination rate (0%).
   - Interactive bar charts for outcome status breakdown (Graded, Grade Set, Cannot Grade, Refuted, Absent).
   - Frequency distribution of extracted clinical features (creatinine, care setting, FiO2 %, pain complications, etc.).
   - Extraction & verification funnel chart.

2. **Case Explorer & Clinical Note Inspector**:
   - Interactive case list with search and filter options by selection cohort (seeded vs holdout), outcome ID, and status.
   - Full patient clinical note text with **interactive verbatim quote highlighting** for accepted features.
   - Detailed findings table (feature, value, quote snippet, unit) with click-to-jump quote highlighting.
   - Deterministic rule engine rationale and raw model JSON response inspector.

3. **Refuted & Audit Review Tab**:
   - Discrepancy reviewer sheet for inspecting cases where features were extracted but refuted by clinical rules.
   - Ability to add reviewer notes, flag `truly_absent`, and export updated audit CSVs.

4. **Runs & File Manager**:
   - Drag-and-drop or upload new MedGemma run JSON files to dynamically analyze new experiment outputs.
