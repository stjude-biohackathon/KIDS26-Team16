"""SCOGS dashboard - Shiny entry point.

    python -m shiny run dashboard/interactive_dashboard.py      # from the repository root

Two modes (README.md, "Dashboard"):
  * Explore Saved Runs - inspect a results file written by medgemma_extraction.py;
  * Evaluate Live Note - grade a pasted note through the CLI's own extraction and
    verification (evaluation.py), or grade feature values typed in by hand.

This file only wires the app together: layout.py (the page) and server.py (the
reactive logic), which use evaluation.py, view_state.py, data.py and
highlight.py. Styles and the theme toggle live in static/.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `shiny run dashboard/interactive_dashboard.py` imports this file by path, so
# only `dashboard/` is on sys.path. Add `scripts/` (for `scogs.…` and
# `experiments.…`) and the repository root (for `dashboard.…`): the same two
# roots pyproject.toml gives the tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shiny import App

from dashboard.layout import app_ui
from dashboard.server import server

app = App(app_ui, server)
