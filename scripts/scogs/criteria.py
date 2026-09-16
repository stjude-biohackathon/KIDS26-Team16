"""Presence criteria rules: outcome -> predicate.

Evaluates whether objective diagnostic criteria / definitions in rules.md
are met by extracted features, independent of the model's subjective presence call.
"""
from __future__ import annotations

from .evaluate import CTX, resolve_derived
from .predicates import UNKNOWN, parse

PRESENCE_CRITERIA: dict[str, str] = {
    # rules.md §08 definition: "TRV ≥ 2.5 m/sec is considered elevated"
    "08": "trv >= 2.5",
    # rules.md §12 diagnostic criteria: conditional 170–199, abnormal ≥ 200 cm/s
    "12": "tcd_velocity >= 170",
    # rules.md §19 criteria: increase in sCr >= 0.3 mg/dL within 48h or >= 1.5x baseline
    "19": "creatinine_x_baseline >= 1.5 or creatinine_increase_mg_dl >= 0.3",
    # rules.md §34 criteria: LIC > 2.5 or ferritin > 1000 or organ dysfunction
    "34": "liver_iron_conc > 2.5 or ferritin > 1000 or organ_dysfunction_iron",
    # rules.md §36 definition: temperature >= 38.0 degC
    "36": "temperature >= 38.0",
}

_CACHE: dict[str, object] = {}


def _ast(pred: str):
    if pred not in _CACHE:
        _CACHE[pred] = parse(pred)
    return _CACHE[pred]


def criteria_met(outcome: str, features: dict) -> bool | UNKNOWN:
    """True if objective numeric presence criteria are met; False if refuted; UNKNOWN if missing/none."""
    pred = PRESENCE_CRITERIA.get(outcome)
    if not pred:
        return UNKNOWN
    env = resolve_derived(features)
    node = _ast(pred)
    return node.evaluate(env, CTX)
