"""Applicability rules: outcome -> predicate.

Determines whether an outcome can apply to the patient based on demographic
or baseline features (e.g. biological sex for sex-restricted conditions, or
age exclusions like fever in infants 0-59 days).

Evaluates to True (applicable), False (not applicable), or UNKNOWN (undetermined).
"""
from __future__ import annotations

from .evaluate import CTX, resolve_derived
from .predicates import UNKNOWN, parse

APPLICABILITY: dict[str, str] = {
    "22": "patient_sex == female",
    "23": "patient_sex == male",
    "24": "patient_sex == male",
    "44": "patient_sex == female",    # rules.md §44: "applies to the mother"
    "45": "patient_sex == female",
    "46": "patient_sex == female",    # rules.md §46: "applies to the mother"
    "36": "patient_age >= 0.1642",    # rules.md §36: ages 0-59 days excluded (60/365.25 = 0.16427)
}

_CACHE: dict[str, object] = {}


def _ast(pred: str):
    if pred not in _CACHE:
        _CACHE[pred] = parse(pred)
    return _CACHE[pred]


def applicability(outcome: str, features: dict) -> bool | UNKNOWN:
    pred = APPLICABILITY.get(outcome)
    if not pred:
        return True
    env = resolve_derived(features)
    node = _ast(pred)
    return node.evaluate(env, CTX)
