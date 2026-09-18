# conformal_utils.py
from __future__ import annotations

import math


def normalize_probabilities(raw: dict, labels: list[str]) -> dict[str, float]:
    values = {}
    for label in labels:
        try:
            value = float(raw.get(label, 0))
        except Exception:
            value = 0.0
        values[label] = max(0.0, value)

    total = sum(values.values())
    if total <= 0:
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}

    return {label: value / total for label, value in values.items()}


def qhat_from_scores(scores: list[float], alpha: float) -> float:
    if not scores:
        raise ValueError("No calibration scores supplied.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    ordered = sorted(float(x) for x in scores)
    n = len(ordered)

    # Finite-sample split-conformal order statistic:
    # k = ceil((n+1)*(1-alpha)), 1-indexed.
    k = math.ceil((n + 1) * (1 - alpha))

    if k > n:
        return 1.0

    return max(0.0, min(1.0, ordered[k - 1]))


def conformal_p_value(scores: list[float], test_score: float) -> float:
    if not scores:
        return 0.0
    count = sum(1 for score in scores if float(score) >= float(test_score))
    return (count + 1.0) / (len(scores) + 1.0)


def apply_conformal(
    probabilities: dict[str, float],
    calibration_scores: list[float],
    qhat: float,
):
    prediction_set = []
    p_values = {}

    for label, probability in probabilities.items():
        score = 1.0 - float(probability)

        if score <= qhat + 1e-12:
            prediction_set.append(label)

        p_values[label] = conformal_p_value(
            calibration_scores,
            score,
        )

    ordered_p = sorted(p_values.values(), reverse=True)

    credibility = ordered_p[0] if ordered_p else 0.0
    second_largest = ordered_p[1] if len(ordered_p) > 1 else 0.0
    confidence = 1.0 - second_largest

    return {
        "prediction_set": prediction_set,
        "p_values": p_values,
        "credibility": credibility,
        "confidence": confidence,
    }
