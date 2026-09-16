"""Parser and three-valued logic."""
import pytest

from scogs.predicates import (UNKNOWN, And, Between, Cmp, In, Not, Or, Truth,
                              k_and, k_not, k_or, parse)
from scogs.evaluate import CTX


def ev(src, **env):
    return parse(src).evaluate(env, CTX)


# ------------------------------------------------------------------ Kleene logic

def test_unknown_is_not_a_truth_value():
    with pytest.raises(TypeError):
        bool(UNKNOWN)

@pytest.mark.parametrize("a,b,expected", [
    (True, True, True), (True, False, False), (False, UNKNOWN, False),
    (UNKNOWN, False, False), (True, UNKNOWN, UNKNOWN), (UNKNOWN, UNKNOWN, UNKNOWN),
])
def test_k_and(a, b, expected):
    assert k_and(a, b) is expected

@pytest.mark.parametrize("a,b,expected", [
    (False, False, False), (True, UNKNOWN, True), (UNKNOWN, True, True),
    (False, UNKNOWN, UNKNOWN), (UNKNOWN, UNKNOWN, UNKNOWN),
])
def test_k_or(a, b, expected):
    assert k_or(a, b) is expected

def test_k_not_preserves_unknown():
    assert k_not(UNKNOWN) is UNKNOWN
    assert k_not(True) is False


# ----------------------------------------------------------------------- parsing

def test_bare_name_is_a_truth_test():
    assert parse("death_attributed") == Truth("death_attributed")

def test_chained_band():
    assert parse("2000 <= ferritin <= 4999") == Between(2000.0, "<=", "ferritin", "<=", 4999.0)

def test_value_first_comparison_is_flipped():
    assert parse("20 <= mpap") == Cmp("mpap", ">=", 20.0)

def test_enum_value_on_the_right_stays_a_name():
    assert parse("resp_support >= high_flow") == Cmp("resp_support", ">=", "high_flow")

def test_precedence_is_or_over_and():
    assert parse("a and b or c") == Or((And((Truth("a"), Truth("b"))), Truth("c")))

def test_parentheses_override_precedence():
    assert parse("a and (b or c)") == And((Truth("a"), Or((Truth("b"), Truth("c")))))

@pytest.mark.parametrize("bad", [
    "testes_volume < 3 by 14",     # the malformed row that shipped in the draft tables
    "ferritin >",
    "and symptomatic",
    "(symptomatic",
    "symptomatic ??",
])
def test_malformed_predicates_raise(bad):
    with pytest.raises(SyntaxError):
        parse(bad)


# -------------------------------------------------------------------- evaluation

def test_missing_feature_is_unknown_not_false():
    assert ev("treated") is UNKNOWN
    assert ev("not treated") is UNKNOWN

def test_explicit_false_is_false():
    assert ev("treated", treated=False) is False

def test_unknown_short_circuits_through_and():
    # one false conjunct decides the row even when another is undocumented
    assert ev("treated and splenectomy", splenectomy=False) is False

def test_unknown_short_circuits_through_or():
    assert ev("treated or splenectomy", splenectomy=True) is True

def test_ordinal_compares_by_rank_not_by_string():
    assert ev("resp_support >= high_flow", resp_support="niv_bipap_cpap") is True
    assert ev("resp_support >= high_flow", resp_support="low_flow_o2") is False
    # lexicographically "low_flow_o2" > "high_flow"; by rank it is lower
    assert "low_flow_o2" > "high_flow"

def test_ordinal_rejects_a_value_off_the_ladder():
    with pytest.raises(ValueError):
        ev("resp_support >= high_flow", resp_support="nasal_cannula")

def test_in_list():
    assert ev("goldberg_stage in (4,5)", goldberg_stage="4") is True
    assert ev("goldberg_stage in (4,5)", goldberg_stage="2") is False

def test_numeric_band_boundaries_are_inclusive_as_written():
    assert ev("2000 <= ferritin <= 4999", ferritin=2000) is True
    assert ev("2000 <= ferritin <= 4999", ferritin=4999) is True
    assert ev("2000 <= ferritin <= 4999", ferritin=1999) is False
