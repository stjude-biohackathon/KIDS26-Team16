"""The prompt ablation ladder.

Two things are being protected here. The first is that stage 0 never moves: it is
the baseline arm, and an ablation whose control drifts measures the control. The
second is that each rung turns on exactly its own group and nothing below it, so a
number can be attributed to the change that produced it.
"""
import dataclasses
import hashlib
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))
from medgemma_extraction import (  # noqa: E402
    STAGES, Tally, build_prompt, call_mock, feature_brief, grade_weight,
    order_features, prompt_features, reply_is_usable, reply_schema, stage, verify,
)
from scogs.definitions import outcome_definition, presence_brief  # noqa: E402
from scogs.tables import TABLES  # noqa: E402

NOTE = ("A 14-year-old with HbSS presented with chest pain. FiO2 was escalated to 60%. "
        "He received a simple transfusion of 2 units and was started on norepinephrine.")


# --------------------------------------------------------------- the baseline

# Every stage is measured against stage 0, so stage 0 is the one thing in this
# file that must never change. Editing the prompt and re-pinning this digest
# silently rebases every comparison in `tasks/` onto a different control.
STAGE0_DIGEST = "db00daf41e74046ddcf1bc346d87c927c28344238b54c923d576d23095c4b022"


def test_stage_0_is_the_prompt_that_shipped():
    blob = "".join(build_prompt("NOTE-TEXT", n) for n in sorted(TABLES))
    assert hashlib.sha256(blob.encode()).hexdigest() == STAGE0_DIGEST, (
        "Stage 0 is the baseline arm of the ablation and has changed. If that was "
        "deliberate, every recorded stage comparison is now against a different control.")


def test_the_default_is_stage_0():
    assert build_prompt(NOTE, "48") == build_prompt(NOTE, "48", stage("0"))


# ------------------------------------------------------------------ the rungs

def test_the_rungs_are_cumulative():
    """Each rung is the one below it plus more; nothing is ever turned back off."""
    live = [{k for k, v in dataclasses.asdict(stage(s)).items()
             if k != "name" and v is True} for s in STAGES]
    for lower, upper in zip(live, live[1:]):
        assert lower < upper, f"{lower} is not a strict subset of {upper}"


def test_an_unknown_rung_is_refused():
    with pytest.raises(ValueError, match="unknown prompt stage"):
        stage("2c")


@pytest.mark.parametrize("name,expected", [
    ("0", set()),
    ("1", {"schema", "structure", "flat_penalty", "retry"}),
    ("2a", {"schema", "structure", "flat_penalty", "retry", "presence", "death_both"}),
    ("2b", {"schema", "structure", "flat_penalty", "retry", "presence", "death_both", "cot"}),
])
def test_each_rung_turns_on_its_own_group(name, expected):
    live = {k for k, v in dataclasses.asdict(stage(name)).items()
            if k != "name" and v is True}
    assert live == expected


# ------------------------------------------------------------------ structure

def test_findings_are_asked_for_before_present():
    """Stage 0 puts `present` first, so the field that discards every other result
    is decoded before a single finding exists."""
    p0, p1 = build_prompt(NOTE, "48"), build_prompt(NOTE, "48", stage("1"))
    assert p0.index('"present"') < p0.index('"findings"')
    assert p1.index('"findings"') < p1.index('"present"')


def test_the_note_is_fenced_in_tags_from_stage_1():
    assert '<note>' not in build_prompt(NOTE, "48")
    assert f"<note>\n{NOTE}\n</note>" in build_prompt(NOTE, "48", stage("1"))


def test_the_mock_still_finds_the_note_under_either_fence():
    for st in (stage("0"), stage("3")):
        reply = json.loads(call_mock(build_prompt(NOTE, "48", st), "", ""))
        assert reply["findings"][0]["quote"] in NOTE


def test_the_job_is_stated_positively_from_stage_1():
    assert "NOT assigning a severity grade" in build_prompt(NOTE, "48")
    assert "NOT assigning a severity grade" not in build_prompt(NOTE, "48", stage("1"))


def test_the_empty_answer_is_shown_not_only_described():
    """The stage 0 template only ever renders one populated finding, so the shape
    the model is taught is 'emit an object' - which with nothing to say is a null
    placeholder."""
    assert '"findings": []' not in build_prompt(NOTE, "48")
    assert '"findings": []' in build_prompt(NOTE, "48", stage("1"))


def test_the_note_can_be_put_above_the_instructions():
    p = build_prompt(NOTE, "48", stage("1", note_first=True))
    assert p.startswith("<note>")
    assert not build_prompt(NOTE, "48", stage("1")).startswith("<note>")


# ------------------------------------------------------------ feature ordering

def test_stage_0_orders_features_alphabetically():
    assert prompt_features("48") == sorted(prompt_features("48"))


def test_grade_driving_features_come_before_a_grade_1_only_feature():
    """Alphabetical order opens outcome 48 with `acs_other_support` - one row,
    negated, grade 1 - ahead of the three features grades 2/3/4 separate on."""
    ordered = prompt_features("48", stage("1"))
    assert ordered[-1] == "acs_other_support"
    for n in ("resp_support", "transfusion_type", "fio2_pct"):
        assert ordered.index(n) < ordered.index("acs_other_support")


def test_a_derived_feature_hands_its_weight_to_its_inputs():
    """`life_support` is never asked for; the features it expands into are, and
    they have to inherit the grade it would have carried."""
    w = grade_weight("48")
    assert "life_support" not in w
    assert w["vasopressors"][0] == 4


def test_ordering_never_changes_the_set_of_features_asked_for():
    for n in TABLES:
        assert set(prompt_features(n)) == set(prompt_features(n, stage("3")))


# ------------------------------------------------------- constrained decoding

def test_the_schema_pins_the_feature_name_to_this_outcome():
    sch = reply_schema(prompt_features("36", stage("1")), stage("1"))
    consts = [b["properties"]["feature"]["const"]
              for b in sch["properties"]["findings"]["items"]["anyOf"]]
    assert consts == ["temperature"]


def test_the_schema_forbids_an_empty_or_missing_quote():
    sch = reply_schema(prompt_features("48", stage("1")), stage("1"))
    for branch in sch["properties"]["findings"]["items"]["anyOf"]:
        assert branch["properties"]["quote"] == {"type": "string", "minLength": 1}
        assert set(branch["required"]) == {"feature", "value", "quote"}
        assert branch["additionalProperties"] is False


def test_the_schema_types_a_value_from_the_registry():
    sch = reply_schema(["resp_support", "fio2_pct", "vasopressors"], stage("1"))
    by = {b["properties"]["feature"]["const"]: b["properties"]["value"]
          for b in sch["properties"]["findings"]["items"]["anyOf"]}
    assert by["fio2_pct"] == {"type": "number"}
    assert by["vasopressors"] == {"type": "boolean"}
    assert by["resp_support"]["enum"][0] == "room_air"


def test_evidence_is_required_only_once_it_is_asked_for():
    assert "evidence" not in reply_schema(["temperature"], stage("2a"))["properties"]
    assert "evidence" in reply_schema(["temperature"], stage("2b"))["required"]


def test_present_quote_is_offered_but_never_required():
    """A note that does not evidence an outcome usually contains no sentence
    saying so. Demanding a quote there buys a fabricated one."""
    sch = reply_schema(["temperature"], stage("2a"))
    assert "present_quote" in sch["properties"]
    assert "present_quote" not in sch["required"]


def test_the_schema_enum_matches_the_features_the_prompt_asks_for():
    """A name the prompt asks for and the grammar forbids is an unanswerable
    question; one the grammar allows and the prompt never names is
    `unknown_feature` coming back through the front door."""
    for n in TABLES:
        st = stage("1")
        names = prompt_features(n, st)
        sch = reply_schema(names, st)
        consts = [b["properties"]["feature"]["const"]
                  for b in sch["properties"]["findings"]["items"]["anyOf"]]
        assert consts == names


# ---------------------------------------------------------- the content gate

def test_a_null_placeholder_reply_is_not_usable():
    assert not reply_is_usable(json.dumps(
        {"present": True, "findings": [{"feature": "temperature", "value": None, "quote": None}]}))


def test_a_missing_quote_makes_a_reply_unusable():
    assert not reply_is_usable(json.dumps(
        {"present": True, "findings": [{"feature": "temperature", "value": 39.0}]}))


def test_an_empty_findings_list_is_perfectly_usable():
    assert reply_is_usable(json.dumps({"present": False, "findings": []}))


def test_unparseable_and_shapeless_replies_are_not_usable():
    assert not reply_is_usable("not json at all")
    assert not reply_is_usable(json.dumps({"present": True}))
    assert not reply_is_usable(json.dumps({"findings": "temperature"}))


# --------------------------------------------------------------- present_quote

def _verify(payload, note=NOTE):
    t = Tally()
    verify(json.dumps(payload), note, t)
    return t


def test_a_quoted_present_is_counted_as_grounded():
    t = _verify({"present": True, "findings": [],
                 "present_quote": "presented with chest pain"})
    assert (t.present_true, t.present_quoted) == (1, 1)
    assert t.present_quote_unfound == 0 and t.present_unquoted == 0


def test_a_present_quote_that_is_not_in_the_note_is_counted():
    t = _verify({"present": True, "findings": [],
                 "present_quote": "the patient had a stroke"})
    assert (t.present_true, t.present_quote_unfound) == (1, 1)
    assert t.present_quoted == 0


def test_an_unquoted_present_is_counted_not_silently_accepted():
    """Stage 0's `present` is the only field in the reply escaping the §2 contract."""
    t = _verify({"present": True, "findings": []})
    assert (t.present_true, t.present_unquoted) == (1, 1)


def test_present_false_is_not_asked_to_prove_a_negative():
    t = _verify({"present": False, "findings": []})
    assert t.present_true == 0 and t.present_unquoted == 0


def test_the_evidence_field_reaches_the_detail_sheet():
    _, _, detail = verify(json.dumps({"present": True, "findings": [], "evidence": "reasoning"}),
                          NOTE, Tally())
    assert detail["evidence"] == "reasoning"


# --------------------------------------------------------------- prompt content

def test_the_rubric_definition_appears_only_from_stage_2a():
    frag = "new LOBAR pulmonary infiltrate"
    assert frag not in build_prompt(NOTE, "48", stage("1"))
    assert frag in build_prompt(NOTE, "48", stage("2a"))


def test_every_outcome_has_a_presence_brief_to_inject():
    missing = [n for n in TABLES if not presence_brief(n)]
    assert missing == []


def test_the_definition_reader_drops_citation_markers():
    d = outcome_definition("48")
    assert "[1]" not in d["definition"] and "`OR`" not in (d["criteria"] or "")


def test_death_from_another_cause_is_addressed_only_from_stage_2a():
    frag = "attributes the death to a different cause"
    assert frag not in build_prompt(NOTE, "48", stage("1"))
    assert frag in build_prompt(NOTE, "48", stage("2a"))


def test_the_death_rule_is_absent_where_no_row_grades_on_it():
    """Fever has no death row; asking for it there is a question with no consumer."""
    assert "death_attributed" not in build_prompt(NOTE, "36", stage("2a"))


def test_the_evidence_field_appears_only_from_stage_2b():
    assert '"evidence"' not in build_prompt(NOTE, "48", stage("2a"))
    assert '"evidence"' in build_prompt(NOTE, "48", stage("2b"))


@pytest.mark.parametrize("frag", [
    "## The episode",                       # episode scoping
    "remained afebrile",                    # negation
    "exchange transfusion was considered",  # hypotheticals
    "shortest continuous run",              # quote hardening
])
def test_precision_rules_appear_only_at_stage_3(frag):
    assert frag not in build_prompt(NOTE, "48", stage("2b"))
    assert frag in build_prompt(NOTE, "48", stage("3"))


def test_ordinal_cues_reach_the_features_that_carry_the_grade():
    p = build_prompt(NOTE, "48", stage("3"))
    assert "high-flow nasal cannula" in p and "erythrocytapheresis" in p
    assert "high-flow nasal cannula" not in build_prompt(NOTE, "48", stage("2b"))


def test_the_unit_stops_reading_as_an_instruction_to_convert():
    """Stage 0 renders the schema's unit as `in mg/dL` three lines above a rule
    forbidding conversion."""
    assert ", in %)" in feature_brief("fio2_pct", "48")
    assert "the pipeline converts to %" in feature_brief("fio2_pct", "48", stage("3"))


def test_a_number_must_be_quoted_with_its_unit_from_stage_3():
    """`unit_guard` only speaks when the quote carries a unit token, so the rule it
    depends on has to be a requirement rather than a suggestion."""
    assert "MUST contain both the number and its unit" in build_prompt(NOTE, "48", stage("3"))
