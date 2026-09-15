"""The rules that make this app different are the ones tested hardest here."""

import numpy as np
import pandas as pd
import pytest

import mora_engine as me
from mora_engine.interpret import (BOUNDARY_NOTE, interpret,
                                   language_violations)
from mora_engine.models import Design, VarType, Variable

SCORE = Variable("score", VarType.CONTINUOUS, unit="points")
ARM = Variable("arm", VarType.NOMINAL)


def run(frame, test, variables, design, ledger=None):
    return me.run(frame, test, variables, design, ledger=ledger)


# ---------------------------------------------------------------- structure

def test_three_to_five_sentences(big_effect, null_data, three_groups,
                                 linear_data, table_data):
    cases = [
        (big_effect, "welch_t", [SCORE, ARM]),
        (null_data, "welch_t", [SCORE, ARM]),
        (big_effect, "mann_whitney", [SCORE, ARM]),
        (three_groups, "one_way_anova",
         [SCORE, Variable("site", VarType.NOMINAL)]),
        (three_groups, "kruskal_wallis",
         [SCORE, Variable("site", VarType.NOMINAL)]),
        (linear_data, "pearson_correlation",
         [Variable("hours", VarType.CONTINUOUS),
          Variable("grade", VarType.CONTINUOUS, unit="marks")]),
        (table_data, "chi_square",
         [Variable("attended", VarType.NOMINAL),
          Variable("outcome", VarType.NOMINAL)]),
    ]
    for frame, test, variables in cases:
        _, interpretation = run(frame, test, variables, Design.CORRELATIONAL)
        assert 3 <= len(interpretation.sentences) <= 5, test
        assert interpretation.apa
        assert interpretation.result_rows


def test_output_is_deterministic(big_effect):
    first = run(big_effect, "welch_t", [SCORE, ARM], Design.EXPERIMENTAL)[1]
    second = run(big_effect, "welch_t", [SCORE, ARM], Design.EXPERIMENTAL)[1]
    assert first.plain_text == second.plain_text
    assert first.apa == second.apa
    assert first.provenance == second.provenance


def test_every_sentence_traces_to_a_computed_number(big_effect):
    _, interpretation = run(big_effect, "welch_t", [SCORE, ARM],
                            Design.EXPERIMENTAL)
    keys = {"analysis", "statistic", "p", "n", "effect", "raw_effect", "mde",
            "exaggeration", "design"}
    assert keys.issubset(interpretation.provenance.keys())
    assert interpretation.provenance["effect"]["value"] is not None


# ---------------------------------------------------------------- design

def test_correlational_never_uses_causal_language(big_effect, linear_data,
                                                  table_data):
    cases = [
        (big_effect, "welch_t", [SCORE, ARM]),
        (big_effect, "mann_whitney", [SCORE, ARM]),
        (linear_data, "pearson_correlation",
         [Variable("hours", VarType.CONTINUOUS),
          Variable("grade", VarType.CONTINUOUS, unit="marks")]),
        (table_data, "chi_square",
         [Variable("attended", VarType.NOMINAL),
          Variable("outcome", VarType.NOMINAL)]),
    ]
    for frame, test, variables in cases:
        _, interpretation = run(frame, test, variables, Design.CORRELATIONAL)
        assert language_violations(interpretation, Design.CORRELATIONAL) == [], (
            test, interpretation.plain_text)
        assert "associated with" in interpretation.plain_text


def test_correlational_states_the_reason(big_effect):
    _, interpretation = run(big_effect, "welch_t", [SCORE, ARM],
                            Design.CORRELATIONAL)
    assert "declared correlational" in interpretation.plain_text


def test_experimental_may_speak_of_an_effect(big_effect):
    _, interpretation = run(big_effect, "welch_t", [SCORE, ARM],
                            Design.EXPERIMENTAL)
    assert "effect of arm" in interpretation.plain_text
    assert language_violations(interpretation, Design.EXPERIMENTAL) == []


def test_quasi_experimental_warns_about_pre_existing_groups(big_effect):
    _, interpretation = run(big_effect, "welch_t", [SCORE, ARM],
                            Design.QUASI_EXPERIMENTAL)
    assert "as they already existed" in interpretation.plain_text


def test_no_analysis_claims_importance(big_effect, null_data):
    for frame, design in ((big_effect, Design.EXPERIMENTAL),
                          (null_data, Design.CORRELATIONAL)):
        _, interpretation = run(frame, "welch_t", [SCORE, ARM], design)
        assert language_violations(interpretation, design) == []
    assert "judgements for you" in BOUNDARY_NOTE


# ---------------------------------------------------------------- power

def test_null_result_gets_a_power_statement_not_a_shrug(null_data):
    result, interpretation = run(null_data, "welch_t", [SCORE, ARM],
                                 Design.EXPERIMENTAL)
    assert not result.significant
    text = interpretation.plain_text
    assert "not evidence that there is no effect" in text
    assert "80 times out of 100" in text
    assert result.power.mde is not None
    assert f"{result.power.mde:.2f}".lstrip("0") in text


def test_null_result_never_says_no_difference(null_data):
    _, interpretation = run(null_data, "welch_t", [SCORE, ARM],
                            Design.EXPERIMENTAL)
    lowered = interpretation.plain_text.lower()
    for phrase in ("no difference", "there is no effect between",
                   "the groups are the same", "equally"):
        assert phrase not in lowered


def test_underpowered_significant_result_is_flagged_as_inflated():
    rng = np.random.default_rng(2)
    # a design small enough that only an overstated estimate can clear .05
    frame = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        values = np.r_[rng.normal(50, 10, 10), rng.normal(56, 10, 10)]
        candidate = pd.DataFrame({"score": values,
                                  "arm": ["a"] * 10 + ["b"] * 10})
        result, interpretation = run(candidate, "welch_t", [SCORE, ARM],
                                     Design.EXPERIMENTAL)
        if result.significant:
            frame = candidate
            break
    assert frame is not None
    assert "overstate" in interpretation.plain_text
    assert result.power.exaggeration > 1.1


def test_well_powered_significant_result_is_not_flagged(big_effect):
    result, interpretation = run(big_effect, "welch_t", [SCORE, ARM],
                                 Design.EXPERIMENTAL)
    assert result.significant
    assert "overstate" not in interpretation.plain_text


def test_rank_tests_label_their_power_figure_as_approximate(big_effect):
    result, _ = run(big_effect, "mann_whitney", [SCORE, ARM],
                    Design.CORRELATIONAL)
    assert result.power.approximate


# ---------------------------------------------------------------- magnitude

def test_magnitude_comes_before_significance(big_effect):
    _, interpretation = run(big_effect, "welch_t", [SCORE, ARM],
                            Design.EXPERIMENTAL)
    first = interpretation.sentences[0]
    assert "points" in first
    assert "p = " not in first and "p < " not in first
    assert "significan" not in first.lower()


def test_units_of_the_measure_are_used(linear_data):
    _, interpretation = run(
        linear_data, "pearson_correlation",
        [Variable("hours", VarType.CONTINUOUS),
         Variable("grade", VarType.CONTINUOUS, unit="marks")],
        Design.CORRELATIONAL)
    assert "marks" in interpretation.sentences[0]


def test_apa_sentence_carries_the_essentials(big_effect):
    result, interpretation = run(big_effect, "welch_t", [SCORE, ARM],
                                 Design.EXPERIMENTAL)
    for fragment in ("t(", "p <", "g = ", "95% CI", "SD = "):
        assert fragment in interpretation.apa
