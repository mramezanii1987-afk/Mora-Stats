"""Counter semantics, threat checks and the test suggestion list."""

import numpy as np
import pandas as pd
import pytest

import mora_engine as me
from mora_engine import diagnostics as dx
from mora_engine.models import Design, VarType, Variable
from mora_engine.session import SessionLedger
from mora_engine.suggest import suggest_tests

SCORE = Variable("score", VarType.CONTINUOUS, unit="points")
ARM = Variable("arm", VarType.NOMINAL)
SITE = Variable("site", VarType.NOMINAL)


# ------------------------------------------------------------------ ledger

def test_counter_ignores_descriptives(big_effect):
    ledger = SessionLedger()
    me.run(big_effect, "descriptives", [SCORE], Design.CORRELATIONAL,
           ledger=ledger)
    assert ledger.project_count == 0


def test_same_question_counts_once(big_effect):
    ledger = SessionLedger()
    me.run(big_effect, "welch_t", [SCORE, ARM], Design.EXPERIMENTAL,
           ledger=ledger)
    me.run(big_effect, "student_t", [SCORE, ARM], Design.EXPERIMENTAL,
           ledger=ledger)
    me.run(big_effect, "mann_whitney", [SCORE, ARM], Design.EXPERIMENTAL,
           ledger=ledger)
    assert ledger.project_count == 1


def test_a_different_filter_is_a_different_test(big_effect):
    ledger = SessionLedger()
    me.run(big_effect, "welch_t", [SCORE, ARM], Design.EXPERIMENTAL,
           ledger=ledger)
    me.run(big_effect, "welch_t", [SCORE, ARM], Design.EXPERIMENTAL,
           ledger=ledger, filter_key="score > 50")
    assert ledger.project_count == 2


def test_statement_appears_from_the_third_test(big_effect, three_groups,
                                               linear_data):
    ledger = SessionLedger()
    _, first = me.run(big_effect, "welch_t", [SCORE, ARM],
                      Design.CORRELATIONAL, ledger=ledger)
    assert first.multiplicity is None
    _, second = me.run(three_groups, "one_way_anova", [SCORE, SITE],
                       Design.CORRELATIONAL, ledger=ledger)
    assert second.multiplicity is None
    _, third = me.run(
        linear_data, "pearson_correlation",
        [Variable("hours", VarType.CONTINUOUS),
         Variable("grade", VarType.CONTINUOUS)],
        Design.CORRELATIONAL, ledger=ledger)
    assert third.multiplicity is not None
    assert "inferential test 3" in third.multiplicity
    assert "14%" in third.multiplicity  # 1 - .95 ** 3


def test_family_wise_rate_arithmetic():
    ledger = SessionLedger()
    assert ledger.family_wise_rate(1) == pytest.approx(0.05)
    assert ledger.family_wise_rate(10) == pytest.approx(1 - 0.95 ** 10)


def test_holm_and_bh_match_the_textbook_definition():
    ledger = SessionLedger()
    raw = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.5}
    for key, p in raw.items():
        ledger.entries[key] = type(
            "E", (), {"key": key, "label": "t", "variables": [], "p": p,
                      "session": 1})()
    holm = ledger.holm()
    assert holm["a"] == pytest.approx(0.04)
    assert holm["b"] == pytest.approx(0.06)
    assert holm["c"] == pytest.approx(0.06)
    bh = ledger.benjamini_hochberg()
    assert bh["a"] == pytest.approx(0.04)
    assert bh["c"] == pytest.approx(0.04)
    assert bh["d"] == pytest.approx(0.5)


def test_correction_offer_cannot_be_switched_off(big_effect, three_groups,
                                                 linear_data):
    ledger = SessionLedger()
    for frame, test, variables in (
            (big_effect, "welch_t", [SCORE, ARM]),
            (three_groups, "one_way_anova", [SCORE, SITE]),
            (linear_data, "pearson_correlation",
             [Variable("hours", VarType.CONTINUOUS),
              Variable("grade", VarType.CONTINUOUS)])):
        _, interpretation = me.run(frame, test, variables,
                                   Design.CORRELATIONAL, ledger=ledger)
    assert "Holm" in interpretation.multiplicity
    assert "Benjamini-Hochberg" in interpretation.multiplicity


# ------------------------------------------------------------------ threats

def test_ceiling_effect_is_detected(ceiling_data, scale_var):
    result, _ = me.run(ceiling_data, "welch_t",
                       [scale_var, Variable("group", VarType.NOMINAL)],
                       Design.CORRELATIONAL)
    codes = [t.code for t in result.threats]
    assert "ceiling" in codes


def test_undeclared_bounds_produce_a_note_not_a_guess(ceiling_data):
    undeclared = Variable("satisfaction", VarType.ORDINAL)
    result, _ = me.run(ceiling_data, "welch_t",
                       [undeclared, Variable("group", VarType.NOMINAL)],
                       Design.CORRELATIONAL)
    codes = [t.code for t in result.threats]
    assert "bounds_unknown" in codes
    assert "ceiling" not in codes


def test_small_cells_are_flagged(null_data):
    result, _ = me.run(null_data, "welch_t", [SCORE, ARM],
                       Design.EXPERIMENTAL)
    assert "small_cell" in [t.code for t in result.threats]


def test_unbalanced_groups_are_flagged():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({
        "score": np.r_[rng.normal(50, 10, 60), rng.normal(52, 10, 25)],
        "arm": ["a"] * 60 + ["b"] * 25})
    result, _ = me.run(frame, "welch_t", [SCORE, ARM], Design.EXPERIMENTAL)
    assert "unbalanced" in [t.code for t in result.threats]


def test_expected_counts_below_five_are_flagged():
    frame = pd.DataFrame({
        "a": ["x"] * 12 + ["y"] * 6,
        "b": ["p"] * 9 + ["q"] * 3 + ["p"] * 4 + ["q"] * 2})
    result, _ = me.run(frame, "chi_square",
                       [Variable("a", VarType.NOMINAL),
                        Variable("b", VarType.NOMINAL)],
                       Design.CORRELATIONAL)
    assert "expected_counts" in [t.code for t in result.threats]


def test_an_influential_case_is_found():
    rng = np.random.default_rng(21)
    values = np.r_[rng.normal(50, 5, 25), rng.normal(50, 5, 25)]
    values[-1] = 140.0
    frame = pd.DataFrame({"score": values, "arm": ["a"] * 25 + ["b"] * 25})
    result, _ = me.run(frame, "welch_t", [SCORE, ARM], Design.EXPERIMENTAL)
    assert "influential_case" in [t.code for t in result.threats]


def test_every_threat_states_an_action(ceiling_data, scale_var, null_data):
    frames = [
        (ceiling_data, [scale_var, Variable("group", VarType.NOMINAL)]),
        (null_data, [SCORE, ARM]),
    ]
    for frame, variables in frames:
        result, _ = me.run(frame, "welch_t", variables, Design.CORRELATIONAL)
        for threat in result.threats:
            assert threat.action.strip(), threat.code
            assert len(threat.action) > 25, threat.code


def test_missing_values_are_counted_and_reported():
    rng = np.random.default_rng(4)
    values = np.r_[rng.normal(50, 10, 40), rng.normal(55, 10, 40)]
    values[:8] = np.nan
    frame = pd.DataFrame({"score": values, "arm": ["a"] * 40 + ["b"] * 40})
    result, interpretation = me.run(frame, "welch_t", [SCORE, ARM],
                                    Design.EXPERIMENTAL)
    assert result.n_missing == 8
    assert "missing" in [t.code for t in result.threats]
    assert "8 dropped as missing" in " ".join(interpretation.result_rows)


# ------------------------------------------------------------------ suggest

def test_suggestion_greys_out_the_wrong_test_with_a_reason(three_groups):
    suggestions = {s.test: s for s in suggest_tests(three_groups,
                                                    [SCORE, SITE])}
    assert suggestions["welch_t"].eligible is False
    assert "3 groups" in suggestions["welch_t"].reason
    assert suggestions["one_way_anova"].eligible is True
    assert suggestions["one_way_anova"].recommended is True


def test_two_groups_default_to_welch(big_effect):
    suggestions = {s.test: s for s in suggest_tests(big_effect, [SCORE, ARM])}
    assert suggestions["welch_t"].recommended is True
    assert suggestions["mann_whitney"].recommended is False
    assert suggestions["one_way_anova"].eligible is False


def test_skewed_data_switches_the_recommendation_to_ranks():
    rng = np.random.default_rng(8)
    skewed = np.r_[rng.exponential(5, 30) ** 2, rng.exponential(6, 30) ** 2]
    frame = pd.DataFrame({"score": skewed, "arm": ["a"] * 30 + ["b"] * 30})
    suggestions = {s.test: s for s in suggest_tests(frame, [SCORE, ARM])}
    assert suggestions["mann_whitney"].recommended is True
    assert suggestions["welch_t"].recommended is False
    assert "normal shape" in suggestions["mann_whitney"].reason


def test_two_nominal_variables_get_chi_square(table_data):
    suggestions = {s.test: s for s in suggest_tests(
        table_data, [Variable("attended", VarType.NOMINAL),
                     Variable("outcome", VarType.NOMINAL)])}
    assert suggestions["chi_square"].recommended is True


def test_ordinal_outcome_prefers_ranks(ceiling_data, scale_var):
    suggestions = {s.test: s for s in suggest_tests(
        ceiling_data, [scale_var, Variable("group", VarType.NOMINAL)])}
    assert suggestions["mann_whitney"].recommended is True
    assert "ordinal" in suggestions["mann_whitney"].reason
