"""Statistical correctness, checked against published or hand-computed values.

Where a classic dataset has printed values in the literature, the test uses
those. Where it does not, the expected value is computed from the definition
inside the test rather than from the same code path the engine uses.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import mora_engine as me
from mora_engine import effects, power
from mora_engine.models import Design, VarType, Variable


def test_sleep_data_matches_published_values(sleep_data, sleep_vars):
    outcome, group = sleep_vars
    result, _ = me.run(sleep_data, "welch_t", [outcome, group],
                       Design.EXPERIMENTAL)
    assert result.statistic == pytest.approx(-1.8608, abs=1e-4)
    assert result.df == pytest.approx(17.776, abs=1e-3)
    assert result.p == pytest.approx(0.0794, abs=1e-4)

    student, _ = me.run(sleep_data, "student_t", [outcome, group],
                        Design.EXPERIMENTAL)
    assert student.statistic == pytest.approx(-1.8608, abs=1e-4)
    assert student.df == 18


def test_hedges_g_matches_definition(sleep_data):
    a = sleep_data.loc[sleep_data.drug == "drug 1", "extra"].to_numpy()
    b = sleep_data.loc[sleep_data.drug == "drug 2", "extra"].to_numpy()
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                 / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    assert effects.hedges_g(a, b) == pytest.approx(d * j, rel=1e-12)


def test_hedges_g_is_zero_for_identical_groups():
    a = np.array([1.0, 2, 3, 4, 5])
    assert effects.hedges_g(a, a.copy()) == pytest.approx(0.0)


def test_hedges_g_interval_brackets_the_estimate(big_effect):
    a = big_effect.loc[big_effect.arm == "control", "score"].to_numpy()
    b = big_effect.loc[big_effect.arm == "treatment", "score"].to_numpy()
    g = effects.hedges_g(a, b)
    lo, hi = effects.hedges_g_ci(a, b)
    assert lo < g < hi
    assert hi - lo < 1.0


def test_chi_square_matches_the_two_by_two_formula(table_data):
    result, _ = me.run(
        table_data, "chi_square",
        [Variable("attended", VarType.NOMINAL),
         Variable("outcome", VarType.NOMINAL)],
        Design.CORRELATIONAL)
    table = pd.crosstab(table_data.attended, table_data.outcome).to_numpy()
    (a, b), (c, d) = table
    n = a + b + c + d
    expected = n * (a * d - b * c) ** 2 / (
        (a + b) * (c + d) * (a + c) * (b + d))
    assert result.statistic == pytest.approx(expected, rel=1e-10)
    assert result.effect.value == pytest.approx(np.sqrt(expected / n),
                                                rel=1e-10)


def test_anova_sums_of_squares(three_groups):
    arrays = [three_groups.loc[three_groups.site == s, "score"].to_numpy()
              for s in ("east", "north", "south")]
    parts = effects.anova_effects(arrays)
    grand = np.concatenate(arrays).mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in arrays)
    assert parts["ss_between"] == pytest.approx(ss_between, rel=1e-10)
    assert parts["omega_squared"] < parts["eta_squared"]


def test_correlation_recovers_a_perfect_line():
    x = np.arange(1.0, 21.0)
    frame = pd.DataFrame({"x": x, "y": 2.5 * x + 4})
    result, _ = me.run(frame, "pearson_correlation",
                       [Variable("x", VarType.CONTINUOUS),
                        Variable("y", VarType.CONTINUOUS)],
                       Design.CORRELATIONAL)
    assert result.effect.value == pytest.approx(1.0)
    assert result.raw_effect.value == pytest.approx(2.5)


def test_rank_biserial_bounds():
    a = np.array([10.0, 11, 12, 13])
    b = np.array([1.0, 2, 3, 4])
    assert effects.rank_biserial(a, b) == pytest.approx(1.0)
    assert effects.rank_biserial(b, a) == pytest.approx(-1.0)


def test_epsilon_squared_definition():
    assert effects.epsilon_squared(10.0, 50) == pytest.approx(
        10.0 * 51 / (2500 - 1))


@pytest.mark.parametrize("n", [6, 10, 25, 40, 120, 500])
def test_mde_round_trips_to_target_power(n):
    mde = power.mde_two_sample_t(n, n)
    assert mde is not None
    assert power.power_two_sample_t(mde, n, n) == pytest.approx(0.80, abs=1e-3)


def test_mde_falls_as_sample_grows():
    values = [power.mde_two_sample_t(n, n) for n in (10, 30, 100, 300)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_exaggeration_is_near_one_when_well_powered():
    ratio = power.exaggeration_t(0.8, 400, 400)
    assert ratio == pytest.approx(1.0, abs=0.02)


def test_exaggeration_is_large_when_underpowered():
    ratio = power.exaggeration_t(0.3, 15, 15)
    assert ratio > 1.5


def test_bootstrap_is_reproducible(big_effect):
    a = big_effect.loc[big_effect.arm == "control", "score"].to_numpy()
    b = big_effect.loc[big_effect.arm == "treatment", "score"].to_numpy()
    first = effects.bootstrap_ci(effects.hedges_g, [a, b])
    second = effects.bootstrap_ci(effects.hedges_g, [a, b])
    assert first == second


def test_every_analysis_produces_an_effect_with_an_interval(
        big_effect, three_groups, linear_data, table_data):
    score = Variable("score", VarType.CONTINUOUS, unit="points")
    cases = [
        (big_effect, "welch_t", [score, Variable("arm", VarType.NOMINAL)]),
        (big_effect, "mann_whitney", [score, Variable("arm", VarType.NOMINAL)]),
        (three_groups, "one_way_anova",
         [score, Variable("site", VarType.NOMINAL)]),
        (three_groups, "kruskal_wallis",
         [score, Variable("site", VarType.NOMINAL)]),
        (linear_data, "pearson_correlation",
         [Variable("hours", VarType.CONTINUOUS),
          Variable("grade", VarType.CONTINUOUS)]),
        (linear_data, "spearman_correlation",
         [Variable("hours", VarType.CONTINUOUS),
          Variable("grade", VarType.CONTINUOUS)]),
        (table_data, "chi_square",
         [Variable("attended", VarType.NOMINAL),
          Variable("outcome", VarType.NOMINAL)]),
    ]
    for frame, test, variables in cases:
        result, _ = me.run(frame, test, variables, Design.CORRELATIONAL)
        assert result.effect is not None, test
        assert result.effect.ci_low is not None, test
        assert result.effect.ci_high is not None, test
        assert result.effect.ci_low <= result.effect.value + 1e-9, test
        assert result.effect.ci_high >= result.effect.value - 1e-9, test
        assert result.power.mde is not None, test


def test_noncentral_intervals_exclude_zero_when_the_test_is_significant(
        three_groups, table_data):
    """Regression test for a lower bound that always collapsed to zero."""
    anova, _ = me.run(three_groups, "one_way_anova",
                      [Variable("score", VarType.CONTINUOUS),
                       Variable("site", VarType.NOMINAL)],
                      Design.CORRELATIONAL)
    assert anova.p < 0.05
    assert anova.effect.ci_low > 0
    assert anova.effect.ci_low < anova.effect.value < anova.effect.ci_high

    chi, _ = me.run(table_data, "chi_square",
                    [Variable("attended", VarType.NOMINAL),
                     Variable("outcome", VarType.NOMINAL)],
                    Design.CORRELATIONAL)
    assert chi.p < 0.05
    assert chi.effect.ci_low > 0


def test_noncentral_interval_includes_zero_when_the_test_is_not():
    frame = pd.DataFrame({
        "a": ["x"] * 20 + ["y"] * 20,
        "b": (["p"] * 10 + ["q"] * 10) * 2,
    })
    result, _ = me.run(frame, "chi_square",
                       [Variable("a", VarType.NOMINAL),
                        Variable("b", VarType.NOMINAL)],
                       Design.CORRELATIONAL)
    assert result.p > 0.05
    assert result.effect.ci_low == pytest.approx(0.0, abs=1e-8)
